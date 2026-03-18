package com.example.flutter_local_gemma

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import android.system.Os
import android.system.OsConstants
import android.util.Log
import com.google.ai.edge.litertlm.*
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile

/**
 * GemmaPlugin
 *
 * Manages the LiteRT-LM inference engine lifecycle for on-device Gemma 3 text generation.
 *
 * ## Lifecycle
 * Flutter calls these methods in order:
 *   1. `createModel`   – loads the model binary from disk into an [Engine]
 *   2. `createSession` – creates a [Conversation] with sampling parameters
 *   3. `addQueryChunk` / `addImage` / `addAudio` – accumulates content for the next turn
 *   4. `generateResponseAsync` – streams tokens via [EventChannel] ("gemma_stream")
 *      OR `generateResponseSync` – returns the full string in one call
 *   5. `clearContext`  – resets the conversation (keeps the engine loaded)
 *   6. `closeModel`    – fully unloads engine + conversation from memory
 *
 * ## Local file loading — two strategies, tried in order
 *
 * When [createModel] receives a `content://` URI (Android Storage Access Framework):
 *
 * ### Strategy A — True zero-copy (preferred)
 * `ContentResolver.openFileDescriptor(uri, "r")` opens the file's own FD.
 * For files on real local storage (Downloads, Documents, internal storage, SD card)
 * the provider hands back an FD to the actual ext4/F2FS inode.
 * We verify with `Os.fstat()` that it is a regular file, then call
 * `Os.readlink("/proc/self/fd/<N>")` to obtain the **real on-disk path**
 * (e.g. `/storage/emulated/0/Download/gemma-3n-int4.litertlm`).
 * The FD is immediately closed and the real path is handed to LiteRT.
 * **No bytes are written to app storage. Zero storage overhead. Instant.**
 *
 * > Why not pass `/proc/self/fd/<N>` directly?
 * > LiteRT's native JNI selects the format parser by file extension.
 * > A path like `/proc/self/fd/185` has no extension, so LiteRT returns
 * > `INVALID_ARGUMENT: Unsupported or unknown file format` even though the
 * > inode is valid. `readlink` gives us the real path with the correct extension.
 *
 * ### Strategy B — Copy + delete after init (fallback for cloud/MTP)
 * Used when Strategy A fails because the provider uses a pipe or virtual FS
 * (Google Drive, MTP, cloud-backed files) — `fstat()` returns a non-regular mode.
 * We copy to `filesDir/saf_tmp/` keeping the original filename (preserving the
 * `.litertlm` extension), then **delete the temp file immediately after
 * `Engine.initialize()` succeeds**.
 * Blocks freed when engine closes (LiteRT mmap) or immediately (RAM load).
 * A `gemma_copy_progress` EventChannel emits [0, 100] during the copy.
 * Strategy A emits nothing (no copy needed).
 */
class GemmaPlugin : FlutterPlugin, MethodChannel.MethodCallHandler, EventChannel.StreamHandler {

    companion object {
        private const val TAG                   = "GemmaPlugin"
        private const val METHOD_CHANNEL        = "gemma_bundled"
        private const val EVENT_CHANNEL         = "gemma_stream"
        private const val COPY_PROGRESS_CHANNEL = "gemma_copy_progress"

        /** Minimum valid model size (50 MB). Rejects empty / corrupt files early. */
        private const val MIN_MODEL_BYTES = 50L * 1024 * 1024

        /** Bytes read when sniffing the file header for HTML/JSON detection. */
        private const val HEADER_SNIFF_BYTES = 16

        /** Copy-buffer size: 8 MB gives good throughput without RAM pressure. */
        private const val COPY_BUFFER_BYTES = 8 * 1024 * 1024
    }

    // ── Flutter channels ──────────────────────────────────────────────────────
    private lateinit var methodChannel:       MethodChannel
    private lateinit var eventChannel:        EventChannel
    private lateinit var copyProgressChannel: EventChannel
    private lateinit var appContext:          Context

    /** Token-stream sink. Null when no Flutter listener is subscribed. */
    @Volatile private var eventSink: EventChannel.EventSink? = null

    /** Copy-progress sink (0–100). Null when no Flutter listener is subscribed. */
    @Volatile private var copyProgressSink: EventChannel.EventSink? = null

    // ── SAF / FD state ────────────────────────────────────────────────────────

    // openSafFd removed: Strategy A now resolves the real path via Os.readlink()
    // and closes the FD immediately — no FD is held open across initialize().

    /**
     * Temp-file reference kept for Strategy C (copy → real path → delete after init).
     *
     * Set in [resolveModelPath] when Strategies A and B both fail.
     * Deleted in [loadEngine] after `Engine.initialize()` succeeds, or in the
     * error path. Null in all other states.
     */
    @Volatile private var pendingTempFile: File? = null

    /**
     * The filename of the model copy placed in `saf_tmp/` by Strategy B
     * (e.g. `"gemma-3n-E2B-it-int4.litertlm"`). Used after [engine.close()] to
     * delete the OpenCL GPU-kernel cache files that LiteRT writes alongside the
     * model — named `<stem>_<hash>.bin` — which can reach 1–2 GB for a
     * 3.5 GB model and are useless once the temp copy is gone.
     *
     * Set in [resolveModelPath] (Strategy B only). Cleared in
     * [deleteGpuCacheFiles] after engine unload.
     */
    @Volatile private var pendingModelStem: String? = null

    // ── Coroutine scope ───────────────────────────────────────────────────────
    private val supervisorJob = SupervisorJob()
    private val ioScope       = CoroutineScope(supervisorJob + Dispatchers.IO)

    /** Holds the currently active streaming-generation coroutine. */
    private var generationJob: Job? = null

    // ── LiteRT-LM state ───────────────────────────────────────────────────────
    /** Non-null only between a successful `createModel` and `closeModel`. */
    private var engine: Engine? = null

    /** Non-null only between `createSession` and `clearContext` / `closeModel`. */
    private var conversation: Conversation? = null

    // ── Per-turn content buffer ───────────────────────────────────────────────
    private val pendingContent = mutableListOf<Content>()

    // ── Auto-stop / loop detection ────────────────────────────────────────────
    private var autoStopEnabled  = true
    private var maxRepetitions   = 5
    private val minRepeatCharLen = 3
    private val responseBuffer   = StringBuilder()

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext    = binding.applicationContext
        methodChannel = MethodChannel(binding.binaryMessenger, METHOD_CHANNEL)
        methodChannel.setMethodCallHandler(this)

        eventChannel = EventChannel(binding.binaryMessenger, EVENT_CHANNEL)
        eventChannel.setStreamHandler(this)

        // Register a separate stream handler for copy-progress updates.
        copyProgressChannel = EventChannel(binding.binaryMessenger, COPY_PROGRESS_CHANNEL)
        copyProgressChannel.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                copyProgressSink = events
            }
            override fun onCancel(arguments: Any?) {
                copyProgressSink = null
            }
        })
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        copyProgressSink = null
        ioScope.launch { releaseNativeResources() }.invokeOnCompletion {
            supervisorJob.cancel()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel.MethodCallHandler
    // ─────────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {

            "createModel" -> {
                val modelPath    = call.argument<String>("modelPath") ?: ""
                val maxTokens    = call.argument<Int>("maxTokens") ?: 1024
                val backendIdx   = call.argument<Int>("preferredBackend") ?: 0
                val supportAudio = call.argument<Boolean>("supportAudio") ?: false
                ioScope.launch {
                    safeResultCall(result) {
                        loadEngine(modelPath, maxTokens, backendIdx, supportAudio)
                        null
                    }
                }
            }

            "closeModel" -> {
                ioScope.launch {
                    safeResultCall(result) {
                        releaseNativeResources()
                        Log.i(TAG, "Model fully unloaded.")
                        null
                    }
                }
            }

            /**
             * Deletes orphaned files left by interrupted loads.
             * Returns bytes freed as Long.
             */
            "purgeModelCache" -> {
                ioScope.launch {
                    safeResultCall(result) { purgeCachedModels() }
                }
            }

            "createSession" -> {
                val temperature  = call.argument<Double>("temperature")?.toFloat() ?: 0.8f
                val topP         = call.argument<Double>("topP")?.toFloat() ?: 0.95f
                val topK         = call.argument<Int>("topK") ?: 40
                val systemPrompt = call.argument<String>("systemPrompt")
                autoStopEnabled  = call.argument<Boolean>("autoStopEnabled") ?: true
                maxRepetitions   = call.argument<Int>("maxRepetitions") ?: 5
                ioScope.launch {
                    safeResultCall(result) {
                        openConversation(temperature, topP, topK, systemPrompt)
                        null
                    }
                }
            }

            "addQueryChunk" -> {
                call.argument<String>("prompt")?.takeIf { it.isNotEmpty() }
                    ?.let { synchronized(pendingContent) { pendingContent.add(Content.Text(it)) } }
                result.success(null)
            }

            "addImage" -> {
                call.argument<ByteArray>("imageBytes")
                    ?.let { synchronized(pendingContent) { pendingContent.add(Content.ImageBytes(it)) } }
                result.success(null)
            }

            "addAudio" -> {
                call.argument<ByteArray>("audioBytes")
                    ?.let { synchronized(pendingContent) { pendingContent.add(Content.AudioBytes(it)) } }
                result.success(null)
            }

            "generateResponseAsync" -> {
                startStreamingGeneration()
                result.success(null)
            }

            "generateResponseSync" -> {
                ioScope.launch { safeResultCall(result) { generateBlocking() } }
            }

            "stopGeneration" -> {
                cancelGeneration()
                result.success(null)
            }

            "clearContext" -> {
                ioScope.launch {
                    safeResultCall(result) {
                        closeConversation()
                        pendingContent.clear()
                        responseBuffer.setLength(0)
                        null
                    }
                }
            }

            "countTokens" -> {
                val text            = call.argument<String>("text") ?: ""
                val imageCount      = call.argument<Int>("imageCount") ?: 0
                val audioDurationMs = call.argument<Int>("audioDurationMs") ?: 0
                val estimate = (imageCount * 257) +
                               (audioDurationMs / 150) +
                               (text.split(Regex("\\s+")).size * 1.3).toInt() +
                               (text.length * 0.15).toInt()
                result.success(estimate)
            }

            else -> result.notImplemented()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EventChannel.StreamHandler  (token-generation stream)
    // ─────────────────────────────────────────────────────────────────────────

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { eventSink = events }
    override fun onCancel(arguments: Any?) { cancelGeneration(); eventSink = null }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – engine lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Loads the LiteRT-LM engine.
     *
     * If [path] is a `content://` URI, [resolveModelPath] is called first.
     * See the class-level KDoc for the three strategies it may use.
     *
     * After `Engine.initialize()` succeeds:
     * - Strategies A/B: our FD is closed; LiteRT's mmap is the sole reference.
     * - Strategy C: the temp file is deleted immediately.
     *
     * On any error, all FDs and temp files are cleaned up before re-throwing.
     */
    private fun loadEngine(path: String, maxTokens: Int, backendIdx: Int, audio: Boolean) {
        if (engine != null) {
            Log.w(TAG, "Stale engine reference – releasing before reload.")
            releaseNativeResources()
        }

        val resolvedPath = resolveModelPath(path)

        try {
            // /proc/self/fd paths cannot be stat()'d by java.io.File (dir entry gone
            // for Strategies A/B), so skip file-level validation for those paths.
            if (!resolvedPath.startsWith("/proc/self/fd/")) {
                validateModelFile(resolvedPath)
            }

            try { System.loadLibrary("litertlm_jni") }
            catch (e: UnsatisfiedLinkError) { Log.w(TAG, "JNI auto-loaded by AAR: ${e.message}") }

            val backend = if (backendIdx == 1) Backend.GPU else Backend.CPU
            val config  = EngineConfig(
                modelPath     = resolvedPath,
                maxNumTokens  = maxTokens,
                backend       = backend,
                visionBackend = Backend.GPU,
                audioBackend  = if (audio) Backend.CPU else null,
            )

            val newEngine = Engine(config)
            newEngine.initialize()   // throws on failure → caught below
            engine = newEngine       // only reached on success

            // Delete the Strategy B temp file now that the engine has it open.
            // LiteRT likely mmap()'d it; storage freed when the engine closes.
            deletePendingTempFile("model loaded successfully")

            Log.i(TAG, "Engine ready – backend=$backend  maxTokens=$maxTokens")
        } catch (e: Exception) {
            deletePendingTempFile("init failed – cleaning up")
            deleteGpuCacheFiles("init failed – cleaning up")
            throw e
        }
    }

    private fun openConversation(temp: Float, topP: Float, topK: Int, systemPrompt: String?) {
        val eng = engine ?: throw IllegalStateException("Engine not initialised. Call createModel first.")
        closeConversation()
        // ConversationConfig only accepts an optional system Message.
        // temperature / topP / topK are sampling parameters that belong on the
        // generation call side; ConversationConfig does not expose them as
        // named constructor arguments in the LiteRT-LM SDK.
        val sysMsg: Message? = systemPrompt
            ?.takeIf { it.isNotBlank() }
            ?.let { Message.of(listOf(Content.Text(it))) }
        conversation = eng.createConversation(ConversationConfig(systemMessage = sysMsg))
        Log.i(TAG, "Session opened – temp=$temp  topP=$topP  topK=$topK")
    }

    private fun closeConversation() {
        try { conversation?.close() } catch (e: Exception) { Log.w(TAG, "conversation.close: ${e.message}") }
        conversation = null
        synchronized(pendingContent) { pendingContent.clear() }
        responseBuffer.setLength(0)
    }

    private fun releaseNativeResources() {
        cancelGeneration()
        closeConversation()
        try { engine?.close() } catch (e: Exception) { Log.w(TAG, "engine.close: ${e.message}") }
        engine = null
        deletePendingTempFile("releaseNativeResources")
        deleteGpuCacheFiles("releaseNativeResources")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – SAF URI → mmap-able path (three strategies)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Resolves a `content://` URI to a path that LiteRT can open and parse.
     *
     * For plain absolute paths this is a no-op.
     * For SAF `content://` URIs two strategies are tried in order:
     *
     * ```
     *  content:// URI
     *      │
     *      ├─ Strategy A (zero-copy): openFileDescriptor → fstat (S_ISREG) →
     *      │    Os.readlink("/proc/self/fd/N") → real on-disk path
     *      │    ✓ Returns the actual file path (e.g. /storage/emulated/0/Download/model.litertlm)
     *      │    ✓ No bytes written, no storage overhead, instant.
     *      │    ✗ Falls through if the provider uses a pipe/virtual FS (Google Drive, MTP).
     *      │
     *      └─ Strategy B (copy + delete after init): copy to filesDir/saf_tmp/ →
     *           return real path → loadEngine() deletes file after Engine.initialize().
     *           File present on disk only during the copy + init window.
     * ```
     *
     * ## Why not /proc/self/fd/N directly?
     * LiteRT's native JNI uses the path string to detect the model format
     * (by file extension). `/proc/self/fd/185` has no extension, so LiteRT
     * rejects it with "Unsupported or unknown file format" even though the
     * inode is valid. Using the real path (Strategy A) or a named temp file
     * (Strategy B) preserves the `.litertlm` / `.bin` extension.
     */
    private fun resolveModelPath(rawPath: String): String {
        if (!rawPath.startsWith("content://")) return rawPath

        val uri = Uri.parse(rawPath)

        // ══════════════════════════════════════════════════════════════════════
        // Strategy A — true zero-copy via real path resolution
        //
        // Open the URI's file descriptor, verify via fstat() that it is a
        // regular on-disk file, then resolve the actual file path by reading
        // the /proc/self/fd/<N> symlink. This gives us the real path
        // (e.g. /storage/emulated/0/Download/gemma-3n-int4.litertlm) which:
        //   • LiteRT can open directly (correct extension, no copies)
        //   • requires zero extra storage
        //   • works for Downloads, internal storage, SD card, OEM file managers
        //
        // Fails for cloud/virtual providers (Google Drive, MTP) where the FD
        // is a pipe — fstat() returns a non-regular mode, so we fall through.
        // ══════════════════════════════════════════════════════════════════════
        try {
            val pfd = appContext.contentResolver.openFileDescriptor(uri, "r")
                ?: throw IOException("ContentResolver returned null PFD for $uri")

            val fd  = pfd.detachFd()  // take raw ownership
            val jfd = buildJavaFd(fd)

            val stat      = Os.fstat(jfd)
            val isRegular = (stat.st_mode and OsConstants.S_IFMT) == OsConstants.S_IFREG
            val size      = stat.st_size

            if (!isRegular) {
                closeRawFd(fd)
                throw IOException(
                    "FD mode=0${stat.st_mode.toString(8)} — not a regular file " +
                    "(pipe/socket/virtual FS, e.g. Google Drive)"
                )
            }

            if (size < MIN_MODEL_BYTES) {
                closeRawFd(fd)
                throw IOException("File too small: $size bytes")
            }

            // Resolve the real on-disk path via the /proc/self/fd symlink.
            // This is the path LiteRT will receive — it keeps the original
            // file extension so LiteRT's format detector works correctly.
            val realPath: String = Os.readlink("/proc/self/fd/$fd")

            // Validate header (read via the still-open FD, then close it).
            val header = ByteArray(HEADER_SNIFF_BYTES)
            Os.read(jfd, header, 0, HEADER_SNIFF_BYTES)
            closeRawFd(fd)  // FD no longer needed — using real path from now on
            validateHeader(header, "Strategy-A")

            // Sanity-check the resolved path is actually readable.
            val f = File(realPath)
            if (!f.exists() || !f.canRead()) {
                throw IOException("Resolved path not readable: $realPath")
            }

            Log.i(TAG, "✓ Strategy A (zero-copy): ${size / 1024 / 1024} MB → $realPath")
            return realPath   // pass straight to LiteRT — no copy, no FD tricks

        } catch (e: Exception) {
            Log.w(TAG, "Strategy A failed (${e.message}) — falling back to Strategy B (copy).")
        }

        // ══════════════════════════════════════════════════════════════════════
        // Strategy B — copy to internal storage, delete after Engine.initialize()
        //
        // Used for cloud/virtual/MTP providers where Strategy A cannot get a
        // real on-disk path. We copy to filesDir/saf_tmp/ keeping the original
        // filename (so the .litertlm extension is preserved for LiteRT), then
        // record the file in pendingTempFile. loadEngine() deletes it the
        // instant Engine.initialize() returns — whether that succeeds or fails.
        //
        // The file is on disk only during the copy + init window. After init
        // the directory entry is gone and storage blocks are freed when the
        // engine closes (LiteRT likely mmap'd the file).
        // ══════════════════════════════════════════════════════════════════════
        val displayName = queryDisplayName(uri)
        val safTmpDir   = File(appContext.filesDir, "saf_tmp").also { it.mkdirs() }
        val tmpFile     = File(safTmpDir, displayName).also { if (it.exists()) it.delete() }

        copyUriToFile(uri, tmpFile)  // emits progress on gemma_copy_progress channel

        pendingTempFile  = tmpFile
        pendingModelStem = tmpFile.name  // track for GPU cache cleanup on unload
        Log.i(TAG, "✓ Strategy B (copy+delete-after-init): ${tmpFile.absolutePath}")
        Log.w(TAG, "  Temp file + GPU cache will be deleted after Engine.initialize().")
        return tmpFile.absolutePath
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – copy helper with progress reporting
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Copies [uri]'s content to [dest], emitting progress values in [0, 99]
     * on the `gemma_copy_progress` EventChannel as it goes.
     * (100 is emitted by [loadEngine] indirectly when init completes.)
     */
    private fun copyUriToFile(uri: Uri, dest: File) {
        val totalBytes: Long = try {
            appContext.contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1L
        } catch (_: Exception) { -1L }

        val sizeDesc = if (totalBytes > 0) "${totalBytes / 1024 / 1024} MB" else "unknown size"
        Log.i(TAG, "Copy start: $sizeDesc → ${dest.name}")
        emitCopyProgress(0.0)

        appContext.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output ->
                val buf     = ByteArray(COPY_BUFFER_BYTES)
                var copied  = 0L
                var lastPct = -1
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    output.write(buf, 0, n)
                    copied += n
                    if (totalBytes > 0) {
                        // Emit every ~2 % to keep event volume low.
                        val pct = (copied * 100.0 / totalBytes).toInt().coerceIn(0, 99)
                        if (pct >= lastPct + 2) {
                            lastPct = pct
                            emitCopyProgress(pct.toDouble())
                        }
                    }
                }
                output.flush()
            }
        } ?: throw IOException("ContentResolver returned null InputStream for: $uri")

        Log.i(TAG, "Copy done: ${dest.length() / 1024 / 1024} MB → ${dest.name}")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – raw FD helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Constructs a [java.io.FileDescriptor] that wraps the raw Linux int [fd].
     * Uses reflection because the `descriptor` field is private in the SDK.
     */
    private fun buildJavaFd(fd: Int): java.io.FileDescriptor {
        val jfd   = java.io.FileDescriptor()
        val field = java.io.FileDescriptor::class.java.getDeclaredField("descriptor")
        field.isAccessible = true
        field.setInt(jfd, fd)
        return jfd
    }

    /**
     * Closes a raw Linux FD obtained via [ParcelFileDescriptor.detachFd].
     * Safe to call even if already closed (logs a warning).
     */
    private fun closeRawFd(fd: Int) {
        try { Os.close(buildJavaFd(fd)) }
        catch (e: Exception) { Log.w(TAG, "closeRawFd($fd): ${e.message}") }
    }

    // closeSafFd() removed — no longer needed since Strategy A returns a real
    // path and closes the FD before returning, rather than keeping it open.

    // ─────────────────────────────────────────────────────────────────────────
    // Private – temp-file helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Deletes [pendingTempFile] if set and clears the reference. */
    private fun deletePendingTempFile(reason: String) {
        val f = pendingTempFile ?: return
        pendingTempFile = null
        if (!f.exists()) return
        val bytes = f.length()
        if (f.delete()) Log.i(TAG, "Temp file deleted ($reason): freed ${bytes / 1024 / 1024} MB")
        else            Log.w(TAG, "Could not delete temp file ($reason): ${f.absolutePath}")
    }

    /**
     * Deletes the OpenCL GPU-kernel cache `.bin` files that LiteRT writes into
     * `filesDir/saf_tmp/` while compiling GPU delegates for a Strategy-B model.
     *
     * LiteRT names these files `<modelFilename>_<uint64hash>.bin` in the same
     * directory as the model (the "serialization dir"). For a 3.5 GB model they
     * can total 1–2 GB. They only speed up *subsequent* loads of the same file;
     * since the model temp copy is deleted after every load, the cache is useless
     * and must be cleaned up to reclaim the storage.
     *
     * Called right after [Engine.close()]:  at that point the engine has released
     * all file handles, so the .bin files can be safely deleted.
     *
     * Only acts when [pendingModelStem] is set (i.e. Strategy B was used).
     * Clears [pendingModelStem] after running so it does not re-run on a
     * subsequent unload.
     */
    private fun deleteGpuCacheFiles(reason: String) {
        val stem = pendingModelStem ?: return
        pendingModelStem = null

        val safTmpDir = File(appContext.filesDir, "saf_tmp")
        if (!safTmpDir.exists()) return

        var freed = 0L
        // LiteRT cache files follow the pattern:  <modelFilename>_<uint64>.bin
        // Example: gemma-3n-E2B-it-int4.litertlm_6790221948153149531.bin
        safTmpDir.listFiles { f ->
            f.name.startsWith("${stem}_") && f.name.endsWith(".bin")
        }?.forEach { file ->
            val size = file.length()
            if (file.delete()) {
                freed += size
                Log.i(TAG, "GPU cache deleted ($reason): ${file.name} (${size / 1024 / 1024} MB)")
            } else {
                Log.w(TAG, "GPU cache delete failed ($reason): ${file.absolutePath}")
            }
        }

        if (freed > 0) Log.i(TAG, "GPU cache purge freed ${freed / 1024 / 1024} MB  ($reason).")
        else           Log.d(TAG, "GPU cache: no .bin files found for stem=$stem  ($reason).")
    }

    /**
     * Deletes orphaned files from `filesDir/saf_tmp/` and the legacy
     * `filesDir/models/` directory. Returns total bytes freed.
     */
    private fun purgeCachedModels(): Long {
        var freed = 0L
        listOf(
            File(appContext.filesDir, "saf_tmp"),
            File(appContext.filesDir, "models"),
        ).forEach { dir ->
            if (!dir.exists()) return@forEach
            dir.listFiles()?.forEach { file ->
                val size = file.length()
                if (file.delete()) {
                    freed += size
                    Log.i(TAG, "Purged: ${file.name} ($size bytes)")
                }
            }
        }
        if (freed > 0) Log.i(TAG, "Cache purge freed ${freed / 1024 / 1024} MB total.")
        return freed
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – validation helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Checks that [header] bytes do not look like an HTML / JSON error page.
     * Throws [IllegalArgumentException] if they do (common when a download
     * token is invalid and the server returns a 4xx HTML page).
     */
    private fun validateHeader(header: ByteArray, source: String) {
        val text = String(header).trim().lowercase()
        if (text.startsWith("<!doctype") || text.startsWith("<html") ||
            text.startsWith("<?xml")     || text.startsWith("{")) {
            throw IllegalArgumentException(
                "[$source] File header looks like HTML/JSON, not a LiteRT binary. " +
                "Check your download token or re-download the model."
            )
        }
    }

    /**
     * Validates a regular file at [path] for size, readability, and header.
     * Not called for `/proc/self/fd/` paths (directory entry already deleted).
     */
    private fun validateModelFile(path: String) {
        if (path.isEmpty()) throw IllegalArgumentException("Model path must not be empty.")
        val file = File(path)
        if (!file.exists() || !file.canRead())
            throw IllegalArgumentException("Model file not found or unreadable: $path")
        if (file.length() < MIN_MODEL_BYTES)
            throw IllegalArgumentException(
                "File too small (${file.length()} B). " +
                "A valid LiteRT model must be ≥ ${MIN_MODEL_BYTES / 1024 / 1024} MB."
            )
        RandomAccessFile(file, "r").use { raf ->
            val header = ByteArray(HEADER_SNIFF_BYTES)
            raf.read(header)
            validateHeader(header, path)
        }
    }

    /** Queries the display name for a content URI; falls back to a timestamp name. */
    private fun queryDisplayName(uri: Uri): String = try {
        appContext.contentResolver
            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
            ?: "model_${System.currentTimeMillis()}.litertlm"
    } catch (_: Exception) { "model_${System.currentTimeMillis()}.litertlm" }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – copy-progress event helper
    // ─────────────────────────────────────────────────────────────────────────

    /** Posts a progress value in [0, 100] to the Flutter copy-progress stream. */
    private fun emitCopyProgress(progress: Double) {
        val sink = copyProgressSink ?: return
        ioScope.launch(Dispatchers.Main) {
            try { sink.success(progress) } catch (_: Exception) {}
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – generation
    // ─────────────────────────────────────────────────────────────────────────

    private fun startStreamingGeneration() {
        val conv = conversation ?: run { sendStreamError("Session not initialised."); return }

        val parts: List<Content>
        synchronized(pendingContent) {
            if (pendingContent.isEmpty()) { sendStreamError("No content staged for this turn."); return }
            parts = pendingContent.toList()
            pendingContent.clear()
        }
        responseBuffer.setLength(0)

        generationJob?.cancel()
        generationJob = ioScope.launch {
            try {
                conv.sendMessageAsync(Message.of(parts))
                    .catch { e ->
                        if (e !is CancellationException && e.message?.contains("CANCELLED") == false)
                            withContext(Dispatchers.Main) { sendStreamError(e.message ?: "Generation error") }
                    }
                    .onCompletion { withContext(Dispatchers.Main) { sendStreamDone() } }
                    .collect { partial ->
                        val token = partial.toString()
                        if (autoStopEnabled && isRepetitionLoop(token)) {
                            Log.w(TAG, "Repetition loop detected – stopping.")
                            cancelGeneration()
                            withContext(Dispatchers.Main) {
                                sendStreamPartial(" [Stopped: repetition detected]")
                            }
                            this.cancel()
                        } else {
                            withContext(Dispatchers.Main) { sendStreamPartial(token) }
                        }
                    }
            } catch (_: CancellationException) {
                // Normal cancellation – swallow silently.
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected generation error: ${e.message}", e)
                withContext(Dispatchers.Main) { sendStreamError(e.message ?: "Unknown error") }
            }
        }
    }

    private fun generateBlocking(): String {
        val conv = conversation ?: throw IllegalStateException("Session not initialised.")
        val parts: List<Content>
        synchronized(pendingContent) { parts = pendingContent.toList(); pendingContent.clear() }
        return conv.sendMessage(Message.of(parts)).toString()
    }

    private fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
        try { conversation?.cancelProcess() } catch (e: Exception) { Log.w(TAG, "cancelProcess: ${e.message}") }
        ioScope.launch(Dispatchers.Main) { sendStreamDone() }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – loop detection
    // ─────────────────────────────────────────────────────────────────────────

    private fun isRepetitionLoop(chunk: String): Boolean {
        responseBuffer.append(chunk)
        if (responseBuffer.length < 20) return false
        val tail  = responseBuffer.takeLast(200)
        val regex = Regex("(.{$minRepeatCharLen,}?)\\1{${maxRepetitions - 1},}$")
        return regex.containsMatchIn(tail)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – EventSink helpers (must be called from Main thread)
    // ─────────────────────────────────────────────────────────────────────────

    private fun sendStreamPartial(text: String) {
        eventSink?.success(mapOf("partialResult" to text, "done" to false))
    }

    private fun sendStreamDone() {
        eventSink?.success(mapOf("partialResult" to "", "done" to true))
    }

    private fun sendStreamError(message: String) {
        eventSink?.success(mapOf("error" to message))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – safe coroutine result wrapper
    // ─────────────────────────────────────────────────────────────────────────

    private suspend fun <T> safeResultCall(
        result: MethodChannel.Result,
        errorCode: String = "GEMMA_ERROR",
        block: suspend () -> T,
    ) {
        try {
            val value = block()
            withContext(Dispatchers.Main) { result.success(value) }
        } catch (e: Exception) {
            Log.e(TAG, "$errorCode: ${e.message}", e)
            withContext(Dispatchers.Main) { result.error(errorCode, e.message, null) }
        }
    }
}