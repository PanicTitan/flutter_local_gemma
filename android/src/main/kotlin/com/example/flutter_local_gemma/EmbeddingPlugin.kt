package com.example.flutter_local_gemma

import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import android.system.Os
import android.system.OsConstants
import android.util.Log
import com.google.ai.edge.localagents.rag.models.EmbedData
import com.google.ai.edge.localagents.rag.models.EmbeddingRequest
import com.google.ai.edge.localagents.rag.models.GemmaEmbeddingModel
import com.google.common.collect.ImmutableList
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.*
import kotlinx.coroutines.guava.await
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * EmbeddingPlugin
 *
 * Wraps [GemmaEmbeddingModel] from the `localagents-rag` library to expose
 * on-device text embeddings to Flutter via a [MethodChannel].
 *
 * ## Lifecycle
 * Flutter calls:
 *   1. `initEmbeddingModel`  – loads model + tokeniser (resolving SAF URIs if needed).
 *   2. `getEmbedding`        – returns a `List<Double>` embedding vector for a text string.
 *   3. `closeEmbeddingModel` – unloads the model and frees native memory.
 *   4. `purgeModelCache`     – deletes orphaned files from the temp directory.
 *
 * ## Local file loading — two strategies, tried in order
 *
 * Both `modelPath` and `tokenizerPath` may arrive as `content://` URIs from the
 * Android file picker (SAF). When they do, [resolveFilePath] is called first:
 *
 * ### Strategy A — True zero-copy (preferred)
 * `ContentResolver.openFileDescriptor(uri, "r")` asks the provider for the file's
 * own FD. For files on real local storage (Downloads, Documents, SD card, internal
 * storage) the provider hands back an FD to the actual ext4/F2FS inode.
 * `Os.fstat()` confirms it is a regular file, then `Os.readlink("/proc/self/fd/<N>")`
 * resolves the **real on-disk path** (e.g. `.../Download/embed.tflite`).
 * The FD is closed immediately and the real path is passed to [GemmaEmbeddingModel].
 * **No bytes are written to app storage. Zero storage overhead. Instant.**
 *
 * ### Strategy B — Copy + delete after init (fallback for cloud/MTP)
 * Used when Strategy A fails because the provider uses a pipe or virtual FS
 * (Google Drive, MTP). We copy to `filesDir/emb_tmp/` preserving the original
 * filename, pass the real path to [GemmaEmbeddingModel], then **delete the temp
 * file immediately after the model loads**.
 * A `embedding_copy_progress` EventChannel emits [0, 100] during the copy.
 *
 * ## Threading model
 * A [SupervisorJob] + [Dispatchers.IO] scope ensures all heavy work stays off the
 * Flutter/UI thread. Coroutine failures are isolated by [SupervisorJob].
 *
 * ## Memory strategy
 * [GemmaEmbeddingModel] has no public `close()` method in the RAG API; nulling
 * the reference allows the GC to reclaim native buffers.
 */
class EmbeddingPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {

    companion object {
        private const val TAG                   = "EmbeddingPlugin"
        private const val CHANNEL               = "embedding_plugin"
        private const val COPY_PROGRESS_CHANNEL = "embedding_copy_progress"

        /** Copy-buffer size: 4 MB — embedding models are small, no need for 8 MB. */
        private const val COPY_BUFFER_BYTES = 4 * 1024 * 1024
    }

    // ── Flutter channels ──────────────────────────────────────────────────────
    private lateinit var channel:             MethodChannel
    private lateinit var copyProgressChannel: EventChannel
    private lateinit var appContext:          Context

    /** Copy-progress sink (0–100). Null when no Flutter listener is subscribed. */
    @Volatile private var copyProgressSink: EventChannel.EventSink? = null

    // ── Coroutine scope ───────────────────────────────────────────────────────
    private val supervisorJob = SupervisorJob()
    private val ioScope = CoroutineScope(supervisorJob + Dispatchers.IO)

    // ── Model state ───────────────────────────────────────────────────────────

    /** Non-null only between a successful `initEmbeddingModel` and `closeEmbeddingModel`. */
    @Volatile private var embeddingModel: GemmaEmbeddingModel? = null

    /**
     * Temp files (model copy and/or tokenizer copy) created by Strategy B.
     * Deleted immediately after [GemmaEmbeddingModel] is constructed successfully,
     * or on any error path, or on [unloadModel].
     */
    private val pendingTempFiles = mutableListOf<File>()

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext

        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)

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
        channel.setMethodCallHandler(null)
        copyProgressSink = null
        ioScope.launch { unloadModel() }.invokeOnCompletion { supervisorJob.cancel() }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel.MethodCallHandler
    // ─────────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {

            /**
             * Initialises the embedding model.
             *
             * Arguments:
             *   - `modelPath`     (String) – absolute path **or** `content://` URI
             *                               to the `.tflite` embedding model.
             *   - `tokenizerPath` (String) – absolute path **or** `content://` URI
             *                               to the SentencePiece tokeniser file.
             *   - `useGpu`        (Boolean, optional) – default false; ignored on emulators.
             */
            "initEmbeddingModel" -> {
                val modelPath     = call.argument<String>("modelPath")
                val tokenizerPath = call.argument<String>("tokenizerPath")
                val useGpu        = call.argument<Boolean>("useGpu") ?: false

                if (modelPath.isNullOrEmpty() || tokenizerPath.isNullOrEmpty()) {
                    result.error("INVALID_ARG", "modelPath and tokenizerPath must not be empty.", null)
                    return
                }

                ioScope.launch {
                    safeResultCall(result, "EMBED_INIT_ERROR") {
                        loadModel(modelPath, tokenizerPath, useGpu)
                        null
                    }
                }
            }

            /**
             * Generates a semantic embedding vector for the given text.
             *
             * Arguments:
             *   - `text` (String) – the input text to embed.
             *
             * Returns: `List<Double>` – the embedding vector.
             */
            "getEmbedding" -> {
                val text = call.argument<String>("text")
                if (text == null) {
                    result.error("INVALID_ARG", "text must not be null.", null)
                    return
                }
                ioScope.launch {
                    safeResultCall(result, "EMBED_ERROR") { computeEmbedding(text) }
                }
            }

            /**
             * Unloads the model and frees native memory.
             * Also deletes any Strategy-B temp files that somehow survived init.
             * Safe to call even when no model is loaded.
             */
            "closeEmbeddingModel" -> {
                ioScope.launch {
                    safeResultCall(result, "EMBED_CLOSE_ERROR") {
                        unloadModel()
                        null
                    }
                }
            }

            /**
             * Deletes all files from `filesDir/emb_tmp/`.
             * Returns bytes freed as Long.
             */
            "purgeModelCache" -> {
                ioScope.launch {
                    safeResultCall(result, "EMBED_PURGE_ERROR") { purgeEmbeddingCache() }
                }
            }

            else -> result.notImplemented()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – model lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Resolves both paths (model + tokenizer), constructs [GemmaEmbeddingModel],
     * then deletes any Strategy-B temp copies immediately.
     *
     * On any error, temp files are also cleaned up before re-throwing.
     */
    private fun loadModel(rawModelPath: String, rawTokenizerPath: String, useGpu: Boolean) {
        if (embeddingModel != null) {
            Log.w(TAG, "Embedding model already loaded. Call closeEmbeddingModel first.")
            return
        }

        // Resolve content:// URIs → real paths (or copies in emb_tmp/).
        // The tokenizer is always small so we don't emit copy-progress for it
        // (it's usually just a few MB and resolves instantly).
        val modelPath     = resolveFilePath(rawModelPath,     isTokenizer = false)
        val tokenizerPath = resolveFilePath(rawTokenizerPath, isTokenizer = true)

        try {
            val effectiveGpu = useGpu && !isRunningOnEmulator()
            if (useGpu && !effectiveGpu) Log.w(TAG, "Emulator detected — forcing CPU.")

            embeddingModel = GemmaEmbeddingModel(modelPath, tokenizerPath, effectiveGpu)
            Log.i(TAG, "Embedding model loaded — gpu=$effectiveGpu")

            // Strategy B cleanup: delete temp copies now that the model holds them.
            deletePendingTempFiles("model loaded successfully")

        } catch (e: Exception) {
            deletePendingTempFiles("init failed — cleaning up")
            throw e
        }
    }

    /**
     * Nulls the model reference so the GC can reclaim native buffers.
     * [GemmaEmbeddingModel] has no public `close()` so releasing the reference
     * is sufficient. Also cleans up any surviving temp files.
     */
    private fun unloadModel() {
        embeddingModel = null
        deletePendingTempFiles("unloadModel")
        Log.i(TAG, "Embedding model unloaded.")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – SAF URI → real path (Strategy A) or copy (Strategy B)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Resolves a raw path or `content://` URI to an absolute file-system path
     * that [GemmaEmbeddingModel] can open.
     *
     * For plain absolute paths this is a no-op.
     * For SAF `content://` URIs:
     *
     * ```
     *  content:// URI
     *      │
     *      ├─ Strategy A (zero-copy): openFileDescriptor → fstat (S_ISREG)
     *      │    → Os.readlink("/proc/self/fd/N") → real on-disk path
     *      │    ✓ No bytes written. Instant. Works for local storage.
     *      │    ✗ Falls through for pipes/virtual FS (Google Drive, MTP).
     *      │
     *      └─ Strategy B (copy + delete after init): copy to filesDir/emb_tmp/
     *           → real path → deleted by loadModel() after GemmaEmbeddingModel()
     * ```
     *
     * @param isTokenizer  When true, skips the model-size minimum check
     *                     (tokenizer files are legitimately small).
     */
    private fun resolveFilePath(rawPath: String, isTokenizer: Boolean): String {
        if (!rawPath.startsWith("content://")) return rawPath

        val uri = Uri.parse(rawPath)

        // ══════════════════════════════════════════════════════════════════════
        // Strategy A — zero-copy via real path resolution
        // ══════════════════════════════════════════════════════════════════════
        try {
            val pfd = appContext.contentResolver.openFileDescriptor(uri, "r")
                ?: throw IOException("ContentResolver returned null PFD for $uri")

            val fd  = pfd.detachFd()
            val jfd = buildJavaFd(fd)

            val stat      = Os.fstat(jfd)
            val isRegular = (stat.st_mode and OsConstants.S_IFMT) == OsConstants.S_IFREG

            if (!isRegular) {
                closeRawFd(fd)
                throw IOException(
                    "FD mode=0${stat.st_mode.toString(8)} — not a regular file " +
                    "(pipe/socket/virtual FS, e.g. Google Drive)"
                )
            }

            val realPath = Os.readlink("/proc/self/fd/$fd")
            closeRawFd(fd)   // FD no longer needed

            val f = File(realPath)
            if (!f.exists() || !f.canRead()) {
                throw IOException("Resolved path not readable: $realPath")
            }

            val label = if (isTokenizer) "tokenizer" else "model"
            Log.i(TAG, "✓ Strategy A ($label zero-copy): ${stat.st_size / 1024 / 1024} MB → $realPath")
            return realPath

        } catch (e: Exception) {
            val label = if (isTokenizer) "tokenizer" else "model"
            Log.w(TAG, "Strategy A failed for $label (${e.message}) — falling back to Strategy B.")
        }

        // ══════════════════════════════════════════════════════════════════════
        // Strategy B — copy to emb_tmp/ + delete after GemmaEmbeddingModel()
        // ══════════════════════════════════════════════════════════════════════
        val displayName = queryDisplayName(uri)
        val embTmpDir   = File(appContext.filesDir, "emb_tmp").also { it.mkdirs() }
        val tmpFile     = File(embTmpDir, displayName).also { if (it.exists()) it.delete() }

        // Only emit copy-progress events for the model (not the tiny tokenizer).
        copyUriToFile(uri, tmpFile, emitProgress = !isTokenizer)

        // Register for deletion after GemmaEmbeddingModel() is constructed.
        synchronized(pendingTempFiles) { pendingTempFiles.add(tmpFile) }

        val label = if (isTokenizer) "tokenizer" else "model"
        Log.i(TAG, "✓ Strategy B ($label copy+delete-after-init): ${tmpFile.absolutePath}")
        return tmpFile.absolutePath
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – copy helper with optional progress reporting
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Copies [uri]'s content to [dest].
     * If [emitProgress] is true, emits progress values in [0, 99] on the
     * `embedding_copy_progress` EventChannel.
     */
    private fun copyUriToFile(uri: Uri, dest: File, emitProgress: Boolean) {
        val totalBytes: Long = try {
            appContext.contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1L
        } catch (_: Exception) { -1L }

        val sizeDesc = if (totalBytes > 0) "${totalBytes / 1024 / 1024} MB" else "unknown size"
        Log.i(TAG, "Emb copy start: $sizeDesc → ${dest.name}")
        if (emitProgress) emitCopyProgress(0.0)

        appContext.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output ->
                val buf     = ByteArray(COPY_BUFFER_BYTES)
                var copied  = 0L
                var lastPct = -1
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    output.write(buf, 0, n)
                    copied += n
                    if (emitProgress && totalBytes > 0) {
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

        Log.i(TAG, "Emb copy done: ${dest.length() / 1024 / 1024} MB → ${dest.name}")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – temp-file helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Deletes all registered temp files and clears the list. */
    private fun deletePendingTempFiles(reason: String) {
        val toDelete: List<File>
        synchronized(pendingTempFiles) {
            toDelete = pendingTempFiles.toList()
            pendingTempFiles.clear()
        }
        var freed = 0L
        for (f in toDelete) {
            if (!f.exists()) continue
            val size = f.length()
            if (f.delete()) {
                freed += size
                Log.i(TAG, "Emb temp deleted ($reason): ${f.name} freed ${size / 1024 / 1024} MB")
            } else {
                Log.w(TAG, "Emb temp delete failed ($reason): ${f.absolutePath}")
            }
        }
        if (freed > 0) Log.i(TAG, "Emb temp cleanup freed ${freed / 1024 / 1024} MB  ($reason).")
    }

    /**
     * Deletes all files from `filesDir/emb_tmp/`.
     * Called by the `purgeModelCache` method-channel handler.
     * Returns bytes freed.
     */
    private fun purgeEmbeddingCache(): Long {
        var freed = 0L
        val embTmpDir = File(appContext.filesDir, "emb_tmp")
        if (!embTmpDir.exists()) return 0L
        embTmpDir.listFiles()?.forEach { file ->
            val size = file.length()
            if (file.delete()) {
                freed += size
                Log.i(TAG, "Emb purge: ${file.name} (${size / 1024 / 1024} MB)")
            }
        }
        if (freed > 0) Log.i(TAG, "Emb cache purge freed ${freed / 1024 / 1024} MB total.")
        return freed
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – raw FD helpers  (identical pattern to GemmaPlugin)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Wraps a raw Linux int [fd] in a [java.io.FileDescriptor] via reflection
     * so it can be passed to [Os] syscalls.
     */
    private fun buildJavaFd(fd: Int): java.io.FileDescriptor {
        val jfd   = java.io.FileDescriptor()
        val field = java.io.FileDescriptor::class.java.getDeclaredField("descriptor")
        field.isAccessible = true
        field.setInt(jfd, fd)
        return jfd
    }

    /** Closes a raw Linux FD obtained via [android.os.ParcelFileDescriptor.detachFd]. */
    private fun closeRawFd(fd: Int) {
        try { Os.close(buildJavaFd(fd)) }
        catch (e: Exception) { Log.w(TAG, "closeRawFd($fd): ${e.message}") }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – misc helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Queries the display name for a content URI; falls back to a timestamp name. */
    private fun queryDisplayName(uri: Uri): String = try {
        appContext.contentResolver
            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
            ?: "embedding_${System.currentTimeMillis()}.tflite"
    } catch (_: Exception) { "embedding_${System.currentTimeMillis()}.tflite" }

    /** Posts a progress value in [0, 100] to the Flutter copy-progress stream. */
    private fun emitCopyProgress(progress: Double) {
        val sink = copyProgressSink ?: return
        ioScope.launch(Dispatchers.Main) {
            try { sink.success(progress) } catch (_: Exception) {}
        }
    }

    /**
     * Heuristic emulator check. Covers the stock Android emulator (goldfish/ranchu)
     * and most CI / cloud virtual device setups.
     */
    private fun isRunningOnEmulator(): Boolean =
        Build.FINGERPRINT.startsWith("generic") ||
        Build.MODEL.contains("Emulator", ignoreCase = true) ||
        Build.PRODUCT.contains("sdk", ignoreCase = true) ||
        Build.HARDWARE.contains("goldfish", ignoreCase = true) ||
        Build.HARDWARE.contains("ranchu", ignoreCase = true)

    // ─────────────────────────────────────────────────────────────────────────
    // Private – embedding computation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs the embedding model over [text] and returns the float vector as
     * `List<Double>` (Flutter's standard number type).
     *
     * Uses [SEMANTIC_SIMILARITY] task type, appropriate for RAG retrieval.
     *
     * @throws IllegalStateException if the model is not loaded.
     */
    private suspend fun computeEmbedding(text: String): List<Double> {
        val model = embeddingModel
            ?: throw IllegalStateException("Embedding model not initialised. Call initEmbeddingModel first.")

        val embedData = EmbedData.builder<String>()
            .setData(text)
            .setTask(EmbedData.TaskType.SEMANTIC_SIMILARITY)
            .build()

        val request    = EmbeddingRequest.create(ImmutableList.of(embedData))
        val embeddings = model.getEmbeddings(request).await()

        if (embeddings.isEmpty()) {
            Log.w(TAG, "getEmbeddings returned empty for: '${text.take(60)}'")
            return emptyList()
        }

        return embeddings.map { it.toDouble() }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – safe coroutine result wrapper
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Executes [block] and routes success/failure back to [result] on the Main thread.
     * Exceptions are caught so no unhandled exception propagates to Android's handler.
     */
    private suspend fun <T> safeResultCall(
        result: MethodChannel.Result,
        errorCode: String = "EMBED_ERROR",
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