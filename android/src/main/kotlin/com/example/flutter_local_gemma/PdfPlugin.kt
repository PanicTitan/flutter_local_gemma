package com.example.flutter_local_gemma

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.*
import org.openpdf.text.pdf.PdfReader
import org.openpdf.text.pdf.parser.PdfTextExtractor
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * PdfPlugin
 *
 * Provides on-device PDF text extraction and page rendering for Flutter via a [MethodChannel].
 * This plugin is intentionally **stateless**: every `extractPdf` call is fully self-contained
 * – there is no persistent object held between calls, which means there is nothing to leak.
 *
 * ## Methods
 * | Method         | Description                                              |
 * |----------------|----------------------------------------------------------|
 * | `extractPdf`   | Extract text and/or render page images from a PDF blob.  |
 * | `clearCache`   | Deletes any `.pdf` temp files left in the app cache dir. |
 *
 * ## Threading model
 * A [SupervisorJob] + [Dispatchers.IO] scope keeps all PDF work off the Flutter thread.
 * Because there is no shared mutable state between calls, concurrency is trivially safe.
 *
 * ## Text extraction strategy
 * | Android version | Library used                         |
 * |-----------------|--------------------------------------|
 * | API 35+         | `PdfRenderer.Page.textContents` (native) |
 * | API < 35        | OpenPDF (`PdfTextExtractor`)         |
 *
 * After raw extraction the text is passed through [cleanExtractedText] to repair common
 * kerning artefacts where individual characters are space-separated (e.g. "H e l l o").
 *
 * ## Render scale
 * The `renderScale` argument controls bitmap DPI. A value of `2.0` (default) gives
 * 2× the PDF's logical pixel size, which is crisp on most phone screens without being
 * excessively memory hungry. Higher values increase quality but also RAM usage linearly.
 */
class PdfPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {

    companion object {
        private const val TAG     = "PdfPlugin"
        private const val CHANNEL = "pdf_plugin"
        private const val TEMP_PREFIX = "flutter_pdf_"
        private const val TEMP_SUFFIX = ".pdf"
    }

    // ── Flutter channel ───────────────────────────────────────────────────────
    private lateinit var channel: MethodChannel

    // ── Application context (needed for cache dir) ────────────────────────────
    private var appContext: Context? = null

    // ── Coroutine scope ───────────────────────────────────────────────────────
    private val supervisorJob = SupervisorJob()
    private val ioScope = CoroutineScope(supervisorJob + Dispatchers.IO)

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        channel   = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        appContext = null
        supervisorJob.cancel()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel.MethodCallHandler
    // ─────────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {

            /**
             * Extracts content from a PDF supplied as a raw byte array.
             *
             * Arguments:
             *   - `bytes`       (ByteArray, required) – the PDF binary.
             *   - `mode`        (String)  – extraction mode (see below). Default: `"auto"`.
             *   - `filter`      (String)  – page filter: `"all"`, `"odd"`, `"even"`, `"range"`. Default: `"all"`.
             *   - `startPage`   (num)     – first page (1-based, inclusive). Used when filter=`"range"`.
             *   - `endPage`     (num)     – last page  (1-based, inclusive). Used when filter=`"range"`.
             *   - `renderScale` (num)     – bitmap scale multiplier. Default: `2.0`.
             *
             * Extraction modes:
             *   - `"auto"`          – extracts text; falls back to rendering if no text is found.
             *   - `"textOnly"`      – text extraction only (no rendering).
             *   - `"imagesOnly"`    – page rendering only (no text).
             *   - `"fullRender"`    – always renders every page as an image.
             *   - `"textAndImages"` – always does both.
             *
             * Returns: `List<Map<String, Any>>` where each map has:
             *   - `"type"` → `"text"` or `"image"`
             *   - `"data"` → `String` (for text) or `ByteArray` (PNG bytes, for image)
             */
            "extractPdf" -> {
                val bytes       = call.argument<ByteArray>("bytes")
                val mode        = call.argument<String>("mode") ?: "auto"
                val filter      = call.argument<String>("filter") ?: "all"
                val startPage   = call.argument<Any>("startPage")?.toString()?.toDoubleOrNull()?.toInt()
                val endPage     = call.argument<Any>("endPage")?.toString()?.toDoubleOrNull()?.toInt()
                val renderScale = call.argument<Any>("renderScale")?.toString()?.toDoubleOrNull() ?: 2.0

                if (bytes == null) {
                    result.error("INVALID_ARG", "bytes must not be null.", null)
                    return
                }

                ioScope.launch {
                    safeResultCall(result, "PDF_ERROR") {
                        extractContent(bytes, mode, filter, startPage, endPage, renderScale)
                    }
                }
            }

            /**
             * Deletes any stale temporary PDF files left in the app cache directory.
             * Useful if a previous extraction crashed before cleaning up.
             */
            "clearCache" -> {
                ioScope.launch {
                    safeResultCall(result, "PDF_CACHE_ERROR") {
                        deleteCachedTempFiles()
                        null
                    }
                }
            }

            else -> result.notImplemented()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – extraction orchestration
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Core extraction function. Writes [bytes] to a temp file, opens it with
     * [PdfRenderer], processes each page, then cleans up unconditionally via a
     * `finally` block.
     *
     * All [Closeable] resources ([PdfRenderer], [PdfReader], [ParcelFileDescriptor])
     * are closed in the `finally` block so memory is released even if an exception
     * is thrown mid-way.
     */
    @SuppressLint("NewApi")
    private fun extractContent(
        bytes: ByteArray,
        mode: String,
        filter: String,
        startPage: Int?,
        endPage: Int?,
        renderScale: Double
    ): List<Map<String, Any>> {
        val context = appContext
            ?: throw IllegalStateException("Plugin is not attached to an engine.")

        val parts = mutableListOf<Map<String, Any>>()

        // Write bytes to a temp file so PdfRenderer (which needs a file descriptor) can open it.
        val tempFile = File.createTempFile(TEMP_PREFIX, TEMP_SUFFIX, context.cacheDir)

        // These are held as vars so the finally block can close them regardless of where
        // execution reaches.
        var fileDescriptor: ParcelFileDescriptor? = null
        var pdfRenderer: PdfRenderer? = null
        var openPdfReader: PdfReader? = null

        try {
            tempFile.writeBytes(bytes)

            fileDescriptor = ParcelFileDescriptor.open(tempFile, ParcelFileDescriptor.MODE_READ_ONLY)
            pdfRenderer    = PdfRenderer(fileDescriptor)

            val totalPages   = pdfRenderer.pageCount
            val needsText    = mode in setOf("textOnly", "textAndImages", "auto")
            val needsRender  = mode in setOf("imagesOnly", "fullRender", "textAndImages", "auto")

            // OpenPDF is used as a text fallback on pre-API-35 devices.
            if (needsText && Build.VERSION.SDK_INT < 35) {
                openPdfReader = tryOpenPdfReader(bytes)
            }
            val openPdfExtractor = openPdfReader?.let {
                try { PdfTextExtractor(it) } catch (e: Throwable) {
                    Log.w(TAG, "PdfTextExtractor init failed: ${e.message}")
                    null
                }
            }

            for (pageIndex in 0 until totalPages) {
                val pageNumber = pageIndex + 1

                // Apply page filter before doing any work for this page.
                if (!shouldProcessPage(pageNumber, filter, startPage, endPage)) continue

                processPage(
                    pdfRenderer    = pdfRenderer,
                    pageIndex      = pageIndex,
                    pageNumber     = pageNumber,
                    needsText      = needsText,
                    needsRender    = needsRender,
                    mode           = mode,
                    renderScale    = renderScale,
                    openExtractor  = openPdfExtractor,
                    output         = parts
                )
            }

        } finally {
            // Always clean up, even if an exception was thrown above.
            safeClose("PdfReader")   { openPdfReader?.close() }
            safeClose("PdfRenderer") { pdfRenderer?.close() }
            safeClose("FileDescriptor") { fileDescriptor?.close() }
            safeClose("TempFile")    { tempFile.delete() }
        }

        return parts
    }

    /**
     * Processes a single PDF page: extracts text and/or renders a bitmap,
     * then appends the result(s) to [output].
     *
     * The [PdfRenderer.Page] is always closed in a finally block so that the
     * renderer never ends up with an open page handle if rendering fails.
     */
    @SuppressLint("NewApi")
    private fun processPage(
        pdfRenderer:   PdfRenderer,
        pageIndex:     Int,
        pageNumber:    Int,
        needsText:     Boolean,
        needsRender:   Boolean,
        mode:          String,
        renderScale:   Double,
        openExtractor: PdfTextExtractor?,
        output:        MutableList<Map<String, Any>>
    ) {
        var page: PdfRenderer.Page? = null
        try {
            page = pdfRenderer.openPage(pageIndex)

            // ── Text extraction ───────────────────────────────────────────────
            var pageText = ""
            if (needsText) {
                pageText = extractPageText(page, pageNumber, openExtractor)
                if (pageText.isNotBlank()) {
                    pageText = cleanExtractedText(pageText)
                }
            }

            // In "auto" mode: render only if no text was found (image-based page).
            val doText   = needsText   && (mode != "auto" || pageText.isNotEmpty())
            val doRender = needsRender && (mode != "auto" || pageText.isEmpty())

            if (doText && pageText.isNotEmpty()) {
                output.add(mapOf("type" to "text", "data" to "--- Page $pageNumber ---\n$pageText\n"))
            }

            if (doRender) {
                val pngBytes = renderPageToPng(page, renderScale)
                output.add(mapOf("type" to "image", "data" to pngBytes))
            }

        } catch (e: Throwable) {
            // Log and skip the page – we do not want one bad page to abort the whole document.
            Log.e(TAG, "Failed to process page $pageNumber: ${e.message}", e)
        } finally {
            safeClose("Page[$pageNumber]") { page?.close() }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – text extraction
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Extracts raw text from [page] using the best available API for the device.
     * Returns an empty string if extraction fails or produces no content.
     */
    @SuppressLint("NewApi")
    private fun extractPageText(
        page:          PdfRenderer.Page,
        pageNumber:    Int,
        openExtractor: PdfTextExtractor?
    ): String {
        return if (Build.VERSION.SDK_INT >= 35) {
            // Native API: joins all text content blocks with newlines to preserve
            // paragraph boundaries and prevent words from being concatenated.
            try {
                page.textContents.joinToString("\n") { it.text }
            } catch (e: Throwable) {
                Log.w(TAG, "Native text extraction failed on page $pageNumber: ${e.message}")
                ""
            }
        } else {
            // OpenPDF fallback for older Android versions.
            try {
                openExtractor?.getTextFromPage(pageNumber) ?: ""
            } catch (e: Throwable) {
                Log.w(TAG, "OpenPDF extraction failed on page $pageNumber: ${e.message}")
                ""
            }
        }
    }

    /**
     * Renders [page] to a PNG-encoded [ByteArray] at [scale]× the page's logical resolution.
     * The bitmap is recycled immediately after encoding to release its native heap allocation.
     */
    private fun renderPageToPng(page: PdfRenderer.Page, scale: Double): ByteArray {
        val width  = (page.width  * scale).toInt().coerceAtLeast(1)
        val height = (page.height * scale).toInt().coerceAtLeast(1)

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.WHITE)
        page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)

        return ByteArrayOutputStream().use { stream ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            bitmap.recycle() // free native heap immediately
            stream.toByteArray()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – text cleanup heuristic
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Repairs common PDF kerning artefacts where a font's glyph spacing causes
     * individual characters to be emitted with separating spaces by the extractor.
     *
     * Two modes:
     * - **Aggressive** (when double-spaces are present): treats 2+ consecutive spaces
     *   as real word boundaries and removes all single spaces (phantom kerning gaps).
     * - **Heuristic** (no double-spaces): removes spaces that appear between a letter
     *   and a following lowercase letter (strong signal of a phantom gap, not a real word break).
     *
     * Example input:  `"C á l c u l o  D i f e r e n c i a l"`
     * Example output: `"Cálculo Diferencial"`
     */
    private fun cleanExtractedText(rawText: String): String {
        if (rawText.isBlank()) return ""

        // Normalise non-breaking spaces to regular spaces.
        var text = rawText.replace('\u00A0', ' ')

        val hasDoubleSpaces = text.contains("  ")

        text = if (hasDoubleSpaces) {
            // Aggressive mode: 2+ spaces = real gap; single space = kerning artefact.
            text.replace(Regex(" {2,}"), "\u0000")   // mark real gaps
                .replace(" ", "")                    // strip phantom spaces
                .replace("\u0000", " ")              // restore real gaps
        } else {
            // Heuristic mode: remove space before a lowercase letter.
            text.replace(Regex("(?<=[\\p{L}\\p{N}]) (?=[\\p{Ll}])"), "")
        }

        // Final pass: collapse multiple spaces/tabs per line, trim edges.
        return text.lines().joinToString("\n") { line ->
            line.replace(Regex("[ \\t]{2,}"), " ").trim()
        }.trim()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – cache management
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Deletes any temp files matching [TEMP_PREFIX] in the app cache directory.
     * These should be cleaned up automatically after each call, but this method
     * provides a manual safety net.
     */
    private fun deleteCachedTempFiles() {
        val cacheDir = appContext?.cacheDir ?: return
        val staleFiles = cacheDir.listFiles { f -> f.name.startsWith(TEMP_PREFIX) } ?: return
        var deleted = 0
        for (f in staleFiles) {
            if (f.delete()) deleted++
        }
        if (deleted > 0) Log.i(TAG, "Cleared $deleted stale PDF temp file(s).")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private – helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Returns true if [pageNumber] should be processed given the [filter] and
     * optional [startPage] / [endPage] bounds.
     */
    private fun shouldProcessPage(pageNumber: Int, filter: String, startPage: Int?, endPage: Int?): Boolean {
        return when (filter) {
            "odd"   -> pageNumber % 2 != 0
            "even"  -> pageNumber % 2 == 0
            "range" -> (startPage == null || pageNumber >= startPage) &&
                       (endPage   == null || pageNumber <= endPage)
            else    -> true  // "all" and any unknown value
        }
    }

    /**
     * Attempts to construct a [PdfReader] from [bytes]. Returns null if the library
     * is unavailable or the PDF is malformed.
     */
    private fun tryOpenPdfReader(bytes: ByteArray): PdfReader? {
        return try {
            PdfReader(bytes)
        } catch (e: Throwable) {
            Log.w(TAG, "OpenPDF PdfReader init failed: ${e.message}")
            null
        }
    }

    /**
     * Closes [block] silently, logging any exceptions under [label].
     * Used in `finally` blocks to ensure resources are always freed.
     */
    private inline fun safeClose(label: String, block: () -> Unit) {
        try { block() } catch (e: Throwable) { Log.w(TAG, "Close failed [$label]: ${e.message}") }
    }

    /**
     * Executes [block] and routes success/failure to [result] on the Main thread.
     * Prevents any unhandled exception from reaching Android's crash handler.
     */
    private suspend fun <T> safeResultCall(
        result: MethodChannel.Result,
        errorCode: String = "PDF_ERROR",
        block: suspend () -> T
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