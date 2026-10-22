


package com.example.flutter_local_gemma

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.mediapipe.tasks.components.containers.AudioData
import com.google.mediapipe.tasks.core.Part
import com.google.mediapipe.tasks.core.TextPart
import com.google.mediapipe.tasks.retrieval.components.SqliteVectorStore
import com.google.mediapipe.tasks.retrieval.components.VectorStore
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.sqrt

/**
 * SemanticRetrievalPlugin
 *
 * Exposes MediaPipe's on-device [SqliteVectorStore] and semantic vector retrieval
 * to Flutter via MethodChannel 'semantic_retrieval_plugin'.
 */
class SemanticRetrievalPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {

    companion object {
        private const val TAG = "SemanticRetrievalPlugin"
        private const val CHANNEL = "semantic_retrieval_plugin"
    }

    private lateinit var channel: MethodChannel
    private lateinit var appContext: Context

    private val supervisorJob = SupervisorJob()
    private val ioScope = CoroutineScope(supervisorJob + Dispatchers.IO)

    @Volatile private var vectorStore: VectorStore? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        ioScope.launch {
            vectorStore?.close()
            vectorStore = null
        }.invokeOnCompletion {
            supervisorJob.cancel()
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "initRetriever" -> {
                val dbName = call.argument<String>("databaseName") ?: "semantic_retrieval.db"
                val dimension = call.argument<Int>("dimension") ?: 768
                ioScope.launch {
                    safeResultCall(result, "RETRIEVER_INIT_ERROR") {
                        try {
                            vectorStore?.close()
                            vectorStore = SqliteVectorStore(appContext, dbName, dimension)
                            Log.i(TAG, "SqliteVectorStore initialized with: $dbName (dim=$dimension)")
                            true
                        } catch (e: Exception) {
                            Log.w(TAG, "SqliteVectorStore failed ($e), using InMemoryVectorStore fallback.")
                            vectorStore = FallbackInMemoryVectorStore()
                            true
                        }
                    }
                }
            }

            "getAllRecordIds" -> {
                ioScope.launch {
                    safeResultCall(result, "GET_IDS_ERROR") {
                        vectorStore?.getAllRecordIds() ?: emptyList<String>()
                    }
                }
            }

            "insertVectorRecord" -> {
                val id = call.argument<String>("id")
                val embeddingsList = call.argument<List<Double>>("embeddings")
                val metadata = call.argument<Map<String, String>>("metadata") ?: emptyMap()

                if (id.isNullOrEmpty() || embeddingsList == null) {
                    result.error("INVALID_ARG", "id and embeddings must not be null.", null)
                    return
                }

                ioScope.launch {
                    safeResultCall(result, "INSERT_RECORD_ERROR") {
                        val floatArray = embeddingsList.map { it.toFloat() }.toFloatArray()
                        val record = RetrievalRecord(id, listOf<Part>(TextPart(id)), floatArray, metadata)
                        vectorStore?.upsert(listOf(record))
                        true
                    }
                }
            }

            "addMultimodalRecord" -> {
                val id = call.argument<String>("id")
                val parts = call.argument<List<Map<String, Any>>>("parts")
                val metadata = call.argument<Map<String, String>>("metadata") ?: emptyMap()

                if (id.isNullOrEmpty() || parts == null) {
                    result.error("INVALID_ARG", "id and parts must not be null.", null)
                    return
                }

                ioScope.launch {
                    val embedder = EmbeddingPlugin.activeUniversalEmbedder
                    if (embedder == null) {
                        withContext(Dispatchers.Main) {
                            result.error("EMBEDDER_NOT_INITIALIZED", "EmbeddingGemma 2 UniversalEmbedder not initialised.", null)
                        }
                        return@launch
                    }
                    safeResultCall(result, "ADD_MULTIMODAL_ERROR") {
                        val content = parsePartsToContent(parts)
                        val res = embedder.embedContent(content)
                        val floats = res.embeddings().firstOrNull()?.floatEmbedding()
                            ?: throw RuntimeException("Failed to generate embedding for parts.")
                        val record = RetrievalRecord(id, listOf<Part>(TextPart(id)), floats, metadata)
                        vectorStore?.upsert(listOf(record))
                        true
                    }
                }
            }

            "addRecordsBatch" -> {
                val records = call.argument<List<Map<String, Any>>>("records")
                if (records.isNullOrEmpty()) {
                    result.success(true)
                    return
                }

                ioScope.launch {
                    val embedder = EmbeddingPlugin.activeUniversalEmbedder
                    if (embedder == null) {
                        withContext(Dispatchers.Main) {
                            result.error("EMBEDDER_NOT_INITIALIZED", "EmbeddingGemma 2 UniversalEmbedder not initialised.", null)
                        }
                        return@launch
                    }
                    safeResultCall(result, "BATCH_ADD_ERROR") {
                        val retrievalRecords = mutableListOf<RetrievalRecord>()
                        for (r in records) {
                            val id = r["id"] as? String ?: continue
                            val images = (r["images"] as? List<*>)?.filterIsInstance<ByteArray>() ?: emptyList()
                            val metadata = (r["metadata"] as? Map<*, *>)?.mapNotNull { (k, v) ->
                                if (k != null && v != null) k.toString() to v.toString() else null
                            }?.toMap() ?: emptyMap()

                            if (images.isNotEmpty()) {
                                val res = embedder.embedContent(images.map { it as Any })
                                val floats = res.embeddings().firstOrNull()?.floatEmbedding()
                                if (floats != null) {
                                    retrievalRecords.add(RetrievalRecord(id, listOf<Part>(TextPart(id)), floats, metadata))
                                }
                            }
                        }
                        if (retrievalRecords.isNotEmpty()) {
                            vectorStore?.upsert(retrievalRecords)
                        }
                        true
                    }
                }
            }

            "retrieveEntities" -> {
                val queryText = call.argument<String>("queryText")
                val limit = call.argument<Int>("limit") ?: 100
                val metadataFilter = call.argument<Map<String, String>>("metadataFilter") ?: emptyMap()

                if (queryText.isNullOrEmpty()) {
                    result.success(emptyList<Map<String, Any>>())
                    return
                }

                ioScope.launch {
                    val embedder = EmbeddingPlugin.activeUniversalEmbedder
                    if (embedder == null) {
                        withContext(Dispatchers.Main) {
                            result.error("EMBEDDER_NOT_INITIALIZED", "EmbeddingGemma 2 UniversalEmbedder not initialised.", null)
                        }
                        return@launch
                    }
                    safeResultCall(result, "RETRIEVE_ENTITIES_ERROR") {
                        val res = embedder.embedContent(listOf(queryText))
                        val queryEmbedding = res.embeddings().firstOrNull()?.floatEmbedding()
                            ?: return@safeResultCall emptyList<Map<String, Any>>()
                        searchVectorStore(queryEmbedding, limit, null, metadataFilter)
                    }
                }
            }

            "retrieveEntitiesByImage" -> {
                val imageData = call.argument<ByteArray>("imageData")
                val limit = call.argument<Int>("limit") ?: 100
                val seedId = call.argument<String>("seedId")
                val metadataFilter = call.argument<Map<String, String>>("metadataFilter") ?: emptyMap()

                if (imageData == null) {
                    result.error("INVALID_ARG", "imageData must not be null.", null)
                    return
                }

                ioScope.launch {
                    val embedder = EmbeddingPlugin.activeUniversalEmbedder
                    if (embedder == null) {
                        withContext(Dispatchers.Main) {
                            result.error("EMBEDDER_NOT_INITIALIZED", "EmbeddingGemma 2 UniversalEmbedder not initialised.", null)
                        }
                        return@launch
                    }
                    safeResultCall(result, "RETRIEVE_IMAGE_ERROR") {
                        val res = embedder.embedContent(listOf(imageData))
                        val queryEmbedding = res.embeddings().firstOrNull()?.floatEmbedding()
                            ?: return@safeResultCall emptyList<Map<String, Any>>()
                        searchVectorStore(queryEmbedding, limit, seedId, metadataFilter)
                    }
                }
            }

            "retrieveEntitiesByVector" -> {
                val queryVector = call.argument<List<Double>>("queryVector")
                val limit = call.argument<Int>("limit") ?: 100
                val seedId = call.argument<String>("seedId")
                val metadataFilter = call.argument<Map<String, String>>("metadataFilter") ?: emptyMap()

                if (queryVector == null) {
                    result.error("INVALID_ARG", "queryVector must not be null.", null)
                    return
                }

                ioScope.launch {
                    safeResultCall(result, "RETRIEVE_VECTOR_ERROR") {
                        val floats = queryVector.map { it.toFloat() }.toFloatArray()
                        searchVectorStore(floats, limit, seedId, metadataFilter)
                    }
                }
            }

            "deleteRecords" -> {
                val ids = call.argument<List<String>>("ids") ?: emptyList()
                ioScope.launch {
                    safeResultCall(result, "DELETE_RECORDS_ERROR") {
                        vectorStore?.delete(ids)
                        true
                    }
                }
            }

            "deleteRecordsWithIdPrefix" -> {
                val idPrefix = call.argument<String>("idPrefix") ?: ""
                ioScope.launch {
                    safeResultCall(result, "DELETE_PREFIX_ERROR") {
                        val allIds = vectorStore?.getAllRecordIds() ?: emptyList()
                        val matching = allIds.filter { it.startsWith(idPrefix) }
                        if (matching.isNotEmpty()) {
                            vectorStore?.delete(matching)
                        }
                        matching.size
                    }
                }
            }

            "deleteAllRecords" -> {
                ioScope.launch {
                    safeResultCall(result, "DELETE_ALL_ERROR") {
                        vectorStore?.deleteAll()
                        true
                    }
                }
            }

            "extractVideoFrame" -> {
                val videoPath = call.argument<String>("videoPath")
                val timeMs = (call.argument<Number>("timeMs"))?.toLong() ?: 0L
                if (videoPath.isNullOrEmpty()) {
                    result.error("INVALID_ARG", "videoPath must not be null.", null)
                    return
                }
                ioScope.launch {
                    try {
                        val retriever = MediaMetadataRetriever()
                        if (videoPath.startsWith("content://")) {
                            retriever.setDataSource(appContext, Uri.parse(videoPath))
                        } else {
                            retriever.setDataSource(videoPath)
                        }
                        val bitmap = retriever.getFrameAtTime(timeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                        retriever.release()
                        if (bitmap != null) {
                            val stream = ByteArrayOutputStream()
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
                            val bytes = stream.toByteArray()
                            bitmap.recycle()
                            withContext(Dispatchers.Main) { result.success(bytes) }
                        } else {
                            withContext(Dispatchers.Main) { result.success(null) }
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "extractVideoFrame notice: ${e.message}")
                        withContext(Dispatchers.Main) { result.success(null) }
                    }
                }
            }

            "extractVideoMetadata" -> {
                val videoPath = call.argument<String>("videoPath")
                if (videoPath.isNullOrEmpty()) {
                    result.error("INVALID_ARG", "videoPath must not be null.", null)
                    return
                }
                ioScope.launch {
                    try {
                        val retriever = MediaMetadataRetriever()
                        if (videoPath.startsWith("content://")) {
                            retriever.setDataSource(appContext, Uri.parse(videoPath))
                        } else {
                            retriever.setDataSource(videoPath)
                        }
                        val durStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        val widthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                        val heightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                        retriever.release()
                        val durationMs = durStr?.toLongOrNull() ?: 0L
                        val width = widthStr?.toIntOrNull() ?: 0
                        val height = heightStr?.toIntOrNull() ?: 0
                        withContext(Dispatchers.Main) {
                            result.success(mapOf(
                                "durationMs" to durationMs,
                                "width" to width,
                                "height" to height
                            ))
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) { result.success(null) }
                    }
                }
            }

            "supportsAudioEmbedding" -> {
                result.success(true)
            }

            "closeRetriever" -> {
                ioScope.launch {
                    safeResultCall(result, "CLOSE_RETRIEVER_ERROR") {
                        vectorStore?.close()
                        vectorStore = null
                        true
                    }
                }
            }

            else -> result.notImplemented()
        }
    }

    private fun searchVectorStore(
        queryEmbedding: FloatArray,
        limit: Int,
        seedId: String?,
        metadataFilter: Map<String, String>
    ): List<Map<String, Any>> {
        val store = vectorStore ?: return emptyList()
        val records = store.search(queryEmbedding, limit + (if (seedId != null) 1 else 0), metadataFilter)

        val results = mutableListOf<Map<String, Any>>()
        for (r in records) {
            if (seedId != null && r.id == seedId) continue
            val emb = r.embeddings
            val score = if (emb.isNotEmpty()) cosineSimilarity(queryEmbedding, emb) else 0f
            results.add(
                mapOf(
                    "id" to r.id,
                    "embeddings" to emb.map { it.toDouble() },
                    "metadata" to (r.metadata ?: emptyMap<String, String>()),
                    "similarityScore" to score.toDouble()
                )
            )
        }
        results.sortByDescending { (it["similarityScore"] as? Double) ?: 0.0 }
        return if (results.size > limit) results.subList(0, limit) else results
    }

    private fun parsePartsToContent(parts: List<Map<String, Any>>): List<Any> {
        val content = mutableListOf<Any>()
        for (part in parts) {
            val type = part["type"] as? String ?: continue
            when (type) {
                "text" -> {
                    val text = part["text"] as? String
                    if (!text.isNullOrEmpty()) content.add(text)
                }
                "image" -> {
                    val bytes = part["bytes"] as? ByteArray
                    if (bytes != null && bytes.isNotEmpty()) content.add(bytes)
                }
                "audio" -> {
                    val bytes = part["bytes"] as? ByteArray
                    val sampleRate = (part["sampleRateHz"] as? Number)?.toInt() ?: 16000
                    if (bytes != null && bytes.isNotEmpty()) {
                        val shortBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        val shorts = ShortArray(shortBuffer.remaining())
                        shortBuffer.get(shorts)
                        val format = AudioData.AudioDataFormat.builder()
                            .setNumOfChannels(1)
                            .setSampleRate(sampleRate.toFloat())
                            .build()
                        val audioData = AudioData.create(format, shorts.size).apply { load(shorts) }
                        content.add(audioData)
                    }
                }
            }
        }
        return content
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size || v1.isEmpty()) return 0f
        var dot = 0f
        var norm1 = 0f
        var norm2 = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }
        val denom = sqrt(norm1.toDouble()) * sqrt(norm2.toDouble())
        return if (denom > 0) (dot / denom).toFloat() else 0f
    }

    private suspend fun <T> safeResultCall(
        result: MethodChannel.Result,
        errorCode: String = "RETRIEVER_ERROR",
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

/** Fallback in-memory [VectorStore] implementation */
private class FallbackInMemoryVectorStore : VectorStore {
    private val records = ConcurrentHashMap<String, RetrievalRecord>()

    override fun getAllRecordIds(): List<String> = records.keys.toList()

    override fun deleteAll() {
        records.clear()
    }

    override fun upsert(records: List<RetrievalRecord>) {
        for (record in records) {
            this.records[record.id] = record
        }
    }

    override fun delete(ids: List<String>) {
        for (id in ids) {
            records.remove(id)
        }
    }

    override fun delete(metadataFilter: Map<String, String>) {
        if (metadataFilter.isEmpty()) return
        val toDelete = records.values.filter { matchesFilter(it, metadataFilter) }.map { it.id }
        delete(toDelete)
    }

    override fun search(queryEmbedding: FloatArray, topK: Int): List<RetrievalRecord> {
        return search(queryEmbedding, topK, metadataFilter = emptyMap())
    }

    override fun search(
        queryEmbedding: FloatArray,
        topK: Int,
        metadataFilter: Map<String, String>
    ): List<RetrievalRecord> {
        return records.values
            .filter { matchesFilter(it, metadataFilter) }
            .mapNotNull { record ->
                val emb = record.embeddings
                if (emb.isEmpty()) return@mapNotNull null
                val score = cosineSimilarity(queryEmbedding, emb)
                record to score
            }
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    override fun get(ids: List<String>): List<RetrievalRecord> = ids.mapNotNull { records[it] }

    override fun close() {
        records.clear()
    }

    private fun matchesFilter(record: RetrievalRecord, filter: Map<String, String>): Boolean {
        if (filter.isEmpty()) return true
        val metadata = record.metadata ?: return false
        return filter.all { (k, v) -> metadata[k] == v }
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size || v1.isEmpty()) return 0f
        var dot = 0f
        var norm1 = 0f
        var norm2 = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }
        val denom = sqrt(norm1.toDouble()) * sqrt(norm2.toDouble())
        return if (denom > 0) (dot / denom).toFloat() else 0f
    }
}
