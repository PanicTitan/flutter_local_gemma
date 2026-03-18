package com.example.flutter_local_gemma

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/**
 * FlutterLocalGemmaPlugin
 *
 * The top-level Flutter plugin entry point. Its sole responsibility is to:
 *   1. Register itself with the Flutter engine.
 *   2. Delegate plugin lifecycle events to each sub-plugin.
 *   3. Handle the generic `getPlatformVersion` query on its own channel.
 *
 * ## Sub-plugin architecture
 * Each feature area is encapsulated in its own plugin class with a dedicated
 * [MethodChannel] (and [EventChannel] where streaming is needed). This means:
 *   - Each sub-plugin manages its own resources and lifecycle independently.
 *   - A crash or resource error in one sub-plugin cannot affect the others.
 *   - Sub-plugins can be unloaded individually without touching the others.
 *
 * | Class                    | Channel               | Purpose                          |
 * |--------------------------|-----------------------|----------------------------------|
 * | [GemmaPlugin]            | `gemma_bundled`       | LiteRT-LM text/multimodal inference |
 * |                          | `gemma_stream`        | Token streaming via EventChannel |
 * | [EmbeddingPlugin]        | `embedding_plugin`    | GemmaEmbedding on-device vectors |
 * | [PdfPlugin]              | `pdf_plugin`          | PDF text extraction & rendering  |
 * | [NativeFilePickerPlugin] | `native_file_picker`  | System file picker (content URIs)|
 *
 * ## Lazy initialisation
 * Sub-plugin instances are created eagerly here but their *native resources*
 * (models, renderers, etc.) are created lazily – only when the first relevant
 * method call arrives. This keeps startup memory footprint near zero.
 */
class FlutterLocalGemmaPlugin : FlutterPlugin, MethodCallHandler, ActivityAware {

    // ── Top-level channel ─────────────────────────────────────────────────────
    private lateinit var channel: MethodChannel

    // ── Sub-plugins ───────────────────────────────────────────────────────────
    // Instantiated here so the coordinator owns them, but each registers its own
    // channel inside onAttachedToEngine.
    private val gemmaPlugin           = GemmaPlugin()
    private val embeddingPlugin       = EmbeddingPlugin()
    private val pdfPlugin             = PdfPlugin()
    private val nativeFilePickerPlugin = NativeFilePickerPlugin()

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        // Register the coordinator's own channel for platform-level queries.
        channel = MethodChannel(binding.binaryMessenger, "flutter_local_gemma")
        channel.setMethodCallHandler(this)

        // Propagate engine attachment to every sub-plugin so each can register
        // its own channel.
        gemmaPlugin.onAttachedToEngine(binding)
        embeddingPlugin.onAttachedToEngine(binding)
        pdfPlugin.onAttachedToEngine(binding)
        nativeFilePickerPlugin.onAttachedToEngine(binding)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)

        // Each sub-plugin is responsible for releasing its own resources.
        gemmaPlugin.onDetachedFromEngine(binding)
        embeddingPlugin.onDetachedFromEngine(binding)
        pdfPlugin.onDetachedFromEngine(binding)
        nativeFilePickerPlugin.onDetachedFromEngine(binding)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodCallHandler (coordinator channel only)
    // ─────────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
            else                 -> result.notImplemented()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ActivityAware  – forwarded entirely to NativeFilePickerPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        nativeFilePickerPlugin.onAttachedToActivity(binding)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        nativeFilePickerPlugin.onDetachedFromActivityForConfigChanges()
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        nativeFilePickerPlugin.onReattachedToActivityForConfigChanges(binding)
    }

    override fun onDetachedFromActivity() {
        nativeFilePickerPlugin.onDetachedFromActivity()
    }
}