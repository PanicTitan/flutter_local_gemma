import 'dart:async';
import 'dart:js_interop';
import '../utils/web_script_loader.dart';

// ─── JS Interop declarations ──────────────────────────────────────────────────

@JS('initEmbeddingModel')
external JSPromise<JSBoolean> _jsInitEmbedding(
  JSString url,
  JSString base, [
  JSString? token,
]);

@JS('getEmbedding')
external JSPromise<JSAny> _jsGetEmbedding(JSString text);

@JS('unloadEmbeddingModel')
external void _jsUnloadEmbeddingModel();

// ─────────────────────────────────────────────────────────────────────────────

/// Web implementation of the embedding plugin.
///
/// Wraps the LiteRT-based `getEmbedding` JavaScript function exposed by
/// `main.ts`.
class EmbeddingPluginWeb {
  static final EmbeddingPluginWeb _instance = EmbeddingPluginWeb._internal();
  factory EmbeddingPluginWeb() => _instance;
  EmbeddingPluginWeb._internal();

  bool _initialized = false;

  /// Whether the web embedding engine is ready to generate embeddings.
  bool get isInitialized => _initialized;

  /// Initialises the LiteRT model and tokenizer in the browser.
  ///
  /// Subsequent calls with the same [modelUrl] are no-ops (the JS side
  /// caches the compiled model).
  Future<void> init(String modelUrl, {String? token}) async {
    await WebScriptLoader.ensureJsLoaded();
    await _jsInitEmbedding(
      modelUrl.toJS,
      WebScriptLoader.assetBase.toJS,
      token?.toJS,
    ).toDart;
    _initialized = true;
  }

  /// Computes a semantic embedding vector for [text].
  ///
  /// Returns a [List<double>] of length matching the model's output dimension.
  Future<List<double>> getEmbedding(String text) async {
    if (!_initialized) throw StateError('EmbeddingPluginWeb not initialised.');
    final result = await _jsGetEmbedding(text.toJS).toDart;
    return (result as JSFloat32Array).toDart.map((e) => e.toDouble()).toList();
  }

  /// Releases the compiled model and tokenizer from browser memory.
  ///
  /// The LiteRT WASM runtime itself is kept alive (it cannot be unloaded
  /// without a full page reload), but the compiled model graph is freed.
  void unload() {
    _jsUnloadEmbeddingModel();
    _initialized = false;
  }
}