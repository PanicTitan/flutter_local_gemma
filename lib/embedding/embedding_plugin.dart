import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'embedding_plugin_web.dart'
    if (dart.library.io) 'embedding_plugin_stub.dart';

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for loading the embedding model.
class EmbeddingConfig {
  /// On mobile: the absolute file-system path to the `.tflite` model.
  /// On web: the URL from which the model will be fetched.
  final String modelPathOrId;

  /// Absolute path to the tokenizer file.
  /// **Required on Android**; ignored on web (the tokenizer is fetched
  /// automatically from HuggingFace / IndexedDB cache).
  final String? tokenizerPath;

  /// Whether to attempt GPU acceleration. Ignored on emulators (forced CPU).
  final bool useGpu;

  /// Bearer token for authenticated model endpoints (e.g. HuggingFace private
  /// repos). Only used on web.
  final String? token;

  const EmbeddingConfig({
    required this.modelPathOrId,
    this.tokenizerPath,
    this.useGpu = true,
    this.token,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingPlugin  —  cross-platform embedding engine
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-platform singleton for generating on-device text embeddings.
///
/// ## Usage
/// ```dart
/// await EmbeddingPlugin().init(EmbeddingConfig(
///   modelPathOrId: '/data/.../embedding.tflite',   // Android
///   tokenizerPath: '/data/.../tokenizer.json',     // Android only
/// ));
///
/// final vector = await EmbeddingPlugin().getEmbedding('Hello world');
///
/// await EmbeddingPlugin().dispose(); // free memory when done
/// ```
///
/// ## Lifecycle
/// Follows the same load → use → unload pattern as [FlutterLocalGemma].
/// Call [dispose] to release native / GPU memory between sessions.
class EmbeddingPlugin {
  static final EmbeddingPlugin _instance = EmbeddingPlugin._internal();
  factory EmbeddingPlugin() => _instance;
  EmbeddingPlugin._internal();

  static const MethodChannel _channel             = MethodChannel('embedding_plugin');
  static const EventChannel  _copyProgressChannel = EventChannel('embedding_copy_progress');

  bool _isInitialized = false;

  // ── Copy-progress stream ───────────────────────────────────────────────────

  /// Emits copy-progress values in **[0, 100]** while a local embedding model
  /// file from the Android file picker is being copied to internal storage.
  ///
  /// Only emits on Android and only when the picked model file requires a copy
  /// (Strategy B — i.e. Google Drive, MTP, cloud-backed files). For files on
  /// local storage (Downloads, internal storage, SD card) Strategy A is used
  /// instead — zero bytes are copied and this stream stays silent.
  /// The tokenizer is always small enough that no progress is emitted for it.
  ///
  /// Subscribe **before** calling [init]:
  ///
  /// ```dart
  /// StreamSubscription? _copySub;
  ///
  /// Future<void> loadLocalEmbedding(String contentUri, String tokenizerPath) async {
  ///   _copySub = EmbeddingPlugin().copyProgressStream.listen(
  ///     (pct) => setState(() => _copyProgress = pct),
  ///   );
  ///   try {
  ///     await EmbeddingPlugin().init(EmbeddingConfig(
  ///       modelPathOrId: contentUri,
  ///       tokenizerPath: tokenizerPath,
  ///     ));
  ///   } finally {
  ///     await _copySub?.cancel();
  ///     _copySub = null;
  ///   }
  /// }
  /// ```
  ///
  /// On web this stream never emits (no copy step is performed).
  Stream<double> get copyProgressStream {
    if (kIsWeb) return const Stream.empty();
    return _copyProgressChannel
        .receiveBroadcastStream()
        .map((event) => (event as num).toDouble());
  }

  /// Whether the embedding engine is ready to compute vectors.
  bool get isInitialized => _isInitialized;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Initialises the embedding model.
  ///
  /// Calling [init] when already initialised is a no-op; call [dispose] first
  /// if you need to load a different model.
  Future<void> init(EmbeddingConfig config) async {
    if (_isInitialized) return;

    if (kIsWeb) {
      await EmbeddingPluginWeb().init(
        config.modelPathOrId,
        token: config.token,
      );
    } else {
      if (config.tokenizerPath == null) {
        throw ArgumentError(
          'EmbeddingConfig.tokenizerPath is required on Android.',
        );
      }
      await _channel.invokeMethod('initEmbeddingModel', {
        'modelPath':     config.modelPathOrId,
        'tokenizerPath': config.tokenizerPath,
        'useGpu':        config.useGpu,
      });
    }

    _isInitialized = true;
  }

  /// Computes a semantic embedding vector for [text].
  ///
  /// Returns a `List<double>` whose length matches the model's output
  /// dimension (typically 768 for Gemma 300M embeddings).
  ///
  /// Throws [StateError] if [init] has not been called.
  Future<List<double>> getEmbedding(String text) async {
    if (!_isInitialized) {
      throw StateError('EmbeddingPlugin is not initialised. Call init() first.');
    }

    if (kIsWeb) {
      return EmbeddingPluginWeb().getEmbedding(text);
    } else {
      final List<dynamic>? result = await _channel.invokeMethod(
        'getEmbedding',
        {'text': text},
      );
      return result?.cast<double>() ?? [];
    }
  }

  /// Fully releases the embedding model from memory.
  ///
  /// On Android, calls `closeEmbeddingModel` which nulls the native
  /// `GemmaEmbeddingModel` reference so the GC can reclaim its buffers.
  /// On web, calls `unloadEmbeddingModel` to free the LiteRT-compiled graph.
  ///
  /// After [dispose], [init] must be called again before embedding.
  Future<void> dispose() async {
    if (!_isInitialized) return;

    if (kIsWeb) {
      EmbeddingPluginWeb().unload();
    } else {
      await _channel.invokeMethod('closeEmbeddingModel');
    }

    _isInitialized = false;
  }
}