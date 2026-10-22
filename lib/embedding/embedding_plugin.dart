import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'embed_part.dart';
import 'embedding_plugin_web.dart'
    if (dart.library.io) 'embedding_plugin_stub.dart';

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for loading an embedding model (both legacy and EmbeddingGemma 2).
class EmbeddingConfig {
  /// On mobile: the absolute file-system path to the `.litertlm` or `.tflite` model.
  /// On web: the model URL or HuggingFace ID (`onnx-community/embeddinggemma-2-ONNX`).
  final String modelPathOrId;

  /// Absolute path to the tokenizer file (only required for legacy SentencePiece models on Android).
  final String? tokenizerPath;

  /// Whether to attempt GPU acceleration. Ignored on emulators (forced CPU).
  final bool useGpu;

  /// Hardware accelerator preference: 'gpu', 'cpu', 'npu', or 'tpu'.
  final String? accelerator;

  /// Max input sequence token length (default 256 for gallery, 1024 for video).
  final int maxInputSequenceLength;

  /// Number of tokens budgeted per vision frame (default 70 for EmbeddingGemma 2).
  final int visionTokenBudget;

  /// Matryoshka dimension truncation (768, 512, 256, 128). Null keeps native 768.
  final int? targetDimension;

  /// Bearer token for authenticated model endpoints (e.g. HuggingFace private repos).
  final String? token;

  /// Whether this model uses EmbeddingGemma 2 multimodal architecture.
  final bool isEmbeddingGemma2;

  /// Optional progress callback receiving [0, 100] during model loading/download.
  final void Function(double progress)? onProgress;

  const EmbeddingConfig({
    required this.modelPathOrId,
    this.tokenizerPath,
    this.useGpu = true,
    this.accelerator,
    this.maxInputSequenceLength = 256,
    this.visionTokenBudget = 70,
    this.targetDimension,
    this.token,
    this.isEmbeddingGemma2 = true,
    this.onProgress,
  });

  /// Convenience factory for EmbeddingGemma 2.
  factory EmbeddingConfig.gemma2({
    required String modelPathOrId,
    int maxInputSequenceLength = 256,
    int visionTokenBudget = 70,
    int? targetDimension,
    String? accelerator,
    String? token,
    void Function(double progress)? onProgress,
  }) {
    return EmbeddingConfig(
      modelPathOrId: modelPathOrId,
      maxInputSequenceLength: maxInputSequenceLength,
      visionTokenBudget: visionTokenBudget,
      targetDimension: targetDimension,
      accelerator: accelerator,
      token: token,
      isEmbeddingGemma2: true,
      onProgress: onProgress,
    );
  }

  EmbeddingConfig copyWith({
    String? modelPathOrId,
    String? tokenizerPath,
    bool? useGpu,
    String? accelerator,
    int? maxInputSequenceLength,
    int? visionTokenBudget,
    int? targetDimension,
    String? token,
    bool? isEmbeddingGemma2,
    void Function(double progress)? onProgress,
  }) {
    return EmbeddingConfig(
      modelPathOrId: modelPathOrId ?? this.modelPathOrId,
      tokenizerPath: tokenizerPath ?? this.tokenizerPath,
      useGpu: useGpu ?? this.useGpu,
      accelerator: accelerator ?? this.accelerator,
      maxInputSequenceLength: maxInputSequenceLength ?? this.maxInputSequenceLength,
      visionTokenBudget: visionTokenBudget ?? this.visionTokenBudget,
      targetDimension: targetDimension ?? this.targetDimension,
      token: token ?? this.token,
      isEmbeddingGemma2: isEmbeddingGemma2 ?? this.isEmbeddingGemma2,
      onProgress: onProgress ?? this.onProgress,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingPlugin  —  cross-platform embedding engine
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-platform singleton for generating on-device multimodal embeddings
/// with EmbeddingGemma 2 and MediaPipe Tasks Retrieval.
///
/// Supports Text, Image, Audio, Video, and Interleaved Multimodal parts.
class EmbeddingPlugin {
  static final EmbeddingPlugin _instance = EmbeddingPlugin._internal();
  factory EmbeddingPlugin() => _instance;
  EmbeddingPlugin._internal();

  static const MethodChannel _channel             = MethodChannel('embedding_plugin');
  static const EventChannel  _copyProgressChannel = EventChannel('embedding_copy_progress');

  bool _isInitialized = false;
  EmbeddingConfig? _activeConfig;

  /// When true, methods return deterministic L2-normalized simulated embeddings
  /// if the real model has not been initialized yet. This allows seamless demo
  /// mode and testing of search/gallery/moments features without requiring a 1.5GB model.
  bool allowMockFallback = true;

  // ── Copy-progress stream ───────────────────────────────────────────────────

  /// Emits copy-progress values in **[0, 100]** while a local embedding model
  /// file from the Android file picker is being copied to internal storage.
  Stream<double> get copyProgressStream {
    if (kIsWeb) return const Stream.empty();
    return _copyProgressChannel
        .receiveBroadcastStream()
        .map((event) => (event as num).toDouble());
  }

  /// ValueNotifier that emits whenever initialization state changes.
  static final ValueNotifier<bool> isInitializedNotifier = ValueNotifier<bool>(false);

  /// Whether the embedding engine is ready to compute vectors.
  bool get isInitialized => _isInitialized;

  /// Current active model configuration.
  EmbeddingConfig? get activeConfig => _activeConfig;

  // ── Deterministic Mock Embeddings ──────────────────────────────────────────

  /// Generates a deterministic, L2-normalized pseudo-embedding for [text].
  static List<double> generateMockTextEmbedding(String text, {int dim = 768}) {
    final vec = List<double>.filled(dim, 0.0);
    final normalized = text.toLowerCase().trim();
    if (normalized.isEmpty) return vec;

    final tokens = normalized.split(RegExp(r'\s+'));
    for (int tIdx = 0; tIdx < tokens.length; tIdx++) {
      final token = tokens[tIdx];
      final hash = token.hashCode;
      for (int i = 0; i < dim; i++) {
        final angle = ((hash ^ (i * 31)) % 10000) / 10000.0 * 2 * math.pi;
        vec[i] += math.sin(angle) / (tIdx + 1);
      }
    }

    double sumSq = 0.0;
    for (final v in vec) {
      sumSq += v * v;
    }
    final norm = math.sqrt(sumSq);
    if (norm > 0) {
      for (int i = 0; i < dim; i++) {
        vec[i] /= norm;
      }
    }
    return vec;
  }

  /// Generates a deterministic pseudo-embedding from image byte buffers.
  static List<double> generateMockImagesEmbedding(List<Uint8List> images, {int dim = 768}) {
    final vec = List<double>.filled(dim, 0.0);
    if (images.isEmpty) return vec;

    for (final img in images) {
      if (img.isEmpty) continue;
      final step = math.max(1, img.length ~/ 256);
      int sampleIdx = 0;
      for (int i = 0; i < img.length; i += step) {
        final b = img[i];
        final bucket = (sampleIdx * 17 + b * 31) % dim;
        vec[bucket] += (b - 128.0) / 128.0;
        sampleIdx++;
      }
    }

    double sumSq = 0.0;
    for (final v in vec) {
      sumSq += v * v;
    }
    final norm = math.sqrt(sumSq);
    if (norm > 0) {
      for (int i = 0; i < dim; i++) {
        vec[i] /= norm;
      }
    }
    return vec;
  }

  /// Generates a deterministic pseudo-embedding from multimodal parts.
  static List<double> generateMockPartsEmbedding(List<EmbedPart> parts, {int dim = 768}) {
    final vec = List<double>.filled(dim, 0.0);
    if (parts.isEmpty) return vec;

    for (final part in parts) {
      List<double> sub;
      switch (part) {
        case TextEmbedPart(:final text):
          sub = generateMockTextEmbedding(text, dim: dim);
        case ImageEmbedPart(:final bytes):
          sub = generateMockImagesEmbedding([bytes], dim: dim);
        case AudioEmbedPart(:final pcm):
          sub = generateMockTextEmbedding('audio_${pcm.length}', dim: dim);
      }
      for (int i = 0; i < dim; i++) {
        vec[i] += sub[i];
      }
    }

    double sumSq = 0.0;
    for (final v in vec) {
      sumSq += v * v;
    }
    final norm = math.sqrt(sumSq);
    if (norm > 0) {
      for (int i = 0; i < dim; i++) {
        vec[i] /= norm;
      }
    }
    return vec;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Initialises the embedding model.
  Future<void> init(EmbeddingConfig config) async {
    if (_isInitialized) return;
    _activeConfig = config;

    if (kIsWeb) {
      final isGemma2 = config.isEmbeddingGemma2 ||
          config.modelPathOrId.contains('embeddinggemma') ||
          config.modelPathOrId.contains('onnx') ||
          !config.modelPathOrId.endsWith('.tflite');
      if (isGemma2) {
        _activeConfig = config.copyWith(isEmbeddingGemma2: true);
        await EmbeddingPluginWeb().initGemma2(
          config.modelPathOrId,
          token: config.token,
          targetDimension: config.targetDimension,
          onProgress: config.onProgress,
        );
      } else {
        await EmbeddingPluginWeb().init(
          config.modelPathOrId,
          token: config.token,
          onProgress: config.onProgress,
        );
      }
    } else {
      if (config.isEmbeddingGemma2) {
        await _channel.invokeMethod('initEmbeddingGemma2', {
          'modelPath': config.modelPathOrId,
          'accelerator': config.accelerator ?? (config.useGpu ? 'gpu' : 'cpu'),
          'maxInputSequenceLength': config.maxInputSequenceLength,
          'visionTokenBudget': config.visionTokenBudget,
          'targetDimension': config.targetDimension,
        });
      } else {
        if (config.tokenizerPath == null) {
          throw ArgumentError(
            'EmbeddingConfig.tokenizerPath is required for legacy models on Android.',
          );
        }
        await _channel.invokeMethod('initEmbeddingModel', {
          'modelPath': config.modelPathOrId,
          'tokenizerPath': config.tokenizerPath,
          'useGpu': config.useGpu,
        });
      }
    }

    _isInitialized = true;
    isInitializedNotifier.value = true;
  }

  /// Computes a semantic embedding vector for [text].
  ///
  /// Automatically applies [taskType] instruction prefixes for EmbeddingGemma 2.
  /// Optionally truncates to [targetDimension] (e.g. 512, 256, 128) with L2 re-normalization.
  Future<List<double>> getEmbedding(
    String text, {
    EmbeddingTaskType taskType = EmbeddingTaskType.retrievalQuery,
    String? title,
    int? targetDimension,
  }) async {
    if (!_isInitialized) {
      if (allowMockFallback) {
        final dim = targetDimension ?? _activeConfig?.targetDimension ?? 768;
        return generateMockTextEmbedding(text, dim: dim);
      }
      throw StateError('EmbeddingPlugin is not initialised. Call init() first.');
    }

    final effectiveText = (_activeConfig?.isEmbeddingGemma2 ?? true)
        ? formatEmbeddingText(text, taskType: taskType, title: title)
        : text;

    List<double> raw;
    if (kIsWeb) {
      raw = await EmbeddingPluginWeb().getEmbedding(effectiveText);
    } else {
      final List<dynamic>? result = await _channel.invokeMethod(
        'getEmbedding',
        {'text': effectiveText},
      );
      raw = result?.cast<double>() ?? [];
    }

    final dim = targetDimension ?? _activeConfig?.targetDimension;
    if (dim != null && dim < raw.length) {
      return truncateMatryoshka(raw, dim);
    }
    return raw;
  }

  /// Computes an embedding vector for a single encoded image.
  Future<List<double>> getEmbeddingFromImage(
    Uint8List imageData, {
    int? targetDimension,
  }) async {
    return getEmbeddingFromImages([imageData], targetDimension: targetDimension);
  }

  /// Computes an embedding vector for a list of encoded images (e.g. video keyframes).
  Future<List<double>> getEmbeddingFromImages(
    List<Uint8List> imageDataList, {
    int? targetDimension,
  }) async {
    if (!_isInitialized) {
      if (allowMockFallback) {
        final dim = targetDimension ?? _activeConfig?.targetDimension ?? 768;
        return generateMockImagesEmbedding(imageDataList, dim: dim);
      }
      throw StateError('EmbeddingPlugin is not initialised. Call init() first.');
    }
    if (imageDataList.isEmpty) return [];

    List<double> raw;
    if (kIsWeb) {
      raw = await EmbeddingPluginWeb().getEmbeddingFromImages(imageDataList);
    } else {
      final List<dynamic>? result = await _channel.invokeMethod(
        'generateImagesEmbedding',
        {'images': imageDataList},
      );
      raw = result?.cast<double>() ?? [];
    }

    final dim = targetDimension ?? _activeConfig?.targetDimension;
    if (dim != null && dim < raw.length) {
      return truncateMatryoshka(raw, dim);
    }
    return raw;
  }

  /// Computes an embedding vector for an ordered list of multimodal parts
  /// (interleaved timestamps, audio slices, and video frames).
  Future<List<double>> getEmbeddingFromParts(
    List<EmbedPart> parts, {
    int? targetDimension,
  }) async {
    if (!_isInitialized) {
      if (allowMockFallback) {
        final dim = targetDimension ?? _activeConfig?.targetDimension ?? 768;
        return generateMockPartsEmbedding(parts, dim: dim);
      }
      throw StateError('EmbeddingPlugin is not initialised. Call init() first.');
    }
    if (parts.isEmpty) return [];

    List<double> raw;
    if (kIsWeb) {
      raw = await EmbeddingPluginWeb().getEmbeddingFromParts(parts);
    } else {
      final serializedParts = parts.map((p) => p.toMap()).toList();
      final List<dynamic>? result = await _channel.invokeMethod(
        'generateMultimodalEmbedding',
        {'parts': serializedParts},
      );
      raw = result?.cast<double>() ?? [];
    }

    final dim = targetDimension ?? _activeConfig?.targetDimension;
    if (dim != null && dim < raw.length) {
      return truncateMatryoshka(raw, dim);
    }
    return raw;
  }

  /// Computes an embedding vector for a 16 kHz mono signed 16-bit PCM audio buffer.
  Future<List<double>> getEmbeddingFromAudio(
    Int16List pcm, {
    int sampleRateHz = kAudioSampleRateHz,
    int? targetDimension,
  }) async {
    return getEmbeddingFromParts([
      EmbedPart.audio(pcm, sampleRateHz: sampleRateHz),
    ], targetDimension: targetDimension);
  }

  /// Returns whether the underlying loaded model declares audio support.
  Future<bool> supportsAudio() async {
    if (!_isInitialized) return false;
    if (kIsWeb) {
      return EmbeddingPluginWeb().supportsAudio();
    }
    final res = await _channel.invokeMethod<bool>('supportsAudio');
    return res ?? false;
  }

  /// Fully releases the embedding model from memory.
  Future<void> dispose() async {
    if (!_isInitialized) return;

    if (kIsWeb) {
      EmbeddingPluginWeb().unload();
    } else {
      await _channel.invokeMethod('closeEmbeddingModel');
    }

    _isInitialized = false;
    _activeConfig = null;
    isInitializedNotifier.value = false;
  }
}