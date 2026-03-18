import 'package:flutter/foundation.dart';
import '../flutter_local_gemma.dart';

// ── Default model URLs ────────────────────────────────────────────────────────

const _llmUrlWeb =
    'https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4-Web.litertlm?download=true';
const _llmUrlMobile =
    'https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm?download=true';
const _embedUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite?download=true';
const _tokUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model?download=true';

// ─────────────────────────────────────────────────────────────────────────────
// GemmaLoader — pure-function helpers, no state management
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless helpers for downloading, initialising, and unloading the
/// [FlutterLocalGemma] LLM engine and [EmbeddingPlugin] engine.
///
/// Unlike the example-app's `ModelManager`, this class carries no
/// [ChangeNotifier] state — it is designed to be called from any architecture
/// (BLoC, Riverpod, raw `setState`, integration tests, etc.).
///
/// ## Typical usage
/// ```dart
/// // Download + init in one call:
/// await GemmaLoader.loadLlm(
///   token: 'hf_…',
///   onProgress: (p) => print('${p.toStringAsFixed(0)}%'),
/// );
///
/// // Use the engine …
///
/// await GemmaLoader.unloadLlm();
/// ```
abstract final class GemmaLoader {
  // ── LLM ──────────────────────────────────────────────────────────────────

  /// Downloads the LLM model from [networkUrl] (or the built-in default) and
  /// returns the resolved local path / OPFS key.
  ///
  /// [onProgress] receives values in [0, 100] during the download.
  static Future<String> downloadLlm({
    String? networkUrl,
    String? token,
    void Function(double progress)? onProgress,
  }) {
    final url = networkUrl ?? (kIsWeb ? _llmUrlWeb : _llmUrlMobile);
    return FlutterLocalGemma.installModel()
        .fromNetwork(url, token: token)
        .withProgress((p) => onProgress?.call(p))
        .install();
  }

  /// Initialises the LLM engine from an already-resolved [path].
  ///
  /// Calling this when the engine is already initialised is a no-op
  /// ([FlutterLocalGemma.init] is idempotent).
  static Future<void> initLlm({
    required String path,
    int maxTokens = 4096,
    bool useGpu = true,
    bool supportAudio = true,
    int maxNumImages = 10,
  }) =>
      FlutterLocalGemma().init(InferenceConfig(
        modelPath: path,
        maxTokens: maxTokens,
        backend:
            kIsWeb ? null : (useGpu ? PreferredBackend.gpu : PreferredBackend.cpu),
        maxNumImages: maxNumImages,
        supportAudio: supportAudio,
      ));

  /// Convenience wrapper: [downloadLlm] → [initLlm] in a single call.
  ///
  /// [onProgress] covers the download phase; the init phase has no progress.
  static Future<void> loadLlm({
    String? networkUrl,
    String? localPath,
    String? token,
    int maxTokens = 4096,
    bool useGpu = true,
    bool supportAudio = true,
    int maxNumImages = 10,
    void Function(double progress)? onProgress,
  }) async {
    final path = localPath ??
        await downloadLlm(
          networkUrl: networkUrl,
          token: token,
          onProgress: onProgress,
        );
    await initLlm(
      path: path,
      maxTokens: maxTokens,
      useGpu: useGpu,
      supportAudio: supportAudio,
      maxNumImages: maxNumImages,
    );
  }

  /// Releases the LLM engine from memory.
  static Future<void> unloadLlm() => FlutterLocalGemma().dispose();

  // ── Embedding ─────────────────────────────────────────────────────────────

  /// Downloads the embedding model (and tokenizer on mobile) and returns the
  /// resolved local model path / OPFS key.
  ///
  /// [onProgress] receives values in [0, 100].
  /// On web, only the model is downloaded; the tokenizer is fetched
  /// automatically by the native JS layer.
  static Future<({String modelPath, String? tokenizerPath})> downloadEmbedding({
    String? modelUrl,
    String? tokenizerUrl,
    String? token,
    void Function(double progress)? onProgress,
  }) async {
    final mUrl = modelUrl ?? _embedUrl;
    final tUrl = tokenizerUrl ?? _tokUrl;

    if (kIsWeb) {
      // On web the model URL is passed directly; no local copy needed.
      return (modelPath: mUrl, tokenizerPath: null);
    }

    final modelPath = await ModelInstaller(fileName: 'embed.tflite')
        .fromNetwork(mUrl, token: token)
        .withProgress((p) => onProgress?.call(p * 0.8))
        .install();

    final tokenizerPath = await ModelInstaller(fileName: 'tokenizer.spm')
        .fromNetwork(tUrl, token: token)
        .withProgress((p) => onProgress?.call(80 + p * 0.2))
        .install();

    return (modelPath: modelPath, tokenizerPath: tokenizerPath);
  }

  /// Initialises the embedding engine from already-resolved paths.
  static Future<void> initEmbedding({
    required String modelPath,
    String? tokenizerPath,
    bool useGpu = true,
    String? token,
  }) =>
      EmbeddingPlugin().init(EmbeddingConfig(
        modelPathOrId: modelPath,
        tokenizerPath: tokenizerPath,
        useGpu: useGpu,
        token: token,
      ));

  /// Convenience wrapper: [downloadEmbedding] → [initEmbedding].
  static Future<void> loadEmbedding({
    String? modelUrl,
    String? tokenizerUrl,
    String? token,
    bool useGpu = true,
    void Function(double progress)? onProgress,
  }) async {
    final (:modelPath, :tokenizerPath) = await downloadEmbedding(
      modelUrl: modelUrl,
      tokenizerUrl: tokenizerUrl,
      token: token,
      onProgress: onProgress,
    );
    await initEmbedding(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
      useGpu: useGpu,
      token: token,
    );
  }

  /// Releases the embedding engine from memory.
  static Future<void> unloadEmbedding() => EmbeddingPlugin().dispose();

  // ── Cache ─────────────────────────────────────────────────────────────────

  /// Deletes all cached model files.  Returns a human-readable size string.
  static Future<String> purgeCache() async {
    final bytes = await ModelInstaller.purgeCache();
    return _formatBytes(bytes);
  }

  static String _formatBytes(int b) {
    if (b >= 1 << 30) return '${(b / (1 << 30)).toStringAsFixed(2)} GB';
    if (b >= 1 << 20) return '${(b / (1 << 20)).toStringAsFixed(1)} MB';
    if (b >= 1 << 10) return '${(b / (1 << 10)).toStringAsFixed(0)} KB';
    return '$b bytes';
  }
}