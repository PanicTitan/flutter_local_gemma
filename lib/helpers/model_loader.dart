import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import '../flutter_local_gemma.dart';

// ── Default model URLs ────────────────────────────────────────────────────────

const _llmUrlWeb =
    'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task';
const _llmUrlMobile =
    'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm';
const _embedUrl =
    'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/main/embeddinggemma-2-740m.litertlm';
const _tokUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model?download=true';
const _embedGemma2IdWeb = 'onnx-community/embeddinggemma-2-ONNX';

// ─────────────────────────────────────────────────────────────────────────────
// PluginModelLoader — pure-function helpers, no state management
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless helpers for downloading, initialising, and unloading the
/// [FlutterLocalGemma] LLM engine and [EmbeddingPlugin] engine.
///
/// Unlike the example-app's `ModelManager`, this class carries no
/// [ChangeNotifier] state — it is designed to be called from any architecture
/// (BLoC, Riverpod, raw `setState`, integration tests, etc.). For the
/// simpler `GemmaModel`-enum-based catalog API, see `GemmaLoader`
/// (`gemma/gemma_loader.dart`) instead.
///
/// ## Typical usage
/// ```dart
/// // Download + init in one call:
/// await PluginModelLoader.loadLlm(
///   token: 'hf_…',
///   onProgress: (p) => print('${p.toStringAsFixed(0)}%'),
/// );
///
/// // Use the engine …
///
/// await PluginModelLoader.unloadLlm();
/// ```
abstract final class PluginModelLoader {
  // ── LLM ──────────────────────────────────────────────────────────────────

  /// Downloads the LLM model from [networkUrl] (or the built-in default) and
  /// returns the resolved local path / OPFS key.
  ///
  /// [onProgress] receives values in [0, 100] during the download.
  ///
  /// Tracks [GemmaState.loadState] as [ModelLoadState.downloading] for the
  /// duration of the download (this is the *only* place that state is set —
  /// [FlutterLocalGemma.init] only ever resolves an already-local file, it
  /// never downloads). Resolves back to [ModelLoadState.loaded] if a model
  /// happens to already be loaded (e.g. downloading a second model in the
  /// background while the current one is still in use), or
  /// [ModelLoadState.notLoaded] otherwise — either way, accurately
  /// reflecting the engine's real state once the download itself is done.
  /// Sets [ModelLoadState.error] (with [GemmaState.lastError]) if the
  /// download fails.
  static Future<String> downloadLlm({
    String? networkUrl,
    String? token,
    void Function(double progress)? onProgress,
  }) async {
    final url = networkUrl ?? (kIsWeb ? _llmUrlWeb : _llmUrlMobile);
    GemmaState.instance.setLastError(null);
    GemmaState.instance.setLoadState(ModelLoadState.downloading);
    try {
      final String path;
      if (kIsWeb) {
        if (url.contains('onnx') || url.endsWith('.onnx')) {
          path = url;
        } else {
          final uri = Uri.tryParse(url);
          final rawFileName = (uri?.pathSegments.isNotEmpty == true)
              ? uri!.pathSegments.last
              : 'gemma-model.litertlm';
          path = await ModelInstaller(fileName: rawFileName)
              .fromNetwork(url, token: token)
              .withProgress((p) => onProgress?.call(p))
              .install();
        }
      } else {
        final uri = Uri.tryParse(url);
        final rawFileName = (uri?.pathSegments.isNotEmpty == true)
            ? uri!.pathSegments.last
            : '';
        final fileName = rawFileName.isNotEmpty && rawFileName.endsWith('.litertlm')
            ? rawFileName
            : 'gemma-model.litertlm';

        final docs = await getApplicationDocumentsDirectory();
        final legacyFile = File('${docs.path}/gemma-model.litertlm');
        final targetFile = File('${docs.path}/$fileName');

        if (!targetFile.existsSync() &&
            legacyFile.existsSync() &&
            legacyFile.lengthSync() > 100 * 1024 * 1024) {
          onProgress?.call(100.0);
          path = legacyFile.path;
        } else {
          path = await ModelInstaller(fileName: fileName)
              .fromNetwork(url, token: token)
              .withProgress((p) => onProgress?.call(p))
              .install();
        }
      }
      GemmaState.instance.setLoadState(
        FlutterLocalGemma().isInitialized
            ? ModelLoadState.loaded
            : ModelLoadState.notLoaded,
      );
      return path;
    } catch (e) {
      GemmaState.instance.setLastError(e);
      GemmaState.instance.setLoadState(ModelLoadState.error);
      rethrow;
    }
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
    bool enableMtp = true,
    String? modelName,
    void Function(double progress)? onProgress,
  }) => FlutterLocalGemma().init(
    InferenceConfig(
      modelPath: path,
      maxTokens: maxTokens,
      backend: useGpu ? PreferredBackend.gpu : PreferredBackend.cpu,
      maxNumImages: maxNumImages,
      supportAudio: supportAudio,
      enableMtp: enableMtp,
      modelName: modelName,
      onProgress: onProgress,
    ),
  );

  /// Convenience wrapper: [downloadLlm] → [initLlm] in a single call.
  ///
  /// [onProgress] covers the download phase; the init phase has no progress.
  static Future<String> loadLlm({
    String? networkUrl,
    String? localPath,
    String? token,
    int maxTokens = 4096,
    bool useGpu = true,
    bool supportAudio = true,
    int maxNumImages = 10,
    bool enableMtp = true,
    String? modelName,
    void Function(double progress)? onProgress,
  }) async {
    final path =
        localPath ??
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
      enableMtp: enableMtp,
      modelName: modelName,
      onProgress: onProgress,
    );
    return path;
  }

  /// Releases the LLM engine from memory.
  static Future<void> unloadLlm() => FlutterLocalGemma().dispose();

  /// Safely swaps the currently loaded model for a new one.
  ///
  /// Unlike calling [loadLlm]/[initLlm] directly — which are **no-ops** if a
  /// model is already loaded, because [FlutterLocalGemma.init] is
  /// idempotent — this always unloads whatever model is currently loaded
  /// first (if any), so the native engine actually ends up running the new
  /// model. Use this any time the user can pick a *different* model while
  /// one is already active (e.g. a model picker / model manager screen).
  ///
  /// If [GemmaState.isLoadBusy] is true (a download, load, or unload is
  /// already in flight), this throws a [StateError] rather than racing it —
  /// check `GemmaState.instance.loadState` in your UI to disable the
  /// "switch model" action while one is already in progress.
  ///
  /// If [path] is already the currently loaded model
  /// ([FlutterLocalGemma.currentModelPath]) and [forceReload] is false
  /// (the default), this is a no-op — no unnecessary unload/reload cycle.
  ///
  /// ```dart
  /// // In a model-picker "Load" button handler:
  /// await PluginModelLoader.switchLlm(
  ///   path: newModelPath,
  ///   maxTokens: 4096,
  ///   useGpu: true,
  /// );
  /// ```
  static Future<void> switchLlm({
    required String path,
    int maxTokens = 4096,
    bool useGpu = true,
    bool supportAudio = true,
    int maxNumImages = 10,
    bool enableMtp = true,
    String? modelName,
    bool forceReload = false,
  }) async {
    if (GemmaState.instance.isLoadBusy) {
      throw StateError(
        'A model download/load/unload is already in progress '
        '(GemmaState.instance.loadState == '
        '${GemmaState.instance.loadState.value}). Wait for it to finish '
        'before switching models.',
      );
    }

    final gemma = FlutterLocalGemma();
    final alreadyThisModel =
        gemma.isInitialized && gemma.currentModelPath == path;
    if (alreadyThisModel && !forceReload) return;

    if (gemma.isInitialized) {
      // Unload whatever is currently loaded first — init() would otherwise
      // silently no-op and leave the *old* model running natively while the
      // UI thinks the new one is active.
      await gemma.dispose();
    }

    await initLlm(
      path: path,
      maxTokens: maxTokens,
      useGpu: useGpu,
      supportAudio: supportAudio,
      maxNumImages: maxNumImages,
      enableMtp: enableMtp,
      modelName: modelName,
    );
  }

  /// [downloadLlm] → [switchLlm] in one call: downloads (if needed), then
  /// safely swaps out any currently-loaded model for the new one.
  ///
  /// This is the model-switching counterpart to [loadLlm] — prefer this one
  /// in any UI where the user can choose between multiple models while one
  /// may already be loaded.
  static Future<String> switchToLlm({
    String? networkUrl,
    String? localPath,
    String? token,
    int maxTokens = 4096,
    bool useGpu = true,
    bool supportAudio = true,
    int maxNumImages = 10,
    bool enableMtp = true,
    String? modelName,
    bool forceReload = false,
    void Function(double progress)? onProgress,
  }) async {
    final path =
        localPath ??
        await downloadLlm(
          networkUrl: networkUrl,
          token: token,
          onProgress: onProgress,
        );
    await switchLlm(
      path: path,
      maxTokens: maxTokens,
      useGpu: useGpu,
      supportAudio: supportAudio,
      maxNumImages: maxNumImages,
      enableMtp: enableMtp,
      modelName: modelName,
      forceReload: forceReload,
    );
    return path;
  }

  /// Returns the local cache date for a model file, or null if not cached.
  static Future<DateTime?> modelCachedAt(GemmaModel model) async {
    if (kIsWeb) {
      if (model.isOnnx) {
        final pattern = model.webRepo.isNotEmpty ? model.webRepo : model.webFile;
        final isCached = await ModelInstaller.isTransformersModelCached(pattern);
        return isCached ? DateTime.now() : null;
      }
      try {
        final cache = await ModelInstaller.listWebCache();
        for (final item in cache) {
          if (item['name'] == model.webFile) {
            return DateTime.now(); // OPFS doesn't expose modified dates, but the file exists
          }
        }
      } catch (_) {}
      return null;
    }
    try {
      final docs = await getApplicationDocumentsDirectory();
      final file = File('${docs.path}/${model.androidFile}');
      if (await file.exists()) {
        final stat = await file.stat();
        return stat.modified;
      }
    } catch (_) {}
    return null;
  }

  /// Returns true if the model file is cached AND its MTP shard is current
  /// (for Gemma 4: download date must be >= 2026-05-05).
  static Future<bool> isModelReady(GemmaModel model) async {
    final cachedAt = await modelCachedAt(model);
    if (cachedAt == null) return false;
    if (model.isGemma4 && !model.isOnnx && cachedAt.isBefore(DateTime(2026, 5, 5))) {
      return false;
    }
    return true;
  }

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

    if (mUrl.endsWith('.litertlm')) {
      final path = await downloadEmbeddingGemma2(
        modelUrlOrId: mUrl,
        token: token,
        onProgress: onProgress,
      );
      return (modelPath: path, tokenizerPath: null);
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
    void Function(double progress)? onProgress,
  }) => EmbeddingPlugin().init(
    EmbeddingConfig(
      modelPathOrId: modelPath,
      tokenizerPath: tokenizerPath,
      useGpu: useGpu,
      token: token,
      isEmbeddingGemma2: false,
      onProgress: onProgress,
    ),
  );

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

  // ── EmbeddingGemma 2 ──────────────────────────────────────────────────────

  /// Downloads or resolves the EmbeddingGemma 2 model.
  /// On mobile: downloads the `.litertlm` model bundle.
  /// On web: returns the HuggingFace model ID (`onnx-community/embeddinggemma-2-ONNX`).
  ///
  /// When [modelUrlOrId] is null the generic build is used. Set
  /// [preferChipsetVariant] to first try the build compiled for this device's
  /// SoC (Qualcomm / MediaTek / Tensor); if it is unavailable the generic
  /// build is downloaded instead.
  static Future<String> downloadEmbeddingGemma2({
    String? modelUrlOrId,
    String? token,
    bool preferChipsetVariant = false,
    void Function(double progress)? onProgress,
  }) async {
    if (kIsWeb) {
      return modelUrlOrId ?? _embedGemma2IdWeb;
    }
    if (modelUrlOrId != null) {
      if (!modelUrlOrId.startsWith('http://') &&
          !modelUrlOrId.startsWith('https://')) {
        return modelUrlOrId;
      }
      return await ModelInstaller(fileName: 'embeddinggemma-2-740m.litertlm')
          .fromNetwork(modelUrlOrId, token: token)
          .withProgress((p) => onProgress?.call(p))
          .install();
    }

    final soc = preferChipsetVariant
        ? await EmbeddingGemma2Catalog.detectSoc()
        : const DeviceSocInfo();
    final candidates = EmbeddingGemma2Catalog.candidates(
      soc,
      preferChipset: preferChipsetVariant,
    );
    Object? lastError;
    for (final variant in candidates) {
      try {
        return await ModelInstaller(fileName: variant.fileName)
            .fromNetwork(variant.url, token: token)
            .withProgress((p) => onProgress?.call(p))
            .install();
      } catch (e) {
        lastError = e;
        debugPrint('[PluginModelLoader] ${variant.label} download failed: $e');
      }
    }
    throw lastError ?? Exception('EmbeddingGemma 2 download failed');
  }

  /// Initialises the EmbeddingGemma 2 engine from a resolved path or model ID.
  static Future<void> initEmbeddingGemma2({
    required String modelPathOrId,
    int maxInputSequenceLength = 256,
    int visionTokenBudget = 70,
    int? targetDimension,
    String? accelerator,
    String? token,
    void Function(double progress)? onProgress,
  }) => EmbeddingPlugin().init(
    EmbeddingConfig.gemma2(
      modelPathOrId: modelPathOrId,
      maxInputSequenceLength: maxInputSequenceLength,
      visionTokenBudget: visionTokenBudget,
      targetDimension: targetDimension,
      accelerator: accelerator,
      token: token,
      onProgress: onProgress,
    ),
  );

  /// Downloads (if needed) and initialises EmbeddingGemma 2 in one call.
  static Future<String> loadEmbeddingGemma2({
    String? modelUrlOrId,
    String? token,
    bool preferChipsetVariant = false,
    int maxInputSequenceLength = 256,
    int visionTokenBudget = 70,
    int? targetDimension,
    String? accelerator,
    void Function(double progress)? onProgress,
  }) async {
    final path = await downloadEmbeddingGemma2(
      modelUrlOrId: modelUrlOrId,
      token: token,
      preferChipsetVariant: preferChipsetVariant,
      onProgress: onProgress,
    );
    await initEmbeddingGemma2(
      modelPathOrId: path,
      maxInputSequenceLength: maxInputSequenceLength,
      visionTokenBudget: visionTokenBudget,
      targetDimension: targetDimension,
      accelerator: accelerator,
      token: token,
      onProgress: onProgress,
    );
    return path;
  }

  /// Deletes a specific model file from local cache.
  static Future<bool> deleteModel(GemmaModel model) async {
    if (kIsWeb) {
      if (model.isOnnx) {
        final pattern = model.webRepo.isNotEmpty ? model.webRepo : model.webFile;
        return await ModelInstaller.deleteTransformersModel(pattern);
      }
      return false;
    }
    try {
      final docs = await getApplicationDocumentsDirectory();
      final file = File('${docs.path}/${model.androidFile}');
      if (await file.exists()) {
        await file.delete();
        return true;
      }
    } catch (_) {}
    return false;
  }

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