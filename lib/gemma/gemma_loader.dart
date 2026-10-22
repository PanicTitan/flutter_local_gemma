import 'package:flutter/foundation.dart';
import '../embedding/embedding_plugin.dart';
import '../helpers/model_loader.dart';
import 'gemma.dart';
import 'gemma_model.dart';
import 'gemma_state.dart';

/// High-level model lifecycle helper.
///
/// Provides one-liner helpers that eliminate boilerplate when downloading and
/// managing known [GemmaModel] variants. For custom model URLs use
/// [ModelInstaller] directly.
///
/// ## Quick start
/// ```dart
/// // Download + initialise in one call:
/// await GemmaLoader.loadGemmaModel(
///   GemmaModel.gemma4E2B,
///   onProgress: (p) => print('${p.toStringAsFixed(0)}%'),
/// );
///
/// // Check before downloading:
/// if (!await GemmaLoader.isModelReady(GemmaModel.gemma4E2B)) {
///   await GemmaLoader.loadGemmaModel(GemmaModel.gemma4E2B);
/// }
///
/// // Clean up a specific model:
/// await GemmaLoader.purgeModel(GemmaModel.gemma4E2B);
/// ```
class GemmaLoader {
  GemmaLoader._();

  // ── Download + init ────────────────────────────────────────────────────────

  /// Downloads (if not already cached) and initialises the inference engine
  /// for the given [model].
  ///
  /// The correct URL for the current platform is selected automatically:
  /// - Android: uses [GemmaModel.androidUrl]
  /// - Web: uses [GemmaModel.webUrl]
  ///
  /// [token] is required for gated models ([GemmaModel.isGated] == true).
  /// [maxTokens] sets the KV-cache capacity.
  /// [useGpu] selects the GPU backend (set to false on unsupported devices).
  /// [onProgress] receives download progress in [0.0, 100.0].
  /// [supportAudio] overrides whether the audio backend is initialised;
  /// defaults to [GemmaModel.supportsAudio] (pass `false` to skip it even on
  /// an audio-capable model, e.g. to save memory when you know you won't
  /// use audio input).
  ///
  /// **Safely swaps models.** If a *different* model is already loaded, that
  /// model is disposed before the new one is initialised — calling
  /// [FlutterLocalGemma.init] directly again with a different model would
  /// otherwise silently no-op (it's a no-op whenever a model is already
  /// loaded) and leave the old model running natively while callers assume
  /// [model] is now active. If [model] is already the one that's loaded,
  /// this returns immediately without downloading or reloading anything
  /// (pass [forceReload] to force a reload anyway).
  ///
  /// Tracks [GemmaState.loadState] throughout: [ModelLoadState.downloading]
  /// during the download, then [ModelLoadState.loading] /
  /// [ModelLoadState.loaded] / [ModelLoadState.error] exactly as
  /// [FlutterLocalGemma.init] does. Throws a [StateError] instead of racing
  /// an already-in-flight download/load/unload — check
  /// `GemmaState.instance.isLoadBusy` in your UI to disable the "load"
  /// action while one is in progress.
  ///
  /// Returns the local path of the installed model file.
  static Future<String> loadGemmaModel(
    GemmaModel model, {
    String? token,
    int maxTokens = 4096,
    bool useGpu = true,
    bool? supportAudio,
    void Function(double)? onProgress,
    bool forceReload = false,
  }) async {
    if (GemmaState.instance.isLoadBusy) {
      throw StateError(
        'A model download/load/unload is already in progress '
        '(GemmaState.instance.loadState == '
        '${GemmaState.instance.loadState.value}). Wait for it to finish '
        'before loading another model.',
      );
    }

    final url = kIsWeb ? model.webUrl : model.androidUrl;
    final fileName = kIsWeb ? model.webFile : model.androidFile;

    final installer = ModelInstaller(fileName: fileName);
    String path;

    GemmaState.instance.setLastError(null);
    GemmaState.instance.setLoadState(ModelLoadState.downloading);
    try {
      if (kIsWeb && model.isOnnx) {
        path = model.webUrl;
      } else if (onProgress != null) {
        path = await installer
            .fromNetwork(url, token: token)
            .withProgress(onProgress)
            .install();
      } else {
        path = await installer.fromNetwork(url, token: token).install();
      }
    } catch (e) {
      GemmaState.instance.setLastError(e);
      GemmaState.instance.setLoadState(ModelLoadState.error);
      rethrow;
    }

    if (model.isEmbedding) {
      await EmbeddingPlugin().init(EmbeddingConfig.gemma2(
        modelPathOrId: path,
        token: token,
        onProgress: onProgress,
      ));
      GemmaState.instance.setLoadState(ModelLoadState.loaded);
      return path;
    }

    final gemma = FlutterLocalGemma();
    final alreadyThisModel = gemma.isInitialized && gemma.currentModelPath == path;
    if (alreadyThisModel && !forceReload) {
      // Nothing else to do — restore `loaded` since we parked the state at
      // `downloading` above, even though nothing was actually downloaded
      // beyond a cache check.
      GemmaState.instance.setLoadState(ModelLoadState.loaded);
      return path;
    }

    if (gemma.isInitialized) {
      // Unload whatever is currently loaded first — init() would otherwise
      // silently no-op and leave the *old* model running natively while
      // the caller assumes [model] is now active.
      await gemma.dispose();
    }

    // Initialise the engine after successful download. init() takes over
    // GemmaState tracking from here (downloading -> loading -> loaded/error).
    await gemma.init(InferenceConfig(
      modelPath: path,
      maxTokens: maxTokens,
      backend: useGpu ? PreferredBackend.gpu : PreferredBackend.cpu,
      supportAudio: supportAudio ?? model.supportsAudio,
      enableMtp: model.supportsMtp, // auto-detected on Android anyway
      modelName: fileName,
      onProgress: onProgress,
    ));

    return path;
  }

  // ── Cache queries ──────────────────────────────────────────────────────────

  /// Returns true if the model file appears to be cached locally.
  ///
  /// On Web: checks the OPFS listing or browser Cache Storage for the model.
  /// On Android: always returns `true` when the engine is already initialised
  /// with this model (i.e. [FlutterLocalGemma.currentModelPath] ends with the
  /// model filename). A full cache check requires a file-system stat not
  /// currently exposed by [ModelInstaller]; use [loadGemmaModel] to let the
  /// installer determine whether to download or use the cached file.
  static Future<bool> isModelCached(GemmaModel model) async {
    if (kIsWeb) {
      if (model.isOnnx) {
        return (await PluginModelLoader.modelCachedAt(model)) != null;
      }
      final fileName = model.webFile;
      try {
        final files = await ModelInstaller.listWebCache();
        return files.any((f) => (f['name'] as String?) == fileName);
      } catch (_) {
        return false;
      }
    } else {
      if (model.isEmbedding) {
        return (await PluginModelLoader.modelCachedAt(model)) != null;
      }
      final path = FlutterLocalGemma().currentModelPath;
      if (path != null && path.endsWith(model.androidFile)) return true;
      return (await PluginModelLoader.modelCachedAt(model)) != null;
    }
  }

  /// Returns true if the model is cached AND ready for use (no re-download needed).
  static Future<bool> isModelReady(GemmaModel model) =>
      PluginModelLoader.isModelReady(model);

  /// Returns the local cache date for the model file, or null if not available.
  static Future<DateTime?> modelCachedAt(GemmaModel model) =>
      PluginModelLoader.modelCachedAt(model);

  // ── Cache management ───────────────────────────────────────────────────────

  /// Deletes the cached file for a specific [model].
  ///
  /// This currently calls [ModelInstaller.purgeCache] which deletes **all**
  /// cached models (the [ModelInstaller] API does not yet support single-file
  /// deletion). A future update will narrow this to the specific model file.
  ///
  /// The engine must be disposed before calling this:
  /// ```dart
  /// await FlutterLocalGemma().dispose();
  /// await GemmaLoader.purgeModel(GemmaModel.gemma4E2B);
  /// ```
  static Future<void> purgeModel(GemmaModel model) async {
    await PluginModelLoader.deleteModel(model);
  }

  /// Deletes all cached model files.
  ///
  /// Returns the number of bytes freed.
  static Future<int> purgeCache() => ModelInstaller.purgeCache();
}