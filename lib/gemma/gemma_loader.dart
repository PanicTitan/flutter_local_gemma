import 'package:flutter/foundation.dart';
import 'gemma.dart';
import 'gemma_model.dart';

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
  ///
  /// Returns the local path of the installed model file.
  static Future<String> loadGemmaModel(
    GemmaModel model, {
    String? token,
    int maxTokens = 4096,
    bool useGpu = true,
    void Function(double)? onProgress,
  }) async {
    final url = kIsWeb ? model.webUrl : model.androidUrl;
    final fileName = kIsWeb ? model.webFile : model.androidFile;

    final installer = ModelInstaller(fileName: fileName);
    String path;

    if (onProgress != null) {
      path = await installer
          .fromNetwork(url, token: token)
          .withProgress(onProgress)
          .install();
    } else {
      path = await installer.fromNetwork(url, token: token).install();
    }

    // Initialise the engine after successful download.
    await FlutterLocalGemma().init(InferenceConfig(
      modelPath: path,
      maxTokens: maxTokens,
      backend: useGpu ? PreferredBackend.gpu : PreferredBackend.cpu,
      supportAudio: model.supportsAudio,
      enableMtp: model.supportsMtp, // auto-detected on Android anyway
      modelName: fileName,
    ));

    return path;
  }

  // ── Cache queries ──────────────────────────────────────────────────────────

  /// Returns true if the model file appears to be cached locally.
  ///
  /// On Web: checks the OPFS listing for the model filename.
  /// On Android: always returns `true` when the engine is already initialised
  /// with this model (i.e. [FlutterLocalGemma.currentModelPath] ends with the
  /// model filename). A full cache check requires a file-system stat not
  /// currently exposed by [ModelInstaller]; use [loadGemmaModel] to let the
  /// installer determine whether to download or use the cached file.
  static Future<bool> isModelCached(GemmaModel model) async {
    if (kIsWeb) {
      final fileName = model.webFile;
      try {
        final files = await ModelInstaller.listWebCache();
        return files.any((f) => (f['name'] as String?) == fileName);
      } catch (_) {
        return false;
      }
    } else {
      // Android: check if the engine currently has this model loaded.
      final path = FlutterLocalGemma().currentModelPath;
      if (path != null && path.endsWith(model.androidFile)) return true;
      // Cannot stat the file without additional platform channel support;
      // return false and let loadGemmaModel decide at install time.
      return false;
    }
  }

  /// Returns true if the model is cached AND ready for use (no re-download needed).
  static Future<bool> isModelReady(GemmaModel model) => isModelCached(model);

  /// Returns the local cache date for the model file, or null if not available.
  ///
  /// Currently only supported on Web via OPFS metadata.
  static Future<DateTime?> modelCachedAt(GemmaModel model) async {
    if (!kIsWeb) return null; // Android path not yet exposed
    try {
      final files = await ModelInstaller.listWebCache();
      final entry = files.firstWhere(
        (f) => (f['name'] as String?) == model.webFile,
        orElse: () => {},
      );
      final ts = entry['lastModified'];
      if (ts is int) return DateTime.fromMillisecondsSinceEpoch(ts);
      return null;
    } catch (_) {
      return null;
    }
  }

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
    // TODO: When ModelInstaller exposes single-file deletion, use it here.
    // For now we fall through to purgeCache which clears everything.
    await ModelInstaller.purgeCache();
  }

  /// Deletes all cached model files.
  ///
  /// Returns the number of bytes freed.
  static Future<int> purgeCache() => ModelInstaller.purgeCache();
}