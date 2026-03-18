import 'package:cross_file/cross_file.dart';
import 'model_installer_stub.dart'
    if (dart.library.io) 'model_installer_mobile.dart'
    if (dart.library.js_interop) 'model_installer_web.dart';

// ─────────────────────────────────────────────────────────────────────────────
// ModelInstaller — public API for all model install paths
// ─────────────────────────────────────────────────────────────────────────────

class ModelInstaller {
  final String _fileName;

  ModelInstaller({String fileName = 'gemma-model.litertlm'}) : _fileName = fileName;

  /// Downloads [url] into the platform model cache directory.
  ModelDownloaderBuilder fromNetwork(String url, {String? token, String? directoryPath}) {
    final name = url.split('/').last.split('?').first;
    return createDownloader(url, name.isNotEmpty ? name : _fileName,
        token: token, directoryPath: directoryPath);
  }

  /// Installs from an [XFile] (file picker result). Zero-copy on all platforms.
  Future<String> fromXFile(XFile file) => createXFileInstaller(file);

  /// **Web only.** Installs from a raw JS `Blob`. Zero-copy, no RAM spike.
  Future<String> fromWebBlob(Object blob) => createBlobInstaller(blob);

  // ── Static cache management ───────────────────────────────────────────────

  /// Deletes all cached model files on the current platform.
  ///
  /// **Android:**
  ///   - `filesDir/models/` — SAF copy cache (via native `purgeModelCache`
  ///      method channel call into GemmaPlugin).
  ///   - `applicationDocumentsDirectory/` — downloaded `.litertlm`/`.tflite`
  ///     files placed there by the installer.
  ///
  /// **Web:**
  ///   - Calls `purgeOpfsCache([])` in JS which deletes every file from the
  ///     Origin Private File System root.
  ///
  /// Returns total bytes freed. Safe to call while no model is loaded.
  static Future<int> purgeCache() => purgeModelCacheImpl();

  /// **Web only.** Lists all files currently stored in the OPFS cache.
  ///
  /// Returns `[{'name': String, 'size': int}, ...]`.
  /// Returns `[]` on Android or when OPFS is unavailable.
  static Future<List<Map<String, dynamic>>> listWebCache() async {
    try {
      return await listOpfsCacheImpl();
    } catch (_) {
      return [];
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelDownloaderBuilder — fluent download builder
// ─────────────────────────────────────────────────────────────────────────────

abstract class ModelDownloaderBuilder {
  final String url;
  final String fileName;
  final String? token;
  final String? directoryPath;
  Function(double)? onProgressCallback;

  ModelDownloaderBuilder(this.url, this.fileName, {this.token, this.directoryPath});

  /// Registers a progress callback that receives values in [0, 100].
  ModelDownloaderBuilder withProgress(Function(double) cb) {
    onProgressCallback = cb;
    return this;
  }

  /// Runs the installation and returns the resolved model path.
  Future<String> install();
}
