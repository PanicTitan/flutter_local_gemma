import 'dart:async';
import 'dart:io';
import 'package:background_downloader/background_downloader.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:cross_file/cross_file.dart';
import 'model_installer.dart';

// ── Private channel (same name as GemmaPlugin's METHOD_CHANNEL) ───────────────
const _gemmaChannel = MethodChannel('gemma_bundled');

ModelDownloaderBuilder createDownloader(
  String url,
  String fileName, {
  String? token,
  String? directoryPath,
}) => MobileModelDownloader(url, fileName, token: token, directoryPath: directoryPath);

Future<String> createXFileInstaller(XFile file) async => file.path;

Future<String> createBlobInstaller(Object blob) async =>
    throw UnsupportedError('Web Blobs are not supported on mobile.');

// Web-only stub.
Future<List<Map<String, dynamic>>> listOpfsCacheImpl() async => [];

/// Deletes all cached model files on Android.
///
/// Two locations are cleaned:
///  1. `filesDir/models/`  — SAF-URI copies made by GemmaPlugin (via the
///     native `purgeModelCache` method channel call).
///  2. `applicationDocumentsDirectory/` — files downloaded by the installer.
///
/// Returns total bytes freed.
Future<int> purgeModelCacheImpl() async {
  int freed = 0;

  // 1. Native cache (filesDir/models/ inside GemmaPlugin).
  try {
    final bytes = await _gemmaChannel.invokeMethod<int>('purgeModelCache');
    freed += bytes ?? 0;
  } catch (_) {
    // Channel may not be ready if no model has ever been loaded. Safe to ignore.
  }

  // 2. Downloaded model + embedding files in applicationDocumentsDirectory.
  try {
    final docs = await getApplicationDocumentsDirectory();
    for (final entity in docs.listSync()) {
      if (entity is File) {
        final name = entity.path.split('/').last.toLowerCase();
        if (name.endsWith('.litertlm') ||
            name.endsWith('.tflite')   ||
            name.endsWith('.bin')      ||
            name.endsWith('.spm')) {
          freed += entity.lengthSync();
          entity.deleteSync();
        }
      }
    }
  } catch (_) {}

  // 3. Also purge the embedding model native channel cache if it exists.
  //    EmbeddingPlugin stores its model in the same location. The channel
  //    call is a best-effort no-op if EmbeddingPlugin hasn't been loaded.
  try {
    const embCh = MethodChannel('embedding_plugin');
    final bytes = await embCh.invokeMethod<int>('purgeModelCache');
    freed += bytes ?? 0;
  } catch (_) {}

  return freed;
}

class MobileModelDownloader extends ModelDownloaderBuilder {
  MobileModelDownloader(
    super.url,
    super.fileName, {
    super.token,
    super.directoryPath,
  });

  @override
  Future<String> install() async {
    String folderPath;
    BaseDirectory baseDir;

    if (directoryPath != null) {
      folderPath = directoryPath!;
      baseDir    = BaseDirectory.root;
      final dir  = Directory(folderPath);
      if (!await dir.exists()) await dir.create(recursive: true);
    } else {
      final docs = await getApplicationDocumentsDirectory();
      folderPath = docs.path;
      baseDir    = BaseDirectory.applicationDocuments;
    }

    final fullPath = '$folderPath/$fileName';
    final file     = File(fullPath);

    if (await file.exists() && await file.length() > 1 * 1024 * 1024) {
      onProgressCallback?.call(100.0);
      return fullPath;
    }
    if (await file.exists()) await file.delete();

    final task = DownloadTask(
      url:           url,
      filename:      fileName,
      headers:       token != null ? {'Authorization': 'Bearer $token'} : {},
      baseDirectory: baseDir,
      directory:     directoryPath ?? '',
      updates:       Updates.statusAndProgress,
      allowPause:    true,
    );

    final result = await FileDownloader().download(
      task,
      onProgress: (p) => onProgressCallback?.call(p * 100),
    );

    if (result.status == TaskStatus.complete) return fullPath;
    throw Exception('Download failed: ${result.status}');
  }
}