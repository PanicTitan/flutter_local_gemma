import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'package:cross_file/cross_file.dart';
import 'package:web/web.dart' as web;
import '../utils/web_script_loader.dart';
import 'model_installer.dart';

// ─── JS interop ───────────────────────────────────────────────────────────────

@JS('downloadModelWithProgress')
external JSPromise<JSString> _download(JSString url, JSString? token, JSFunction onProgress);

@JS('purgeOpfsCache')
external JSPromise<JSNumber> _jsDeleteOpfsFiles(JSArray<JSString> fileNames);

@JS('listOpfsCache')
external JSPromise<JSArray<JSObject>> _jsListOpfsCache();

@JS('purgeEmbeddingCache')
external JSPromise<JSNumber> _jsPurgeEmbeddingCache(JSArray<JSString> patterns);

// ─────────────────────────────────────────────────────────────────────────────

ModelDownloaderBuilder createDownloader(
  String url,
  String fileName, {
  String? token,
  String? directoryPath,
}) => WebModelDownloader(url, fileName, token: token);

Future<String> createXFileInstaller(XFile xFile) async {
  final web.File blob = (xFile as dynamic).webFile;
  return web.URL.createObjectURL(blob);
}

/// Creates an object-URL pointing directly at the user's local disk file.
/// Zero-copy – the browser streams directly from disk.
Future<String> createBlobInstaller(Object blob) async =>
    web.URL.createObjectURL(blob as web.Blob);

/// Purges cached model files on web.
///
/// Clears two storage locations:
/// 1. OPFS (Origin Private File System) — where `downloadModelWithProgress`
///    stores `.litertlm` model files.
/// 2. Cache Storage — where transformers.js caches tokenizer JSON and ONNX
///    shards for the embedding model.
///
/// Returns total bytes freed.
Future<int> purgeModelCacheImpl() async {
  await WebScriptLoader.ensureJsLoaded();
  int freed = 0;

  // 1. Clear OPFS model files.
  try {
    final opfsFree = await _jsDeleteOpfsFiles(JSArray<JSString>()).toDart;
    freed += opfsFree.toDartDouble.toInt();
  } catch (_) {}

  // 2. Clear Cache Storage entries for the embedding model.
  // Match URLs containing these patterns (covers HF and ONNX community paths).
  try {
    final patterns = ['embeddinggemma', 'onnx-community', 'litert-community']
        .map((p) => p.toJS)
        .toList();
    final jsArr = patterns.fold(JSArray<JSString>(), (arr, p) { arr.add(p); return arr; });
    final embedFree = await _jsPurgeEmbeddingCache(jsArr).toDart;
    freed += embedFree.toDartDouble.toInt();
  } catch (_) {}

  return freed;
}

/// Lists all files in the OPFS cache.
/// Returns `[{ 'name': String, 'size': int }, ...]`.
Future<List<Map<String, dynamic>>> listOpfsCacheImpl() async {
  await WebScriptLoader.ensureJsLoaded();
  try {
    final jsArray = await _jsListOpfsCache().toDart;
    final result  = <Map<String, dynamic>>[];
    for (int i = 0; i < jsArray.length; i++) {
      final obj  = jsArray.getProperty(i.toJS) as JSObject;
      final name = (obj.getProperty('name'.toJS) as JSString).toDart;
      final size = (obj.getProperty('size'.toJS) as JSNumber).toDartDouble.toInt();
      result.add({'name': name, 'size': size});
    }
    return result;
  } catch (_) {
    return [];
  }
}

class WebModelDownloader extends ModelDownloaderBuilder {
  WebModelDownloader(super.url, super.fileName, {super.token, super.directoryPath});

  @override
  Future<String> install() async {
    await WebScriptLoader.ensureJsLoaded();
    final void Function(JSNumber) onProg =
        (JSNumber p) => onProgressCallback?.call(p.toDartDouble);
    try {
      final result = await _download(url.toJS, token?.toJS, onProg.toJS).toDart;
      return result.toDart;
    } catch (e) {
      throw Exception('Web download failed: $e');
    }
  }
}