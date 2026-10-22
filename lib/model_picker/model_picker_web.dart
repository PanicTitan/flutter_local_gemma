import 'dart:async';
import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// Opens an `<input type="file">` element and installs the picked file into
/// OPFS via [FlutterLocalGemma.installModel().fromWebBlob()].
Future<String?> pickModelImpl() async {
  final input = web.document.createElement('input') as web.HTMLInputElement
    ..type = 'file'
    ..accept = '.litertlm,.bin,.tflite';

  final completer = Completer<web.File?>();
  input.onchange = ((web.Event _) {
    final files = input.files;
    completer.complete(files != null && files.length > 0 ? files.item(0) : null);
  }).toJS;
  input.click();

  final file = await completer.future;
  if (file == null) return null;
  final blobUrl = await FlutterLocalGemma.installModel().fromWebBlob(file);
  return '$blobUrl#${file.name}';
}

/// Opens an `<input type="file">` for videos on Web.
Future<String?> pickVideoImpl() async {
  final input = web.document.createElement('input') as web.HTMLInputElement
    ..type = 'file'
    ..accept = 'video/*';

  final completer = Completer<web.File?>();
  input.onchange = ((web.Event _) {
    final files = input.files;
    completer.complete(files != null && files.length > 0 ? files.item(0) : null);
  }).toJS;
  input.click();

  final file = await completer.future;
  if (file == null) return null;
  final blobUrl = web.URL.createObjectURL(file);
  return '$blobUrl#${file.name}';
}

/// Opens an `<input type="file">` directory selector on Web.
Future<String?> pickFolderImpl() async {
  final input = web.document.createElement('input') as web.HTMLInputElement
    ..type = 'file'
    ..setAttribute('webkitdirectory', '')
    ..setAttribute('directory', '');

  final completer = Completer<String?>();
  input.onchange = ((web.Event _) {
    final files = input.files;
    if (files != null && files.length > 0) {
      completer.complete('web_folder_${DateTime.now().millisecondsSinceEpoch}');
    } else {
      completer.complete(null);
    }
  }).toJS;
  input.click();
  return completer.future;
}

/// Opens an `<input type="file">` with multiple media selection on Web.
Future<List<String>> pickMultipleMediaImpl() async {
  final input = web.document.createElement('input') as web.HTMLInputElement
    ..type = 'file'
    ..multiple = true
    ..accept = 'image/*,video/*';

  final completer = Completer<List<String>>();
  input.onchange = ((web.Event _) {
    final files = input.files;
    final urls = <String>[];
    if (files != null) {
      for (int i = 0; i < files.length; i++) {
        final f = files.item(i);
        if (f != null) {
          urls.add('${web.URL.createObjectURL(f)}#${f.name}');
        }
      }
    }
    completer.complete(urls);
  }).toJS;
  input.click();
  return completer.future;
}

/// Fallback for scanning folder on Web (typically handled by file input change event).
Future<List<Map<String, dynamic>>> listFilesInFolderImpl(String folderUri) async {
  return [];
}