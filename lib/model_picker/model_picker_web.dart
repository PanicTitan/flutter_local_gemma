import 'dart:async';
import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// Opens an `<input type="file">` element and installs the picked file into
/// OPFS via [FlutterLocalGemma.installModel().fromWebBlob()].
///
/// Returns the OPFS key / path string, or `null` if the user dismisses the
/// picker without selecting a file.
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
  return FlutterLocalGemma.installModel().fromWebBlob(file);
}