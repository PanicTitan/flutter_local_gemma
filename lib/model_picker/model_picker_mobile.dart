import 'package:flutter/services.dart';

const MethodChannel _ch = MethodChannel('native_file_picker');

/// Opens the native Android file picker and returns the selected file's
/// `content://` URI string, or `null` if the user cancelled.
///
/// The URI is passed directly to [FlutterLocalGemma().init] or
/// [EmbeddingPlugin().init]; the native plugin resolves it via
/// `ContentResolver`, preferring a zero-copy FD strategy and falling back to
/// an in-process copy when the provider does not expose a mappable descriptor.
Future<String?> pickModelImpl() async {
  try {
    return await _ch.invokeMethod<String>('pickFile');
  } on PlatformException catch (e) {
    throw Exception('File picker error [${e.code}]: ${e.message}');
  }
}