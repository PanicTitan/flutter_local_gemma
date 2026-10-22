import 'package:flutter/services.dart';

const MethodChannel _ch = MethodChannel('native_file_picker');

/// Opens the native Android file picker and returns the selected file's
/// `content://` URI string, or `null` if the user cancelled.
Future<String?> pickModelImpl() async {
  try {
    return await _ch.invokeMethod<String>('pickFile');
  } on PlatformException catch (e) {
    throw Exception('File picker error [${e.code}]: ${e.message}');
  }
}

/// Opens the native Android video picker and returns the selected video's
/// `content://` URI string, or `null` if cancelled.
Future<String?> pickVideoImpl() async {
  try {
    return await _ch.invokeMethod<String>('pickVideo');
  } on PlatformException catch (e) {
    throw Exception('Video picker error [${e.code}]: ${e.message}');
  }
}

/// Opens the native Android directory picker (`ACTION_OPEN_DOCUMENT_TREE`)
/// and returns the selected folder's persistable tree URI string, or `null`.
Future<String?> pickFolderImpl() async {
  try {
    return await _ch.invokeMethod<String>('pickFolder');
  } on PlatformException catch (e) {
    throw Exception('Folder picker error [${e.code}]: ${e.message}');
  }
}

/// Opens the native Android media picker allowing multi-selection of photos and videos.
Future<List<String>> pickMultipleMediaImpl() async {
  try {
    final List<dynamic>? list = await _ch.invokeMethod('pickMultipleMedia');
    return list?.cast<String>() ?? [];
  } on PlatformException catch (e) {
    throw Exception('Media picker error [${e.code}]: ${e.message}');
  }
}

/// Scans child media files from a given tree URI on Android.
Future<List<Map<String, dynamic>>> listFilesInFolderImpl(String folderUri) async {
  try {
    final List<dynamic>? list = await _ch.invokeMethod('listFilesInFolder', {'folderUri': folderUri});
    if (list == null) return [];
    return list.map((e) => Map<String, dynamic>.from(e as Map)).toList();
  } on PlatformException catch (e) {
    throw Exception('List files error [${e.code}]: ${e.message}');
  }
}