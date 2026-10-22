import 'model_picker_stub.dart'
    if (dart.library.io) 'model_picker_mobile.dart'
    if (dart.library.js_interop) 'model_picker_web.dart';

// ─────────────────────────────────────────────────────────────────────────────
// GemmaModelPicker — cross-platform local-file & media picker
// ─────────────────────────────────────────────────────────────────────────────

/// Opens the platform's native file picker and returns an installed model path
/// (or OPFS key on web) ready to pass to [GemmaLoader.initLlm] /
/// [GemmaLoader.initEmbedding].
///
/// Also provides helpers for media gallery selection, video selection,
/// and directory picking for Smart Album and Video Moment Finder.
abstract final class GemmaModelPicker {
  /// Picks a single model file.
  static Future<String?> pick() => pickModelImpl();

  /// Picks a single video file.
  static Future<String?> pickVideo() => pickVideoImpl();

  /// Picks a media directory/folder.
  static Future<String?> pickFolder() => pickFolderImpl();

  /// Picks multiple image and video files.
  static Future<List<String>> pickMultipleMedia() => pickMultipleMediaImpl();

  /// Lists media files within a selected folder tree URI.
  static Future<List<Map<String, dynamic>>> listFilesInFolder(String folderUri) =>
      listFilesInFolderImpl(folderUri);
}
