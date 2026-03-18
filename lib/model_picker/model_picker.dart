import 'model_picker_stub.dart'
    if (dart.library.io) 'model_picker_mobile.dart'
    if (dart.library.js_interop) 'model_picker_web.dart';

// ─────────────────────────────────────────────────────────────────────────────
// GemmaModelPicker — cross-platform local-file picker
// ─────────────────────────────────────────────────────────────────────────────

/// Opens the platform's native file picker and returns an installed model path
/// (or OPFS key on web) ready to pass to [GemmaLoader.initLlm] /
/// [GemmaLoader.initEmbedding].
///
/// Returns `null` if the user cancels.
///
/// ## Platform behaviour
/// - **Android:** Opens the native SAF file picker via a `MethodChannel`.
///   The plugin attempts a zero-copy `content://` URI approach first.
///   If the provider does not expose a mappable FD (e.g. Google Drive), the
///   file is transparently copied to internal storage.
/// - **Web:** Shows an `<input type="file">` element and installs the picked
///   file into OPFS via `FlutterLocalGemma.installModel().fromWebBlob()`.
/// - **Other / desktop:** Returns `null` (stub).
///
/// ## Usage
/// ```dart
/// final path = await GemmaModelPicker.pick();
/// if (path != null) {
///   await GemmaLoader.initLlm(path: path);
/// }
/// ```
abstract final class GemmaModelPicker {
  static Future<String?> pick() => pickModelImpl();
}
