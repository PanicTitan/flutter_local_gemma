/// Cross-platform HTTP text fetch utility.
///
/// Uses conditional exports to select the appropriate implementation:
/// - `http_fetch_io.dart` for native platforms (dart:io)
/// - `http_fetch_web.dart` for web (dart:js_interop)
/// - `http_fetch_stub.dart` as fallback
export 'http_fetch_stub.dart'
    if (dart.library.io) 'http_fetch_io.dart'
    if (dart.library.js_interop) 'http_fetch_web.dart';
