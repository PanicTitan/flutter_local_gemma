import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'package:web/web.dart' as web;
import 'dart:ui_web' as ui_web;

class WebScriptLoader {
  static String? _cachedAssetBase;
  static Completer<void>? _loadCompleter;

  static String get assetBase {
    if (_cachedAssetBase != null) return _cachedAssetBase!;
    final baseHref = ui_web.BrowserPlatformLocation().getBaseHref() ?? '/';
    _cachedAssetBase = Uri.parse(
      web.window.location.href,
    ).resolve('${baseHref}assets/packages/flutter_local_gemma/web/').toString();
    return _cachedAssetBase!;
  }

  static Future<void> ensureJsLoaded() async {
    if (_loadCompleter != null) return _loadCompleter!.future;
    _loadCompleter = Completer<void>();

    // Initialize pdfjsLib shim for other plugins
    if (web.window.getProperty('pdfjsLib'.toJS).isUndefinedOrNull) {
      web.window.setProperty('pdfjsLib'.toJS, JSObject());
    }

    final script =
        web.document.createElement('script') as web.HTMLScriptElement;
    script.type = 'module';
    script.src = '${assetBase}dist/gemma_web.js';

    script.onload = (web.Event e) {
      if (!_loadCompleter!.isCompleted) _loadCompleter!.complete();
    }.toJS;

    web.document.head!.appendChild(script);
    return _loadCompleter!.future;
  }
}
