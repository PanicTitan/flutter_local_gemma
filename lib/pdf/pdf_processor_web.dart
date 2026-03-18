import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import '../utils/web_script_loader.dart';

@JS('initPdfWorker')
external JSPromise<JSBoolean> _initPdfWorker(JSString assetBase);
@JS('extractPdf')
external JSPromise<JSArray<JSAny>> _extractPdf(
  JSUint8Array b,
  JSString m,
  JSString f,
  JSNumber? s,
  JSNumber? e,
  JSNumber r,
);

class PdfProcessorWeb {
  static bool _workerReady = false;

  Future<void> initPdfWorker() async {
    if (_workerReady) return;
    await WebScriptLoader.ensureJsLoaded();
    await _initPdfWorker(WebScriptLoader.assetBase.toJS).toDart;
    _workerReady = true;
  }

  Future<List<Map<String, dynamic>>> extractPdf(
    Uint8List bytes,
    String mode,
    String filter,
    int? start,
    int? end,
    double scale,
  ) async {
    await initPdfWorker();
    final jsArray = await _extractPdf(
      bytes.toJS,
      mode.toJS,
      filter.toJS,
      start?.toJS,
      end?.toJS,
      scale.toJS,
    ).toDart;
    final res = <Map<String, dynamic>>[];
    for (int i = 0; i < jsArray.length; i++) {
      final item = jsArray.getProperty(i.toJS) as JSObject;
      final type = (item.getProperty('type'.toJS) as JSString).toDart;
      final data = type == 'text'
          ? (item.getProperty('data'.toJS) as JSString).toDart
          : (item.getProperty('data'.toJS) as JSUint8Array).toDart;
      res.add({'type': type, 'data': data});
    }
    return res;
  }
}
