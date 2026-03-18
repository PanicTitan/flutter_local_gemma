import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:web/web.dart' as web;
import '../types/content_parts.dart';
import '../utils/web_script_loader.dart';
import 'gemma.dart';

// ─── JS Interop declarations ──────────────────────────────────────────────────

@JS('initLLM')
external JSPromise<JSBoolean> _initLLM(JSAny options);

@JS('generateResponse')
external void _generateResponse(JSAny parts, JSFunction callback);

@JS('cancelProcessing')
external void _cancelProcessing();

@JS('unloadLLM')
external void _unloadLLM();

@JS('countTokens')
external JSPromise<JSNumber> _countTokens(JSString text);

// ─────────────────────────────────────────────────────────────────────────────

/// Web implementation of the FlutterLocalGemma engine interface.
///
/// Bridges Dart calls to the MediaPipe GenAI JavaScript API via `dart:js_interop`.
/// This class is a singleton; the JS engine is also a singleton inside `main.ts`.
class FlutterLocalGemmaWeb {
  static final FlutterLocalGemmaWeb _instance = FlutterLocalGemmaWeb._internal();
  factory FlutterLocalGemmaWeb() => _instance;
  FlutterLocalGemmaWeb._internal();

  /// Staging buffer for content parts accumulated between [addToBuffer] calls
  /// and consumed on [generateResponse].
  final List<ContentPart> _webBuffer = [];

  /// The controller for the currently active streaming generation.
  /// Null when idle.
  StreamController<Map<String, dynamic>>? _activeController;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Initialises the MediaPipe LLM engine in the browser.
  Future<void> init(InferenceConfig config) async {
    await WebScriptLoader.ensureJsLoaded();

    final options    = JSObject();
    final baseOpts   = JSObject();

    options.setProperty('assetBase'.toJS,  WebScriptLoader.assetBase.toJS);
    baseOpts.setProperty('modelPath'.toJS, config.modelPath.toJS);
    options.setProperty('baseOptions'.toJS, baseOpts);
    options.setProperty('maxTokens'.toJS,   config.maxTokens.toJS);
    if (config.maxNumImages != null) options.setProperty('maxNumImages'.toJS, config.maxNumImages!.toJS);
    options.setProperty('supportAudio'.toJS, config.supportAudio.toJS);

    await _initLLM(options).toDart;
  }

  /// Releases the LLM instance and all associated GPU memory from the browser.
  /// After this call [init] must be called again before generating.
  void unload() {
    _activeController?.close();
    _activeController = null;
    _webBuffer.clear();
    _unloadLLM();
  }

  // ── Buffer helpers ────────────────────────────────────────────────────────

  /// Appends [parts] to the in-memory staging buffer.
  void addToBuffer(List<ContentPart> parts) => _webBuffer.addAll(parts);

  /// Clears the staging buffer without triggering generation.
  void clearBuffer() => _webBuffer.clear();

  // ── Token counting ────────────────────────────────────────────────────────

  /// Returns the estimated token count for [text] via the JS tokenizer.
  Future<int> countTokensWeb(String text) async {
    await WebScriptLoader.ensureJsLoaded();
    final result = await _countTokens(text.toJS).toDart;
    return result.toDartInt;
  }

  // ── Generation ────────────────────────────────────────────────────────────

  /// Cancels the active generation. Emits a `[Stopped]` token and closes the
  /// stream controller so the Dart `await for` loop terminates cleanly.
  Future<void> cancelProcessing() async {
    _cancelProcessing();
    if (_activeController != null && !_activeController!.isClosed) {
      _activeController!.add({'partialResult': ' [Stopped]', 'done': true});
      _activeController!.close();
    }
    _activeController = null;
  }

  /// Consumes [_webBuffer], calls `generateResponse` on the JS side, and
  /// returns a broadcast stream of token events.
  ///
  /// Each event is a map with `'partialResult'` (String) and `'done'` (bool).
  Stream<Map<String, dynamic>> generateResponse({
    required AutoStopConfig autoStopConfig,
  }) {
    final controller = StreamController<Map<String, dynamic>>();
    _activeController = controller;

    // Serialise content parts into the JS array format expected by MediaPipe.
    final jsArray = JSArray();
    for (final part in _webBuffer) {
      if (part is TextPart) {
        jsArray.add(part.text.toJS);
      } else if (part is ImagePart) {
        jsArray.add(_bytesToBlobObj(part.bytes, 'imageSource', 'image/png'));
      } else if (part is AudioPart) {
        jsArray.add(_bytesToBlobObj(part.bytes, 'audioSource', 'audio/wav'));
      }
    }
    _webBuffer.clear();

    // The callback is typed as `void Function(JSAny, JSAny)` to satisfy
    // dart:js_interop's strict typing rules.
    final void Function(JSAny, JSAny) cb = (JSAny res, JSAny done) {
      if (controller.isClosed) return;
      final text   = (res  as JSString).toDart;
      final isDone = (done as JSBoolean).toDart;
      controller.add({'partialResult': text, 'done': isDone});
      if (isDone) {
        controller.close();
        _activeController = null;
      }
    };

    _generateResponse(jsArray, cb.toJS);

    return controller.stream;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /// Wraps [bytes] in a JS Blob and returns a JSObject with [key] set to the
  /// object-URL string, which MediaPipe can consume as a media source.
  JSObject _bytesToBlobObj(Uint8List bytes, String key, String mime) {
    final blob = web.Blob(
      [bytes.toJS].toJS,
      web.BlobPropertyBag(type: mime),
    );
    final obj = JSObject();
    obj.setProperty(key.toJS, web.URL.createObjectURL(blob).toJS);
    return obj;
  }
}