import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'dart:convert';
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

@JS('executeJsSkill')
external JSPromise<JSString> _executeJsSkill(JSString scriptHtml, JSString argsJson, JSString secret, JSNumber timeoutMs);

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

  bool _isGemma4 = false;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Initialises the MediaPipe LLM engine in the browser.
  Future<void> init(InferenceConfig config) async {
    await WebScriptLoader.ensureJsLoaded();

    final options = JSObject();
    options.setProperty('modelPath'.toJS, config.modelPath.toJS);
    options.setProperty('assetBase'.toJS, WebScriptLoader.assetBase.toJS);

    // Limit Web maxTokens to 8192 (GPU memory limit).
    final int safeMaxTokens = config.maxTokens.clamp(512, 8192);
    options.setProperty('maxTokens'.toJS, safeMaxTokens.toJS);

    // IMPORTANT: Do NOT pass supportAudio=true for Gemma 4 web models.
    // Gemma 4's WASM binary does not include the audio modality — passing
    // supportAudio causes a 'RuntimeError: memory access out of bounds' crash
    // during GPU init.  We detect Gemma 4 by its well-known path substrings.
    // Note: E2B and E4B are Gemma 3 variants and DO support audio/images.
    _isGemma4 = FlutterLocalGemma().currentModelIsGemma4;

    if (config.maxNumImages != null) {
      options.setProperty('maxNumImages'.toJS, _isGemma4 ? 0.toJS : config.maxNumImages!.toJS);
    }

    final effectiveSupportAudio = config.supportAudio && !_isGemma4;
    options.setProperty('supportAudio'.toJS, effectiveSupportAudio.toJS);
    options.setProperty('isGemma4'.toJS, _isGemma4.toJS);

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
       
        final base64String = base64Encode(part.bytes);
        final mimeType = _detectImageFormat(part.bytes);
        final dataUrl = 'data:$mimeType;base64,$base64String';
       
        final obj = JSObject();
        obj.setProperty('imageSource'.toJS, dataUrl.toJS);
        jsArray.add(obj);
      } else if (part is AudioPart) {
       
        final base64String = base64Encode(part.bytes);
        final dataUrl = 'data:audio/wav;base64,$base64String';
       
        final obj = JSObject();
        obj.setProperty('audioSource'.toJS, dataUrl.toJS);
        jsArray.add(obj);
      }
    }
    _webBuffer.clear();

    // The callback is typed to satisfy dart:js_interop's strict typing rules.
    void cb(JSAny res, JSAny done) {
      if (controller.isClosed) return;
      final text   = (res  as JSString).toDart;
      final isDone = (done as JSBoolean).toDart;
      controller.add({'partialResult': text, 'done': isDone});
      if (isDone) {
        controller.close();
        _activeController = null;
      }
    }

    _generateResponse(jsArray, cb.toJS);

    return controller.stream;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  String _detectImageFormat(Uint8List bytes) {
    if (bytes.length >= 2) {
      if (bytes[0] == 0xFF && bytes[1] == 0xD8) return 'image/jpeg';
      if (bytes[0] == 0x89 && bytes[1] == 0x50) return 'image/png';
      if (bytes[0] == 0x47 && bytes[1] == 0x49) return 'image/gif';
      if (bytes.length >= 12 &&
          bytes[8] == 0x57 &&
          bytes[9] == 0x45 &&
          bytes[10] == 0x42 &&
          bytes[11] == 0x50) return 'image/webp';
    }
    return 'image/jpeg'; // Default fallback
  }

  /// Executes a JS-based skill inside a sandboxed iframe.
  Future<String> executeJsSkill(String scriptHtml, String argsJson, String secret, int timeoutMs) async {
    await WebScriptLoader.ensureJsLoaded();
    final result = await _executeJsSkill(scriptHtml.toJS, argsJson.toJS, secret.toJS, timeoutMs.toJS).toDart;
    return result.toDart;
  }
}