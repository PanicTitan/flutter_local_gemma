import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../model_installer/model_installer.dart';
export '../model_installer/model_installer.dart';

import '../types/content_parts.dart';
import 'gemma_state.dart';
import 'gemma_web.dart' if (dart.library.io) 'gemma_stub.dart';

// ─── Configuration types ──────────────────────────────────────────────────────

enum PreferredBackend { cpu, gpu }

enum ModelType { gemmaIt, gemmaBase }

/// Controls the built-in repetition-loop detector.
class AutoStopConfig {
  /// Whether to enable loop detection at all.
  final bool enabled;

  /// How many consecutive repetitions of a pattern trigger a stop.
  final int maxRepetitions;

  /// Minimum length (in characters) of a repeating unit to count as a loop.
  ///
  /// **Android note:** this value is NOT forwarded to the native LiteRT-LM SDK.
  /// The Kotlin side hardcodes 3 regardless of what you set here.
  /// Only [maxRepetitions] is applied on Android.
  final int minRepeatCharLen;

  /// Hard cap on generated tokens (0 = no cap).
  final int maxOutputTokens;

  const AutoStopConfig({
    this.enabled          = true,
    this.maxRepetitions   = 5,
    this.minRepeatCharLen = 3,
    this.maxOutputTokens  = 8192,
  });
}

/// Configuration for loading the inference engine.
class InferenceConfig {
  final String modelPath;

  /// KV-cache capacity. Larger values use more RAM.
  final int maxTokens;

  /// Preferred compute backend for text generation *and* vision encoding.
  ///
  /// Defaults to [PreferredBackend.gpu] on Android: GPU is meaningfully
  /// faster for both LLM decoding and image-token encoding on the vast
  /// majority of devices, and [FlutterLocalGemma.init] automatically falls
  /// back to CPU if GPU delegate initialisation fails (missing/broken
  /// OpenCL, emulator, etc.) — so this is safe to leave at its default.
  ///
  /// Pass [PreferredBackend.cpu] explicitly to force CPU (e.g. for
  /// reproducible benchmarking, or on devices you know lack usable GPU
  /// delegates).
  ///
  /// **Audio note:** [supportAudio] is always served by the CPU backend
  /// regardless of this setting — LiteRT-LM does not currently ship a GPU
  /// audio encoder.
  final PreferredBackend backend;

  final int? maxNumImages;
  final bool supportAudio;
  /// Whether to enable MTP (Multi-Token Prediction / speculative decoding).
  ///
  /// **Android note:** This value is **ignored**. MTP is auto-detected from
  /// the model file via `Capabilities(path).hasSpeculativeDecodingSupport()`.
  /// If the model contains the MTP drafter shard it will always be enabled;
  /// if not, it silently remains off. No flag needed from your code.
  ///
  /// **Web note:** MTP is not available in the current MediaPipe build and
  /// this value is silently ignored.
  final bool enableMtp;

  /// Optional model name used to identify the model version (e.g. for Gemma 4 
  /// detection on Web when the OPFS URL strips the file name).
  final String? modelName;

  /// Optional download/initialisation progress callback.
  final void Function(double progress)? onProgress;

  InferenceConfig({
    required this.modelPath,
    this.maxTokens   = 4096,
    this.backend     = PreferredBackend.gpu,
    this.maxNumImages,
    this.supportAudio = false,
    this.enableMtp    = true,
    this.modelName,
    this.onProgress,
  });
}

/// Per-session sampling parameters.
///
/// **Android sampling note:** [temperature], [topP], and [topK] are currently
/// not applied on Android. The LiteRT-LM SDK does not expose per-generation
/// sampling parameters on its Conversation API — these values are logged but
/// have no effect on the output distribution. They are fully respected on Web.
/// This will be resolved in a future LiteRT-LM SDK update.
class SessionConfig {
  /// Sampling temperature. **Android: no-op** (see class doc).
  final double temperature;

  /// Top-p nucleus sampling. **Android: no-op** (see class doc).
  final double topP;

  /// Top-k sampling. **Android: no-op** (see class doc).
  final int topK;
  final int? randomSeed;
  final String? systemPrompt;
  final AutoStopConfig autoStopConfig;
  final bool enableThinking;
  final bool enableMultimodality;
  final List<Object> skills;

  /// Controls the tool-call architecture used when [AgentChat] is active.
  ///
  /// When `false` (default): JSON prompt injection + regex parsing on the Dart side.
  /// The model outputs `<|tool_call>{"name":...,"arguments":{...}}<tool_call|>` tokens
  /// which are parsed by [AgentChat].
  ///
  /// When `true` (Android only): the LiteRT-LM native constrained decoding path via
  /// `GemmaAgentToolSet`. The C++ engine calls `load_skill`, `run_js`, and
  /// `execute_skill` natively. JS skills run in a hidden WebView pool.
  /// Dart skills receive callbacks via MethodChannel. Falls back to JSON
  /// prompt mode on Web.
  ///
  /// **Note:** This does NOT affect [temperature], [topP], or [topK] — those
  /// are still no-ops on Android regardless of this value.
  final bool nativeToolCalling;

  SessionConfig({
    this.temperature    = 0.8,
    this.topP           = 0.95,
    this.topK           = 40,
    this.randomSeed,
    this.systemPrompt,
    this.autoStopConfig = const AutoStopConfig(),
    this.enableThinking = false,
    this.enableMultimodality = true,
    this.skills = const [],
    this.nativeToolCalling = false,
  });

  // Sentinel object for clearing nullable fields in copyWith.
  static const _unset = Object();

  /// Returns a copy of this config with the given fields replaced.
  ///
  /// To **clear** [randomSeed] or [systemPrompt] back to null, pass
  /// `clearRandomSeed: true` or `clearSystemPrompt: true`:
  /// ```dart
  /// config.copyWith(clearSystemPrompt: true)
  /// ```
  SessionConfig copyWith({
    double? temperature,
    double? topP,
    int? topK,
    Object? randomSeed = _unset,
    Object? systemPrompt = _unset,
    AutoStopConfig? autoStopConfig,
    bool? enableThinking,
    bool? enableMultimodality,
    List<Object>? skills,
    bool? nativeToolCalling,
  }) {
    return SessionConfig(
      temperature: temperature ?? this.temperature,
      topP: topP ?? this.topP,
      topK: topK ?? this.topK,
      randomSeed: identical(randomSeed, _unset)
          ? this.randomSeed
          : randomSeed as int?,
      systemPrompt: identical(systemPrompt, _unset)
          ? this.systemPrompt
          : systemPrompt as String?,
      autoStopConfig: autoStopConfig ?? this.autoStopConfig,
      enableThinking: enableThinking ?? this.enableThinking,
      enableMultimodality: enableMultimodality ?? this.enableMultimodality,
      skills: skills ?? this.skills,
      nativeToolCalling: nativeToolCalling ?? this.nativeToolCalling,
    );
  }
}

/// Snapshot of the KV-cache / context window usage for a [ChatSession].
class TokenStats {
  /// Rough estimate of tokens currently consumed by context + generated text.
  final int estimatedUsed;

  /// Maximum token capacity declared in [InferenceConfig.maxTokens].
  final int maxContext;

  /// How many tokens remain before the context window is full.
  int get remaining => (maxContext - estimatedUsed).clamp(0, maxContext);

  /// Percentage of the context window used (0–100).
  double get usedPercent =>
      maxContext == 0 ? 0 : (estimatedUsed / maxContext * 100).clamp(0, 100);

  const TokenStats({required this.estimatedUsed, required this.maxContext});

  @override
  String toString() =>
      'TokenStats($estimatedUsed / $maxContext used, $remaining remaining)';
}

// ─────────────────────────────────────────────────────────────────────────────
// FlutterLocalGemma  —  singleton engine wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// The main entry point for on-device inference.
///
/// ## Lifecycle
/// ```dart
/// final gemma = FlutterLocalGemma();
/// await gemma.init(InferenceConfig(modelPath: '...'));
///
/// final session = await gemma.createSession(config: SessionConfig());
/// final response = await session.generateResponseFuture([TextPart('Hello!')]);
/// await session.dispose();
///
/// await gemma.dispose(); // releases all native / GPU memory
/// ```
class FlutterLocalGemma {
  static final FlutterLocalGemma _instance = FlutterLocalGemma._internal();
  factory FlutterLocalGemma() => _instance;
  FlutterLocalGemma._internal();

  static const MethodChannel _channel             = MethodChannel('gemma_bundled');
  static const EventChannel  _eventChannel        = EventChannel('gemma_stream');
  static const EventChannel  _copyProgressChannel = EventChannel('gemma_copy_progress');

  bool _isInitialized = false;
  int  _maxTokens     = 0;
  String? _currentModelPath;
  String? _currentModelName;

  Future<dynamic> Function(Map<String, dynamic>)? _nativeToolHandler;

  void setNativeToolHandler(Future<dynamic> Function(Map<String, dynamic>)? handler) {
    _nativeToolHandler = handler;
  }

  Future<dynamic> _handleNativeMethodCall(MethodCall call) async {
    if (call.method == 'executeTool') {
      if (_nativeToolHandler != null) {
        final args = Map<String, dynamic>.from(call.arguments as Map);
        return await _nativeToolHandler!(args);
      }
    }
    throw PlatformException(code: 'Unimplemented', message: 'Method not implemented');
  }

  // ── Copy-progress stream ───────────────────────────────────────────────────

  /// Emits copy-progress values in **[0, 100]** while a local model file
  /// from the Android file picker is being copied to internal storage.
  ///
  /// This stream only emits on Android and only when the picked file requires
  /// a copy (Strategy B/C — e.g. Google Drive, MTP, cloud-backed files).
  /// For files on local storage (Downloads, internal storage, SD card)
  /// the plugin uses Strategy A instead — zero bytes are copied and this
  /// stream stays silent.
  ///
  /// Subscribe **before** calling [init] with a `content://` URI:
  ///
  /// ```dart
  /// StreamSubscription? _copySub;
  ///
  /// Future<void> loadLocalModel(String contentUri) async {
  ///   _copySub = FlutterLocalGemma().copyProgressStream.listen(
  ///     (pct) => setState(() => _copyProgress = pct),
  ///   );
  ///   try {
  ///     await FlutterLocalGemma().init(InferenceConfig(modelPath: contentUri));
  ///   } finally {
  ///     await _copySub?.cancel();
  ///     _copySub = null;
  ///   }
  /// }
  /// ```
  ///
  /// On web and iOS this stream never emits (no copy step is performed).
  Stream<double> get copyProgressStream {
    if (kIsWeb) return const Stream.empty();
    return _copyProgressChannel
        .receiveBroadcastStream()
        .map((event) => (event as num).toDouble());
  }

  /// Persistent subscription to the Android EventChannel stream.
  ///
  /// Subscribed once in [init] and kept alive for the engine lifetime to avoid
  /// race conditions where the sink becomes null between successive generations.
  StreamSubscription? _eventSubscription;

  /// Broadcast controller that multiplexes the single Android EventChannel
  /// stream to any number of [ChatSession] listeners.
  final _eventController =
      StreamController<Map<dynamic, dynamic>>.broadcast();

  // ── Public API ─────────────────────────────────────────────────────────────

  /// Convenience factory for [ModelInstaller].
  static ModelInstaller installModel({ModelType? modelType}) =>
      ModelInstaller(fileName: 'gemma-model.litertlm');

  /// Whether the engine has been successfully initialised.
  bool get isInitialized => _isInitialized;

  /// The maximum context-window size declared during [init].
  int get maxTokens => _maxTokens;

  /// The path of the currently loaded model, or null if none is loaded.
  String? get currentModelPath => _currentModelPath;

  /// The name of the currently loaded model, or null if unknown.
  String? get currentModelName => _currentModelName;

  /// Whether the currently loaded model is identified as Gemma 4.
  bool get currentModelIsGemma4 {
    final str = (_currentModelName ?? _currentModelPath ?? '').toLowerCase();
    return str.contains('gemma-4') ||
           str.contains('gemma_4') ||
           str.contains('gemma4');
  }

  /// Whether the currently loaded model is identified as an ONNX model.
  bool get currentModelIsOnnx =>
      (_currentModelName ?? _currentModelPath ?? '').toLowerCase().contains('onnx');

  /// Whether the currently loaded model supports image/vision inputs.
  bool get currentModelSupportsVision {
    if (kIsWeb) {
      // On Web, only ONNX models or Gemma 3n models support vision inputs.
      // Standard LiteRT / MediaPipe web models (e.g. Gemma 4 .task / .litertlm)
      // do not have Web vision calculator models and cannot process images.
      if (currentModelIsOnnx) return true;
      final name = (_currentModelName ?? _currentModelPath ?? '').toLowerCase();
      if (name.contains('gemma-3n') || name.contains('gemma_3n') || name.contains('gemma3n')) {
        return true;
      }
      return false;
    }
    return true;
  }

  /// Whether the currently loaded model supports audio inputs.
  bool get currentModelSupportsAudio {
    if (kIsWeb) {
      // On Web, only ONNX models or Gemma 3n models support audio inputs.
      // Standard LiteRT / MediaPipe web models (e.g. Gemma 4 .task / .litertlm)
      // do not have Web audio processing support.
      if (currentModelIsOnnx) return true;
      final name = (_currentModelName ?? _currentModelPath ?? '').toLowerCase();
      if (name.contains('gemma-3n') || name.contains('gemma_3n') || name.contains('gemma3n')) {
        return true;
      }
      return false;
    }
    final name = (_currentModelName ?? _currentModelPath ?? '').toLowerCase();
    if (name.contains('e4b') && !name.contains('3n')) {
      return false;
    }
    return true;
  }

  /// Initialises the inference engine.
  ///
  /// Calling [init] when the engine is already initialised is a no-op; call
  /// [dispose] first if you need to swap to a different model — or use
  /// `PluginModelLoader.switchLlm(...)` (in `helpers/model_loader.dart`),
  /// which does that for you safely.
  ///
  /// Tracks [GemmaState.loadState] throughout: [ModelLoadState.loading]
  /// while this call is in flight, [ModelLoadState.loaded] on success, and
  /// [ModelLoadState.error] (with [GemmaState.lastError] set) if it throws.
  /// Note: [ModelLoadState.downloading] is set by the *helpers* that
  /// download a model first (`PluginModelLoader`/`GemmaLoader`) — `init()`
  /// itself only ever resolves an already-local file, it doesn't download.
  Future<void> init(InferenceConfig config) async {
    if (_isInitialized) {
      if (_currentModelPath == null && config.modelPath.isNotEmpty) {
        _currentModelPath = config.modelPath;
      }
      _currentModelName ??= config.modelName ?? _extractNameFromPath(config.modelPath);
      return;
    }
    if (GemmaState.instance.isLoadBusy) {
      throw StateError(
        'A model load/download/unload is already in progress '
        '(GemmaState.instance.loadState == '
        '${GemmaState.instance.loadState.value}). Await it (or check '
        'GemmaState.instance.loadState) before calling init() again.',
      );
    }

    GemmaState.instance.setLastError(null);
    GemmaState.instance.setLoadState(ModelLoadState.loading);

    try {
      _maxTokens = config.maxTokens;
      _currentModelPath = config.modelPath;
      _currentModelName = config.modelName ?? _extractNameFromPath(config.modelPath);

      if (kIsWeb) {
        await FlutterLocalGemmaWeb().init(config);
      } else {
        // Subscribe to the Android EventChannel once here so there is never a
        // window where the sink is null between two consecutive generations.
        _eventSubscription ??= _eventChannel.receiveBroadcastStream().listen(
          (event) {
            if (event is Map) {
              _eventController.add(event);
            }
          },
          onError: _eventController.addError,
        );

        await _channel.invokeMethod('createModel', {
          'modelPath':       config.modelPath,
          'maxTokens':       config.maxTokens,
          'preferredBackend': config.backend.index,
          'maxNumImages':    config.maxNumImages,
          'supportAudio':    config.supportAudio,
          'enableMtp':       config.enableMtp,
        });
        _channel.setMethodCallHandler(_handleNativeMethodCall);
      }

      _isInitialized = true;
      GemmaState.instance.setLoadState(ModelLoadState.loaded);
    } catch (e) {
      _isInitialized = false;
      _currentModelPath = null;
      _currentModelName = null;
      GemmaState.instance.setLastError(e);
      GemmaState.instance.setLoadState(ModelLoadState.error);
      rethrow;
    }
  }

  String? _extractNameFromPath(String path) {
    if (path.isEmpty) return null;
    try {
      // First, check if there's a hash fragment (used by the Web picker).
      if (path.contains('#')) {
        final hash = path.split('#').last;
        if (hash.isNotEmpty) return hash;
      }
      // Otherwise, parse the URI and get the last path segment.
      final uri = Uri.parse(path);
      if (uri.pathSegments.isNotEmpty) {
        final name = uri.pathSegments.last;
        if (name.isNotEmpty) return name;
      }
      return path.split('/').last;
    } catch (_) {
      return path.split('/').last;
    }
  }

  /// Creates a new [ChatSession] with the given sampling parameters.
  ///
  /// On mobile, a new native `Conversation` object is created each time.
  /// On web, the LLM instance is reused and only the content buffer is reset.
  Future<ChatSession> createSession({SessionConfig? config}) async {
    if (!_isInitialized) {
      throw StateError('FlutterLocalGemma is not initialised. Call init() first.');
    }

    final conf = config ?? SessionConfig();

    if (kIsWeb) {
      return ChatSession(maxContext: _maxTokens, webConfig: conf);
    } else {
      await _channel.invokeMethod('createSession', {
        'temperature':     conf.temperature,
        'topP':            conf.topP,
        'topK':            conf.topK,
        'randomSeed':      conf.randomSeed,
        'systemPrompt':    conf.systemPrompt,
        'autoStopEnabled': conf.autoStopConfig.enabled,
        'maxRepetitions':  conf.autoStopConfig.maxRepetitions,
        'enableThinking':  conf.enableThinking,
        'nativeToolCalling': conf.nativeToolCalling,
      });
      return ChatSession(
        channel:    _channel,
        stream:     _eventChannel,
        maxContext: _maxTokens,
        webConfig:  null,
      );
    }
  }

  /// Generates a response for [input] in a single, self-contained call.
  ///
  /// Creates a temporary session, generates synchronously (or collects the
  /// full stream), then disposes the session. Suitable for one-off queries
  /// where you don't need to maintain conversational context.
  Future<String> computeSingle(
    List<ContentPart> input, {
    SessionConfig? config,
  }) async {
    final session = await createSession(config: config);
    try {
      return await session.generateResponseFuture(input);
    } finally {
      await session.dispose();
    }
  }

  /// Fully releases the engine and all associated memory.
  ///
  /// On Android, calls `closeModel` which closes the `Conversation` and the
  /// `Engine` (native JNI heap is freed immediately).
  /// On web, calls `unloadLLM` which releases GPU/WASM memory.
  ///
  /// After dispose, [init] must be called again before any further use.
  /// Tracks [GemmaState.loadState] ([ModelLoadState.unloading] while in
  /// flight, [ModelLoadState.notLoaded] on success).
  Future<void> dispose() async {
    if (!_isInitialized) return;

    GemmaState.instance.setLoadState(ModelLoadState.unloading);
    try {
      if (kIsWeb) {
        await FlutterLocalGemmaWeb().unload();
      } else {
        await _channel.invokeMethod('closeModel');
        await _eventSubscription?.cancel();
        _eventSubscription = null;
      }

      _isInitialized = false;
      _currentModelPath = null;
      _currentModelName = null;
      GemmaState.instance.setLoadState(ModelLoadState.notLoaded);
      // A stray session could theoretically be mid-generation if the caller
      // disposes without stopping it first — reset so the UI doesn't get
      // stuck showing a busy state for a model that no longer exists.
      GemmaState.instance.setInferenceState(GemmaInferenceState.idle);
    } catch (e) {
      GemmaState.instance.setLastError(e);
      GemmaState.instance.setLoadState(ModelLoadState.error);
      rethrow;
    }
  }

  // ── Internal accessor used by ChatSession ──────────────────────────────────

  Stream<Map<dynamic, dynamic>> get _nativeEventStream =>
      _eventController.stream;
}

// ─────────────────────────────────────────────────────────────────────────────
// ChatSession  —  a single conversation turn holder
// ─────────────────────────────────────────────────────────────────────────────

/// A handle to a single native `Conversation` (mobile) or content buffer (web).
///
/// Created via [FlutterLocalGemma.createSession]. Supports streaming and blocking
/// generation, token counting, and explicit context clearing.
///
/// **Thread-safety:** All public methods are safe to call from the Flutter UI
/// isolate. Heavy work runs on IO threads in native code or in a JS worker.
class ChatSession {
  final MethodChannel? _channel;

  // Kept for API-compatibility reference only — EventChannel subscriptions
  // are now managed centrally by [FlutterLocalGemma._nativeEventStream].
  // ignore: unused_field
  final EventChannel? _stream;
  final int maxContext;
  final SessionConfig? webConfig;

  /// Estimated token usage across all [addToContext] and generation calls.
  int _usageCounter = 0;

  ChatSession({
    required this.maxContext,
    this.webConfig,
    MethodChannel? channel,
    EventChannel? stream,
  })  : _channel = channel,
        _stream  = stream;

  // ── Token tracking ────────────────────────────────────────────────────────

  /// Current token usage / capacity snapshot.
  TokenStats get stats =>
      TokenStats(estimatedUsed: _usageCounter, maxContext: maxContext);

  /// Estimated tokens consumed so far.
  int get usedTokens => _usageCounter;

  /// Estimated tokens still available in the context window.
  int get remainingTokens => (maxContext - _usageCounter).clamp(0, maxContext);

  /// Returns true when used tokens exceed 80% of the context window.
  bool get isNearContextLimit => _usageCounter >= (maxContext * 0.8).toInt();

  /// Estimates the token count for [text] and optional media payloads.
  ///
  /// Uses the native tokenizer on mobile (accurate) and the JS tokenizer on
  /// web (accurate when loaded, otherwise a word-count heuristic).
  Future<int> countTokens(
    String text, {
    int imageCount     = 0,
    int audioDurationMs = 0,
  }) async {
    // 257 tokens per image (Google's documented constant for Gemma vision).
    // 1 token per 150 ms of audio.
    int base = (imageCount * 257) + (audioDurationMs / 150).ceil();
    if (text.isEmpty) return base;

    if (kIsWeb) {
      return base + await FlutterLocalGemmaWeb().countTokensWeb(text);
    } else {
      final int? count = await _channel?.invokeMethod<int>('countTokens', {
        'text':            text,
        'imageCount':      imageCount,
        'audioDurationMs': audioDurationMs,
      });
      return count ?? base;
    }
  }

  // ── Content accumulation ─────────────────────────────────────────────────

  /// Pushes [parts] into the native content buffer (mobile) or the web staging
  /// buffer, and updates the internal token counter.
  Future<void> addToContext(List<ContentPart> parts) async {
    if (kIsWeb) {
      FlutterLocalGemmaWeb().addToBuffer(parts);
    } else {
      await _sendPartsNative(parts);
    }
    _usageCounter += await _estimatePartTokens(parts);
  }

  // ── Generation ─────────────────────────────────────────────────────────────

  /// Streams tokens as they are generated. Closes the returned stream when
  /// generation finishes (or is cancelled).
  ///
  /// Tracks [GemmaState.inferenceState]: [GemmaInferenceState.processingInput]
  /// while [input] is being pushed to the native side,
  /// [GemmaInferenceState.answering] once tokens start arriving, and back to
  /// [GemmaInferenceState.idle] when the stream closes (or
  /// [GemmaInferenceState.error] if it fails). This raw stream has no
  /// concept of "thinking" vs "answering" — use
  /// [GemmaChat.sendMessageResponseStream] or [AgentChat.run] for that
  /// distinction (and for [GemmaInferenceState.toolCalling]).
  Stream<String> generateResponseStream(List<ContentPart> input) {
    final controller = StreamController<String>();
    bool reachedTerminalState = false;

    void toIdleOrError(Object? error) {
      if (reachedTerminalState) return;
      reachedTerminalState = true;
      if (error != null) {
        GemmaState.instance.setLastError(error);
        GemmaState.instance.setInferenceState(GemmaInferenceState.error);
      } else {
        GemmaState.instance.setInferenceState(GemmaInferenceState.idle);
      }
    }

    Future<void> run() async {
      try {
        GemmaState.instance.setInferenceState(GemmaInferenceState.processingInput);
        await addToContext(input);
        if (kIsWeb) {
          await _streamWeb(controller);
        } else {
          await _streamMobile(controller);
        }
      } catch (e, st) {
        toIdleOrError(e);
        if (!controller.isClosed) {
          controller.addError(e, st);
          controller.close();
        }
      }
    }

    run();

    return controller.stream.transform(
      StreamTransformer<String, String>.fromHandlers(
        handleData: (data, sink) {
          if (GemmaState.instance.inferenceState.value !=
              GemmaInferenceState.answering) {
            GemmaState.instance.setInferenceState(GemmaInferenceState.answering);
          }
          sink.add(data);
        },
        handleError: (e, st, sink) {
          toIdleOrError(e);
          sink.addError(e, st);
        },
        handleDone: (sink) {
          toIdleOrError(null);
          sink.close();
        },
      ),
    );
  }

  /// Generates a full response and returns it as a single [String].
  ///
  /// On mobile, uses the synchronous native path (`generateResponseSync`).
  /// On web, collects the full streaming response.
  ///
  /// Tracks [GemmaState.inferenceState] the same way [generateResponseStream]
  /// does: [GemmaInferenceState.processingInput] →
  /// [GemmaInferenceState.answering] → [GemmaInferenceState.idle] (or
  /// [GemmaInferenceState.error]). On mobile this is one blocking native
  /// call, so there's no token-by-token granularity — the whole call is
  /// reported as `answering`.
  Future<String> generateResponseFuture(List<ContentPart> input) async {
    if (kIsWeb) {
      // Routes through generateResponseStream, which already tracks state.
      final buffer = StringBuffer();
      await for (final chunk in generateResponseStream(input)) {
        buffer.write(chunk);
      }
      return buffer.toString();
    } else {
      GemmaState.instance.setInferenceState(GemmaInferenceState.processingInput);
      try {
        await addToContext(input);
        GemmaState.instance.setInferenceState(GemmaInferenceState.answering);
        final String? result =
            await _channel?.invokeMethod<String>('generateResponseSync');
        GemmaState.instance.setInferenceState(GemmaInferenceState.idle);
        return result ?? '';
      } catch (e) {
        GemmaState.instance.setLastError(e);
        GemmaState.instance.setInferenceState(GemmaInferenceState.error);
        rethrow;
      }
    }
  }

  /// Signals the native side to abort the current generation mid-stream.
  ///
  /// Sets [GemmaState.inferenceState] to [GemmaInferenceState.cancelling];
  /// it returns to [GemmaInferenceState.idle] once the stream actually
  /// closes.
  Future<void> stopGeneration() async {
    GemmaState.instance.setInferenceState(GemmaInferenceState.cancelling);
    if (kIsWeb) {
      await FlutterLocalGemmaWeb().cancelProcessing();
    } else {
      await _channel?.invokeMethod('stopGeneration');
    }
  }

  // ── Context management ────────────────────────────────────────────────────

  /// Resets the conversation context on both mobile and web.
  ///
  /// On mobile, this closes the native `Conversation` object; a new
  /// `createSession` call is required before generating again.
  Future<void> clearContext() async {
    _usageCounter = 0;
    if (kIsWeb) {
      FlutterLocalGemmaWeb().clearBuffer();
    } else {
      await _channel?.invokeMethod('clearContext');
    }
  }

  /// Disposes this session, releasing the native Conversation and clearing
  /// all buffers. The [FlutterLocalGemma] engine itself remains loaded.
  Future<void> dispose() async => clearContext();

  // ── Private helpers ───────────────────────────────────────────────────────

  Future<void> _sendPartsNative(List<ContentPart> parts) async {
    if (_channel == null) return;
    for (final part in parts) {
      if (part is TextPart) {
        await _channel.invokeMethod('addQueryChunk', {'prompt': part.text});
      } else if (part is ImageFilePart) {
        // Zero-copy: native reads the file directly, no byte-array transfer.
        await _channel.invokeMethod('addImageFile', {'path': part.path});
      } else if (part is ImagePart) {
        await _channel.invokeMethod('addImage', {'imageBytes': part.bytes});
      } else if (part is AudioFilePart) {
        await _channel.invokeMethod('addAudioFile', {'path': part.path});
      } else if (part is AudioPart) {
        await _channel.invokeMethod('addAudio', {'audioBytes': part.bytes});
      }
    }
  }

  /// Fixed duration estimate (ms) used for [AudioFilePart]s that don't
  /// specify [AudioFilePart.durationMs]. Deliberately conservative — pass
  /// the real duration when you know it for an accurate token count.
  static const int _defaultAudioFileDurationMs = 5000;

  Future<int> _estimatePartTokens(List<ContentPart> parts) async {
    final textBuf    = StringBuffer();
    int imgCount     = 0;
    int audDurationMs = 0;
    for (final p in parts) {
      if (p is TextPart) textBuf.write(p.text);
      if (p is ImagePart || p is ImageFilePart) imgCount++;
      // Standard 16 kHz 16-bit mono PCM: 32 bytes per millisecond.
      if (p is AudioPart) audDurationMs += p.bytes.length ~/ 32;
      if (p is AudioFilePart) audDurationMs += p.durationMs ?? _defaultAudioFileDurationMs;
    }
    return countTokens(
      textBuf.toString(),
      imageCount:      imgCount,
      audioDurationMs: audDurationMs,
    );
  }

  Future<void> _streamWeb(StreamController<String> controller) async {
    // Safety net: if the caller cancels the returned stream before generation
    // finishes (widget disposed, navigated away, etc.), don't leave
    // GemmaState.inferenceState stuck on `answering`/`processingInput`.
    controller.onCancel = () {
      if (GemmaState.instance.inferenceState.value != GemmaInferenceState.error) {
        GemmaState.instance.setInferenceState(GemmaInferenceState.idle);
      }
    };

    FlutterLocalGemmaWeb()
        .generateResponse(autoStopConfig: webConfig?.autoStopConfig ?? const AutoStopConfig())
        .listen(
          (event) {
            if (controller.isClosed) return;
            if (event['error'] != null) {
              controller.addError(event['error'] as Object);
            } else {
              final text   = (event['partialResult'] as String?) ?? '';
              final isDone = (event['done'] as bool?) ?? false;
              if (text.isNotEmpty) {
                _usageCounter++;
                controller.add(text);
              }
              if (isDone && !controller.isClosed) controller.close();
            }
          },
          onError: (Object e) {
            if (!controller.isClosed) {
              controller.addError(e);
              controller.close();
            }
          },
          onDone: () {
            if (!controller.isClosed) controller.close();
          },
        );
  }

  Future<void> _streamMobile(StreamController<String> controller) async {
    StreamSubscription? sub;
    bool generationDone = false;

    try {
      sub = FlutterLocalGemma()._nativeEventStream.listen(
        (event) {
          if (controller.isClosed) return;
          if (event.containsKey('error')) {
            controller.addError(event['error'] as Object);
            sub?.cancel();
            controller.close();
          } else {
            final text   = (event['partialResult'] as String?) ?? '';
            final isDone = (event['done'] as bool?) ?? false;
            if (text.isNotEmpty) {
              _usageCounter++;
              controller.add(text);
            }
            if (isDone) {
              generationDone = true;
              sub?.cancel();
              controller.close();
            }
          }
        },
        onError: (Object e) {
          controller.addError(e);
          sub?.cancel();
          controller.close();
        },
      );

      await _channel?.invokeMethod('generateResponseAsync');

      // If the user cancels the stream before generation finishes, send
      // stopGeneration so the native side doesn't keep computing.
      controller.onCancel = () {
        if (!generationDone) {
          _channel?.invokeMethod('stopGeneration');
        }
        sub?.cancel();
        // Safety net: don't leave GemmaState.inferenceState stuck on
        // `answering`/`processingInput` if the caller cancels early.
        if (GemmaState.instance.inferenceState.value != GemmaInferenceState.error) {
          GemmaState.instance.setInferenceState(GemmaInferenceState.idle);
        }
      };
    } catch (e) {
      controller.addError(e);
      await sub?.cancel();
      controller.close();
    }
  }
}