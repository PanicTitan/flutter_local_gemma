import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../model_installer/model_installer.dart';
export '../model_installer/model_installer.dart';

import '../types/content_parts.dart';
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

  /// null = let the native side decide (usually CPU).
  final PreferredBackend? backend;

  final int? maxNumImages;
  final bool supportAudio;

  InferenceConfig({
    required this.modelPath,
    this.maxTokens   = 4096,
    this.backend,
    this.maxNumImages = 10,
    this.supportAudio = true,
  });
}

/// Per-session sampling parameters.
class SessionConfig {
  final double temperature;
  final double topP;
  final int topK;
  final int? randomSeed;
  final String? systemPrompt;
  final AutoStopConfig autoStopConfig;

  SessionConfig({
    this.temperature    = 0.8,
    this.topP           = 0.95,
    this.topK           = 40,
    this.randomSeed,
    this.systemPrompt,
    this.autoStopConfig = const AutoStopConfig(),
  });
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
      'TokenStats($estimatedUsed / $maxContext used, ${remaining} remaining)';
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

  /// Initialises the inference engine.
  ///
  /// Calling [init] when the engine is already initialised is a no-op; call
  /// [dispose] first if you need to swap to a different model.
  Future<void> init(InferenceConfig config) async {
    if (_isInitialized) return;
    _maxTokens = config.maxTokens;

    if (kIsWeb) {
      await FlutterLocalGemmaWeb().init(config);
    } else {
      // Subscribe to the Android EventChannel once here so there is never a
      // window where the sink is null between two consecutive generations.
      _eventSubscription ??= _eventChannel.receiveBroadcastStream().listen(
        (event) {
          if (event is Map) {
            _eventController.add(event as Map<dynamic, dynamic>);
          }
        },
        onError: _eventController.addError,
      );

      await _channel.invokeMethod('createModel', {
        'modelPath':       config.modelPath,
        'maxTokens':       config.maxTokens,
        'preferredBackend': config.backend?.index ?? 0,
        'maxNumImages':    config.maxNumImages,
        'supportAudio':    config.supportAudio,
      });
    }

    _isInitialized = true;
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
  Future<void> dispose() async {
    if (!_isInitialized) return;

    if (kIsWeb) {
      FlutterLocalGemmaWeb().unload();
    } else {
      await _channel.invokeMethod('closeModel');
      await _eventSubscription?.cancel();
      _eventSubscription = null;
    }

    _isInitialized = false;
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
  final EventChannel?  _stream;   // kept for API-compatibility reference only
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
  Stream<String> generateResponseStream(List<ContentPart> input) {
    final controller = StreamController<String>();

    Future<void> run() async {
      try {
        await addToContext(input);
        if (kIsWeb) {
          await _streamWeb(controller);
        } else {
          await _streamMobile(controller);
        }
      } catch (e, st) {
        if (!controller.isClosed) {
          controller.addError(e, st);
          controller.close();
        }
      }
    }

    run();
    return controller.stream;
  }

  /// Generates a full response and returns it as a single [String].
  ///
  /// On mobile, uses the synchronous native path (`generateResponseSync`).
  /// On web, collects the full streaming response.
  Future<String> generateResponseFuture(List<ContentPart> input) async {
    if (kIsWeb) {
      final buffer = StringBuffer();
      await for (final chunk in generateResponseStream(input)) {
        buffer.write(chunk);
      }
      return buffer.toString();
    } else {
      await addToContext(input);
      final String? result =
          await _channel?.invokeMethod<String>('generateResponseSync');
      return result ?? '';
    }
  }

  /// Signals the native side to abort the current generation mid-stream.
  Future<void> stopGeneration() async {
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
        await _channel!
            .invokeMethod('addQueryChunk', {'prompt': part.text});
      } else if (part is ImagePart) {
        await _channel!
            .invokeMethod('addImage', {'imageBytes': part.bytes});
      } else if (part is AudioPart) {
        await _channel!
            .invokeMethod('addAudio', {'audioBytes': part.bytes});
      }
    }
  }

  Future<int> _estimatePartTokens(List<ContentPart> parts) async {
    final textBuf    = StringBuffer();
    int imgCount     = 0;
    int audDurationMs = 0;
    for (final p in parts) {
      if (p is TextPart)  textBuf.write(p.text);
      if (p is ImagePart) imgCount++;
      // Standard 16 kHz 16-bit mono PCM: 32 bytes per millisecond.
      if (p is AudioPart) audDurationMs += p.bytes.length ~/ 32;
    }
    return countTokens(
      textBuf.toString(),
      imageCount:      imgCount,
      audioDurationMs: audDurationMs,
    );
  }

  Future<void> _streamWeb(StreamController<String> controller) async {
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
      };
    } catch (e) {
      controller.addError(e);
      await sub?.cancel();
      controller.close();
    }
  }
}