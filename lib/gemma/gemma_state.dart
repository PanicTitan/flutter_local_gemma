import 'package:flutter/foundation.dart';

/// Lifecycle state of the native inference engine / currently loaded model.
///
/// Since [FlutterLocalGemma] is a singleton and the underlying native engine
/// can only ever have **one** model loaded at a time, this single enum
/// (surfaced through [GemmaState.loadState]) is enough to describe the whole
/// plugin's load lifecycle — there is no scenario where two models are
/// loaded, or two [FlutterLocalGemma.init] calls are legitimately "in
/// flight" at once.
enum ModelLoadState {
  /// No model is loaded and nothing is downloading. Initial state, and the
  /// state after a successful [FlutterLocalGemma.dispose].
  notLoaded,

  /// A model file is being downloaded/copied (e.g. via `ModelInstaller`,
  /// `PluginModelLoader.downloadLlm`, or `GemmaLoader.loadGemmaModel`).
  /// This is *before* the native engine even starts initialising — see
  /// [loading] for that phase. Not every code path goes through this state:
  /// only helpers that download go through it, and only for local files
  /// (Android `content://`/SAF copies also included; a model that's already
  /// fully cached and only needs `init()` skips straight there).
  downloading,

  /// [FlutterLocalGemma.init] is in progress: resolving the model file
  /// (including any SAF copy on Android), initialising the native engine,
  /// setting up the GPU/CPU delegate, etc. This can take anywhere from
  /// under a second to tens of seconds for a cold multi-GB model load.
  loading,

  /// The engine is initialised and ready — [FlutterLocalGemma.createSession]
  /// / [FlutterLocalGemma.computeSingle] can be called.
  loaded,

  /// [FlutterLocalGemma.dispose] is in progress.
  unloading,

  /// The last download, [FlutterLocalGemma.init], or [FlutterLocalGemma.dispose]
  /// call threw. See [GemmaState.lastError] for the underlying error. The
  /// engine is not usable in this state; call [FlutterLocalGemma.dispose]
  /// (safe to call even after a failed init) and then [FlutterLocalGemma.init]
  /// again to retry.
  error,
}

/// What the loaded model is currently doing during a generation turn.
///
/// Populated automatically by [ChatSession], [GemmaChat], and [AgentChat] —
/// you never need to set this yourself. Only [GemmaChat.sendMessageResponseStream]
/// and [AgentChat.run] can distinguish [thinking] from [answering] (they
/// consume the typed [GemmaResponse]/`AgentTurn` streams); the plain
/// string-based [GemmaChat.sendMessageStream] / [ChatSession.generateResponseStream]
/// only ever report [answering] while busy, since they have no way to tell
/// "thinking" tokens apart from the final answer.
enum GemmaInferenceState {
  /// Not generating anything right now.
  idle,

  /// A prompt/context is being pushed to the native side just before
  /// generation starts (tokenizing text, transferring image/audio bytes or
  /// file paths, etc). Typically very brief.
  processingInput,

  /// Streaming `<think>…</think>` / `<|channel>thought…<channel|>` content
  /// (Gemma 4 "thinking" mode). Only reachable through
  /// [GemmaChat.sendMessageResponseStream] or [AgentChat.run].
  thinking,

  /// Streaming the final answer tokens that should be shown to the user.
  answering,

  /// [AgentChat] has parsed a tool/skill call out of the model's output and
  /// is waiting for the skill's result before continuing generation.
  toolCalling,

  /// [ChatSession.stopGeneration] / [GemmaChat.stop] / [AgentChat.stop] was
  /// called and the engine is winding down the current turn.
  cancelling,

  /// The last generation turn ended with an error. See [GemmaState.lastError].
  error,
}

/// Global, reactive snapshot of what [FlutterLocalGemma] is doing right now.
///
/// There is only ever one [FlutterLocalGemma] engine and one turn generating
/// at a time (see [ModelLoadState] docs), so a single process-wide state
/// holder is all that's needed — you don't need to (and can't) juggle
/// per-instance state for "which model" or "which session".
///
/// [loadState] and [inferenceState] are plain [ValueListenable]s — the same
/// reactive primitive `ValueNotifier` / `ValueListenableBuilder` already use
/// everywhere in Flutter, so wiring them up is just standard Flutter:
///
/// ```dart
/// // Rebuild a widget subtree automatically when the load state changes:
/// ValueListenableBuilder<ModelLoadState>(
///   valueListenable: GemmaState.instance.loadState,
///   builder: (context, state, _) => Text(state.name),
/// )
///
/// // Or listen imperatively — e.g. to drive a foreground service:
/// GemmaState.instance.inferenceState.addListener(() {
///   final busy = GemmaState.instance.isBusy;
///   busy ? startForegroundService() : stopForegroundService();
/// });
/// ```
class GemmaState {
  GemmaState._internal();

  /// The single global instance.
  static final GemmaState instance = GemmaState._internal();

  final ValueNotifier<ModelLoadState> _loadState =
      ValueNotifier(ModelLoadState.notLoaded);
  final ValueNotifier<GemmaInferenceState> _inferenceState =
      ValueNotifier(GemmaInferenceState.idle);
  final ValueNotifier<Object?> _lastError = ValueNotifier(null);

  /// Current model-loading lifecycle state. Read-only from outside the
  /// package; use [ValueListenable.addListener] or [ValueListenableBuilder]
  /// to react to changes.
  ValueListenable<ModelLoadState> get loadState => _loadState;

  /// Current per-turn inference activity.
  ValueListenable<GemmaInferenceState> get inferenceState => _inferenceState;

  /// The error from the last failed load or generation turn, if any. Reset
  /// to null at the start of the next [FlutterLocalGemma.init] or
  /// generation call.
  ValueListenable<Object?> get lastError => _lastError;

  /// True whenever [inferenceState] is anything other than
  /// [GemmaInferenceState.idle]. Handy as a single flag to gate a
  /// foreground service, wake lock, "busy" spinner, disabling input, etc.
  bool get isBusy => _inferenceState.value != GemmaInferenceState.idle;

  /// True whenever [loadState] is [ModelLoadState.loaded].
  bool get isModelLoaded => _loadState.value == ModelLoadState.loaded;

  /// True whenever a download or engine load/unload is in flight
  /// ([ModelLoadState.downloading], [ModelLoadState.loading], or
  /// [ModelLoadState.unloading]) — handy for disabling a "load"/"switch
  /// model" action in the UI so it can't race an operation already running.
  bool get isLoadBusy => switch (_loadState.value) {
    ModelLoadState.downloading || ModelLoadState.loading || ModelLoadState.unloading => true,
    _ => false,
  };

  /// Fires whenever *either* [loadState] or [inferenceState] changes.
  /// Convenient for a single listener/`AnimatedBuilder` that needs to react
  /// to any plugin activity without manually merging two notifiers.
  Listenable get listenable => Listenable.merge([_loadState, _inferenceState]);

  // ── Framework-internal mutators ──────────────────────────────────────────
  // Called by FlutterLocalGemma / ChatSession / GemmaChat / AgentChat only.
  // Application code should treat these as internal — read the
  // ValueListenables above instead of calling these directly.

  void setLoadState(ModelLoadState state) => _loadState.value = state;

  void setInferenceState(GemmaInferenceState state) =>
      _inferenceState.value = state;

  void setLastError(Object? error) => _lastError.value = error;
}