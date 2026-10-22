// lib/services/foreground_task_controller_web.dart
//
// Web implementation. There is no OS-level "foreground service" concept in
// a browser tab, so we use a well-known trick instead: play a real (but
// very quiet) looping <audio> element for as long as the model is busy.
//
// Most browsers exempt a tab that's actively playing *audible* audio from
// the background-tab throttling that otherwise slows down (or nearly
// freezes) timers/rAF once the tab loses focus or is backgrounded/minimized
// — which would otherwise starve a long-running WASM generation.
//
// IMPORTANT: this must NOT be perfect digital silence (every sample at the
// zero-amplitude midpoint). An element can be technically "playing" while
// producing exactly zero measurable output, and Chrome's tab-audio
// indicator — and, more importantly, the actual background-throttling
// exemption, which appears to key off the same real audio-energy signal —
// doesn't count that as audible. So the WAV embedded below is a very quiet
// (±6 out of a possible ±127 on 8-bit PCM, roughly -27 dBFS) low-frequency
// tone: genuinely non-zero output, but far too quiet to be perceptible on
// virtually any device/speaker. We also deliberately do NOT mute the
// element or set volume to 0 — that would defeat the same purpose from the
// other direction.
//
// This file is only ever compiled/loaded on web — see
// foreground_task_controller.dart for the conditional-export setup that
// guarantees this.

import 'dart:js_interop';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:web/web.dart' as web;

/// A ~0.25s, 8kHz/8-bit, very quiet (±6/127, ~-27 dBFS) 20 Hz tone, looped
/// indefinitely while the model is busy — see the file-level comment above
/// for why it's a quiet tone rather than pure silence. Small enough to
/// inline as a data URI (no network request, no extra asset to bundle).
const _silentWavDataUri =
    'data:audio/wav;base64,UklGRvQHAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0Yd'
    'AHAACAgICAgICBgYGBgYGBgYGBgYKCgoKCgoKCgoKCg4ODg4ODg4ODg4ODhISEhISEhISEhISEhI'
    'SFhYWFhYWFhYWFhYWFhYWFhYWFhYaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGho'
    'aGhoaGhoaGhoaGhoaGhoaGhYWFhYWFhYWFhYWFhYWFhYWFhYWEhISEhISEhISEhISEhIODg4ODg4'
    'ODg4ODg4KCgoKCgoKCgoKCgYGBgYGBgYGBgYGAgICAgICAgICAgH9/f39/f39/f39/fn5+fn5+fn'
    '5+fn59fX19fX19fX19fX18fHx8fHx8fHx8fHx8fHt7e3t7e3t7e3t7e3t7e3t7e3t7enp6enp6en'
    'p6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp7e3t7e3t7e3t7e3'
    't7e3t7e3t7e3x8fHx8fHx8fHx8fHx8fX19fX19fX19fX19fn5+fn5+fn5+fn5/f39/f39/f39/f4'
    'CAgICAgICAgICAgYGBgYGBgYGBgYGCgoKCgoKCgoKCgoODg4ODg4ODg4ODg4SEhISEhISEhISEhI'
    'SEhYWFhYWFhYWFhYWFhYWFhYWFhYWGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGho'
    'aGhoaGhoaGhoaGhoaGhoaGhoWFhYWFhYWFhYWFhYWFhYWFhYWFhISEhISEhISEhISEhISDg4ODg4'
    'ODg4ODg4OCgoKCgoKCgoKCgoGBgYGBgYGBgYGBgICAgICAgICAgIB/f39/f39/f39/f35+fn5+fn'
    '5+fn5+fX19fX19fX19fX19fHx8fHx8fHx8fHx8fHx7e3t7e3t7e3t7e3t7e3t7e3t7e3p6enp6en'
    'p6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6e3t7e3t7e3t7e3'
    't7e3t7e3t7e3t8fHx8fHx8fHx8fHx8fH19fX19fX19fX19fX5+fn5+fn5+fn5+f39/f39/f39/f3'
    '+AgICAgICAgICAgIGBgYGBgYGBgYGBgoKCgoKCgoKCgoKDg4ODg4ODg4ODg4OEhISEhISEhISEhI'
    'SEhIWFhYWFhYWFhYWFhYWFhYWFhYWFhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGho'
    'aGhoaGhoaGhoaGhoaGhoaGhoaFhYWFhYWFhYWFhYWFhYWFhYWFhYSEhISEhISEhISEhISEg4ODg4'
    'ODg4ODg4ODgoKCgoKCgoKCgoKBgYGBgYGBgYGBgYCAgICAgICAgICAf39/f39/f39/f39+fn5+fn'
    '5+fn5+fn19fX19fX19fX19fXx8fHx8fHx8fHx8fHx8e3t7e3t7e3t7e3t7e3t7e3t7e3t6enp6en'
    'p6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6ent7e3t7e3t7e3'
    't7e3t7e3t7e3t7fHx8fHx8fHx8fHx8fHx9fX19fX19fX19fX1+fn5+fn5+fn5+fn9/f39/f39/f3'
    '9/gICAgICAgICAgICBgYGBgYGBgYGBgYKCgoKCgoKCgoKCg4ODg4ODg4ODg4ODhISEhISEhISEhI'
    'SEhISFhYWFhYWFhYWFhYWFhYWFhYWFhYaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGho'
    'aGhoaGhoaGhoaGhoaGhoaGhoaGhYWFhYWFhYWFhYWFhYWFhYWFhYWEhISEhISEhISEhISEhIODg4'
    'ODg4ODg4ODg4KCgoKCgoKCgoKCgYGBgYGBgYGBgYGAgICAgICAgICAgH9/f39/f39/f39/fn5+fn'
    '5+fn5+fn59fX19fX19fX19fX18fHx8fHx8fHx8fHx8fHt7e3t7e3t7e3t7e3t7e3t7e3t7enp6en'
    'p6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp7e3t7e3t7e3'
    't7e3t7e3t7e3t7e3x8fHx8fHx8fHx8fHx8fX19fX19fX19fX19fn5+fn5+fn5+fn5/f39/f39/f3'
    '9/f4CAgICAgICAgICAgYGBgYGBgYGBgYGCgoKCgoKCgoKCgoODg4ODg4ODg4ODg4SEhISEhISEhI'
    'SEhISEhYWFhYWFhYWFhYWFhYWFhYWFhYWGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGho'
    'aGhoaGhoaGhoaGhoaGhoaGhoaGhoWFhYWFhYWFhYWFhYWFhYWFhYWFhISEhISEhISEhISEhISDg4'
    'ODg4ODg4ODg4OCgoKCgoKCgoKCgoGBgYGBgYGBgYGBgICAgICAgICAgIB/f39/f39/f39/f35+fn'
    '5+fn5+fn5+fX19fX19fX19fX19fHx8fHx8fHx8fHx8fHx7e3t7e3t7e3t7e3t7e3t7e3t7e3p6en'
    'p6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6enp6e3t7e3t7e3'
    't7e3t7e3t7e3t7e3t8fHx8fHx8fHx8fHx8fH19fX19fX19fX19fX5+fn5+fn5+fn5+f39/f39/f3'
    '9/f3+AgICAgA==';


/// Starts/stops the silent-audio keep-alive loop in lock-step with
/// [GemmaState.instance.inferenceState].
///
/// Same public shape as the io implementation
/// (foreground_task_controller_io.dart) so `main.dart`/`chat_screen.dart`
/// can use either one without caring which platform they're on.
class ForegroundTaskController {
  ForegroundTaskController._();
  static final instance = ForegroundTaskController._();

  bool enabled = true;

  web.HTMLAudioElement? _audio;
  bool _unlocked = false;
  JSFunction? _unlockListener;

  /// Registers the state listener and arms a one-time "unlock" listener on
  /// the very first user interaction with the page.
  Future<void> init() async {
    GemmaState.instance.inferenceState.addListener(_onInferenceStateChanged);
    _armUnlockListener();
  }

  // ── Autoplay unlock ──────────────────────────────────────────────────────
  //
  // Browsers require a *direct* user gesture the first time a page plays
  // non-muted media — a play() call from inside a GemmaState listener
  // callback (which is several `await`s removed from the actual click/tap
  // that started generation) can be rejected by that policy, silently,
  // which is very likely why no audio ever actually started. To fix this
  // for real: listen for the first raw pointerdown/keydown/touchstart
  // anywhere on the page, and use *that* real gesture to play() (then
  // immediately pause, if we're not actually supposed to be busy yet) the
  // exact same <audio> element we'll reuse later. Once an element has been
  // successfully played from a real gesture, browsers allow subsequent
  // programmatic play() calls on it without requiring another gesture —
  // which is what lets `_start()` below work when it's triggered from a
  // GemmaState change instead of a raw DOM event.

  void _armUnlockListener() {
    if (_unlockListener != null) return;
    void handleGesture(web.Event event) {
      _removeUnlockListener();
      _unlock();
    }

    _unlockListener = handleGesture.toJS;
    for (final type in ['pointerdown', 'keydown', 'touchstart']) {
      web.document.addEventListener(type, _unlockListener);
    }
  }

  void _removeUnlockListener() {
    final listener = _unlockListener;
    if (listener == null) return;
    for (final type in ['pointerdown', 'keydown', 'touchstart']) {
      web.document.removeEventListener(type, listener);
    }
    _unlockListener = null;
  }

  void _unlock() {
    if (_unlocked) return;
    final audio = _ensureAudioElement();
    audio
        .play()
        .toDart
        .then((_) {
          _unlocked = true;
          debugPrint('[ForegroundTaskController] Audio unlocked.');
          if (!GemmaState.instance.isBusy) audio.pause();
        })
        .catchError((Object e) {
          // Not fatal — _armUnlockListener re-arms below so the next
          // interaction gets another attempt.
          debugPrint('[ForegroundTaskController] Audio unlock failed: $e');
          _armUnlockListener();
        });
  }

  bool _manualTaskRunning = false;

  /// Whether the keepalive audio is currently playing.
  bool get isServiceRunning => _audio != null && !_audio!.paused;

  /// Keeps the web app alive in background via silent looping audio
  Future<void> startTask({
    String title = 'Processing in background…',
    String text = 'Tap to return to the app',
  }) async {
    if (!enabled) return;
    _manualTaskRunning = true;
    _start();
  }

  Future<void> updateTask({
    required String title,
    required String text,
  }) async {
    // Web audio keepalive runs in background without Android notifications
  }

  Future<void> stopTask() async {
    _manualTaskRunning = false;
    if (!GemmaState.instance.isBusy) {
      _stop();
    }
  }

  void _onInferenceStateChanged() {
    if (!enabled) return;
    if (GemmaState.instance.isBusy || _manualTaskRunning) {
      _start();
    } else {
      _stop();
    }
  }

  web.HTMLAudioElement _ensureAudioElement() {
    final existing = _audio;
    if (existing != null) return existing;

    // Note: `HTMLAudioElement()`'s own constructor is known to throw at
    // runtime in package:web (it lowers to `new HTMLAudioElement()`, which
    // isn't a valid JS constructor) — `document.createElement` is the
    // documented, future-proof workaround. See
    // https://github.com/dart-lang/web/issues/124.
    final audio = web.document.createElement('audio') as web.HTMLAudioElement;
    audio.src = _silentWavDataUri;
    audio.loop = true;
    audio.setAttribute('playsinline', 'true');
    // Some browsers are stricter about background/detached elements —
    // attach it (hidden) to the document so it behaves like "real" media
    // rather than a dangling object the tab-audio/throttling-exemption
    // logic might not pick up.
    audio.style.cssText = 'display:none';
    web.document.body?.appendChild(audio);
    _audio = audio;
    return audio;
  }

  void _start() {
    final audio = _ensureAudioElement();
    audio
        .play()
        .toDart
        .then((_) {
          _unlocked = true;
          debugPrint('[ForegroundTaskController] Keep-alive audio started.');
        })
        .catchError((Object e) {
          // Most likely cause: no user gesture has unlocked audio yet.
          // _armUnlockListener (called from init()) should catch the next
          // interaction and retry via _unlock(); nothing more to do here.
          debugPrint(
            '[ForegroundTaskController] play() rejected (probably not '
            'unlocked yet — will retry on next user interaction): $e',
          );
        });
  }

  void _stop() {
    // Pause but keep the element around (and in the DOM) — recreating it
    // on every turn would mean re-unlocking autoplay every time too.
    _audio?.pause();
  }

  void dispose() {
    GemmaState.instance.inferenceState.removeListener(_onInferenceStateChanged);
    _removeUnlockListener();
    _audio?.pause();
    _audio?.remove();
    _audio = null;
  }
}