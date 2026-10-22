// lib/services/foreground_task_controller_io.dart
//
// Non-web implementation. Only ever compiled/loaded when `dart:io` is
// available — see foreground_task_controller.dart for why that matters.
//
// Demonstrates wiring the plugin's reactive `GemmaState.instance.inferenceState`
// (from package:flutter_local_gemma) to `flutter_foreground_task`
// (https://pub.dev/packages/flutter_foreground_task), so the Android app
// keeps running in a foreground service for as long as the model is
// generating/processing — and stops the service the moment it goes back to
// idle. This is example-app wiring only: the core plugin has no dependency
// on flutter_foreground_task.
//
// Only meaningful on Android — flutter_foreground_task's iOS/desktop support
// is limited/no-op-ish, and `_onInferenceStateChanged` below only acts when
// `defaultTargetPlatform == TargetPlatform.android`.

import 'package:flutter/foundation.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// The task handler that actually runs inside the foreground service
/// process. It does no real work itself — the LLM inference happens on the
/// main isolate exactly as it always does; this handler's only job is to
/// exist so Android treats the app as doing user-visible foreground work
/// (long model generations, tool-calling loops, etc.) instead of being
/// eligible for background throttling/kill.
@pragma('vm:entry-point')
void startGemmaForegroundTask() {
  FlutterForegroundTask.setTaskHandler(_GemmaForegroundTaskHandler());
}

class _GemmaForegroundTaskHandler extends TaskHandler {
  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {}

  @override
  void onRepeatEvent(DateTime timestamp) {}

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {}
}

/// Starts/stops the Android foreground service in lock-step with
/// [GemmaState.instance.inferenceState], so the app isn't killed or throttled
/// mid-generation when backgrounded.
///
/// Usage (see `main.dart`):
/// ```dart
/// void main() async {
///   WidgetsFlutterBinding.ensureInitialized();
///   await ForegroundTaskController.instance.init();
///   runApp(const AIPlaygroundApp());
/// }
/// ```
///
/// [enabled] gates whether the controller reacts at all — wire this to a
/// user-facing settings toggle (see the "Keep app active while generating"
/// switch in `chat_screen.dart`'s settings drawer) so users who don't want
/// a persistent notification can opt out.
class ForegroundTaskController {
  ForegroundTaskController._();
  static final instance = ForegroundTaskController._();

  bool enabled = true;
  bool _serviceRunning = false;

  bool get _isAndroid => defaultTargetPlatform == TargetPlatform.android;

  /// Registers the notification channel/options and starts listening to
  /// [GemmaState.instance.inferenceState]. Safe to call once at startup;
  /// does nothing on non-Android platforms.
  Future<void> init() async {
    if (!_isAndroid) return;

    // Required once by flutter_foreground_task before any of its other
    // APIs are used.
    FlutterForegroundTask.initCommunicationPort();

    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'gemma_inference_service',
        channelName: 'Model activity',
        channelDescription:
            'Shown while the on-device model is generating or processing '
            'so the app keeps running in the background.',
        onlyAlertOnce: true,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: false,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(),
        autoRunOnBoot: false,
        autoRunOnMyPackageReplaced: false,
        allowWakeLock: true,
        allowWifiLock: false,
      ),
    );

    // React to every inference-state change. GemmaState.instance.isBusy is
    // exactly "not idle", which is the single condition we care about here.
    GemmaState.instance.inferenceState.addListener(_onInferenceStateChanged);
  }

  bool _manualTaskRunning = false;

  /// Whether the background service is currently active.
  bool get isServiceRunning => _serviceRunning;

  /// Starts or keeps the background service alive for a custom long-running
  /// workload (such as background embedding batch jobs).
  Future<void> startTask({
    String title = 'Processing in background…',
    String text = 'Tap to return to the app',
  }) async {
    if (!enabled || !_isAndroid) return;
    _manualTaskRunning = true;
    _serviceRunning = true;

    final notificationPermission =
        await FlutterForegroundTask.checkNotificationPermission();
    if (notificationPermission != NotificationPermission.granted) {
      await FlutterForegroundTask.requestNotificationPermission();
    }

    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.updateService(
        notificationTitle: title,
        notificationText: text,
      );
      return;
    }

    await FlutterForegroundTask.startService(
      serviceId: 501,
      notificationTitle: title,
      notificationText: text,
      callback: startGemmaForegroundTask,
    );
  }

  /// Updates the foreground notification text while a background task is running.
  Future<void> updateTask({
    required String title,
    required String text,
  }) async {
    if (!enabled || !_isAndroid) return;
    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.updateService(
        notificationTitle: title,
        notificationText: text,
      );
    }
  }

  /// Stops the manual background task and stops the service if inference is idle.
  Future<void> stopTask() async {
    _manualTaskRunning = false;
    if (!GemmaState.instance.isBusy) {
      await _stop();
    }
  }

  Future<void> _onInferenceStateChanged() async {
    if (!enabled || !_isAndroid) return;

    final busy = GemmaState.instance.isBusy || _manualTaskRunning;
    if (busy && !_serviceRunning) {
      await _start();
    } else if (!busy && _serviceRunning) {
      await _stop();
    }
  }

  Future<void> _start() async {
    _serviceRunning = true; // set optimistically to avoid double-start races

    final notificationPermission =
        await FlutterForegroundTask.checkNotificationPermission();
    if (notificationPermission != NotificationPermission.granted) {
      await FlutterForegroundTask.requestNotificationPermission();
    }

    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.restartService();
      return;
    }

    await FlutterForegroundTask.startService(
      serviceId: 501,
      notificationTitle: 'Generating response…',
      notificationText: 'Tap to return to the app',
      callback: startGemmaForegroundTask,
    );
  }

  Future<void> _stop() async {
    _serviceRunning = false;
    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.stopService();
    }
  }

  /// Call from a State.dispose() if you ever tear down the whole app's
  /// listener graph (not needed for a normal single-Activity app, since this
  /// controller lives for the process lifetime).
  void dispose() {
    GemmaState.instance.inferenceState.removeListener(_onInferenceStateChanged);
  }
}