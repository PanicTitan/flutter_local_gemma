import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Centralised debug-logging utility for `flutter_local_gemma`.
///
/// Disabled by default. Enable at app startup:
/// ```dart
/// void main() async {
///   GemmaDebug.enabled = kDebugMode;
///   await GemmaDebug.setNativeDebug(kDebugMode);
///   runApp(MyApp());
/// }
/// ```
///
/// Per-module tag constants allow you to filter logcat / console output:
/// ```bash
/// # Android logcat
/// adb logcat -s GemmaEngine GemmaChat AgentChat SkillDispatch
/// ```
class GemmaDebug {
  GemmaDebug._();

  /// Master switch. When `false`, all [log] calls are no-ops.
  static bool enabled = false;

  // ── Per-module tag constants ───────────────────────────────────────────────

  static const String tagEngine     = 'GemmaEngine';
  static const String tagSession    = 'ChatSession';
  static const String tagChat       = 'GemmaChat';
  static const String tagAgent      = 'AgentChat';
  static const String tagThinking   = 'ThinkingParser';
  static const String tagToolCall   = 'ToolCallParser';
  static const String tagSkills     = 'SkillDispatch';
  static const String tagLoader     = 'GemmaLoader';
  static const String tagMultimodal = 'MultimodalGuard';

  // ── Dart-side logging ─────────────────────────────────────────────────────

  /// Logs [message] to the Flutter debug console when [enabled] is `true`.
  ///
  /// Messages are prefixed with `[tag]` for easy filtering:
  /// ```dart
  /// GemmaDebug.log(GemmaDebug.tagAgent, 'init — skills=${skills.length}');
  /// // prints: [AgentChat] init — skills=3
  /// ```
  static void log(String tag, String message) {
    if (enabled) debugPrint('[$tag] $message');
  }

  /// Same as [log] but only emits on errors (always logged regardless of [enabled]).
  static void logError(String tag, String message, [Object? error, StackTrace? st]) {
    debugPrint('[$tag] ERROR: $message${error != null ? ' — $error' : ''}');
    if (st != null) debugPrint(st.toString());
  }

  // ── Native-side logging ────────────────────────────────────────────────────

  /// Enables verbose `Log.d(TAG, ...)` output in Kotlin (`GemmaPlugin`,
  /// `JsSkillExecutor`, `GemmaAgentToolSet`) and equivalent console output
  /// in the TypeScript web bundle.
  ///
  /// Call this at startup, before [FlutterLocalGemma.init]:
  /// ```dart
  /// await GemmaDebug.setNativeDebug(kDebugMode);
  /// ```
  static Future<void> setNativeDebug(bool value) async {
    if (kIsWeb) return; // Web bundle reads GemmaDebug.enabled via JS interop
    const ch = MethodChannel('gemma_bundled');
    try {
      await ch.invokeMethod<void>('setDebugLogging', {'enabled': value});
    } catch (_) {
      // Silently ignore if the native side hasn't been initialised yet.
    }
  }
}
