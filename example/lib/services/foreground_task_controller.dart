// lib/services/foreground_task_controller.dart
//
// Public, platform-agnostic entry point. Import *this* file only — never
// foreground_task_controller_io.dart / _web.dart directly.
//
// Mirrors the exact conditional-import idiom the plugin itself uses for
// gemma_web.dart/gemma_stub.dart: the default branch is the web
// implementation, overridden by the io implementation whenever `dart:io`
// is available (i.e. every non-web target — Android, iOS, desktop).
//
// This split exists for a real reason, not just style: `package:
// flutter_foreground_task` is Android/iOS-only (its own example imports
// `dart:io` directly) and is never proven to be web-safe. If we imported it
// unconditionally from a file that's part of the web build's import graph,
// `flutter build web` / `flutter run -d chrome` could fail to compile
// entirely. By keeping it confined to foreground_task_controller_io.dart —
// which this conditional export guarantees is *never even parsed* for a web
// target — that risk is completely eliminated, not just guarded at runtime.
export 'foreground_task_controller_web.dart'
    if (dart.library.io) 'foreground_task_controller_io.dart';