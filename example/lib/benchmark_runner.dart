import 'dart:async';

/// A single timed measurement.
class BenchmarkResult {
  final String name;
  final Duration duration;
  final String? detail;
  final String? error;
  final Map<String, dynamic> meta;

  const BenchmarkResult({
    required this.name,
    required this.duration,
    this.detail,
    this.error,
    this.meta = const {},
  });

  bool get isSuccess => error == null;

  String get durationLabel {
    final ms = duration.inMilliseconds;
    if (ms >= 60000) return '${(ms / 60000).toStringAsFixed(1)} min';
    if (ms >= 1000)  return '${(ms / 1000).toStringAsFixed(2)} s';
    return '$ms ms';
  }

  @override
  String toString() =>
      '${isSuccess ? "✓" : "✗"} $name  [$durationLabel]'
      '${detail != null ? "  – $detail" : ""}'
      '${error  != null ? "  ⚠ $error"  : ""}';
}

/// Lightweight timer that measures the wall-clock duration of async operations
/// and collects the results in a list.
///
/// ## Usage
/// ```dart
/// final runner = BenchmarkRunner();
///
/// final modelPath = await runner.measure('Download model', () => installer.install());
/// await runner.measure('Load engine', () => FlutterLocalGemma().init(...));
/// final reply = await runner.measure('First inference', () => chat.generate('Hello'));
///
/// runner.printSummary();
/// ```
class BenchmarkRunner {
  BenchmarkRunner({this.onUpdate});

  /// Called after every completed measurement so the UI can rebuild in real-time.
  final VoidCallback? onUpdate;

  final List<BenchmarkResult> results = [];

  int get totalMs =>
      results.fold(0, (sum, r) => sum + r.duration.inMilliseconds);

  /// Executes [fn] and records the elapsed time under [name].
  ///
  /// On success, the return value of [fn] is passed through.
  /// On error, the exception is re-thrown after recording the failure.
  Future<T> measure<T>(
    String name,
    Future<T> Function() fn, {
    String Function(T result)? detailFn,
  }) async {
    final start = DateTime.now();
    try {
      final value    = await fn();
      final duration = DateTime.now().difference(start);
      final detail   = detailFn?.call(value);
      results.add(BenchmarkResult(name: name, duration: duration, detail: detail));
      onUpdate?.call();
      return value;
    } catch (e) {
      final duration = DateTime.now().difference(start);
      results.add(BenchmarkResult(name: name, duration: duration, error: e.toString()));
      onUpdate?.call();
      rethrow;
    }
  }

  /// Resets all collected results.
  void clear() => results.clear();

  /// Prints a human-readable summary to the console.
  void printSummary() {
    // ignore: avoid_print
    print('══ Benchmark Results ══');
    for (final r in results) print(r);
    final totalSec = (totalMs / 1000).toStringAsFixed(2);
    // ignore: avoid_print
    print('── Total: ${totalSec}s ──');
  }
}

typedef VoidCallback = void Function();