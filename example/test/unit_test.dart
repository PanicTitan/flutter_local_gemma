// test/unit_test.dart
//
// Pure-Dart unit tests — no device or emulator required.
//
//   flutter test test/unit_test.dart
//
// Covers everything that doesn't need the native plugin:
//   - BenchmarkRunner timing + error capture
//   - TokenCounterBar / TokenStats math
//   - ChatMessage JSON round-trip serialisation

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_gemma_example/benchmark_runner.dart';

// Pull in the ChatMessage / TokenStats types from the plugin directly.
// Adjust the import path to match your pubspec dependency name.
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

void main() {
  // ── BenchmarkRunner ────────────────────────────────────────────────────────

  group('BenchmarkRunner', () {
    test('measure() records duration on success', () async {
      final runner = BenchmarkRunner();
      final result = await runner.measure('noop', () async {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        return 42;
      });
      expect(result, 42);
      expect(runner.results, hasLength(1));
      expect(runner.results.first.isSuccess, isTrue);
      expect(runner.results.first.name, 'noop');
      expect(runner.results.first.duration.inMilliseconds, greaterThanOrEqualTo(40));
    });

    test('measure() records error on failure', () async {
      final runner = BenchmarkRunner();
      expect(
        () => runner.measure('boom', () async => throw Exception('test error')),
        throwsException,
      );
      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(runner.results.first.isSuccess, isFalse);
      expect(runner.results.first.error, contains('test error'));
    });

    test('totalMs sums all result durations', () async {
      final runner = BenchmarkRunner();
      await runner.measure('a', () async => Future<void>.delayed(const Duration(milliseconds: 20)));
      await runner.measure('b', () async => Future<void>.delayed(const Duration(milliseconds: 30)));
      expect(runner.totalMs, greaterThanOrEqualTo(50));
    });

    test('clear() empties results', () async {
      final runner = BenchmarkRunner();
      await runner.measure('x', () async {});
      expect(runner.results, hasLength(1));
      runner.clear();
      expect(runner.results, isEmpty);
    });

    test('detailFn is called with return value', () async {
      final runner = BenchmarkRunner();
      await runner.measure(
        'detail-test',
        () async => 'hello world',
        detailFn: (s) => '${s.split(' ').length} words',
      );
      expect(runner.results.first.detail, '2 words');
    });

    test('onUpdate callback fires after each measurement', () async {
      int callCount = 0;
      final runner = BenchmarkRunner(onUpdate: () => callCount++);
      await runner.measure('u1', () async {});
      await runner.measure('u2', () async {});
      expect(callCount, 2);
    });

    test('durationLabel formats correctly', () {
      final ms = BenchmarkResult(name: 'a', duration: const Duration(milliseconds: 500));
      expect(ms.durationLabel, contains('ms'));

      final sec = BenchmarkResult(name: 'b', duration: const Duration(seconds: 3));
      expect(sec.durationLabel, contains('s'));

      final min = BenchmarkResult(name: 'c', duration: const Duration(minutes: 2));
      expect(min.durationLabel, contains('min'));
    });
  });

  // ── TokenStats ─────────────────────────────────────────────────────────────

  group('TokenStats', () {
    test('remaining clamps to zero', () {
      const ts = TokenStats(estimatedUsed: 5000, maxContext: 4096);
      expect(ts.remaining, 0);
    });

    test('usedPercent never exceeds 100', () {
      const ts = TokenStats(estimatedUsed: 9999, maxContext: 100);
      expect(ts.usedPercent, 100.0);
    });

    test('usedPercent is 0 when maxContext is 0', () {
      const ts = TokenStats(estimatedUsed: 0, maxContext: 0);
      expect(ts.usedPercent, 0.0);
    });

    test('normal usage', () {
      const ts = TokenStats(estimatedUsed: 1024, maxContext: 4096);
      expect(ts.remaining, 3072);
      expect(ts.usedPercent, closeTo(25.0, 0.1));
    });
  });

  // ── ChatMessage serialization ──────────────────────────────────────────────

  group('ChatMessage JSON round-trip', () {
    test('text-only message', () {
      final original = const ChatMessage(role: 'user', text: 'Hello, world!');
      final json = original.toJson();
      final restored = ChatMessage.fromJson(json);

      expect(restored.role, original.role);
      expect(restored.text, original.text);
      expect(restored.images, isEmpty);
      expect(restored.audios, isEmpty);
    });

    test('message with images', () {
      final bytes = [1, 2, 3, 4, 5];
      final original = ChatMessage(
        role: 'user',
        text: 'Look at this',
        images: [Uint8List.fromList(bytes)],
      );
      final restored = ChatMessage.fromJson(original.toJson());
      expect(restored.images, hasLength(1));
      expect(restored.images.first, Uint8List.fromList(bytes));
    });

    test('exportHistory / importHistory round-trip (via GemmaChat internal format)', () {
      // We can't call GemmaChat without the engine so we test the JSON
      // shape manually — same format that exportHistory produces.
      final messages = [
        const ChatMessage(role: 'user', text: 'Q'),
        const ChatMessage(role: 'model', text: 'A'),
      ];
      final payload = jsonEncode({
        'version': 1,
        'systemPrompt': 'sys',
        'maxContext': 4096,
        'exportedAt': DateTime.now().toIso8601String(),
        'messages': messages.map((m) => m.toJson()).toList(),
      });

      final decoded = jsonDecode(payload) as Map<String, dynamic>;
      expect(decoded['version'], 1);
      final restored = (decoded['messages'] as List)
          .map((e) => ChatMessage.fromJson(e as Map<String, dynamic>))
          .toList();
      expect(restored, hasLength(2));
      expect(restored[0].role, 'user');
      expect(restored[1].text, 'A');
    });
  });

  // ── SessionConfig defaults ─────────────────────────────────────────────────

  group('SessionConfig', () {
    test('has sensible defaults', () {
      final c = SessionConfig();
      expect(c.temperature, 0.8);
      expect(c.topP, 0.95);
      expect(c.topK, 40);
      expect(c.randomSeed, isNull);
      expect(c.systemPrompt, isNull);
      expect(c.autoStopConfig.enabled, isTrue);
    });

    test('custom values are stored', () {
      final c = SessionConfig(temperature: 0.1, topK: 1, randomSeed: 42);
      expect(c.temperature, 0.1);
      expect(c.topK, 1);
      expect(c.randomSeed, 42);
    });
  });

  // ── AutoStopConfig ─────────────────────────────────────────────────────────

  group('AutoStopConfig', () {
    test('defaults', () {
      const cfg = AutoStopConfig();
      expect(cfg.enabled, isTrue);
      expect(cfg.maxRepetitions, 5);
      expect(cfg.minRepeatCharLen, 3);
      expect(cfg.maxOutputTokens, 8192);
    });

    test('disabled', () {
      const cfg = AutoStopConfig(enabled: false);
      expect(cfg.enabled, isFalse);
    });
  });
}