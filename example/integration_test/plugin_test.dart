// integration_test/plugin_test.dart
//
// Runs the full plugin test suite automatically, driven by the
// integration_test harness (no human interaction needed).
//
// ## Running
//
// Android:
//   flutter test integration_test/plugin_test.dart \
//     --timeout=none -d <device-id>
//
// Web (requires chromedriver running on port 4444):
//   flutter drive \
//     --driver=test_driver/integration_test_driver.dart \
//     --target=integration_test/plugin_test.dart \
//     -d chrome
//
// The test downloads the real model files, so expect 5–15 min on first run
// depending on network speed. Subsequent runs are faster because the files
// are cached in OPFS (web) or app-documents (Android).
//
// ## What is tested
//   Phase 1  LLM download + init
//   Phase 2  Single-turn inference, streaming, stop-mid-stream,
//            image/audio/PDF multimodal, cache clear, settings update
//   Phase 3  JSON schema output
//   Phase 4  Multi-turn chat: context, history export/import,
//            sliding-window strategy, edit/delete history
//   Phase 5  LLM unload + reload
//   Phase 6  Embedding download + init, vector generation,
//            cosine similarity, semantic search, unload
//   Phase 7  Cache purge

import 'package:flutter/material.dart';
import 'package:flutter_local_gemma_example/screens/test_runner_screen.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:flutter_local_gemma_example/main.dart' as app;
import 'package:flutter_local_gemma_example/testing/test_suite.dart';

void main() {
  // Bind the integration-test framework to Flutter's rendering pipeline.
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Report performance results to the harness when running on Android.
  binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;

  // ── One big test that mirrors the visual test runner ─────────────────────
  // We use a single testWidgets so we can pump the app, navigate to the
  // Tests tab, tap Run, and assert results — all in one deterministic flow.

  testWidgets(
    'flutter_local_gemma plugin — full feature suite',
    (tester) async {
      // 1. Boot the app.
      app.main();
      await tester.pumpAndSettle(const Duration(seconds: 2));

      // 2. Navigate to the Tests tab (last NavigationDestination).
      final testTabFinder = find.byIcon(Icons.science_outlined);
      await tester.tap(testTabFinder);
      await tester.pumpAndSettle();

      // 3. Tap "Run All Tests" FAB.
      final fabFinder = find.byKey(const Key('fab_run_all'));
      expect(fabFinder, findsOneWidget, reason: 'Run All Tests FAB not found');
      await tester.tap(fabFinder);

      // 4. Wait for all tests to finish.
      //    The suite runs sequentially; tests include model downloads so we
      //    use a very generous timeout (none = controlled by the test runner
      //    flag --timeout=none we pass on the CLI).
      //
      //    We poll every 5 seconds until the TestSuite reports it's done.
      const pollInterval = Duration(seconds: 5);
      const maxWait = Duration(minutes: 30);
      final deadline = DateTime.now().add(maxWait);

      while (DateTime.now().isBefore(deadline)) {
        await tester.pump(pollInterval);

        // Find the TestRunnerScreen's state to inspect the suite.
        final screenState = tester
            .state<_TestRunnerScreenState>(find.byType(TestRunnerScreen));
        final suite = screenState._suitePublic;

        if (!suite.isRunning) break; // All done.
      }

      // 5. Pump one more time and collect results.
      await tester.pumpAndSettle();

      final screenState = tester
          .state<_TestRunnerScreenState>(find.byType(TestRunnerScreen));
      final suite = screenState._suitePublic;

      // 6. Assert — skip-typed tests don't count as failures.
      final failures = suite.cases.where(
        (c) => c.status == TestStatus.failed,
      );

      if (failures.isNotEmpty) {
        final summary = failures
            .map((c) => '  [${c.id}] ${c.name}\n    ${c.error}')
            .join('\n');
        fail('${failures.length} test(s) failed:\n$summary');
      }

      final passed = suite.cases.where((c) => c.status == TestStatus.passed).length;
      final skipped = suite.cases.where((c) => c.status == TestStatus.skipped).length;
      debugPrint('✅ $passed passed, $skipped skipped, 0 failed.');
    },
    // Timeout is set to none here; control it via the CLI flag so the same
    // file works for quick CI smoke-tests (shorter timeout) and full runs.
    timeout: Timeout.none,
  );
}

// Allow the integration test to access the private state for inspection.
// This extension method mirrors how the real screen stores the suite.
extension on _TestRunnerScreenState {
  TestSuite get _suitePublic => _suitePublic;
}

// Re-export the private type so the extension above compiles.
// (The actual class is in test_runner_screen.dart.)
typedef _TestRunnerScreenState = State<TestRunnerScreen>;