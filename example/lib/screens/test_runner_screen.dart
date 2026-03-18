// lib/screens/test_runner_screen.dart
//
// Visual test-runner tab. Each row shows live status, duration, and (on pass)
// a one-line detail summary that collapses into the row rather than overflowing.

import 'package:flutter/material.dart';
import '../testing/test_suite.dart';

class TestRunnerScreen extends StatefulWidget {
  const TestRunnerScreen({super.key});
  @override
  State<TestRunnerScreen> createState() => _TestRunnerScreenState();
}

class _TestRunnerScreenState extends State<TestRunnerScreen> {
  late final TestSuite _suite;
  final ScrollController _scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _suite = TestSuite(onUpdate: () { if (mounted) setState(() {}); });
    _suite.ctx.onProgress = (_) { if (mounted) setState(() {}); };
  }

  @override
  void dispose() {
    _scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs      = Theme.of(context).colorScheme;
    final cases   = _suite.cases;
    final running = _suite.isRunning;

    final passed  = _suite.passCount;
    final failed  = _suite.failCount;
    final skipped = _suite.skippedCount;
    final done    = _suite.totalDone;
    final total   = cases.length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Plugin Tests'),
        actions: [
          if (running)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Center(
                child: SizedBox(
                  width: 18, height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
            ),
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reset all',
            onPressed: running ? null : _suite.reset,
          ),
        ],
      ),
      body: Column(
        children: [
          // ── Summary bar ──────────────────────────────────────────────────
          _SummaryBar(
            total: total, done: done,
            passed: passed, failed: failed, skipped: skipped,
            running: running,
          ),

          // ── Download progress (shown during model downloads) ─────────────
          if (running && _suite.ctx.downloadProgress > 0 &&
              _suite.ctx.downloadProgress < 100) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  LinearProgressIndicator(value: _suite.ctx.downloadProgress / 100),
                  const SizedBox(height: 2),
                  Text(
                    'Downloading… ${_suite.ctx.downloadProgress.toStringAsFixed(0)}%',
                    style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
                  ),
                ],
              ),
            ),
          ],

          const Divider(height: 8),

          // ── Test list ────────────────────────────────────────────────────
          Expanded(
            child: ListView.separated(
              controller: _scroll,
              padding: const EdgeInsets.fromLTRB(0, 4, 0, 120),
              itemCount: cases.length,
              separatorBuilder: (_, __) =>
                  const Divider(height: 1, indent: 56),
              itemBuilder: (_, i) => _TestRow(
                tc:     cases[i],
                onRerun: running ? null : () => _suite.runOne(cases[i]),
              ),
            ),
          ),
        ],
      ),

      floatingActionButton: FloatingActionButton.extended(
        key: const Key('fab_run_all'),
        onPressed: running ? null : _suite.runAll,
        icon: running
            ? const SizedBox(
                width: 20, height: 20,
                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
              )
            : const Icon(Icons.play_arrow),
        label: Text(running ? 'Running…' : 'Run All Tests'),
      ),
    );
  }
}

// ─── Summary bar ─────────────────────────────────────────────────────────────

class _SummaryBar extends StatelessWidget {
  final int total, done, passed, failed, skipped;
  final bool running;
  const _SummaryBar({
    required this.total, required this.done, required this.passed,
    required this.failed, required this.skipped, required this.running,
  });

  @override
  Widget build(BuildContext context) {
    final cs    = Theme.of(context).colorScheme;
    final ratio = total == 0 ? 0.0 : (done / total).clamp(0.0, 1.0);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            _StatusChip('$passed passed',  Colors.green),
            if (failed > 0)  ...[const SizedBox(width: 6), _StatusChip('$failed failed',  Colors.red)],
            if (skipped > 0) ...[const SizedBox(width: 6), _StatusChip('$skipped skipped', Colors.orange)],
            const Spacer(),
            Text('$done / $total',
                style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
          ]),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              value: ratio,
              minHeight: 5,
              backgroundColor: cs.surfaceContainerHighest,
              valueColor: AlwaysStoppedAnimation(
                failed > 0 ? cs.error : Colors.green,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final String label;
  final Color color;
  const _StatusChip(this.label, this.color);
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        decoration: BoxDecoration(
          color:  color.withValues(alpha: 0.12),
          border: Border.all(color: color.withValues(alpha: 0.4)),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(label,
            style: TextStyle(
                color: color, fontSize: 11, fontWeight: FontWeight.w600)),
      );
}

// ─── Single test row ──────────────────────────────────────────────────────────

class _TestRow extends StatelessWidget {
  final TestCase tc;
  final VoidCallback? onRerun;
  const _TestRow({required this.tc, this.onRerun});

  @override
  Widget build(BuildContext context) {
    final cs     = Theme.of(context).colorScheme;
    final status = tc.status;

    // ── Per-status appearance ──────────────────────────────────────────────
    final (IconData icon, Color color, bool spin) = switch (status) {
      TestStatus.pending  => (Icons.radio_button_unchecked, cs.outline, false),
      TestStatus.running  => (Icons.sync, cs.primary, true),
      TestStatus.passed   => (Icons.check_circle, Colors.green, false),
      TestStatus.failed   => (Icons.cancel, cs.error, false),
      TestStatus.skipped  => (Icons.skip_next, Colors.orange, false),
    };

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status icon / spinner
          SizedBox(
            width: 28,
            child: spin
                ? SizedBox(
                    width: 20, height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2, color: color),
                  )
                : Icon(icon, size: 20, color: color),
          ),
          const SizedBox(width: 8),

          // Content
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Name + duration
                Row(children: [
                  Expanded(
                    child: Text(
                      tc.name,
                      style: TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                          color: color),
                    ),
                  ),
                  if (tc.duration != null)
                    Text(
                      _fmt(tc.duration!),
                      style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
                    ),
                ]),
                const SizedBox(height: 2),

                // Description
                Text(tc.description,
                    style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),

                // Detail (model reply, extracted text count, etc.)
                if (tc.detail != null && tc.detail!.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    tc.detail!,
                    style: TextStyle(fontSize: 11, color: cs.primary),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],

                // Error / skip message
                if (tc.error != null) ...[
                  const SizedBox(height: 4),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: status == TestStatus.skipped
                          ? Colors.orange.withValues(alpha: 0.12)
                          : cs.errorContainer,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      tc.error!,
                      style: TextStyle(
                        fontSize: 11,
                        color: status == TestStatus.skipped
                            ? Colors.orange.shade800
                            : cs.onErrorContainer,
                      ),
                    ),
                  ),
                ],

                // Retry button (failed only)
                if (status == TestStatus.failed && onRerun != null)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: onRerun,
                      icon: const Icon(Icons.replay, size: 14),
                      label: const Text('Retry', style: TextStyle(fontSize: 12)),
                      style: TextButton.styleFrom(
                        visualDensity: VisualDensity.compact,
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _fmt(Duration d) {
    final ms = d.inMilliseconds;
    if (ms >= 60000) return '${(ms / 60000).toStringAsFixed(1)} min';
    if (ms >= 1000)  return '${(ms / 1000).toStringAsFixed(2)} s';
    return '$ms ms';
  }
}