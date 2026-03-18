import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

import '../app_state.dart';
import '../benchmark_runner.dart';
import '../utils/model_loader.dart';
import '../widgets/model_status_chip.dart';

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});
  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  late BenchmarkRunner _runner;
  bool _isRunning = false;
  double _loadProgress = 0;

  final _promptCtrl = TextEditingController(text: 'Explain quantum entanglement in 3 sentences.');

  /// The last generated response, shown after inference benchmarks.
  String? _lastResponse;

  @override
  void initState() {
    super.initState();
    _runner = BenchmarkRunner(onUpdate: () => setState(() {}));
    // _runner.onUpdate = () => setState(() {});
    ModelManager.instance.addListener(_onManagerUpdate);
  }

  @override
  void dispose() {
    ModelManager.instance.removeListener(_onManagerUpdate);
    _promptCtrl.dispose();
    super.dispose();
  }

  void _onManagerUpdate() => setState(() {});

  // ── Benchmark suites ───────────────────────────────────────────────────────

  Future<void> _runFullSuite() async {
    _runner.clear();
    _lastResponse = null;
    setState(() => _isRunning = true);

    final mgr = ModelManager.instance;

    try {
      // 1. Download (if not already loaded)
      if (mgr.llmStatus == ModelStatus.unloaded) {
        final url = kIsWeb
            ? 'https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4-Web.litertlm?download=true'
            : 'https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm?download=true';

        mgr.setLlmStatus(ModelStatus.downloading);
        final path = await _runner.measure(
          'Model download',
          () async {
            return await FlutterLocalGemma.installModel()
                .fromNetwork(url, token: 'hf_YOUR_TOKEN_HERE')
                .withProgress((p) => setState(() => _loadProgress = p))
                .install();
          },
          detailFn: (p) => p,
        );
        setState(() => _loadProgress = 0);

        // 2. Model load
        mgr.setLlmStatus(ModelStatus.loading);
        await _runner.measure('Model load (cold)', () async {
          await FlutterLocalGemma().init(InferenceConfig(
            modelPath:    path,
            maxTokens:    mgr.maxTokens,
            backend:      kIsWeb ? null : (mgr.useGpu ? PreferredBackend.gpu : PreferredBackend.cpu),
            supportAudio: false,
          ));
        });
        mgr.setLlmStatus(ModelStatus.ready);
      }

      // 3. First inference (cold session)
      final chat = SingleTurnChat(config: mgr.sessionConfig);
      final prompt = _promptCtrl.text.trim();

      final reply = await _runner.measure(
        'First inference (text)',
        () => chat.generate(prompt),
        detailFn: (r) => '${r.split(' ').length} words',
      );
      _lastResponse = reply;

      // 4. Second inference (warm session, same engine)
      await _runner.measure(
        'Second inference (warm)',
        () => chat.generate('Continue with one more sentence.'),
        detailFn: (r) => '${r.split(' ').length} words',
      );

      // 5. Token estimation
      await _runner.measure(
        'Token estimation (100 chars)',
        () => chat.estimateTokens(prompt),
        detailFn: (count) => '$count tokens',
      );

      // 6. Unload
      await _runner.measure('Model unload', () async {
        await FlutterLocalGemma().dispose();
        mgr.setLlmStatus(ModelStatus.unloaded);
      });

      // 7. Reload (warm – WASM/JNI already initialized)
      mgr.setLlmStatus(ModelStatus.loading);
      await _runner.measure('Model reload (warm)', () async {
        await loadLlm(onProgress: (p) => setState(() => _loadProgress = p));
      });
      setState(() => _loadProgress = 0);

      // 8. Inference after reload
      await _runner.measure(
        'Inference after reload',
        () => SingleTurnChat(config: mgr.sessionConfig).generate(prompt),
        detailFn: (r) => '${r.split(' ').length} words',
      );

      _runner.printSummary();
      _snack('Benchmark complete!');
    } catch (e) {
      _snack('Benchmark error: $e');
    } finally {
      setState(() => _isRunning = false);
    }
  }

  /// Quick single inference benchmark (assumes model already loaded).
  Future<void> _runInferenceOnly() async {
    _runner.clear();
    _lastResponse = null;
    setState(() => _isRunning = true);

    final mgr = ModelManager.instance;
    if (mgr.llmStatus != ModelStatus.ready) {
      _snack('Load the model first via the Chat tab or the full suite.');
      setState(() => _isRunning = false);
      return;
    }

    try {
      final prompt = _promptCtrl.text.trim();
      final chat   = SingleTurnChat(config: mgr.sessionConfig);

      for (int i = 1; i <= 3; i++) {
        final reply = await _runner.measure(
          'Inference run #$i',
          () => chat.generate(prompt),
          detailFn: (r) => '${r.split(' ').length} words',
        );
        _lastResponse = reply;
      }

      _runner.printSummary();
      _snack('Done – 3 runs completed.');
    } catch (e) {
      _snack('Error: $e');
    } finally {
      setState(() => _isRunning = false);
    }
  }

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final mgr = ModelManager.instance;
    final cs  = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          const Text('Benchmark'),
          const SizedBox(width: 8),
          ModelStatusChip(mgr.llmStatus),
        ]),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Prompt input
            const Text('Test Prompt', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            TextField(
              controller: _promptCtrl,
              maxLines: 3, minLines: 1,
              decoration: const InputDecoration(border: OutlineInputBorder(), isDense: true),
            ),
            const SizedBox(height: 12),

            // Action buttons
            Row(children: [
              Expanded(child: FilledButton.icon(
                onPressed: _isRunning ? null : _runFullSuite,
                icon: const Icon(Icons.play_arrow, size: 18),
                label: const Text('Full Suite'),
              )),
              const SizedBox(width: 8),
              Expanded(child: FilledButton.icon(
                onPressed: _isRunning ? null : _runInferenceOnly,
                icon: const Icon(Icons.speed, size: 18),
                label: const Text('Inference ×3'),
              )),
              const SizedBox(width: 8),
              OutlinedButton(
                onPressed: _isRunning ? null : () { _runner.clear(); setState(() => _lastResponse = null); },
                child: const Text('Clear'),
              ),
            ]),

            // Download progress
            if (_loadProgress > 0) ...[
              const SizedBox(height: 8),
              LinearProgressIndicator(value: _loadProgress / 100),
              const SizedBox(height: 2),
              Text('${_loadProgress.toStringAsFixed(0)}%', style: const TextStyle(fontSize: 11)),
            ],

            const Divider(height: 24),

            // Results table header
            if (_runner.results.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(children: [
                  const Expanded(child: Text('Test', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold))),
                  const Text('Time', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  if (_runner.results.any((r) => r.detail != null))
                    const Padding(padding: EdgeInsets.only(left: 12), child: Text('Detail', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold))),
                ]),
              ),

            // Results list
            Expanded(
              child: _runner.results.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.timer_outlined, size: 48, color: Colors.grey),
                          const SizedBox(height: 8),
                          Text('No results yet. Run a benchmark.', style: TextStyle(color: cs.onSurfaceVariant)),
                        ],
                      ),
                    )
                  : Column(
                      children: [
                        Expanded(
                          child: ListView.separated(
                            itemCount: _runner.results.length,
                            separatorBuilder: (_, __) => const Divider(height: 1),
                            itemBuilder: (_, i) {
                              final r = _runner.results[i];
                              return _ResultRow(r);
                            },
                          ),
                        ),
                        // Summary footer
                        Container(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          child: Row(children: [
                            const Text('Total', style: TextStyle(fontWeight: FontWeight.bold)),
                            const Spacer(),
                            Text(
                              _runner.results.isNotEmpty
                                  ? _formatDuration(Duration(milliseconds: _runner.totalMs))
                                  : '',
                              style: const TextStyle(fontWeight: FontWeight.bold),
                            ),
                          ]),
                        ),
                      ],
                    ),
            ),

            // Last response preview
            if (_lastResponse != null) ...[
              const Divider(),
              const Text('Last Response', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              const SizedBox(height: 4),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: cs.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SelectableText(
                  _lastResponse!,
                  style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
                  maxLines: 4,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _formatDuration(Duration d) {
    final ms = d.inMilliseconds;
    if (ms >= 60000) return '${(ms / 60000).toStringAsFixed(1)} min';
    if (ms >= 1000)  return '${(ms / 1000).toStringAsFixed(2)} s';
    return '$ms ms';
  }
}

class _ResultRow extends StatelessWidget {
  final BenchmarkResult result;
  const _ResultRow(this.result);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final color = result.isSuccess ? cs.onSurface : cs.error;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(result.isSuccess ? Icons.check_circle_outline : Icons.error_outline, size: 16, color: result.isSuccess ? Colors.green : cs.error),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(result.name, style: TextStyle(fontSize: 13, color: color)),
                if (result.error != null)
                  Text(result.error!, style: TextStyle(fontSize: 11, color: cs.error), maxLines: 2, overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(result.durationLabel, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: color)),
              if (result.detail != null)
                Text(result.detail!, style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            ],
          ),
        ],
      ),
    );
  }
}