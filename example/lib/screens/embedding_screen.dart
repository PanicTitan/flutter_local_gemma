import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

import '../app_state.dart';
import '../utils/model_loader.dart';
import '../widgets/model_status_chip.dart';

// ── Domain types ───────────────────────────────────────────────────────────────

class _Document {
  final String text;
  final List<double> embedding;
  final int embeddingDimension;
  final Duration inferenceTime;
  final double? similarity;

  const _Document({
    required this.text,
    required this.embedding,
    required this.embeddingDimension,
    required this.inferenceTime,
    this.similarity,
  });

  _Document withSimilarity(double sim) => _Document(
        text: text, embedding: embedding,
        embeddingDimension: embeddingDimension,
        inferenceTime: inferenceTime,
        similarity: sim,
      );
}

// ── Screen ─────────────────────────────────────────────────────────────────────

class EmbeddingScreen extends StatefulWidget {
  const EmbeddingScreen({super.key});
  @override
  State<EmbeddingScreen> createState() => _EmbeddingScreenState();
}

class _EmbeddingScreenState extends State<EmbeddingScreen> {
  final _searchCtrl = TextEditingController();
  final _docCtrl    = TextEditingController();

  final List<_Document> _db = [];
  bool  _isWorking    = false;
  double _loadProgress = 0;
  Duration? _lastSearchTime;
  String?   _infoMsg;

  @override
  void initState() {
    super.initState();
    ModelManager.instance.addListener(_onManagerUpdate);
  }

  @override
  void dispose() {
    ModelManager.instance.removeListener(_onManagerUpdate);
    _searchCtrl.dispose();
    _docCtrl.dispose();
    super.dispose();
  }

  void _onManagerUpdate() => setState(() {});

  // ── Model ops ──────────────────────────────────────────────────────────────

  Future<void> _loadModel({required bool local}) async {
    try {
      await loadEmbedding(
        local: local,
        onProgress: (p) => setState(() => _loadProgress = p),
      );
      await _addSeedDocuments();
    } catch (e) {
      _snack('Error: ${e.toString().replaceAll('Exception: ', '')}');
    } finally {
      setState(() { _loadProgress = 0; _isWorking = false; });
    }
  }

  Future<void> _unloadModel() async {
    setState(() { _db.clear(); });
    await unloadEmbedding();
  }

  // ── Document ops ───────────────────────────────────────────────────────────

  Future<void> _addSeedDocuments() async {
    const seeds = [
      'Apples are red and grow on trees.',
      'The sky is blue on clear days.',
      'Flutter is a cross-platform UI framework by Google.',
      'Neural networks learn from large datasets.',
      'Cats are popular household pets worldwide.',
    ];
    for (final s in seeds) await _embedAndAdd(s);
  }

  Future<void> _embedAndAdd(String text) async {
    setState(() { _isWorking = true; _infoMsg = 'Embedding…'; });
    try {
      final sw  = Stopwatch()..start();
      final vec = await EmbeddingPlugin().getEmbedding(text);
      sw.stop();
      _db.add(_Document(
        text: text, embedding: vec,
        embeddingDimension: vec.length,
        inferenceTime: sw.elapsed,
      ));
      setState(() {
        _infoMsg = 'Added in ${sw.elapsedMilliseconds} ms  (dim=${vec.length})';
      });
    } catch (e) {
      _snack('Embed error: $e');
    } finally {
      setState(() => _isWorking = false);
    }
  }

  Future<void> _addDocument() async {
    final text = _docCtrl.text.trim();
    if (text.isEmpty) return;
    _docCtrl.clear();
    await _embedAndAdd(text);
  }

  Future<void> _search() async {
    final query = _searchCtrl.text.trim();
    if (query.isEmpty || _db.isEmpty) return;

    setState(() { _isWorking = true; _infoMsg = 'Searching…'; });
    try {
      final sw    = Stopwatch()..start();
      final qvec  = await EmbeddingPlugin().getEmbedding(query);
      sw.stop();
      _lastSearchTime = sw.elapsed;

      final scored = _db
          .map((d) => d.withSimilarity(_cosine(qvec, d.embedding)))
          .toList()
        ..sort((a, b) => (b.similarity ?? 0).compareTo(a.similarity ?? 0));

      setState(() {
        _db
          ..clear()
          ..addAll(scored);
        _infoMsg = 'Query embedded in ${sw.elapsedMilliseconds} ms';
      });
    } catch (e) {
      _snack('Search error: $e');
    } finally {
      setState(() => _isWorking = false);
    }
  }

  double _cosine(List<double> a, List<double> b) {
    double dot = 0, nA = 0, nB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      nA  += a[i] * a[i];
      nB  += b[i] * b[i];
    }
    final denom = sqrt(nA) * sqrt(nB);
    return denom == 0 ? 0 : dot / denom;
  }

  void _snack(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final mgr    = ModelManager.instance;
    final status = mgr.embeddingStatus;
    final cs     = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          const Text('Vector DB'),
          const SizedBox(width: 8),
          ModelStatusChip(status, errorText: mgr.embeddingError),
        ]),
        actions: [
          if (status != ModelStatus.unloaded)
            IconButton(
              icon: const Icon(Icons.power_settings_new),
              tooltip: 'Unload model',
              onPressed: status.isBusy ? null : _unloadModel,
            ),
          Builder(builder: (c) => IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => Scaffold.of(c).openEndDrawer(),
          )),
        ],
      ),
      endDrawer: _buildInfoDrawer(mgr),
      body: status == ModelStatus.unloaded || status == ModelStatus.error
          ? _buildLoadView(mgr)
          : status.isBusy && _db.isEmpty
              ? _buildLoadingView(status)
              : _buildMainContent(cs),
    );
  }

  Widget _buildLoadView(ModelManager mgr) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (mgr.embeddingStatus == ModelStatus.error) ...[
              Icon(Icons.error_outline, color: Theme.of(context).colorScheme.error, size: 40),
              const SizedBox(height: 8),
              Text(mgr.embeddingError ?? '', textAlign: TextAlign.center, style: TextStyle(color: Theme.of(context).colorScheme.error)),
              const SizedBox(height: 16),
            ],
            FilledButton.icon(
              onPressed: () => _loadModel(local: false),
              icon: const Icon(Icons.cloud_download_outlined),
              label: const Text('Download Embedding Model'),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: () => _loadModel(local: true),
              icon: const Icon(Icons.folder_open_outlined),
              label: const Text('Load Local File'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingView(ModelStatus status) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(status.label),
          if (_loadProgress > 0) ...[
            const SizedBox(height: 8),
            SizedBox(width: 200, child: LinearProgressIndicator(value: _loadProgress / 100)),
            const SizedBox(height: 4),
            Text('${_loadProgress.toStringAsFixed(0)}%', style: const TextStyle(fontSize: 12)),
          ],
        ],
      ),
    );
  }

  Widget _buildMainContent(ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.all(14),
      child: Column(
        children: [
          // Status banner
          if (_infoMsg != null)
            Container(
              width: double.infinity,
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: cs.secondaryContainer,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(children: [
                if (_isWorking)
                  const Padding(padding: EdgeInsets.only(right: 8), child: SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2))),
                Expanded(child: Text(_infoMsg!, style: TextStyle(fontSize: 12, color: cs.onSecondaryContainer))),
              ]),
            ),

          // Search bar
          Row(children: [
            Expanded(child: TextField(
              controller: _searchCtrl,
              decoration: InputDecoration(
                hintText: 'Semantic search query…',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(28)),
                isDense: true,
              ),
              onSubmitted: (_) => _search(),
            )),
            const SizedBox(width: 8),
            FilledButton.tonal(onPressed: _isWorking ? null : _search, child: const Text('Search')),
          ]),

          if (_lastSearchTime != null)
            Padding(
              padding: const EdgeInsets.only(top: 4, left: 4),
              child: Text('Query: ${_lastSearchTime!.inMilliseconds} ms', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            ),

          const SizedBox(height: 10),

          // Add document
          Row(children: [
            Expanded(child: TextField(
              controller: _docCtrl,
              decoration: InputDecoration(
                hintText: 'Add text to the vector DB…',
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                isDense: true,
              ),
              onSubmitted: (_) => _addDocument(),
            )),
            const SizedBox(width: 8),
            IconButton.filledTonal(
              onPressed: _isWorking ? null : _addDocument,
              icon: const Icon(Icons.add),
            ),
          ]),

          const SizedBox(height: 10),
          Divider(color: cs.outlineVariant),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(children: [
              Text('${_db.length} document(s)', style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
              const Spacer(),
              if (_db.isNotEmpty)
                TextButton.icon(
                  onPressed: () => setState(() { _db.clear(); _infoMsg = null; _lastSearchTime = null; }),
                  icon: const Icon(Icons.clear_all, size: 16),
                  label: const Text('Clear all', style: TextStyle(fontSize: 12)),
                ),
            ]),
          ),

          // Document list
          Expanded(
            child: _db.isEmpty
                ? Center(child: Text('No documents yet.', style: TextStyle(color: cs.onSurfaceVariant)))
                : ListView.builder(
                    itemCount: _db.length,
                    itemBuilder: (_, i) => _DocumentCard(_db[i], onDelete: () => setState(() => _db.removeAt(i))),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoDrawer(ModelManager mgr) {
    final avgInference = _db.isEmpty
        ? null
        : (_db.fold(0, (s, d) => s + d.inferenceTime.inMilliseconds) / _db.length).toStringAsFixed(0);

    return Drawer(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 48, 16, 16),
        children: [
          const Text('DB Stats', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const Divider(height: 24),
          _InfoRow('Status',         mgr.embeddingStatus.label),
          _InfoRow('Documents',      '${_db.length}'),
          if (_db.isNotEmpty) ...[
            _InfoRow('Vector dimension', '${_db.first.embeddingDimension}'),
            _InfoRow('Avg inference', '${avgInference} ms'),
            _InfoRow('Fastest',       '${_db.map((d) => d.inferenceTime.inMilliseconds).reduce(min)} ms'),
            _InfoRow('Slowest',       '${_db.map((d) => d.inferenceTime.inMilliseconds).reduce(max)} ms'),
          ],
          if (_lastSearchTime != null)
            _InfoRow('Last search time', '${_lastSearchTime!.inMilliseconds} ms'),
          const Divider(height: 24),
          const Text('Cache', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: () async {
              await ModelManager.instance.purgeCache();
              if (mounted) Navigator.pop(context);
              _snack('Cache purged.');
            },
            icon: const Icon(Icons.delete_forever_outlined, size: 18),
            label: const Text('Purge Model Cache'),
            style: OutlinedButton.styleFrom(foregroundColor: Theme.of(context).colorScheme.error),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────────────

class _DocumentCard extends StatelessWidget {
  final _Document doc;
  final VoidCallback? onDelete;
  const _DocumentCard(this.doc, {this.onDelete});

  @override
  Widget build(BuildContext context) {
    final cs  = Theme.of(context).colorScheme;
    final sim = doc.similarity;
    final isMatch = (sim ?? 0) > 0.60;

    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      color: isMatch ? cs.primaryContainer : cs.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(doc.text, style: TextStyle(color: isMatch ? cs.onPrimaryContainer : cs.onSurfaceVariant)),
                  const SizedBox(height: 4),
                  Row(children: [
                    Icon(Icons.timer_outlined, size: 12, color: cs.outline),
                    const SizedBox(width: 3),
                    Text('${doc.inferenceTime.inMilliseconds} ms', style: TextStyle(fontSize: 11, color: cs.outline)),
                    const SizedBox(width: 10),
                    Icon(Icons.hub_outlined, size: 12, color: cs.outline),
                    const SizedBox(width: 3),
                    Text('dim=${doc.embeddingDimension}', style: TextStyle(fontSize: 11, color: cs.outline)),
                    if (sim != null) ...[
                      const SizedBox(width: 10),
                      Icon(Icons.percent, size: 12, color: isMatch ? cs.primary : cs.outline),
                      Text('${(sim * 100).toStringAsFixed(1)}%', style: TextStyle(fontSize: 11, fontWeight: isMatch ? FontWeight.bold : FontWeight.normal, color: isMatch ? cs.primary : cs.outline)),
                    ],
                  ]),
                ],
              ),
            ),
            if (onDelete != null)
              IconButton(icon: const Icon(Icons.close, size: 16), onPressed: onDelete, padding: EdgeInsets.zero, constraints: const BoxConstraints()),
          ],
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label, value;
  const _InfoRow(this.label, this.value);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(children: [
          Text(label, style: const TextStyle(fontSize: 13)),
          const Spacer(),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
        ]),
      );
}