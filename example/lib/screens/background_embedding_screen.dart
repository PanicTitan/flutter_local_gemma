import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

import '../app_state.dart';
import '../services/foreground_task_controller.dart';
import '../utils/model_loader.dart';

/// Embedded document record stored during background processing.
class BackgroundItem {
  final String id;
  final String text;
  final String? imagePath;
  final List<double> vector;
  final int latencyMs;
  final DateTime timestamp;
  final bool wasInBackground;

  const BackgroundItem({
    required this.id,
    required this.text,
    this.imagePath,
    required this.vector,
    required this.latencyMs,
    required this.timestamp,
    required this.wasInBackground,
  });
}

/// Screen demonstrating EmbeddingGemma 2 processing large batches of items
/// in the background without being suspended or killed by the OS.
class BackgroundEmbeddingScreen extends StatefulWidget {
  const BackgroundEmbeddingScreen({super.key});

  @override
  State<BackgroundEmbeddingScreen> createState() => _BackgroundEmbeddingScreenState();
}

class _BackgroundEmbeddingScreenState extends State<BackgroundEmbeddingScreen>
    with WidgetsBindingObserver {
  final _plugin = EmbeddingPlugin();
  final _picker = ImagePicker();
  final _searchController = TextEditingController();
  final _customInputController = TextEditingController();

  // App lifecycle state
  bool _isAppInBackground = false;

  // Background keepalive settings
  bool _keepForegroundActive = ForegroundTaskController.instance.enabled;

  // Processing state
  bool _isProcessing = false;
  bool _isPaused = false;
  int _currentIndex = 0;
  int _totalItems = 50;
  final List<BackgroundItem> _processedItems = [];
  final List<String> _eventLogs = [];
  final ScrollController _logScrollController = ScrollController();

  // Benchmark stats
  final List<int> _latencies = [];
  Stopwatch? _batchStopwatch;
  String _statusText = 'Ready to start background embedding';

  // Search results
  List<({BackgroundItem item, double similarity})> _searchResults = [];

  // Workload selection
  int _selectedWorkloadType = 0; // 0: Knowledge Corpus, 1: Custom Text, 2: Device Photos
  final List<XFile> _selectedPhotos = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    ModelManager.instance.addListener(_onModelManagerChanged);
    _logEvent('Background embedding module initialized.');
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    ModelManager.instance.removeListener(_onModelManagerChanged);
    _searchController.dispose();
    _customInputController.dispose();
    _logScrollController.dispose();
    super.dispose();
  }

  void _onModelManagerChanged() {
    if (mounted) setState(() {});
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    final isBackground = state == AppLifecycleState.paused ||
        state == AppLifecycleState.hidden ||
        state == AppLifecycleState.inactive;

    if (isBackground != _isAppInBackground) {
      _isAppInBackground = isBackground;
      if (isBackground) {
        _logEvent('App entered BACKGROUND ($state) — processing continues!');
      } else {
        _logEvent('App RESUMED to foreground ($state).');
      }
    }
  }

  void _logEvent(String msg) {
    final now = DateTime.now();
    final timeStr =
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}';
    final entry = '[$timeStr] $msg';
    debugPrint('[BG-Embed] $entry');

    if (mounted) {
      setState(() {
        _eventLogs.add(entry);
        if (_eventLogs.length > 300) _eventLogs.removeAt(0);
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_logScrollController.hasClients) {
          _logScrollController.animateTo(
            _logScrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 150),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  // ── Workload Preparation ───────────────────────────────────────────────────

  List<String> _buildWorkloadQueue() {
    if (_selectedWorkloadType == 0) {
      return _generateKnowledgeCorpus(_totalItems);
    } else if (_selectedWorkloadType == 1) {
      final lines = _customInputController.text
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();
      return lines.isNotEmpty
          ? lines
          : _generateKnowledgeCorpus(_totalItems);
    } else {
      return _selectedPhotos.map((p) => p.name).toList();
    }
  }

  List<String> _generateKnowledgeCorpus(int count) {
    final topics = [
      'Quantum mechanics describes physics at microscopic atomic and subatomic scales.',
      'Photosynthesis allows plants to convert solar energy and carbon dioxide into oxygen.',
      'Convolutional neural networks extract local visual features across image feature maps.',
      'The James Webb Space Telescope observes the earliest galaxies using infrared sensors.',
      'Flutter compiles Dart code directly to native ARM64 machine instructions for high speed.',
      'The Mediterranean diet includes extra virgin olive oil, fresh vegetables, and whole grains.',
      'Plate tectonics explains seismic activity, volcanic eruptions, and continental drift.',
      'Transformer self-attention mechanisms compute pairwise query-key relationship matrices.',
      'The Great Barrier Reef off Queensland contains thousands of unique coral species.',
      'General relativity models gravitational attraction as curvature of four-dimensional spacetime.',
      'CRISPR Cas9 enables precise enzymatic targeted modifications within genomic sequences.',
      'Renewable offshore wind turbines generate clean electricity via aerodynamic rotor torque.',
      'MediaPipe Tasks enables cross-platform on-device machine learning inference acceleration.',
      'Deep learning embeddings map multi-modal semantic data onto dense mathematical vector spaces.',
      'Vulkan and Impeller provide modern multi-threaded graphics rendering pipelines for mobile.',
      'Honeybees communicate the spatial direction of nectar sources using elaborate waggle dances.',
      'LiteRT on Android leverages hardware NPU and OpenCL GPU acceleration delegates.',
      'Bacteriophages are specialized viruses that target and replicate within bacterial hosts.',
      'Matryoshka Representation Learning produces hierarchical embeddings truncatable to arbitrary sizes.',
      'Superconductivity allows zero electrical resistance below critical cryogenic temperatures.',
    ];

    final queue = <String>[];
    for (int i = 0; i < count; i++) {
      final base = topics[i % topics.length];
      final variation = (i ~/ topics.length) + 1;
      queue.add(variation == 1 ? base : '[$variation] $base');
    }
    return queue;
  }

  // ── Background Execution Pipeline ──────────────────────────────────────────

  Future<void> _startBackgroundProcessing() async {
    if (!_plugin.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please load the EmbeddingGemma 2 model first!'),
          backgroundColor: Colors.redAccent,
        ),
      );
      return;
    }

    final queue = _buildWorkloadQueue();
    if (queue.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Workload queue is empty!')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _isPaused = false;
      _currentIndex = 0;
      _latencies.clear();
      _processedItems.clear();
      _totalItems = queue.length;
      _batchStopwatch = Stopwatch()..start();
      _statusText = 'Starting background keepalive service…';
    });

    _logEvent('Starting batch embedding of ${queue.length} items.');

    // 1. Start background keepalive service (Foreground service on Android, silent audio on Web)
    try {
      await ForegroundTaskController.instance.startTask(
        title: 'EmbeddingGemma 2 Batch',
        text: 'Preparing to embed ${queue.length} items in background…',
      );
      _logEvent('Foreground keepalive service active (Wakelock / Audio keepalive enabled).');
    } catch (e) {
      _logEvent('Notice: keepalive service start note: $e');
    }

    // 2. Process items sequentially
    for (int i = 0; i < queue.length; i++) {
      if (!_isProcessing) break;

      while (_isPaused && _isProcessing) {
        await Future.delayed(const Duration(milliseconds: 200));
      }
      if (!_isProcessing) break;

      final itemText = queue[i];
      final itemNumber = i + 1;
      final wasInBg = _isAppInBackground;

      final sw = Stopwatch()..start();
      try {
        List<double> vector;

        if (_selectedWorkloadType == 2 && i < _selectedPhotos.length) {
          // Multimodal image embedding
          final photoBytes = await _selectedPhotos[i].readAsBytes();
          vector = await _plugin.getEmbeddingFromImage(photoBytes);
        } else {
          // Text embedding
          vector = await _plugin.getEmbedding(
            itemText,
            taskType: EmbeddingTaskType.retrievalDocument,
          );
        }
        sw.stop();

        final latency = sw.elapsedMilliseconds;
        _latencies.add(latency);

        final item = BackgroundItem(
          id: 'bg_item_${DateTime.now().millisecondsSinceEpoch}_$i',
          text: itemText,
          imagePath: _selectedWorkloadType == 2 && i < _selectedPhotos.length
              ? _selectedPhotos[i].path
              : null,
          vector: vector,
          latencyMs: latency,
          timestamp: DateTime.now(),
          wasInBackground: wasInBg,
        );

        _processedItems.add(item);

        final progressPct = ((itemNumber / queue.length) * 100).toInt();
        final avgMs = (_latencies.reduce((a, b) => a + b) / _latencies.length).round();

        // Update Android Foreground Notification
        final notifText = 'Embedded $itemNumber/${queue.length} ($progressPct%) • avg ${avgMs}ms/item';
        ForegroundTaskController.instance.updateTask(
          title: 'EmbeddingGemma 2 Processing ($progressPct%)',
          text: notifText,
        );

        if (itemNumber % 5 == 0 || itemNumber == queue.length || wasInBg) {
          _logEvent('Item #$itemNumber/${queue.length} embedded (${latency}ms, ${vector.length}d)${wasInBg ? " [BG]" : ""}');
        }

        if (mounted) {
          setState(() {
            _currentIndex = itemNumber;
            _statusText = notifText;
          });
        }
      } catch (e) {
        sw.stop();
        _logEvent('Error embedding item #$itemNumber: $e');
      }

      // Small yield to allow UI and OS event pumping
      await Future.delayed(const Duration(milliseconds: 20));
    }

    _batchStopwatch?.stop();
    final elapsedSec = ((_batchStopwatch?.elapsedMilliseconds ?? 0) / 1000).toStringAsFixed(1);
    final totalAvg = _latencies.isEmpty
        ? 0
        : (_latencies.reduce((a, b) => a + b) / _latencies.length).round();

    _logEvent('Finished! Processed ${_processedItems.length}/${queue.length} items in ${elapsedSec}s (avg ${totalAvg}ms/item).');

    // 3. Stop background keepalive service
    try {
      await ForegroundTaskController.instance.stopTask();
      _logEvent('Foreground keepalive service stopped.');
    } catch (e) {
      debugPrint('stopTask note: $e');
    }

    if (mounted) {
      setState(() {
        _isProcessing = false;
        _isPaused = false;
        _statusText = 'Completed ${_processedItems.length} items in ${elapsedSec}s';
      });
    }
  }

  void _pauseProcessing() {
    setState(() {
      _isPaused = true;
      _statusText = 'Processing paused';
    });
    _logEvent('Batch processing paused by user.');
  }

  void _resumeProcessing() {
    setState(() {
      _isPaused = false;
      _statusText = 'Resuming processing…';
    });
    _logEvent('Batch processing resumed.');
  }

  Future<void> _stopProcessing() async {
    _logEvent('Stopping batch processing…');
    setState(() {
      _isProcessing = false;
      _isPaused = false;
      _statusText = 'Processing cancelled';
    });
    await ForegroundTaskController.instance.stopTask();
  }

  // ── Verification Search ────────────────────────────────────────────────────

  Future<void> _searchEmbeddedItems(String query) async {
    final q = query.trim();
    if (q.isEmpty || _processedItems.isEmpty || !_plugin.isInitialized) {
      setState(() => _searchResults = []);
      return;
    }

    try {
      final qVector = await _plugin.getEmbedding(
        q,
        taskType: EmbeddingTaskType.retrievalQuery,
      );
      final results = <({BackgroundItem item, double similarity})>[];

      for (final item in _processedItems) {
        final sim = _cosineSimilarity(qVector, item.vector);
        results.add((item: item, similarity: sim));
      }

      results.sort((a, b) => b.similarity.compareTo(a.similarity));
      setState(() {
        _searchResults = results.take(10).toList();
      });
    } catch (e) {
      _logEvent('Search error: $e');
    }
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return (dot / (math.sqrt(normA) * math.sqrt(normB))).clamp(-1.0, 1.0);
  }

  // ── Photos Picker ──────────────────────────────────────────────────────────

  Future<void> _pickPhotosForWorkload() async {
    try {
      final picked = await _picker.pickMultiImage();
      if (picked.isEmpty) return;
      setState(() {
        _selectedPhotos.clear();
        _selectedPhotos.addAll(picked);
        _totalItems = picked.length;
      });
      _logEvent('Selected ${picked.length} photos for background embedding.');
    } catch (e) {
      _logEvent('Error picking photos: $e');
    }
  }

  // ── UI Builder ─────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final isModelReady = _plugin.isInitialized;
    final progressVal = _totalItems > 0 ? (_currentIndex / _totalItems).clamp(0.0, 1.0) : 0.0;
    final avgLatency = _latencies.isEmpty
        ? 0
        : (_latencies.reduce((a, b) => a + b) / _latencies.length).round();

    return Scaffold(
      backgroundColor: const Color(0xFF0F1218),
      appBar: AppBar(
        backgroundColor: const Color(0xFF151922),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Colors.teal, Colors.cyanAccent],
                ),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.sync_rounded, color: Colors.black87, size: 20),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Background Embedding',
                  style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                Text(
                  'Android Foreground Service & Web Audio Keepalive',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Clear Log & Store',
            icon: const Icon(Icons.delete_sweep_outlined, color: Colors.white70),
            onPressed: () {
              setState(() {
                _eventLogs.clear();
                _processedItems.clear();
                _latencies.clear();
                _searchResults.clear();
                _currentIndex = 0;
                _statusText = 'Database & logs cleared';
              });
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── 1. Model Status Banner ──────────────────────────────────────
            _buildModelStatusCard(isModelReady),
            const SizedBox(height: 14),

            // ── 2. Background Keepalive Configuration Card ──────────────────
            _buildKeepaliveConfigCard(),
            const SizedBox(height: 14),

            // ── 3. Workload Selector Card ───────────────────────────────────
            _buildWorkloadSelectorCard(),
            const SizedBox(height: 14),

            // ── 4. Live Progress & Execution Controller ─────────────────────
            _buildProgressAndControlsCard(progressVal, avgLatency),
            const SizedBox(height: 14),

            // ── 5. Live Execution Event Logs ────────────────────────────────
            _buildLiveEventLogCard(),
            const SizedBox(height: 14),

            // ── 6. Semantic Search Verification ─────────────────────────────
            _buildSearchVerificationCard(),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  // ── Section 1: Model Status Card ───────────────────────────────────────────

  Widget _buildModelStatusCard(bool isModelReady) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF161B24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isModelReady
              ? Colors.teal.withValues(alpha: 0.4)
              : Colors.amber.withValues(alpha: 0.4),
        ),
      ),
      child: Row(
        children: [
          Icon(
            isModelReady ? Icons.check_circle_rounded : Icons.warning_amber_rounded,
            color: isModelReady ? Colors.tealAccent : Colors.amberAccent,
            size: 28,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isModelReady
                      ? 'EmbeddingGemma 2 Ready (GPU Accelerated)'
                      : 'Model Not Initialized',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                Text(
                  isModelReady
                      ? 'UniversalEmbedder • 768d dense vectors • Asymmetric instructions'
                      : 'Open the Models tab or tap setup to load the model file',
                  style: const TextStyle(color: Colors.white60, fontSize: 11),
                ),
              ],
            ),
          ),
          if (!isModelReady)
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.amber,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              ),
              onPressed: () => _loadModelQuickly(),
              child: const Text('Load Model', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Future<void> _loadModelQuickly() async {
    try {
      _logEvent('Attempting to initialize EmbeddingGemma 2…');
      await loadEmbedding(
        local: true,
        onProgress: (p) => _logEvent('Loading model: ${(p * 100).toInt()}%'),
      );
      if (mounted) setState(() {});
      _logEvent('EmbeddingGemma 2 initialized successfully!');
    } catch (e) {
      _logEvent('Model initialization failed: $e');
    }
  }

  // ── Section 2: Keepalive Config Card ───────────────────────────────────────

  Widget _buildKeepaliveConfigCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.shield_outlined, color: Colors.cyanAccent, size: 20),
              const SizedBox(width: 8),
              const Text(
                'Background Keepalive Protection',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _keepForegroundActive
                      ? Colors.teal.withValues(alpha: 0.2)
                      : Colors.white10,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  kIsWeb ? 'Web Audio Keepalive' : 'Android Service (#501)',
                  style: TextStyle(
                    color: _keepForegroundActive ? Colors.tealAccent : Colors.grey,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Prevents mobile OS from throttling CPU/GPU or killing the app when you minimize it, switch apps, or turn off the screen. Uses Android Foreground Task on mobile and silent HTML5 loop on Web.',
            style: TextStyle(color: Colors.white60, fontSize: 11),
          ),
          const SizedBox(height: 10),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            dense: true,
            title: const Text('Keep App Active While Processing', style: TextStyle(color: Colors.white, fontSize: 13)),
            subtitle: Text(
              _keepForegroundActive ? 'Active: Will survive screen lock & backgrounding' : 'Disabled: OS may suspend app in background',
              style: const TextStyle(color: Colors.grey, fontSize: 11),
            ),
            value: _keepForegroundActive,
            activeThumbColor: Colors.tealAccent,
            onChanged: (val) {
              setState(() {
                _keepForegroundActive = val;
                ForegroundTaskController.instance.enabled = val;
              });
              _logEvent('Background keepalive set to: $val');
            },
          ),
        ],
      ),
    );
  }

  // ── Section 3: Workload Selector Card ──────────────────────────────────────

  Widget _buildWorkloadSelectorCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Select Workload to Embed in Background',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
          ),
          const SizedBox(height: 12),
          SegmentedButton<int>(
            segments: const [
              ButtonSegment(
                value: 0,
                label: Text('Corpus (Text)', style: TextStyle(fontSize: 12)),
                icon: Icon(Icons.library_books_outlined, size: 16),
              ),
              ButtonSegment(
                value: 1,
                label: Text('Custom Text', style: TextStyle(fontSize: 12)),
                icon: Icon(Icons.edit_note, size: 16),
              ),
              ButtonSegment(
                value: 2,
                label: Text('Device Photos', style: TextStyle(fontSize: 12)),
                icon: Icon(Icons.photo_library, size: 16),
              ),
            ],
            selected: {_selectedWorkloadType},
            onSelectionChanged: (set) {
              setState(() => _selectedWorkloadType = set.first);
            },
          ),
          const SizedBox(height: 14),

          // Sub-options based on type
          if (_selectedWorkloadType == 0) ...[
            Row(
              children: [
                const Text('Queue Batch Size:', style: TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(width: 12),
                for (final size in [25, 50, 100, 200]) ...[
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: ChoiceChip(
                      label: Text('$size items', style: const TextStyle(fontSize: 11)),
                      selected: _totalItems == size,
                      onSelected: (sel) {
                        if (sel) setState(() => _totalItems = size);
                      },
                    ),
                  ),
                ],
              ],
            ),
          ] else if (_selectedWorkloadType == 1) ...[
            TextField(
              controller: _customInputController,
              maxLines: 4,
              style: const TextStyle(color: Colors.white, fontSize: 12),
              decoration: InputDecoration(
                hintText: 'Enter one text line per document to embed…\nExample:\nFlutter on-device AI\nEmbeddingGemma 2 768d\nMediaPipe vector search',
                hintStyle: const TextStyle(color: Colors.white30, fontSize: 11),
                filled: true,
                fillColor: const Color(0xFF101318),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
              ),
            ),
          ] else ...[
            Row(
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.add_photo_alternate_rounded, size: 16),
                  label: const Text('Pick Photos from Device', style: TextStyle(fontSize: 12)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.teal,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: _pickPhotosForWorkload,
                ),
                const SizedBox(width: 12),
                Text(
                  '${_selectedPhotos.length} photos chosen',
                  style: const TextStyle(color: Colors.tealAccent, fontSize: 12),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  // ── Section 4: Progress & Controls Card ────────────────────────────────────

  Widget _buildProgressAndControlsCard(double progressVal, int avgLatency) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Batch Progress: $_currentIndex/$_totalItems (${(progressVal * 100).toInt()}%)',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
              ),
              const Spacer(),
              if (avgLatency > 0)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: Colors.deepOrange.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    '${avgLatency}ms / item',
                    style: const TextStyle(color: Colors.deepOrangeAccent, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progressVal,
              minHeight: 12,
              backgroundColor: const Color(0xFF101318),
              valueColor: const AlwaysStoppedAnimation<Color>(Colors.tealAccent),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _statusText,
            style: const TextStyle(color: Colors.white60, fontSize: 11),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 16),

          // Action Buttons
          Row(
            children: [
              if (!_isProcessing) ...[
                Expanded(
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.play_arrow_rounded),
                    label: const Text('Start Background Embedding'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.teal,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    onPressed: _startBackgroundProcessing,
                  ),
                ),
              ] else ...[
                Expanded(
                  child: ElevatedButton.icon(
                    icon: Icon(_isPaused ? Icons.play_arrow_rounded : Icons.pause_rounded),
                    label: Text(_isPaused ? 'Resume' : 'Pause'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.amber[700],
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    onPressed: _isPaused ? _resumeProcessing : _pauseProcessing,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.stop_rounded),
                    label: const Text('Stop'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.redAccent,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    onPressed: _stopProcessing,
                  ),
                ),
              ],
            ],
          ),

          if (_isProcessing) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.teal.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.tealAccent.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.touch_app_rounded, color: Colors.tealAccent, size: 20),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text(
                      'TIP: Press the Home button or lock your phone now! Notice the Android notification updating in real-time.',
                      style: TextStyle(color: Colors.tealAccent, fontSize: 11, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ── Section 5: Live Execution Log Card ─────────────────────────────────────

  Widget _buildLiveEventLogCard() {
    return Container(
      height: 220,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF101318),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.terminal_rounded, size: 16, color: Colors.cyanAccent),
              const SizedBox(width: 6),
              const Text(
                'Live Execution Log',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
              ),
              const Spacer(),
              Text(
                '${_eventLogs.length} events',
                style: const TextStyle(color: Colors.grey, fontSize: 10),
              ),
            ],
          ),
          const Divider(color: Colors.white12, height: 14),
          Expanded(
            child: ListView.builder(
              controller: _logScrollController,
              itemCount: _eventLogs.length,
              itemBuilder: (context, index) {
                final log = _eventLogs[index];
                final isBg = log.contains('BACKGROUND');
                final isResume = log.contains('RESUMED');
                final isErr = log.contains('Error') || log.contains('failed');

                Color textColor = Colors.white70;
                if (isBg) textColor = Colors.amberAccent;
                if (isResume) textColor = Colors.tealAccent;
                if (isErr) textColor = Colors.redAccent;

                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text(
                    log,
                    style: TextStyle(
                      color: textColor,
                      fontFamily: 'monospace',
                      fontSize: 10.5,
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  // ── Section 6: Semantic Search Verification ────────────────────────────────

  Widget _buildSearchVerificationCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.search_rounded, color: Colors.tealAccent, size: 18),
              const SizedBox(width: 8),
              const Text(
                'Verify Background Embeddings',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
              ),
              const Spacer(),
              Text(
                '${_processedItems.length} items in vector store',
                style: const TextStyle(color: Colors.grey, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Type a test semantic query to retrieve nearest neighbors from the items generated in background:',
            style: TextStyle(color: Colors.white60, fontSize: 11),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _searchController,
            style: const TextStyle(color: Colors.white, fontSize: 13),
            decoration: InputDecoration(
              hintText: 'e.g. quantum physics, plant biology, cross-platform UI…',
              hintStyle: const TextStyle(color: Colors.white30, fontSize: 12),
              prefixIcon: const Icon(Icons.search, color: Colors.tealAccent, size: 20),
              suffixIcon: IconButton(
                icon: const Icon(Icons.arrow_forward_rounded, color: Colors.tealAccent),
                onPressed: () => _searchEmbeddedItems(_searchController.text),
              ),
              filled: true,
              fillColor: const Color(0xFF101318),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            ),
            onSubmitted: _searchEmbeddedItems,
          ),
          if (_searchResults.isNotEmpty) ...[
            const SizedBox(height: 12),
            const Text(
              'Top Semantic Matches:',
              style: TextStyle(color: Colors.white70, fontSize: 11, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 6),
            for (final res in _searchResults)
              Container(
                margin: const EdgeInsets.only(bottom: 6),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFF101318),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.white12),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                      decoration: BoxDecoration(
                        color: res.similarity >= 0.5
                            ? Colors.teal.withValues(alpha: 0.3)
                            : Colors.grey.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        '${(res.similarity * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: res.similarity >= 0.5 ? Colors.tealAccent : Colors.grey,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        res.item.text,
                        style: const TextStyle(color: Colors.white, fontSize: 11.5),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (res.item.wasInBackground)
                      const Tooltip(
                        message: 'Processed while app was in background',
                        child: Icon(Icons.cloud_done_rounded, color: Colors.tealAccent, size: 14),
                      ),
                  ],
                ),
              ),
          ],
        ],
      ),
    );
  }
}
