import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import '../app_state.dart';
import '../utils/model_loader.dart';

class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key});

  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen> {
  double _loadProgress = 0;
  String? _loadingModelId;
  final Map<String, bool> _readiness = {};

  @override
  void initState() {
    super.initState();
    ModelManager.instance.addListener(_onManagerUpdate);
    _refreshCacheStatus();
  }

  Future<void> _refreshCacheStatus() async {
    for (final m in GemmaModel.values) {
      final isReady = await PluginModelLoader.isModelReady(m);
      if (mounted) setState(() => _readiness[m.name] = isReady);
    }
  }

  @override
  void dispose() {
    ModelManager.instance.removeListener(_onManagerUpdate);
    super.dispose();
  }

  void _onManagerUpdate() => setState(() {});

  Future<void> _loadModel(GemmaModel model) async {
    setState(() => _loadingModelId = model.name);
    try {
      if (model.isEmbedding) {
        await loadEmbeddingGemma2(
          local: false,
          onProgress: (p) => setState(() => _loadProgress = p),
        );
      } else {
        await loadLlm(
          local: false, // force download if not cached
          model: model,
          enableMtp: model.supportsMtp,
          onProgress: (p) => setState(() => _loadProgress = p),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed: $e')));
      }
    } finally {
      if (mounted) setState(() { _loadProgress = 0; _loadingModelId = null; });
      _refreshCacheStatus();
    }
  }

  Future<void> _unloadModel() async {
    if (ModelManager.instance.llmStatus != ModelStatus.unloaded) {
      await unloadLlm();
    }
    if (ModelManager.instance.embeddingStatus != ModelStatus.unloaded) {
      await unloadEmbedding();
    }
    setState(() {});
  }

  Future<void> _deleteModel(GemmaModel model) async {
    await PluginModelLoader.deleteModel(model);
    if (model.isEmbedding && ModelManager.instance.embeddingStatus == ModelStatus.ready) {
      await unloadEmbedding();
    } else if (!model.isEmbedding && ModelManager.instance.llmStatus == ModelStatus.ready) {
      await unloadLlm();
    }
    _refreshCacheStatus();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Model file deleted.')));
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final mgr = ModelManager.instance;
    return Scaffold(
      appBar: AppBar(title: const Text('Models Manager')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: GemmaModel.values.map((m) {
          final isLlmActive = !m.isEmbedding &&
              mgr.llmStatus == ModelStatus.ready &&
              mgr.llmModelPath != null &&
              (mgr.llmModelPath!.contains(m.androidFile) ||
                  mgr.llmModelPath!.contains(m.webFile) ||
                  mgr.llmModelPath!.contains(m.webRepo) ||
                  mgr.llmModelPath!.contains(m.name));
          final isEmbeddingActive = m.isEmbedding &&
              mgr.embeddingStatus == ModelStatus.ready &&
              (mgr.embeddingModelPath == null ||
                  mgr.embeddingModelPath!.contains(m.androidFile) ||
                  mgr.embeddingModelPath!.contains(m.webFile) ||
                  mgr.embeddingModelPath!.contains(m.webRepo));
          final isActive = isLlmActive || isEmbeddingActive;
          final isLoadingThis = _loadingModelId == m.name;
          final isBusy = m.isEmbedding ? mgr.embeddingStatus.isBusy : mgr.llmStatus.isBusy;

          return Card(
            elevation: isActive ? 4 : 2,
            shape: isActive
                ? RoundedRectangleBorder(
                    side: const BorderSide(color: Colors.blue, width: 2),
                    borderRadius: BorderRadius.circular(12),
                  )
                : null,
            margin: const EdgeInsets.only(bottom: 12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(m.displayName, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      ),
                      if (_readiness[m.name] == true && !isActive)
                        const Padding(
                          padding: EdgeInsets.only(right: 8.0),
                          child: Chip(
                            label: Text('Downloaded', style: TextStyle(fontSize: 10, color: Colors.white)),
                            visualDensity: VisualDensity.compact,
                            backgroundColor: Colors.green,
                          ),
                        ),
                      if (isActive)
                        const Padding(
                          padding: EdgeInsets.only(right: 8.0),
                          child: Chip(
                            label: Text('Loaded', style: TextStyle(fontSize: 10, color: Colors.white)),
                            visualDensity: VisualDensity.compact,
                            backgroundColor: Colors.blue,
                          ),
                        ),
                      if (m.supportsMtp)
                        const Chip(label: Text('MTP', style: TextStyle(fontSize: 10)), visualDensity: VisualDensity.compact),
                      if (m.isOnnx)
                        const Padding(
                          padding: EdgeInsets.only(left: 4.0),
                          child: Chip(label: Text('ONNX', style: TextStyle(fontSize: 10)), visualDensity: VisualDensity.compact),
                        ),
                      if (m.supportsVision && (m.isOnnx || !kIsWeb))
                        const Padding(
                          padding: EdgeInsets.only(left: 4.0),
                          child: Chip(label: Text('Vision', style: TextStyle(fontSize: 10)), visualDensity: VisualDensity.compact),
                        ),
                      if (m.supportsAudio && (m.isOnnx || !kIsWeb))
                        const Padding(
                          padding: EdgeInsets.only(left: 4.0),
                          child: Chip(label: Text('Audio', style: TextStyle(fontSize: 10)), visualDensity: VisualDensity.compact),
                        ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(m.sizeDescription, style: const TextStyle(color: Colors.grey)),
                  const SizedBox(height: 8),
                  if (isLoadingThis && _loadProgress > 0) ...[
                    LinearProgressIndicator(value: _loadProgress / 100),
                    const SizedBox(height: 8),
                    Text('Downloading: ${_loadProgress.toStringAsFixed(1)}%'),
                  ] else
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        if (_readiness[m.name] == true) ...[
                          TextButton.icon(
                            onPressed: () => _deleteModel(m),
                            icon: const Icon(Icons.delete, color: Colors.red),
                            label: const Text('Delete', style: TextStyle(color: Colors.red)),
                          ),
                          const SizedBox(width: 8),
                          FilledButton.icon(
                            onPressed: isLoadingThis || isBusy ? null : () => _loadModel(m),
                            icon: const Icon(Icons.play_arrow),
                            label: const Text('Load'),
                          ),
                        ] else ...[
                          FilledButton.icon(
                            onPressed: isLoadingThis || isBusy ? null : () => _loadModel(m),
                            icon: const Icon(Icons.download),
                            label: const Text('Download & Load'),
                          ),
                        ]
                      ],
                    ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: (mgr.llmStatus.isBusy ||
                mgr.embeddingStatus.isBusy ||
                (mgr.llmStatus == ModelStatus.unloaded && mgr.embeddingStatus == ModelStatus.unloaded))
            ? null
            : _unloadModel,
        label: const Text('Unload Active Model'),
        icon: const Icon(Icons.eject),
      ),
    );
  }
}
