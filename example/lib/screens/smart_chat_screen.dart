import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

import '../app_state.dart';
import '../utils/model_loader.dart';
import '../widgets/model_status_chip.dart';
import '../widgets/message_bubble.dart';

class SmartChatScreen extends StatefulWidget {
  const SmartChatScreen({super.key});
  @override
  State<SmartChatScreen> createState() => _SmartChatScreenState();
}

class _SmartChatScreenState extends State<SmartChatScreen> {
  GemmaChat? _chat;
  bool _isGenerating = false;
  String _streamingBuffer = '';
  double _llmProgress      = 0;
  double _embedProgress    = 0;

  // Semantic search state
  bool _isSearchMode = false;
  List<ChatMessage> _searchResults = [];
  final _searchCtrl = TextEditingController();
  final _inputCtrl  = TextEditingController();
  final _scrollCtrl = ScrollController();

  // Per-message embeddings for semantic search
  final Map<int, List<double>> _msgEmbeddings = {};

  @override
  void initState() {
    super.initState();
    ModelManager.instance.addListener(_onManagerUpdate);
  }

  @override
  void dispose() {
    ModelManager.instance.removeListener(_onManagerUpdate);
    _chat?.dispose();
    _searchCtrl.dispose();
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _onManagerUpdate() => setState(() {});

  // ── Engine init ────────────────────────────────────────────────────────────

  Future<void> _initEngines() async {
    final mgr = ModelManager.instance;
    try {
      // Load embedding engine first (lighter, faster)
      if (mgr.embeddingStatus == ModelStatus.unloaded ||
          mgr.embeddingStatus == ModelStatus.error) {
        await loadEmbedding(onProgress: (p) => setState(() => _embedProgress = p));
      }

      // Load LLM
      if (mgr.llmStatus == ModelStatus.unloaded ||
          mgr.llmStatus == ModelStatus.error) {
        await loadLlm(onProgress: (p) => setState(() => _llmProgress = p));
      }

      // Build chat session
      _chat = GemmaChat(
        maxContextTokens: mgr.maxTokens,
        systemPrompt:     'You are a highly intelligent assistant with access to semantic search over the conversation history.',
        contextStrategy:  ContextStrategy.slidingWindow,
      );
      await _chat!.init();
      setState(() {});
    } catch (e) {
      _snack('Init failed: $e');
    } finally {
      setState(() { _llmProgress = 0; _embedProgress = 0; });
    }
  }

  Future<void> _unloadAll() async {
    await _chat?.dispose();
    _chat = null;
    _msgEmbeddings.clear();
    await unloadLlm();
    await unloadEmbedding();
    setState(() {});
  }

  // ── Messaging ──────────────────────────────────────────────────────────────

  Future<void> _sendMessage() async {
    final text = _inputCtrl.text.trim();
    if (text.isEmpty) return;

    // Embed user message first (sequential to avoid GPU contention on Android).
    final userIdx = _chat!.history.length;
    try {
      _msgEmbeddings[userIdx] = await EmbeddingPlugin().getEmbedding(text);
    } catch (_) {}

    setState(() { _isGenerating = true; _streamingBuffer = ''; _inputCtrl.clear(); });
    ModelManager.instance.setGenerating(true);
    _scrollToBottom();

    try {
      await for (final chunk in _chat!.sendMessageStream(text: text)) {
        if (!mounted) break;
        setState(() => _streamingBuffer += chunk);
        _scrollToBottom();
      }

      // Embed model response
      final modelIdx = _chat!.history.length - 1;
      try {
        _msgEmbeddings[modelIdx] = await EmbeddingPlugin().getEmbedding(_streamingBuffer);
      } catch (_) {}
    } catch (e) {
      if (!e.toString().contains('Stopped') && mounted) _snack('Error: $e');
    } finally {
      if (mounted) {
        setState(() { _isGenerating = false; _streamingBuffer = ''; });
        ModelManager.instance.setGenerating(false);
      }
    }
  }

  Future<void> _semanticSearch(String query) async {
    if (query.isEmpty) { setState(() => _searchResults.clear()); return; }

    try {
      final qvec = await EmbeddingPlugin().getEmbedding(query);
      final scored = <(double, ChatMessage)>[];

      _msgEmbeddings.forEach((idx, vec) {
        if (idx < (_chat?.history.length ?? 0)) {
          final score = _cosine(qvec, vec);
          // Removed the 0.55 threshold, just add everything
          scored.add((score, _chat!.history[idx]));
        }
      });
      
      // Sort in descending order (highest similarity first)
      scored.sort((a, b) => b.$1.compareTo(a.$1));
      
      // Use .take(3) to get exactly the top 3 results
      setState(() => _searchResults = scored.take(3).map((e) => e.$2).toList());
    } catch (e) {
      debugPrint('Search error: $e');
    }
  }

  double _cosine(List<double> a, List<double> b) {
    double dot = 0, nA = 0, nB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i];
    }
    final d = sqrt(nA) * sqrt(nB);
    return d == 0 ? 0 : dot / d;
  }

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  void _scrollToBottom() => WidgetsBinding.instance.addPostFrameCallback((_) {
    if (_scrollCtrl.hasClients) {
      _scrollCtrl.animateTo(_scrollCtrl.position.maxScrollExtent, duration: const Duration(milliseconds: 100), curve: Curves.easeOut);
    }
  });

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final mgr = ModelManager.instance;
    final bothReady = mgr.llmStatus    == ModelStatus.ready &&
                      mgr.embeddingStatus == ModelStatus.ready;
    final isLoading = mgr.llmStatus.isBusy || mgr.embeddingStatus.isBusy;

    final displayList = _isSearchMode
        ? _searchResults
        : [
            ...(_chat?.history ?? []),
            if (_isGenerating)
              ChatMessage(role: 'model', text: _streamingBuffer.isEmpty ? '…' : _streamingBuffer),
          ];

    return Scaffold(
      appBar: AppBar(
        title: _isSearchMode
            ? TextField(
                controller: _searchCtrl,
                autofocus: true,
                decoration: const InputDecoration(hintText: 'Semantic search…', border: InputBorder.none),
                onChanged: _semanticSearch,
              )
            : Row(children: [
                const Text('Smart Chat'),
                const SizedBox(width: 8),
                ModelStatusChip(mgr.llmStatus),
                const SizedBox(width: 4),
                ModelStatusChip(mgr.embeddingStatus),
              ]),
        actions: [
          if (_chat != null)
            IconButton(
              icon: Icon(_isSearchMode ? Icons.close : Icons.search),
              onPressed: () => setState(() {
                _isSearchMode = !_isSearchMode;
                if (!_isSearchMode) { _searchCtrl.clear(); _searchResults.clear(); }
              }),
            ),
          if (bothReady || mgr.llmStatus != ModelStatus.unloaded)
            IconButton(
              icon: const Icon(Icons.power_settings_new),
              tooltip: 'Unload all',
              onPressed: isLoading ? null : _unloadAll,
            ),
        ],
      ),
      body: !bothReady && !isLoading && _chat == null
          ? _buildInitView(mgr)
          : isLoading && _chat == null
              ? _buildLoadingView()
              : Column(
                  children: [
                    Expanded(
                      child: ListView.builder(
                        controller: _scrollCtrl,
                        padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
                        itemCount: displayList.length,
                        itemBuilder: (_, i) => MessageBubble(
                          displayList[i],
                          highlight: _isSearchMode,
                        ),
                      ),
                    ),
                    if (!_isSearchMode) _buildInputRow(),
                  ],
                ),
    );
  }

  Widget _buildInitView(ModelManager mgr) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.auto_awesome, size: 48),
            const SizedBox(height: 12),
            const Text('Smart Chat', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text('Loads both the LLM and the embedding model.', style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant), textAlign: TextAlign.center),
            const SizedBox(height: 20),
            if (mgr.llmStatus == ModelStatus.error || mgr.embeddingStatus == ModelStatus.error)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(
                  mgr.llmError ?? mgr.embeddingError ?? '',
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                  textAlign: TextAlign.center,
                ),
              ),
            FilledButton.icon(
              onPressed: _initEngines,
              icon: const Icon(Icons.bolt),
              label: const Text('Initialize Dual Engines'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingView() {
    final mgr = ModelManager.instance;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            // Embedding progress
            Row(children: [
              const SizedBox(width: 80, child: Text('Embedding', style: TextStyle(fontSize: 12))),
              Expanded(child: LinearProgressIndicator(value: _embedProgress > 0 ? _embedProgress / 100 : null)),
              const SizedBox(width: 8),
              ModelStatusChip(mgr.embeddingStatus),
            ]),
            const SizedBox(height: 10),
            // LLM progress
            Row(children: [
              const SizedBox(width: 80, child: Text('LLM', style: TextStyle(fontSize: 12))),
              Expanded(child: LinearProgressIndicator(value: _llmProgress > 0 ? _llmProgress / 100 : null)),
              const SizedBox(width: 8),
              ModelStatusChip(mgr.llmStatus),
            ]),
          ],
        ),
      ),
    );
  }

  Widget _buildInputRow() {
    return Container(
      color: Theme.of(context).colorScheme.surface,
      padding: const EdgeInsets.fromLTRB(8, 6, 8, 0),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _inputCtrl,
                maxLines: 4, minLines: 1,
                decoration: InputDecoration(
                  hintText: 'Message…',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                ),
                onSubmitted: (_) => _isGenerating ? null : _sendMessage(),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filled(
              onPressed: _isGenerating ? () => _chat?.stop() : _sendMessage,
              icon: Icon(_isGenerating ? Icons.stop : Icons.send),
            ),
          ],
        ),
      ),
    );
  }
}