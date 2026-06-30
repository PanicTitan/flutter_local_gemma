import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

import '../app_state.dart';
import '../demo_skills.dart';
import '../utils/model_loader.dart';
import '../widgets/model_status_chip.dart';
import '../widgets/token_counter_bar.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});
  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // ── Chat state ─────────────────────────────────────────────────────────────
  /// The active chat session. Typed as [BaseChat] so [GemmaChat] and
  /// [AgentChat] can be used interchangeably without casting.
  BaseChat? _chat;
  bool _isGenerating = false;
  String _streamingBuffer = '';
  AgentTurn? _currentAgentTurn;
  String? _operationLabel; // shown in the status banner

  // ── Input ──────────────────────────────────────────────────────────────────
  final _inputCtrl     = TextEditingController();
  final _scrollCtrl    = ScrollController();
  final _pendingImages = <Uint8List>[];
  final _pendingAudios = <Uint8List>[];

  // PDF parts (text + images) keyed by filename
  final _pendingPdfs = <Map<String, dynamic>>[];

  // ── Settings ───────────────────────────────────────────────────────────────
  bool   _useStream  = true;
  bool   _jsonMode   = false;
  bool   _enableThinking = false;
  bool   _enableSkills = false;
  bool   _nativeToolCalling = true;
  int    _maxToolLoops = 10;
  bool   _enableMtp = true;
  bool   _enableMultimodality = true;
  String _customSchema = '';
  double _loadProgress = 0;

  // ── Token tracking ─────────────────────────────────────────────────────────
  int _inputPendingTokens = 0;

  // ── Skills Tracking ────────────────────────────────────────────────────────
  List<GemmaSkill>? _allSkills;
  final Map<String, bool> _skillToggles = {};

  // ── Schemas ────────────────────────────────────────────────────────────────
  static final _defaultSchema = Schema.object({
    'name':    Schema.string().description('Full name'),
    'age':     Schema.number(),
    'country': Schema.string().description('Country of origin'),
  });

  @override
  void initState() {
    super.initState();
    ModelManager.instance.addListener(_onManagerUpdate);
    _inputCtrl.addListener(_updatePendingTokens);
    _loadAllSkills();
  }

  Future<void> _loadAllSkills() async {
    _allSkills = await buildDemoSkills();
    for (final skill in _allSkills!) {
      _skillToggles[skill.name] = true;
    }
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    ModelManager.instance.removeListener(_onManagerUpdate);
    _chat?.dispose();
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _onManagerUpdate() {
    if (ModelManager.instance.llmStatus == ModelStatus.ready && _chat == null && !_isGenerating && _operationLabel == null) {
      _rebuildChatSafe();
    } else {
      setState(() {});
    }
  }

  Future<void> _updatePendingTokens() async {
    if (_chat != null) {
      final est = await _chat!.estimateNextTurnTokens(
        _inputCtrl.text,
        images: _pendingImages.isEmpty ? null : _pendingImages,
        audios: _pendingAudios.isEmpty ? null : _pendingAudios,
      );
      if (mounted) setState(() => _inputPendingTokens = est - _chat!.currentTokenCount);
    } else {
      final text = ((_inputCtrl.text.length / 4).ceil());
      final imgs = _pendingImages.length * 257;
      if (mounted) setState(() => _inputPendingTokens = text + imgs);
    }
  }

  // ── Model loading ──────────────────────────────────────────────────────────

  Future<void> _loadModel({required bool local}) async {
    try {
      await loadLlm(
        local: local,
        enableMtp: _enableMtp,
        onProgress: (p) => setState(() => _loadProgress = p),
      );
      await _rebuildChat();
    } catch (e) {
      _snack('Failed: ${e.toString().replaceAll('Exception: ', '')}');
    } finally {
      setState(() => _loadProgress = 0);
    }
  }

  Future<void> _unloadModel() async {
    if (_isGenerating) {
      _chat?.stop();
      await Future.delayed(const Duration(milliseconds: 500));
    }
    await _chat?.dispose();
    _chat = null;
    await unloadLlm();
    setState(() {});
  }

  Future<void> _rebuildChatSafe() async {
    if (_isGenerating) {
      _chat?.stop();
      await Future.delayed(const Duration(milliseconds: 500));
    }
    await _rebuildChat();
  }

  void _handleSkillResult(AgentToolCall call) {
    try {
      GemmaDebug.log(GemmaDebug.tagAgent, 'ChatScreen _handleSkillResult called for ${call.skillName}');
      final decoded = jsonDecode(call.resultJson);
      if (decoded is Map && decoded['image'] is Map && decoded['image']['base64'] != null) {
        final b64 = decoded['image']['base64'] as String;
        final cleanB64 = b64.contains(',') ? b64.split(',').last : b64;
        final bytes = base64Decode(cleanB64);
        
        GemmaDebug.log(GemmaDebug.tagAgent, 'ChatScreen _handleSkillResult decoded image bytes: ${bytes.length}');
        
        setState(() {
          _chat?.appendMessage(ChatMessage(
            role: 'model',
            text: '', // Empty text, just the image
            images: [bytes],
            isUiOnly: true, // Don't pollute the LLM context
          ));
        });
        GemmaDebug.log(GemmaDebug.tagAgent, 'ChatScreen _handleSkillResult appended image message to history');
      } else {
        GemmaDebug.log(GemmaDebug.tagAgent, 'ChatScreen _handleSkillResult JSON did not contain image/base64');
      }
    } catch (e) {
      GemmaDebug.logError(GemmaDebug.tagAgent, 'ChatScreen _handleSkillResult error', e);
    }
  }

  Future<void> _rebuildChat() async {
    setState(() => _operationLabel = 'Applying settings…');
    await _chat?.dispose();
    final mgr = ModelManager.instance;
    
    if (_enableSkills) {
      final skills = _allSkills?.where((s) => _skillToggles[s.name] ?? true).toList() ?? demoSkills;
      _chat = AgentChat(
        skills: skills,
        maxLoops: _maxToolLoops,
        onSkillResult: _handleSkillResult,
        config: SessionConfig(
          enableThinking: _enableThinking,
          nativeToolCalling: _nativeToolCalling,
          temperature: mgr.temperature,
          topK: mgr.topK,
          topP: mgr.topP,
        ),
      );
    } else {
      _chat = GemmaChat(
        maxContextTokens: mgr.maxTokens,
        systemPrompt:     mgr.systemPrompt,
        contextStrategy:  ContextStrategy.slidingWindow,
        config: SessionConfig(
          enableThinking: _enableThinking,
          enableMultimodality: _enableMultimodality,
          temperature: mgr.temperature,
          topK: mgr.topK,
          topP: mgr.topP,
        ),
      );
    }
    
    await _chat!.init();
    _pendingImages.clear();
    _pendingAudios.clear();
    _pendingPdfs.clear();
    setState(() => _operationLabel = null);
    _updatePendingTokens();
  }

  // ── Messaging ──────────────────────────────────────────────────────────────

  Future<void> _sendMessage() async {
    final text = _inputCtrl.text.trim();
    if (text.isEmpty && _pendingImages.isEmpty && _pendingAudios.isEmpty && _pendingPdfs.isEmpty) return;
    // Flatten pending PDFs into text + image arrays
    final allImages = List<Uint8List>.from(_pendingImages);
    final allAudios = List<Uint8List>.from(_pendingAudios);

    String payloadText = text;
    for (final pdf in _pendingPdfs) {
      payloadText += '\n\n[ ${pdf['name']}]\n';
      for (final p in pdf['parts'] as List<ContentPart>) {
        if (p is TextPart)  payloadText += '${p.text}\n';
        if (p is ImagePart) allImages.add(p.bytes);
      }
      payloadText += '[END DOC]\n';
    }

    setState(() {
      _isGenerating = true;
      _streamingBuffer = '';
      _currentAgentTurn = null;
      _inputCtrl.clear();
      _pendingImages.clear();
      _pendingAudios.clear();
      _pendingPdfs.clear();
      _inputPendingTokens = 0;
    });
    ModelManager.instance.setGenerating(true);
    _scrollToBottom();

    try {
      if (_jsonMode) {
        Schema? schema;
        String? raw;
        if (_customSchema.trim().isNotEmpty) {
          raw = _customSchema.trim();
        } else {
          schema = _defaultSchema;
        }
        final stream = (_chat as GemmaChat).sendMessageJsonStream(
          text: payloadText, schema: schema, rawSchemaStr: raw,
          images: allImages.isEmpty ? null : allImages,
          audios: allAudios.isEmpty ? null : allAudios,
        );
        await for (final chunk in stream) {
          if (!mounted) break;
          final pretty = chunk;
          setState(() => _streamingBuffer = pretty);
          _scrollToBottom();
        }
      } else if (_chat is AgentChat) {
        String lastRawText = '';
        String lineBuffer = '';
        final stream = (_chat as AgentChat).run(payloadText);
        await for (final turn in stream) {
          if (!mounted) break;
          if (turn.rawText.length > lastRawText.length) {
            final newText = turn.rawText.substring(lastRawText.length);
            lastRawText = turn.rawText;
            lineBuffer += newText;
            while (lineBuffer.contains('\n')) {
              final nlIndex = lineBuffer.indexOf('\n');
              final line = lineBuffer.substring(0, nlIndex);
              GemmaDebug.log(GemmaDebug.tagAgent, '[AgentRawOutput] $line');
              lineBuffer = lineBuffer.substring(nlIndex + 1);
            }
          }
          setState(() => _currentAgentTurn = turn);
          _scrollToBottom();
        }
        if (lineBuffer.isNotEmpty) {
          GemmaDebug.log(GemmaDebug.tagAgent, '[AgentRawOutput] $lineBuffer');
        }
        if (_currentAgentTurn != null) {
          GemmaDebug.log(GemmaDebug.tagAgent, '[AgentRawOutput Full]\n${_currentAgentTurn!.rawText}');
        }
      } else {
        final stream = (_chat as GemmaChat).sendMessageStream(
          text: payloadText,
          images: allImages.isEmpty ? null : allImages,
          audios: allAudios.isEmpty ? null : allAudios,
        );
        await for (final chunk in stream) {
          if (!mounted) break;
          setState(() => _streamingBuffer += chunk);
          _scrollToBottom();
        }
      }
    } catch (e) {
      if (!e.toString().contains('Stopped') && mounted) _snack('Error: $e');
    } finally {
      if (mounted) {
        setState(() { _isGenerating = false; _streamingBuffer = ''; _currentAgentTurn = null; });
        ModelManager.instance.setGenerating(false);
      }
    }
  }

  // ── Attachments ────────────────────────────────────────────────────────────

  Future<void> _pickImage() async {
    final files = await ImagePicker().pickMultiImage();
    for (final f in files) _pendingImages.add(await f.readAsBytes());
    setState(() {});
    _updatePendingTokens();
  }

  Future<void> _pickAudio() async {
    final res = await FilePicker.platform.pickFiles(type: FileType.audio, withData: true);
    if (res != null) {
      _pendingAudios.add(res.files.first.bytes!);
      setState(() {});
    }
  }

  Future<void> _pickPdf() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom, allowedExtensions: ['pdf'], withData: true,
    );
    if (res == null) return;

    final mode = await showDialog<PdfExtractionMode>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Extract PDF'),
        content: const Text('How do you want to extract this PDF?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          TextButton(onPressed: () => Navigator.pop(context, PdfExtractionMode.textOnly), child: const Text('Text Only')),
          TextButton(onPressed: () => Navigator.pop(context, PdfExtractionMode.imagesOnly), child: const Text('Images Only')),
          FilledButton(onPressed: () => Navigator.pop(context, PdfExtractionMode.textAndImages), child: const Text('Text + Images')),
        ],
      ),
    );
    
    if (mode == null) return;

    setState(() => _operationLabel = 'Extracting PDF…');
    try {
      final parts = await PdfProcessor.extract(
        res.files.first.bytes!,
        PdfExtractionConfig(mode: mode),
      );
      _pendingPdfs.add({'name': res.files.first.name, 'parts': parts});
    } catch (e) {
      _snack('PDF error: $e');
    } finally {
      setState(() => _operationLabel = null);
      _updatePendingTokens();
    }
  }

  // ── History ops ────────────────────────────────────────────────────────────

  Future<void> _editMessage(int index, ChatMessage msg) async {
    final ctrl = TextEditingController(text: msg.text);
    final newText = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Edit Message'),
        content: TextField(controller: ctrl, maxLines: 5, decoration: const InputDecoration(border: OutlineInputBorder())),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, ctrl.text), child: const Text('Save')),
        ],
      ),
    );
    if (newText != null && newText != msg.text) {
      setState(() => _operationLabel = 'Rebuilding context…');
      await _chat?.editHistory(index, ChatMessage(role: msg.role, text: newText, images: msg.images, audios: msg.audios));
      setState(() => _operationLabel = null);
    }
  }

  Future<void> _exportHistory() async {
    if (_chat == null) return;
    final json = await _chat!.exportHistory();
    _snack('History exported (${json.length} bytes) – copy from console.');
    debugPrint('--- HISTORY EXPORT ---\n$json\n--- END ---');
  }

  Future<void> _importHistory() async {
    final ctrl = TextEditingController();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Import History JSON'),
        content: TextField(controller: ctrl, maxLines: 6, decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'Paste exported history JSON…')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Import')),
        ],
      ),
    );
    if (confirmed == true && ctrl.text.trim().isNotEmpty) {
      try {
        setState(() => _operationLabel = 'Importing history…');
        await _chat?.importHistory(ctrl.text.trim());
        setState(() {});
      } catch (e) {
        _snack('Import failed: $e');
      } finally {
        setState(() => _operationLabel = null);
      }
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  void _scrollToBottom() => WidgetsBinding.instance.addPostFrameCallback((_) {
    if (_scrollCtrl.hasClients) {
      _scrollCtrl.animateTo(_scrollCtrl.position.maxScrollExtent, duration: const Duration(milliseconds: 100), curve: Curves.easeOut);
    }
  });

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final mgr    = ModelManager.instance;
    final status = mgr.llmStatus;
    final cs     = Theme.of(context).colorScheme;

    // Build the display list including the in-progress streaming bubble.
    final historyList = _chat?.history ?? <ChatMessage>[];
    final displayList = List<dynamic>.from(historyList.where((m) => !m.isHidden));
    if (_isGenerating) {
      if (_currentAgentTurn != null) {
        displayList.add(_currentAgentTurn);
      } else {
        displayList.add(ChatMessage(role: 'model', text: _streamingBuffer.isEmpty ? '…' : _streamingBuffer));
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          const Text('Gemma Chat'),
          const SizedBox(width: 8),
          ModelStatusChip(status, errorText: mgr.llmError),
        ]),
        actions: [
          if (status == ModelStatus.ready || status == ModelStatus.generating)
            IconButton(icon: const Icon(Icons.delete_sweep_outlined), tooltip: 'Clear context', onPressed: _isGenerating ? null : _rebuildChatSafe),
          Builder(builder: (c) => IconButton(icon: const Icon(Icons.settings_outlined), onPressed: () => Scaffold.of(c).openEndDrawer())),
        ],
      ),
      endDrawer: _buildSettingsDrawer(),
      body: Column(
        children: [
          // Download progress bar
          if (_loadProgress > 0)
            LinearProgressIndicator(value: _loadProgress / 100, minHeight: 3),

          // Operation banner (extracting PDF, rebuilding context, etc.)
          if (_operationLabel != null)
            Container(
              width: double.infinity,
              color: cs.tertiaryContainer,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(children: [
                SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: cs.onTertiaryContainer)),
                const SizedBox(width: 8),
                Text(_operationLabel!, style: TextStyle(fontSize: 12, color: cs.onTertiaryContainer)),
              ]),
            ),

          // Token usage bar (only when a chat is active)
          if (_chat != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: TokenCounterBar(
                usedTokens:    _chat!.currentTokenCount,
                maxTokens:     mgr.maxTokens,
                pendingTokens: _inputPendingTokens,
              ),
            ),

          // Main area: idle/loading state OR message list
          Expanded(
            child: status == ModelStatus.unloaded || status == ModelStatus.error
                ? _buildLoadView(mgr)
                : status.isBusy && _chat == null
                    ? _buildLoadingView(status)
                    : ListView.builder(
                        controller: _scrollCtrl,
                        padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
                        itemCount: displayList.length,
                        itemBuilder: (_, i) {
                          final item = displayList[i];
                          if (item is AgentTurn) return AgentTurnBubble(item);

                          final msg = item as ChatMessage;
                          
                          // Check if this is an injected AgentTurn JSON message
                          if (msg.role == 'model' && msg.isUiOnly) {
                            try {
                              final data = jsonDecode(msg.text);
                              if (data is Map<String, dynamic> && data['is_agent_turn'] == true) {
                                return AgentTurnBubble(AgentTurn.fromJson(data));
                              }
                            } catch (_) {
                              // Fall back to normal message bubble if parsing fails
                            }
                          }

                          final historyIndex = _chat?.history.indexOf(msg) ?? -1;
                          final canEditOrDelete = !_isGenerating && historyIndex >= 0;

                          return MessageBubble(
                            msg,
                            onEdit:   canEditOrDelete ? () => _editMessage(historyIndex, msg) : null,
                            onDelete: canEditOrDelete ? () => _chat!.removeHistory(historyIndex).then((_) => setState(() {})) : null,
                          );
                        },
                      ),
          ),

          // Input area (only when chat is ready)
          if (_chat != null) _buildInputArea(cs),
        ],
      ),
    );
  }

  Widget _buildLoadView(ModelManager mgr) {
    final hasError = mgr.llmStatus == ModelStatus.error;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (hasError)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Text(
                  mgr.llmError ?? 'Unknown error',
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                  textAlign: TextAlign.center,
                ),
              ),
            FilledButton.icon(
              onPressed: () => _loadModel(local: false),
              icon: const Icon(Icons.download_outlined),
              label: const Text('Download Default Model'),
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
          Text(status.label, style: const TextStyle(fontSize: 16)),
          if (_loadProgress > 0) ...[
            const SizedBox(height: 8),
            Text('${_loadProgress.toStringAsFixed(0)}%'),
          ],
        ],
      ),
    );
  }

  Widget _buildInputArea(ColorScheme cs) {
    return Container(
      color: cs.surface,
      padding: const EdgeInsets.fromLTRB(8, 4, 8, 0),
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Pending attachment chips
            if (_pendingImages.isNotEmpty || _pendingAudios.isNotEmpty || _pendingPdfs.isNotEmpty)
              SizedBox(
                height: 44,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    ..._pendingPdfs.map((pdf) => Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: Chip(
                            label: Text(pdf['name'] as String),
                            avatar: const Icon(Icons.picture_as_pdf, size: 16),
                            onDeleted: () => setState(() { _pendingPdfs.remove(pdf); _updatePendingTokens(); }),
                            visualDensity: VisualDensity.compact,
                          ),
                        )),
                    ..._pendingImages.asMap().entries.map((e) => Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: Chip(
                            label: Text('Image ${e.key + 1}'),
                            avatar: const Icon(Icons.image_outlined, size: 16),
                            onDeleted: () => setState(() { _pendingImages.removeAt(e.key); _updatePendingTokens(); }),
                            visualDensity: VisualDensity.compact,
                          ),
                        )),
                    ..._pendingAudios.asMap().entries.map((e) => Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: Chip(
                            label: Text('Audio ${e.key + 1}'),
                            avatar: const Icon(Icons.audiotrack_outlined, size: 16),
                            onDeleted: () => setState(() { _pendingAudios.removeAt(e.key); }),
                            visualDensity: VisualDensity.compact,
                          ),
                        )),
                  ],
                ),
              ),

            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                // Attachment buttons
                PopupMenuButton<String>(
                  icon: const Icon(Icons.attach_file),
                  tooltip: 'Attach',
                  onSelected: (v) {
                    if (v == 'image') _pickImage();
                    if (v == 'audio') _pickAudio();
                    if (v == 'pdf')   _pickPdf();
                  },
                  itemBuilder: (_) => const [
                    PopupMenuItem(value: 'image', child: ListTile(leading: Icon(Icons.image_outlined),       title: Text('Image'),    dense: true)),
                    PopupMenuItem(value: 'audio', child: ListTile(leading: Icon(Icons.audiotrack_outlined),  title: Text('Audio'),    dense: true)),
                    PopupMenuItem(value: 'pdf',   child: ListTile(leading: Icon(Icons.picture_as_pdf),       title: Text('PDF'),      dense: true)),
                  ],
                ),

                // Text input
                Expanded(
                  child: TextField(
                    controller: _inputCtrl,
                    maxLines: 5, minLines: 1,
                    decoration: InputDecoration(
                      hintText: _jsonMode ? 'Ask for structured data…' : 'Message…',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    ),
                    onSubmitted: (_) => _isGenerating ? null : _sendMessage(),
                  ),
                ),
                const SizedBox(width: 6),

                // Send / Stop
                IconButton.filled(
                  onPressed: _isGenerating ? () { _chat?.stop(); } : _sendMessage,
                  icon: Icon(_isGenerating ? Icons.stop : Icons.send),
                ),
              ],
            ),
            const SizedBox(height: 4),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingsDrawer() {
    final mgr = ModelManager.instance;

    return Drawer(
      child: ListenableBuilder(
        listenable: mgr,
        builder: (_, __) => ListView(
          padding: const EdgeInsets.fromLTRB(16, 48, 16, 16),
          children: [
            // Header
            Row(children: [
              const Expanded(child: Text('Settings', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold))),
              ModelStatusChip(mgr.llmStatus),
            ]),
            const Divider(height: 24),

            // ── Model Load / Unload ────────────────────────────────────────
            const _SectionHeader('Model'),
            Row(children: [
              Expanded(child: FilledButton.icon(
                onPressed: mgr.llmStatus == ModelStatus.unloaded ? () { Navigator.pop(context); _loadModel(local: false); } : null,
                icon: const Icon(Icons.download_outlined, size: 18),
                label: const Text('Download'),
              )),
              const SizedBox(width: 8),
              Expanded(child: OutlinedButton.icon(
                onPressed: mgr.llmStatus == ModelStatus.unloaded ? () { Navigator.pop(context); _loadModel(local: true); } : null,
                icon: const Icon(Icons.folder_open_outlined, size: 18),
                label: const Text('Local File'),
              )),
            ]),
            if (mgr.llmStatus != ModelStatus.unloaded)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: OutlinedButton.icon(
                  onPressed: mgr.llmStatus.isBusy ? null : () { Navigator.pop(context); _unloadModel(); },
                  icon: const Icon(Icons.power_settings_new, size: 18),
                  label: const Text('Unload Model'),
                  style: OutlinedButton.styleFrom(foregroundColor: Theme.of(context).colorScheme.error),
                ),
              ),
            const Divider(height: 24),

            // ── Chat Modes ─────────────────────────────────────────────────
            const _SectionHeader('Chat'),
            SwitchListTile(dense: true, title: const Text('Stream responses'), value: _useStream, onChanged: (v) => setState(() => _useStream = v)),
            SwitchListTile(
              dense: true,
              title: const Text('JSON Schema mode'),
              subtitle: const Text('Output structured data'),
              value: _jsonMode,
              onChanged: (v) => setState(() => _jsonMode = v),
            ),
            if (_jsonMode) ...[
              const SizedBox(height: 8),
              const Text('Custom Schema (JSON Schema string)', style: TextStyle(fontSize: 12)),
              const SizedBox(height: 4),
              TextField(
                maxLines: 5, minLines: 3,
                decoration: const InputDecoration(
                  hintText: '{"type":"object","properties":{"name":{"type":"string"}}}',
                  border: OutlineInputBorder(),
                  isDense: true,
                ),
                style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
                onChanged: (v) => _customSchema = v,
                controller: TextEditingController(text: _customSchema),
              ),
              const SizedBox(height: 4),
              Text('Leave empty to use the built-in demo schema.', style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant)),
            ],
            const Divider(height: 24),

            // ── Session Config ─────────────────────────────────────────────
            const _SectionHeader('Session (apply without reload)'),
            SwitchListTile(dense: true, title: const Text('Enable Thinking Mode'), value: _enableThinking, onChanged: (v) { setState(() => _enableThinking = v); mgr.updateSessionConfig(enableThinking: v); }),
            SwitchListTile(dense: true, title: const Text('Enable Skills (Tools)'), subtitle: const Text('Disables multimodality natively'), value: _enableSkills, onChanged: (v) { setState(() => _enableSkills = v); mgr.updateSessionConfig(skills: v ? ['time_skill'] : []); }),
            if (_enableSkills && _allSkills != null) ...[
              const SizedBox(height: 8),
              const Padding(
                padding: EdgeInsets.only(left: 16, bottom: 4),
                child: Text('Active Skills:', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
              ),
              ..._allSkills!.map((skill) => SwitchListTile(
                dense: true,
                title: Text(skill.name, style: const TextStyle(fontSize: 13)),
                subtitle: Text(skill.description, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 11)),
                value: _skillToggles[skill.name] ?? true,
                onChanged: (v) {
                  setState(() => _skillToggles[skill.name] = v);
                },
              )),
              const SizedBox(height: 8),
            ],
            SwitchListTile(dense: true, title: const Text('Native Tool Calling'), subtitle: const Text('If on Android, use native bridge'), value: _nativeToolCalling, onChanged: (v) { setState(() => _nativeToolCalling = v); }),
            if (_enableSkills) _SliderTile(label: 'Max Tool Loops', value: _maxToolLoops.toDouble(), min: 1, max: 20, divisions: 19, onChanged: (v) => setState(() => _maxToolLoops = v.toInt())),
            SwitchListTile(dense: true, title: const Text('Enable Multimodality'), subtitle: const Text('Allow Image/Audio processing'), value: _enableMultimodality, onChanged: (v) { setState(() => _enableMultimodality = v); mgr.updateSessionConfig(enableMultimodality: v); }),
            _SliderTile(label: 'Temperature', value: mgr.temperature, min: 0, max: 2, divisions: 40, onChanged: (v) => mgr.updateSessionConfig(temperature: v)),
            _SliderTile(label: 'Top-P', value: mgr.topP, min: 0, max: 1, divisions: 20, onChanged: (v) => mgr.updateSessionConfig(topP: v)),
            _SliderTile(label: 'Top-K', value: mgr.topK.toDouble(), min: 1, max: 100, divisions: 99, onChanged: (v) => mgr.updateSessionConfig(topK: v.toInt())),
            const SizedBox(height: 8),
            const Text('System Prompt', style: TextStyle(fontSize: 12)),
            const SizedBox(height: 4),
            TextField(
              maxLines: 3, minLines: 2,
              decoration: const InputDecoration(border: OutlineInputBorder(), isDense: true),
              controller: TextEditingController(text: mgr.systemPrompt),
              onChanged: (v) => mgr.updateSessionConfig(systemPrompt: v),
            ),
            const SizedBox(height: 8),
            FilledButton.tonalIcon(
              onPressed: _chat != null ? () { Navigator.pop(context); _rebuildChatSafe(); } : null,
              icon: const Icon(Icons.refresh, size: 18),
              label: const Text('Rebuild Session'),
            ),
            const Divider(height: 24),

            // ── Engine Config (need reload) ────────────────────────────────
            const _SectionHeader('Engine (requires reload)'),
            _SliderTile(
              label: 'Max Tokens: ${mgr.maxTokens}',
              value: mgr.maxTokens.toDouble(), min: 1024, max: 8192, divisions: 7,
              onChanged: (v) { mgr.maxTokens = v.toInt(); mgr.update(); },
            ),
            SwitchListTile(dense: true, title: const Text('Use GPU'), value: mgr.useGpu, onChanged: (v) { mgr.useGpu = v; mgr.update(); }),
            SwitchListTile(dense: true, title: const Text('Enable MTP (Gemma 4)'), subtitle: const Text('Speculative decoding (Android only)'), value: _enableMtp, onChanged: (v) { setState(() => _enableMtp = v); }),
            SwitchListTile(dense: true, title: const Text('Audio support'), value: mgr.supportAudio, onChanged: (v) { mgr.supportAudio = v; mgr.update(); }),
            const SizedBox(height: 8),
            FilledButton.tonalIcon(
              onPressed: () { Navigator.pop(context); _loadModel(local: true); },
              icon: const Icon(Icons.power_settings_new, size: 18),
              label: const Text('Reload Engine'),
            ),
            const Divider(height: 24),

            // ── History ────────────────────────────────────────────────────
            const _SectionHeader('History'),
            Row(children: [
              Expanded(child: OutlinedButton.icon(onPressed: _chat != null ? _exportHistory : null, icon: const Icon(Icons.upload_outlined, size: 18), label: const Text('Export JSON'))),
              const SizedBox(width: 8),
              Expanded(child: OutlinedButton.icon(onPressed: _chat != null ? _importHistory : null, icon: const Icon(Icons.download_outlined, size: 18), label: const Text('Import JSON'))),
            ]),
            const Divider(height: 24),

            // ── Cache ──────────────────────────────────────────────────────
            const _SectionHeader('Cache'),
            OutlinedButton.icon(
              onPressed: () async {
                await ModelManager.instance.purgeCache();
                if (mounted) { Navigator.pop(context); _snack('Cache purged.'); }
              },
              icon: const Icon(Icons.delete_forever_outlined, size: 18),
              label: const Text('Purge Model Cache'),
              style: OutlinedButton.styleFrom(foregroundColor: Theme.of(context).colorScheme.error),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Local helpers ─────────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Text(title, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Theme.of(context).colorScheme.primary)),
      );
}

class _SliderTile extends StatelessWidget {
  final String label;
  final double value, min, max;
  final int divisions;
  final ValueChanged<double> onChanged;
  const _SliderTile({required this.label, required this.value, required this.min, required this.max, required this.divisions, required this.onChanged});
  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Padding(padding: const EdgeInsets.only(top: 8, left: 4), child: Text('$label: ${value.toStringAsFixed(value < 10 ? 2 : 0)}', style: const TextStyle(fontSize: 12))),
      Slider(value: value, min: min, max: max, divisions: divisions, onChanged: onChanged),
    ],
  );
}