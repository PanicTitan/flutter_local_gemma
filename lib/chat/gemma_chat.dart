import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';

import '../types/content_parts.dart';
import '../gemma/gemma.dart';
import '../gemma/gemma_response.dart';
import '../gemma/thinking_parser.dart';
import '../json_schema/json_repair.dart';
import '../json_schema/schema.dart';
import 'base_chat.dart';

// ─── Supporting types ─────────────────────────────────────────────────────────

/// Strategy for managing the conversation context when the KV-cache fills up.
enum ContextStrategy {
  /// Never truncate; allow the engine to error when the limit is reached.
  none,

  /// Remove the oldest messages (a rolling window of recent history).
  slidingWindow,

  /// Summarise the oldest half of the conversation into a single digest message.
  summarize,
}

/// A single message in the conversation history.
class ChatMessage {
  final String role;
  final String text;
  final List<Uint8List> images;
  final List<Uint8List> audios;
  final bool isHidden;
  final bool isUiOnly;

  const ChatMessage({
    required this.role,
    required this.text,
    this.images = const [],
    this.audios = const [],
    this.isHidden = false,
    this.isUiOnly = false,
  });

  // ── Serialization ──────────────────────────────────────────────────────────

  /// Converts this message to a JSON-safe map.
  ///
  /// Binary payloads (images / audio) are base64-encoded so they survive
  /// JSON round-trips across file systems, clipboard, or network.
  Map<String, dynamic> toJson() => {
        'role': role,
        'text': text,
        'images': images.map((b) => base64Encode(b)).toList(),
        'audios': audios.map((b) => base64Encode(b)).toList(),
        'isHidden': isHidden,
        'isUiOnly': isUiOnly,
      };

  /// Restores a [ChatMessage] from its [toJson] representation.
  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
        role: json['role'] as String,
        text: json['text'] as String,
        images: (json['images'] as List<dynamic>? ?? [])
            .map((s) => base64Decode(s as String))
            .toList(),
        audios: (json['audios'] as List<dynamic>? ?? [])
            .map((s) => base64Decode(s as String))
            .toList(),
        isHidden: json['isHidden'] as bool? ?? false,
        isUiOnly: json['isUiOnly'] as bool? ?? false,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// GemmaChat  —  stateful multi-turn conversation
// ─────────────────────────────────────────────────────────────────────────────

/// A high-level, stateful conversational wrapper around [FlutterLocalGemma].
///
/// Manages message history, context window overflow, system prompts, and
/// provides streaming + JSON output helpers.
///
/// ## Quick-start
/// ```dart
/// final chat = GemmaChat(systemPrompt: 'You are a helpful assistant.');
/// await chat.init();
///
/// await for (final token in chat.sendMessageStream(text: 'Hello!')) {
///   print(token);
/// }
///
/// await chat.dispose();
/// ```
///
/// ## Token tracking
/// After each call, inspect [tokenStats] for the current usage snapshot:
/// ```dart
/// print(chat.tokenStats); // TokenStats(340 / 4096 used, 3756 remaining)
/// ```
///
/// ## History export / import
/// ```dart
/// final json = await chat.exportHistory();
/// await prefs.setString('chat_history', json);
/// // later…
/// await chat.importHistory(await prefs.getString('chat_history')!);
/// ```
class GemmaChat implements BaseChat {
  final FlutterLocalGemma _engine = FlutterLocalGemma();

  final int maxContextTokens;
  final String? systemPrompt;
  final ContextStrategy contextStrategy;
  final SessionConfig? config;

  /// Fraction of the context window that triggers context management and
  /// the [isNearContextLimit] flag. Default is 0.8 (80%).
  final double contextFillThreshold;

  ChatSession? _nativeSession;
  final List<ChatMessage> _history = [];
  String? _lastUsedSystemPrompt;

  GemmaChat({
    this.maxContextTokens = 4096,
    this.systemPrompt,
    this.contextStrategy = ContextStrategy.none,
    this.config,
    this.contextFillThreshold = 0.8,
  });

  // ── Public state ──────────────────────────────────────────────────────────

  /// An unmodifiable view of the conversation history.
  List<ChatMessage> get history => List.unmodifiable(_history);

  // ── Token tracking ─────────────────────────────────────────────────────────

  /// Estimates the number of tokens consumed by the current conversation.
  ///
  /// This is a fast, synchronous heuristic based on character counts and
  /// Google's documented constants (257 tokens per image, 1 token / 150 ms
  /// of audio). For a precise count use [ChatSession.countTokens].
  int get currentTokenCount {
    int total = 0;
    if (_lastUsedSystemPrompt != null) {
      total += (_lastUsedSystemPrompt!.length / 4).ceil();
    } else if (systemPrompt != null) {
      total += (systemPrompt!.length / 4).ceil();
    }
    for (final msg in _history) {
      total += (msg.text.length / 4).ceil();
      total += msg.images.length * 257;
      total += msg.audios.fold<int>(
        0,
        (sum, a) => sum + (a.length ~/ 32) ~/ 150,
      );
    }
    return total;
  }

  /// Estimated tokens remaining before the context window is full.
  int get remainingTokens =>
      (maxContextTokens - currentTokenCount).clamp(0, maxContextTokens);

  /// Whether the conversation is approaching the context window limit.
  ///
  /// Returns `true` when [currentTokenCount] exceeds
  /// `maxContextTokens * contextFillThreshold`.
  bool get isNearContextLimit =>
      currentTokenCount >= (maxContextTokens * contextFillThreshold).toInt();

  /// A snapshot of the current token usage and capacity.
  TokenStats get tokenStats => TokenStats(
        estimatedUsed: currentTokenCount,
        maxContext: maxContextTokens,
      );

  @override
  Future<int> estimateNextTurnTokens(
    String prompt, {
    List<Uint8List>? images,
    List<Uint8List>? audios,
  }) async {
    int nextTokens = (prompt.length / 4).ceil();
    if (images != null) nextTokens += images.length * 257;
    if (audios != null) {
      nextTokens += audios.fold<int>(0, (sum, a) => sum + (a.length ~/ 32) ~/ 150);
    }
    return currentTokenCount + nextTokens;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Opens the native session. Must be called before [sendMessageStream].
  ///
  /// On web, the first call to [sendMessageStream] handles session creation
  /// automatically; calling [init] early warms up the infrastructure.
  Future<void> init() async {
    if (!kIsWeb && _nativeSession == null) {
      await _initNativeSession();
    }
  }

  /// Releases all resources held by this chat instance.
  ///
  /// The [FlutterLocalGemma] engine itself is NOT disposed here because it is a
  /// singleton shared across the app. Call [FlutterLocalGemma().dispose()] when
  /// you want to fully unload the model.
  Future<void> dispose() async {
    await _nativeSession?.dispose();
    _nativeSession = null;
  }

  // ── Message sending ───────────────────────────────────────────────────────

  /// Sends a message and streams back the model's response token-by-token.
  ///
  /// The user message and the model's full reply are appended to [history]
  /// automatically. Context overflow is handled according to [contextStrategy].
  ///
  /// [overrideSystemPrompt] temporarily replaces the [systemPrompt] for this
  /// turn only; subsequent turns revert to the original.
  Stream<String> sendMessageStream({
    required String text,
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? overrideSystemPrompt,
    bool isHidden = false,
    bool hideModelResponse = false,
  }) async* {
    final targetPrompt = overrideSystemPrompt ?? systemPrompt;
    final promptChanged = targetPrompt != _lastUsedSystemPrompt;
    
    if (promptChanged) {
      _lastUsedSystemPrompt = targetPrompt;
    }

    // Rebuild the native context when the system prompt changes.
    // Pass targetPrompt explicitly so _rebuildNativeContext forwards it to
    // _initNativeSession — without this the JSON override is silently dropped.
    if (promptChanged || (!kIsWeb && _nativeSession == null)) {
      await _rebuildNativeContext(withPrompt: targetPrompt);
    }
    await _manageContextWindow(text, images ?? [], audios ?? []);

    _history.add(ChatMessage(
      role: 'user',
      text: text,
      images: images ?? [],
      audios: audios ?? [],
      isHidden: isHidden,
    ));

    final Stream<String> responseStream = kIsWeb
        ? await _handleWebTurn(targetPrompt)
        : await _handleNativeTurn(text, images, audios);

    final buffer = StringBuffer();
    await for (var chunk in responseStream) {
      if (chunk.contains('<end_of_turn>')) {
        chunk = chunk.replaceAll('<end_of_turn>', '');
      }
      buffer.write(chunk);
      if (chunk.isNotEmpty) yield chunk;
    }

    var finalText = buffer.toString().replaceAll('<end_of_turn>', '').trim();

    _history.add(ChatMessage(role: 'model', text: finalText, isHidden: hideModelResponse));
  }

  /// Appends a message directly to the conversation history and native context.
  Future<void> appendMessage(ChatMessage message) async {
    _history.add(message);
    if (!message.isUiOnly && !kIsWeb && _nativeSession != null) {
      await _nativeSession!.addToContext(_formatMessageParts(message, isLast: false));
    }
  }

  /// Sends a message and returns the model's response as a single [String].
  Future<String> sendMessage({
    required String text,
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? overrideSystemPrompt,
    bool isHidden = false,
  }) async {
    final buffer = StringBuffer();
    await for (final chunk in sendMessageStream(
      text: text,
      images: images,
      audios: audios,
      overrideSystemPrompt: overrideSystemPrompt,
      isHidden: isHidden,
    )) {
      buffer.write(chunk);
    }
    return buffer.toString();
  }

  /// Sends a message and streams back partial-JSON objects as the model fills
  /// in fields according to [schema].
  ///
  /// Each yielded value is a partially-complete but syntactically valid
  /// (repaired) JSON object or array. The final yield is the complete response.
  ///
  /// Use [schema] for type-safe schema definition, or [rawSchemaStr] to pass
  /// a raw JSON Schema string.
  Stream<dynamic> sendMessageJsonStream({
    required String text,
    Schema? schema,
    String? rawSchemaStr,
    List<Uint8List>? images,
    List<Uint8List>? audios,
  }) async* {
    final schemaStr =
        rawSchemaStr ?? jsonEncode(schema?.toJsonSchema() ?? <String, dynamic>{});

    // Full instruction block used as the system prompt override.
    final jsonInstructions = '''
You are a strict data-formatting API. Your ONLY job is to generate a valid JSON object conforming EXACTLY to this JSON Schema:
$schemaStr

CRITICAL RULES:
1. Do NOT wrap the output in markdown code blocks.
2. Return ONLY raw JSON starting with { or [. No explanations.
3. Every field in the schema must be present.
''';

    final base = systemPrompt ?? '';
    final ephemeralPrompt =
        base.isNotEmpty ? '$base\n\n$jsonInstructions' : jsonInstructions;

    // Android fix: on the native path the model sometimes ignores pure system-
    // prompt instructions. Prepend a compact reminder directly in the user
    // message so it appears in the conversation turn the model actually sees.
    // On web this is redundant but harmless.
    final augmentedText = kIsWeb
        ? text
        : '[INSTRUCTION: Reply ONLY with raw JSON matching this schema: $schemaStr]\n$text';

    final rawStream = sendMessageStream(
      text: augmentedText,
      images: images,
      audios: audios,
      overrideSystemPrompt: ephemeralPrompt,
    );

    final buffer = StringBuffer();
    await for (final chunk in rawStream) {
      buffer.write(chunk);
      final repaired = repairJson(buffer.toString(), streamStable: true);
      if (repaired != null && repaired != '') yield repaired;
    }
  }

  /// Stops the current generation mid-stream.
  Future<void> stop() async => _nativeSession?.stopGeneration();

  /// Streams typed [GemmaResponse] events, separating thinking content from
  /// final answer tokens.
  ///
  /// Only meaningful when `SessionConfig.enableThinking = true` (Gemma 4).
  /// When thinking is disabled, only [GemmaTextResponse] events are emitted.
  ///
  /// ```dart
  /// await for (final resp in chat.sendMessageResponseStream(text: 'Solve 17×19')) {
  ///   switch (resp) {
  ///     case GemmaThinkingResponse(:final content):
  ///       print('Thinking: $content');
  ///     case GemmaTextResponse(:final token):
  ///       print(token);
  ///   }
  /// }
  /// ```
  Stream<GemmaResponse> sendMessageResponseStream({
    required String text,
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? overrideSystemPrompt,
    bool isHidden = false,
  }) {
    return ThinkingParser.filterThinkingStream(
      sendMessageStream(
        text: text,
        images: images,
        audios: audios,
        overrideSystemPrompt: overrideSystemPrompt,
        isHidden: isHidden,
      ),
    );
  }

  // ── History management ────────────────────────────────────────────────────

  /// Removes all messages from the history and resets the native context.
  Future<void> clearHistory() async {
    _history.clear();
    await _rebuildNativeContext();
  }

  /// Replaces the message at [index] with [newMessage].
  /// If [rebuildContext] is true, the native context is destroyed and rebuilt.
  Future<void> editHistory(int index, ChatMessage newMessage, {bool rebuildContext = true}) async {
    if (index < 0 || index >= _history.length) return;
    _history[index] = newMessage;
    if (rebuildContext) {
      await _rebuildNativeContext();
    }
  }

  /// Removes the message at [index] from the history and rebuilds the context.
  Future<void> removeHistory(int index) async {
    if (index < 0 || index >= _history.length) return;
    _history.removeAt(index);
    await _rebuildNativeContext();
  }

  // ── History serialization ─────────────────────────────────────────────────

  /// Exports the conversation history to a JSON string.
  ///
  /// Binary payloads (images, audio clips) are base64-encoded so the export
  /// is fully self-contained and can be stored in a file, shared clipboard,
  /// or a backend database.
  ///
  /// ```dart
  /// final json = await chat.exportHistory();
  /// await File('chat.json').writeAsString(json);
  /// ```
  Future<String> exportHistory() async {
    final payload = <String, dynamic>{
      'version':      1,
      'systemPrompt': systemPrompt,
      'maxContext':   maxContextTokens,
      'exportedAt':   DateTime.now().toIso8601String(),
      'messages':     _history.map((m) => m.toJson()).toList(),
    };
    return const JsonEncoder.withIndent('  ').convert(payload);
  }

  /// Imports a conversation history from a JSON string produced by [exportHistory].
  ///
  /// The current history is replaced. The native context is rebuilt so the
  /// model is aware of all imported messages.
  ///
  /// Throws [FormatException] if the JSON is malformed or the version is
  /// incompatible.
  ///
  /// ```dart
  /// final json = await File('chat.json').readAsString();
  /// await chat.importHistory(json);
  /// ```
  Future<void> importHistory(String jsonString) async {
    final Map<String, dynamic> payload;
    try {
      payload = jsonDecode(jsonString) as Map<String, dynamic>;
    } catch (e) {
      throw FormatException('importHistory: invalid JSON. $e');
    }

    final version = payload['version'] as int? ?? 0;
    if (version != 1) {
      throw FormatException(
        'importHistory: unsupported version $version (expected 1).',
      );
    }

    final rawMessages = payload['messages'] as List<dynamic>? ?? [];
    _history
      ..clear()
      ..addAll(
        rawMessages.map(
          (e) => ChatMessage.fromJson(e as Map<String, dynamic>),
        ),
      );

    await _rebuildNativeContext();
  }

  // ── Private – native session management ───────────────────────────────────

  Future<void> _initNativeSession({String? ephemeralPrompt}) async {
    _lastUsedSystemPrompt = ephemeralPrompt ?? systemPrompt;
    _nativeSession = await _engine.createSession(
      config: (config ?? SessionConfig()).copyWith(
        systemPrompt: _lastUsedSystemPrompt,
      ),
    );
  }

  /// Closes the current session, creates a fresh one, then replays the history.
  ///
  /// [withPrompt] overrides the system prompt for this session. If omitted the
  /// current [_lastUsedSystemPrompt] is forwarded so any active override (e.g.
  /// the JSON schema instruction) is preserved across rebuilds.
  Future<void> _rebuildNativeContext({String? withPrompt}) async {
    if (kIsWeb) return;
    // Use explicit override, then the last-used prompt, then the base prompt.
    final effectivePrompt = withPrompt ?? _lastUsedSystemPrompt ?? systemPrompt;
    await _nativeSession?.dispose();
    _nativeSession = null;
    await _initNativeSession(ephemeralPrompt: effectivePrompt);
    for (final msg in _history) {
      if (msg.isUiOnly) continue;
      await _nativeSession?.addToContext(
        _formatMessageParts(msg, isLast: false),
      );
    }
  }

  // ── Private – context window management ───────────────────────────────────

  Future<void> _manageContextWindow(
    String newText,
    List<Uint8List> images,
    List<Uint8List> audios,
  ) async {
    if (contextStrategy == ContextStrategy.none) return;

    final newTokens = (newText.length / 4).ceil() +
        (images.length * 257) +
        audios.fold<int>(0, (s, a) => s + (a.length ~/ 32) ~/ 150);

    int total     = currentTokenCount + newTokens;
    final limit   = (maxContextTokens * 0.8).toInt();
    if (total <= limit) return;

    if (contextStrategy == ContextStrategy.slidingWindow) {
      // Drop oldest messages until we are under the limit.
      while (_history.length > 1 && total > limit) {
        final removed = _history.removeAt(0);
        total -= (removed.text.length / 4).ceil() +
            (removed.images.length * 257) +
            removed.audios.fold<int>(0, (s, a) => s + (a.length ~/ 32) ~/ 150);
      }
      await _rebuildNativeContext();
    } else if (contextStrategy == ContextStrategy.summarize) {
      // Summarise the oldest half of the history into a digest.
      final cutIndex = (_history.length / 2).ceil().clamp(1, _history.length);
      final toSummarise = _history.sublist(0, cutIndex);
      _history.removeRange(0, cutIndex);

      final prompt = StringBuffer(
        'Summarise the following conversation concisely:\n',
      );
      for (final m in toSummarise) prompt.write('${m.role}: ${m.text}\n');

      final summary =
          await _engine.computeSingle([TextPart(prompt.toString())]);

      _history.insert(
        0,
        ChatMessage(
          role: 'system',
          text: '[Previous conversation summary]: $summary',
        ),
      );

      await _rebuildNativeContext();
    }
  }

  // ── Private – turn handlers ───────────────────────────────────────────────

  Future<Stream<String>> _handleNativeTurn(
    String text,
    List<Uint8List>? images,
    List<Uint8List>? audios,
  ) async {
    final parts = <ContentPart>[];
    if (images != null) for (final img in images) parts.add(ImagePart(img));
    if (audios != null) for (final aud in audios) parts.add(AudioPart(aud));
    if (text.isNotEmpty) parts.add(TextPart(text));
    return _nativeSession!.generateResponseStream(parts);
  }

  Future<Stream<String>> _handleWebTurn(String? targetPrompt) async {
    await _nativeSession?.clearContext();
    _nativeSession = await _engine.createSession(
      config: (config ?? SessionConfig()).copyWith(
        systemPrompt: null, // system prompt is injected manually below
      ),
    );

    // Re-inject the system prompt manually in Gemma's chat template format.
    if (targetPrompt != null && targetPrompt.isNotEmpty) {
      await _nativeSession!.addToContext([
        TextPart('<start_of_turn>system\n$targetPrompt<end_of_turn>\n'),
      ]);
    }

    // Replay history (excluding the very last message which is the user turn
    // we are about to generate a response for).
    for (int i = 0; i < _history.length - 1; i++) {
      if (_history[i].isUiOnly) continue;
      await _nativeSession!.addToContext(
        _formatMessageParts(_history[i], isLast: false),
      );
    }

    // Find the last non-UiOnly message (which should be the user message)
    ChatMessage lastMsg = _history.last;
    for (int i = _history.length - 1; i >= 0; i--) {
      if (!_history[i].isUiOnly) {
        lastMsg = _history[i];
        break;
      }
    }

    return _nativeSession!.generateResponseStream(
      _formatMessageParts(lastMsg, isLast: true),
    );
  }

  // ── Private – formatting helpers ──────────────────────────────────────────

  List<ContentPart> _formatMessageParts(
    ChatMessage msg, {
    required bool isLast,
  }) {
    final parts = <ContentPart>[TextPart('<start_of_turn>${msg.role}\n')];
    for (final img in msg.images) parts.add(ImagePart(img));
    for (final aud in msg.audios) parts.add(AudioPart(aud));
    if (msg.text.isNotEmpty) parts.add(TextPart(msg.text));
    parts.add(TextPart(
      isLast && msg.role == 'user'
          ? '<end_of_turn>\n<start_of_turn>model\n'
          : '<end_of_turn>\n',
    ));
    return parts;
  }
}