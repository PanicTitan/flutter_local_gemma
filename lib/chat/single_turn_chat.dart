import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';

import '../types/content_parts.dart';
import '../gemma/gemma.dart';
import '../json_schema/json_repair.dart';
import '../json_schema/schema.dart';

/// A lightweight, **stateless** inference wrapper.
///
/// Unlike [GemmaChat], `SingleTurnChat`:
/// - Keeps **no conversation history** between calls.
/// - Never rebuilds a context window — each [generate] call is fully
///   independent and disposable.
/// - Is ideal for batch processing, RAG retrieval answers, function-calling
///   pipelines, or any use-case where you don't need multi-turn memory.
///
/// ## Usage
/// ```dart
/// final chat = SingleTurnChat(
///   config: SessionConfig(temperature: 0.3, systemPrompt: 'You are a tagger.'),
/// );
///
/// // Plain text
/// final answer = await chat.generate('Summarise this paragraph: ...');
///
/// // Structured JSON
/// final tags = await chat.generateJson(
///   'Extract tags from: "Flutter is great for cross-platform apps"',
///   schema: Schema.object({'tags': Schema.array(items: Schema.string())}),
/// );
///
/// // When you are truly done, release the engine
/// await FlutterLocalGemma().dispose();
/// ```
///
/// ## Token tracking
/// Each call exposes [lastCallStats] immediately after it returns, giving you
/// a snapshot of the tokens consumed by that single turn.
class SingleTurnChat {
  final FlutterLocalGemma _engine = FlutterLocalGemma();
  final SessionConfig _config;

  /// Token statistics for the most recently completed generation.
  /// Null before the first call.
  TokenStats? _lastCallStats;
  TokenStats? get lastCallStats => _lastCallStats;

  /// The maximum context window reported by the engine.
  int get maxContext => _engine.maxTokens;

  SingleTurnChat({SessionConfig? config})
      : _config = config ?? SessionConfig();

  // ─── Plain text ──────────────────────────────────────────────────────────

  /// Generates a text response for a single [prompt].
  ///
  /// Optionally attach [images] or [audios] for multimodal models.
  /// Use [systemPrompt] to override the session-level system prompt for this
  /// call only.
  ///
  /// Each call creates a fresh native session, generates, then releases it.
  Future<String> generate(
    String prompt, {
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? systemPrompt,
  }) async {
    final buffer = StringBuffer();
    await for (final chunk in generateStream(
      prompt,
      images: images,
      audios: audios,
      systemPrompt: systemPrompt,
    )) {
      buffer.write(chunk);
    }
    return buffer.toString();
  }

  /// Streams tokens for a single [prompt], then closes.
  ///
  /// The underlying native session is created at the start of the stream and
  /// disposed when the stream closes (normally or due to error).
  Stream<String> generateStream(
    String prompt, {
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? systemPrompt,
  }) async* {
    if (!_engine.isInitialized) {
      throw StateError(
        'FlutterLocalGemma is not initialised. Call FlutterLocalGemma().init() first.',
      );
    }

    final effectiveConfig = _buildConfig(systemPrompt);
    final session = await _engine.createSession(config: effectiveConfig);

    try {
      final parts = _buildParts(prompt, images, audios);
      await for (final chunk in session.generateResponseStream(parts)) {
        yield chunk;
      }
      _lastCallStats = session.stats;
    } finally {
      await session.dispose();
    }
  }

  // ─── Structured JSON ──────────────────────────────────────────────────────

  /// Generates a structured JSON response for [prompt] and parses it.
  ///
  /// Pass either a typed [Schema] (built with [Schema.object], [Schema.array],
  /// etc.) or a raw JSON Schema string via [rawSchemaStr].
  ///
  /// The model is instructed to return raw JSON only (no markdown wrappers).
  /// The response is parsed and repaired by [repairJson].
  ///
  /// Returns a parsed Dart object (`Map`, `List`, `String`, `num`, or `bool`).
  Future<dynamic> generateJson(
    String prompt, {
    Schema? schema,
    String? rawSchemaStr,
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? systemPrompt,
  }) async {
    final buffer = StringBuffer();
    await for (final _ in generateJsonStream(
      prompt,
      schema: schema,
      rawSchemaStr: rawSchemaStr,
      images: images,
      audios: audios,
      systemPrompt: systemPrompt,
    )) {
      // Collect all partial results; we only want the last (complete) one.
    }
    // Run repair on the final accumulated buffer.
    final raw = await generate(
      prompt,
      images: images,
      audios: audios,
      systemPrompt: _buildJsonSystemPrompt(schema, rawSchemaStr, systemPrompt),
    );
    return repairJson(raw);
  }

  /// Streams partial JSON objects as the model fills in fields.
  ///
  /// Each emitted value is the best-effort parse of the accumulated response
  /// so far. The final emission is the complete, fully-formed JSON value.
  ///
  /// Useful for showing live-updating UI as structured data arrives.
  Stream<dynamic> generateJsonStream(
    String prompt, {
    Schema? schema,
    String? rawSchemaStr,
    List<Uint8List>? images,
    List<Uint8List>? audios,
    String? systemPrompt,
  }) async* {
    final ephemeralPrompt =
        _buildJsonSystemPrompt(schema, rawSchemaStr, systemPrompt);

    final rawStream = generateStream(
      prompt,
      images: images,
      audios: audios,
      systemPrompt: ephemeralPrompt,
    );

    final buffer = StringBuffer();
    await for (final chunk in rawStream) {
      buffer.write(chunk);
      final repaired = repairJson(buffer.toString(), streamStable: true);
      if (repaired != null && repaired != '') yield repaired;
    }
  }

  // ── Token utilities ───────────────────────────────────────────────────────

  /// Estimates how many tokens [text] and optional media payloads will consume.
  ///
  /// This is a heuristic; the native tokenizer is more precise. Use this
  /// before a call to check headroom.
  Future<int> estimateTokens(
    String text, {
    int imageCount = 0,
    int audioDurationMs = 0,
  }) async {
    // We need a temporary session only to call countTokens.
    // On web this is very cheap; on mobile it creates a new Conversation.
    if (!_engine.isInitialized) return _heuristicTokens(text, imageCount, audioDurationMs);

    final session = await _engine.createSession(config: _config);
    try {
      return await session.countTokens(
        text,
        imageCount: imageCount,
        audioDurationMs: audioDurationMs,
      );
    } finally {
      await session.dispose();
    }
  }

  // ─── Private helpers ──────────────────────────────────────────────────────

  /// Returns a [SessionConfig] that overrides [systemPrompt] when provided,
  /// otherwise falls back to the one baked into [_config].
  SessionConfig _buildConfig(String? systemPrompt) {
    if (systemPrompt == null) return _config;
    return SessionConfig(
      temperature:    _config.temperature,
      topP:           _config.topP,
      topK:           _config.topK,
      randomSeed:     _config.randomSeed,
      systemPrompt:   systemPrompt,
      autoStopConfig: _config.autoStopConfig,
    );
  }

  List<ContentPart> _buildParts(
    String text,
    List<Uint8List>? images,
    List<Uint8List>? audios,
  ) {
    final parts = <ContentPart>[];
    if (images != null) for (final img in images) parts.add(ImagePart(img));
    if (audios != null) for (final aud in audios) parts.add(AudioPart(aud));
    if (text.isNotEmpty) parts.add(TextPart(text));
    return parts;
  }

  /// Constructs the ephemeral system prompt that instructs the model to output
  /// only raw JSON conforming to the given schema.
  String _buildJsonSystemPrompt(
    Schema? schema,
    String? rawSchemaStr,
    String? baseSystemPrompt,
  ) {
    final schemaStr =
        rawSchemaStr ?? jsonEncode(schema?.toJsonSchema() ?? <String, dynamic>{});

    final instructions = '''
You are a strict data-formatting API. Your ONLY job is to return a valid JSON object conforming EXACTLY to this JSON Schema:
$schemaStr

CRITICAL RULES:
1. Do NOT use markdown code blocks.
2. Return ONLY raw JSON starting with { or [. No explanations or commentary.
''';

    final base = baseSystemPrompt ?? _config.systemPrompt ?? '';
    return base.isNotEmpty ? '$base\n\n$instructions' : instructions;
  }

  int _heuristicTokens(String text, int imageCount, int audioDurationMs) {
    return (text.split(RegExp(r'\s+')).length * 1.3).toInt() +
        (text.length * 0.15).toInt() +
        imageCount * 257 +
        (audioDurationMs / 150).ceil();
  }
}