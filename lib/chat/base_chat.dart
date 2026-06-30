import 'dart:typed_data';

import '../gemma/gemma.dart';
import 'gemma_chat.dart';

/// Abstract interface for all high-level chat wrappers.
///
/// Both [GemmaChat] and [AgentChat] implement this interface, so you can
/// write UI code that works with either without casting:
///
/// ```dart
/// BaseChat chat = someCondition ? GemmaChat(...) : AgentChat(...);
/// await chat.init();
/// print(chat.tokenStats);
/// print(chat.isNearContextLimit);
/// final json = await chat.exportHistory();
/// await chat.importHistory(json);
/// ```
abstract interface class BaseChat {
  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /// Initialises the underlying engine session.
  Future<void> init();

  /// Releases all resources held by this chat instance.
  /// The [FlutterLocalGemma] engine singleton is NOT disposed.
  Future<void> dispose();

  /// Aborts the current generation mid-stream.
  Future<void> stop();

  // ── History ────────────────────────────────────────────────────────────────

  /// Read-only view of the conversation history.
  List<ChatMessage> get history;

  /// Clears all history and resets the context window.
  Future<void> clearHistory();

  /// Replaces the message at [index] with [msg].
  Future<void> editHistory(int index, ChatMessage msg);

  /// Appends [msg] to the end of the history.
  Future<void> appendMessage(ChatMessage msg);

  /// Removes the message at [index].
  Future<void> removeHistory(int index);

  // ── Token tracking ─────────────────────────────────────────────────────────

  /// Maximum context-window capacity (tokens).
  int get maxContextTokens;

  /// Estimated tokens consumed so far.
  int get currentTokenCount;

  /// Estimated tokens remaining before the context window is full.
  int get remainingTokens;

  /// Whether the conversation is approaching the context limit.
  bool get isNearContextLimit;

  /// A snapshot of the current token usage.
  TokenStats get tokenStats;

  /// Estimates the total token usage for the next turn, including the current
  /// history, the new [prompt], and any multimodal payloads (e.g. [images], [audios]).
  ///
  /// This is useful for UI builders to determine if a prompt will fit in the
  /// context window before actually generating.
  Future<int> estimateNextTurnTokens(
    String prompt, {
    List<Uint8List>? images,
    List<Uint8List>? audios,
  });

  // ── History persistence ────────────────────────────────────────────────────

  /// Exports the conversation history to a JSON string.
  ///
  /// Binary payloads (images, audio) are base64-encoded.
  Future<String> exportHistory();

  /// Imports a previously exported history string.
  ///
  /// Replaces the current in-memory history. Binary payloads are decoded from
  /// base64 automatically.
  Future<void> importHistory(String json);
}