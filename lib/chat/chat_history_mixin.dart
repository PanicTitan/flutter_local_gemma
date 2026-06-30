import 'dart:convert';

import 'gemma_chat.dart';

/// Mixin that provides in-memory history management for chat wrappers.
///
/// Classes that mix this in must supply [history] as a growable [List] and
/// implement [onHistoryChanged] to react to structural changes (e.g., to
/// rebuild the native KV-cache context).
///
/// Used internally by [GemmaChat]. Exposed publicly so you can build custom
/// chat wrappers that reuse the same serialization and mutation logic.
///
/// ```dart
/// class MyChat with ChatHistoryMixin {
///   @override
///   final List<ChatMessage> history = [];
///
///   @override
///   Future<void> onHistoryChanged() async {
///     // Rebuild your context / session here.
///   }
/// }
/// ```
mixin ChatHistoryMixin {
  /// The backing history list. Must be mutable.
  List<ChatMessage> get history;

  /// Called after any structural change to [history] (add, remove, edit, clear).
  ///
  /// Override to rebuild the native KV-cache context or perform any other
  /// housekeeping that must happen when history changes.
  Future<void> onHistoryChanged();

  // ── Mutations ─────────────────────────────────────────────────────────────

  /// Appends [message] to history and triggers [onHistoryChanged].
  Future<void> addMessage(ChatMessage message) async {
    history.add(message);
    await onHistoryChanged();
  }

  /// Replaces the message at [index] with [newMessage] and triggers
  /// [onHistoryChanged].
  ///
  /// No-op if [index] is out of bounds.
  Future<void> editMessage(int index, ChatMessage newMessage) async {
    if (index < 0 || index >= history.length) return;
    history[index] = newMessage;
    await onHistoryChanged();
  }

  /// Removes the message at [index] and triggers [onHistoryChanged].
  ///
  /// No-op if [index] is out of bounds.
  Future<void> removeMessage(int index) async {
    if (index < 0 || index >= history.length) return;
    history.removeAt(index);
    await onHistoryChanged();
  }

  /// Clears all messages and triggers [onHistoryChanged].
  Future<void> clearMessages() async {
    history.clear();
    await onHistoryChanged();
  }

  // ── Serialization ─────────────────────────────────────────────────────────

  /// Exports [history] to a self-contained JSON string.
  ///
  /// Binary payloads (images, audio) are base64-encoded.
  /// Metadata fields: `version`, `exportedAt`, `messageCount`.
  String exportMessages() {
    final payload = <String, dynamic>{
      'version':      1,
      'exportedAt':   DateTime.now().toIso8601String(),
      'messageCount': history.length,
      'messages':     history.map((m) => m.toJson()).toList(),
    };
    return const JsonEncoder.withIndent('  ').convert(payload);
  }

  /// Imports a JSON string produced by [exportMessages], replacing [history].
  ///
  /// Triggers [onHistoryChanged] after import.
  ///
  /// Throws [FormatException] if the JSON is malformed or the `version` field
  /// is missing.
  Future<void> importMessages(String json) async {
    final Map<String, dynamic> payload =
        jsonDecode(json) as Map<String, dynamic>;

    final version = payload['version'] as int?;
    if (version == null || version < 1) {
      throw const FormatException(
          'ChatHistoryMixin: unsupported or missing history version.');
    }

    final raw = payload['messages'] as List<dynamic>? ?? [];
    history
      ..clear()
      ..addAll(raw.map((e) => ChatMessage.fromJson(e as Map<String, dynamic>)));

    await onHistoryChanged();
  }

  // ── Queries ───────────────────────────────────────────────────────────────

  /// Returns true if there are no messages.
  bool get isEmpty => history.isEmpty;

  /// Returns true if there is at least one message.
  bool get isNotEmpty => history.isNotEmpty;

  /// Number of messages in history.
  int get messageCount => history.length;

  /// The last user message, or null if there are none.
  ChatMessage? get lastUserMessage =>
      history.lastWhereOrNull((m) => m.role == 'user');

  /// The last model reply, or null if there are none.
  ChatMessage? get lastModelMessage =>
      history.lastWhereOrNull((m) => m.role == 'model');
}

// ── Private extension ─────────────────────────────────────────────────────────

extension<T> on Iterable<T> {
  T? lastWhereOrNull(bool Function(T) test) {
    T? result;
    for (final e in this) {
      if (test(e)) result = e;
    }
    return result;
  }
}
