/// Structured response types from the Gemma inference engine.
///
/// Mirrors flutter_gemma's ModelResponse pattern to support tool calling,
/// thinking mode, and parallel function calls alongside standard text tokens.
///
/// The currently **emitted** subtypes are:
/// - [GemmaTextResponse] — plain text tokens
/// - [GemmaThinkingResponse] — content from `<think>…</think>` blocks
/// - [GemmaToolResultResponse] — result of a dispatched skill (from [AgentChat])
/// - [GemmaCancelledResponse] — generation aborted by [GemmaChat.stop]
///
/// [GemmaFunctionCallResponse] and [GemmaParallelFunctionCallResponse] are
/// defined but not yet emitted. They are reserved for a future streaming
/// tool-call API. For now, tool calls are handled internally by [AgentChat].
sealed class GemmaResponse {
  const GemmaResponse();
}

/// A single text token from the model's response stream.
final class GemmaTextResponse extends GemmaResponse {
  final String token;
  const GemmaTextResponse(this.token);
}

/// Content from a `<think>…</think>` or `<|channel>thought…<channel|>` block,
/// with the surrounding tags stripped.
///
/// Only emitted when the model supports thinking (Gemma 4) and
/// thinking is enabled in the session config.
final class GemmaThinkingResponse extends GemmaResponse {
  final String content;
  const GemmaThinkingResponse(this.content);
}

/// Emitted by [AgentChat] after a skill has been dispatched and its result
/// received. Carries the full result including optional image and webview.
///
/// ```dart
/// await for (final resp in agent.run('What time is it?')) {
///   if (resp is GemmaToolResultResponse) {
///     print('${resp.skillName} → ${resp.textResult}');
///     if (resp.imageBase64 != null) showImage(resp.imageBase64!);
///   }
/// }
/// ```
final class GemmaToolResultResponse extends GemmaResponse {
  /// Name of the skill that was called.
  final String skillName;

  /// Text result fed back to the model.
  final String textResult;

  /// Optional base64-encoded PNG image returned by the skill.
  final String? imageBase64;

  /// Optional webview panel URL returned by the skill.
  final String? webviewUrl;

  /// Whether the skill completed without an error field.
  final bool succeeded;

  const GemmaToolResultResponse({
    required this.skillName,
    required this.textResult,
    this.imageBase64,
    this.webviewUrl,
    required this.succeeded,
  });
}

/// Emitted when generation is aborted by [GemmaChat.stop] or [AgentChat.stop].
final class GemmaCancelledResponse extends GemmaResponse {
  const GemmaCancelledResponse();
}

// ── Future API (defined, not yet emitted) ─────────────────────────────────────

/// A single function/tool call from the model.
///
/// **Not yet emitted.** Reserved for a future streaming tool-call API where
/// tool requests are surfaced to the caller rather than handled internally
/// by [AgentChat]. Use [AgentChat.run] for current tool-calling functionality.
final class GemmaFunctionCallResponse extends GemmaResponse {
  final String name;
  final Map<String, dynamic> args;
  const GemmaFunctionCallResponse({required this.name, required this.args});
}

/// Multiple function calls requested in parallel.
///
/// **Not yet emitted.** See [GemmaFunctionCallResponse] for details.
final class GemmaParallelFunctionCallResponse extends GemmaResponse {
  final List<GemmaFunctionCallResponse> calls;
  const GemmaParallelFunctionCallResponse(this.calls);
}
