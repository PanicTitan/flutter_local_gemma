import 'dart:convert';

/// Parses Gemma tool-call tokens from raw model output.
///
/// Supports three formats emitted by different Gemma variants and modes:
///
/// **Format 1 — JSON (Prompt mode / Web fallback)**
/// ```
/// <|tool_call|>{"name":"weather","arguments":{"city":"London"}}<tool_call|>
/// ```
///
/// **Format 2 — Native call: format (Gemma 4 native constrained decoding)**
/// ```
/// <|tool_call|>call:weather{"city":"London"}<tool_call|>
/// ```
/// or
/// ```
/// call:weather{data={"city":"London"}}
/// ```
///
/// **Format 3 — Hybrid fallback**
/// ```
/// call:{"name":"weather","arguments":{"city":"London"}}
/// ```
///
/// Usage:
/// ```dart
/// final result = ToolCallParser.parse(modelOutput);
/// if (result != null) {
///   print(result.name);       // "weather"
///   print(result.args);       // {"city": "London"}
///   print(result.endIndex);   // character position after the tool call
///   print(result.isNative);   // true if Format 2 was matched
/// }
/// ```
class ToolCallParser {
  ToolCallParser._();

  // ── Compiled patterns ─────────────────────────────────────────────────────
  //
  // IMPORTANT: Do NOT use \\|? inside these patterns.  In Dart's RE2-based
  // engine a backslash is not a quantifiable atom, so `\\|?` triggers:
  //   FormatException: Nothing to repeat (?:<\|?tool…)
  //
  // Gemma 4 emits three delimiter styles that we must match:
  //   <|tool_call|>  —  standard JSON prompt-injection start
  //   <tool_call|>   —  closing token variant
  //   <|tool|>       —  short alias used in some fine-tunes
  //
  // We match them explicitly via alternation rather than optional backslashes.

  // Matches: [optional open-delim]  {"name":"…","arguments":{…}}  [optional close-delim | EOL]
  static final _jsonPattern = RegExp(
    r'(?:<\|tool_call\|>|<\|tool\|>|<tool_call\|>)?\s*'
    r'(\{[^{}]*"name"\s*:\s*"[^"]*"(?:[^{}]|\{[^{}]*\})*\})'
    r'\s*(?:</tool_call\|>|<tool_call\|>|<\|tool\|>|<\|tool_call\|>|$)',
    dotAll: true,
  );

  // Matches: [optional open-delim] call:SKILL_NAME {args…} [optional close-delim | EOL]
  static final _nativePattern = RegExp(
    r'(?:(?:<\|tool_call\|>|<\|tool\|>|<tool_call\|>)\s*(?:call:)?|call:\s*)'
    r'([a-zA-Z0-9_-]+)\s*(.*?)\s*'
    r'(?:</tool_call\|>|<tool_call\|>|<\|tool\|>|<\|tool_call\|>|$)',
    dotAll: true,
  );

  static final _hybridPattern = RegExp(
    r'call:\s*(\{.*?\})',
    dotAll: true,
  );

  static final _fallbackJsonPattern = RegExp(r'(\{.*\})', dotAll: true);

  // ── Public API ─────────────────────────────────────────────────────────────

  /// Attempts to parse a tool call from [text].
  ///
  /// Returns a [ToolCallResult] if a tool call is found, or `null` if [text]
  /// contains no recognisable tool-call pattern.
  static ToolCallResult? parse(String text) {
    // 1. Standard JSON format
    final jsonMatch = _jsonPattern.firstMatch(text);
    if (jsonMatch != null) {
      try {
        final parsed = jsonDecode(jsonMatch.group(1)!.trim())
            as Map<String, dynamic>;
        return ToolCallResult._fromParsed(parsed, jsonMatch.end, false);
      } catch (_) {}
    }

    // 2. Native call:NAME{ARGS} format
    final nativeMatch = _nativePattern.firstMatch(text);
    if (nativeMatch != null) {
      final name = nativeMatch.group(1)!.trim();
      var argsStr = nativeMatch.group(2)!.trim();
      final args = _parseNativeArgs(argsStr);
      return ToolCallResult(
        name: name,
        args: args,
        endIndex: nativeMatch.end,
        isNative: true,
      );
    }

    // 3. Hybrid fallback: call:{...}
    final hybridMatch = _hybridPattern.firstMatch(text);
    if (hybridMatch != null) {
      try {
        final parsed = jsonDecode(hybridMatch.group(1)!) as Map<String, dynamic>;
        return ToolCallResult._fromParsed(parsed, hybridMatch.end, false);
      } catch (_) {}
    }

    return null;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  static Map<String, dynamic> _parseNativeArgs(String argsStr) {
    if (argsStr.isEmpty) return {};

    // Normalise Gemma's custom quote escaping <|"|>
    argsStr = argsStr.replaceAll('<|"|>', '"');

    // Normalise {data=VALUE} and data=VALUE forms to standard JSON
    if (argsStr.startsWith('{data=') && argsStr.endsWith('}')) {
      argsStr = '{"data": ${argsStr.substring(6, argsStr.length - 1)}}';
    } else if (argsStr.startsWith('data=')) {
      argsStr = '{"data": ${argsStr.substring(5)}}';
    } else if (argsStr.startsWith('{data:') && argsStr.endsWith('}')) {
      argsStr = '{"data": ${argsStr.substring(6, argsStr.length - 1)}}';
    } else if (argsStr.startsWith('data:')) {
      argsStr = '{"data": ${argsStr.substring(5)}}';
    } else if (!argsStr.startsWith('{')) {
      argsStr = '{$argsStr}';
    }

    try {
      return jsonDecode(argsStr) as Map<String, dynamic>;
    } catch (_) {
      // Try to fix unquoted keys (e.g. {a:10,b:10} -> {"a":10,"b":10})
      try {
        final fixedArgsStr = argsStr.replaceAllMapped(
            RegExp(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:'),
            (m) => '${m.group(1)}"${m.group(2)}":');
        return jsonDecode(fixedArgsStr) as Map<String, dynamic>;
      } catch (_) {}

      // Last resort: grab the first {...} substring
      final fallback = _fallbackJsonPattern.firstMatch(argsStr);
      if (fallback != null) {
        try {
          return jsonDecode(fallback.group(1)!) as Map<String, dynamic>;
        } catch (_) {}
      }
      return {};
    }
  }
}

/// The result of a successful [ToolCallParser.parse] call.
class ToolCallResult {
  /// Skill name extracted from the tool-call token.
  final String name;

  /// Parsed arguments. Empty map if the call had no arguments.
  final Map<String, dynamic> args;

  /// Character position immediately after the tool-call token in the original
  /// text. Useful for stripping the token from history or logging.
  final int endIndex;

  /// True when the call was in Gemma 4 native constrained-decoding format
  /// (`call:NAME{ARGS}`) rather than the JSON prompt-injection format.
  final bool isNative;

  const ToolCallResult({
    required this.name,
    required this.args,
    required this.endIndex,
    required this.isNative,
  });

  factory ToolCallResult._fromParsed(
    Map<String, dynamic> parsed,
    int endIndex,
    bool isNative,
  ) {
    final name = parsed['name'] as String? ??
        parsed['function'] as String? ??
        '';

    final rawArgs = parsed['arguments'] ?? parsed['parameters'];
    Map<String, dynamic> args = {};
    if (rawArgs is Map) {
      args = rawArgs.map((k, v) => MapEntry(k.toString(), v));
    }

    // Unwrap nested stringified 'data' argument (common in JS skills)
    if (args.length == 1 && args.containsKey('data') && args['data'] is String) {
      try {
        final nested = jsonDecode(args['data'] as String);
        if (nested is Map<String, dynamic>) args = nested;
      } catch (_) {}
    }

    return ToolCallResult(
      name: name,
      args: args,
      endIndex: endIndex,
      isNative: isNative,
    );
  }

  @override
  String toString() =>
      'ToolCallResult(name=$name, native=$isNative, args=${args.length} keys)';
}