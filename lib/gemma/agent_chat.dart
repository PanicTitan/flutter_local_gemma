import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_local_gemma/gemma/gemma.dart';
import 'package:flutter_local_gemma/gemma/thinking_parser.dart';
import 'package:flutter_local_gemma/utils/gemma_debug.dart';
import 'package:flutter_local_gemma/chat/tool_call_parser.dart';

import 'gemma_response.dart';
import 'gemma_skill.dart';
import '../chat/base_chat.dart';
import '../chat/gemma_chat.dart';
import 'gemma_web.dart' if (dart.library.io) 'gemma_stub.dart';

// ─── Agent System Prompt ──────────────────────────────────────────────────

const _kAgentSystemPrompt = '''
You are a helpful and friendly AI assistant.
You can help the user with a wide variety of tasks, answering questions, or just having a chat.
You have access to a set of skills/tools to help you.

1. If you need a skill to answer the user's request (e.g. checking the weather, fetching time), you MUST call it.
2. If no skill is needed (e.g. telling a joke, general conversation, or math that doesn't need a tool), you can simply provide your final answer directly!

AVAILABLE SKILLS:
___SKILLS___

To call a skill, output ONLY a <|tool_call|> block. For example:
<|tool_call>
{"name": "weather", "arguments": {"city": "London"}}
<tool_call|>
''';

// ─── AgentChat ───────────────────────────────────────────────────────────────

/// An agentic chat wrapper that enforces structured native token output.
///
/// Wraps [GemmaChat] with a multi-loop agent pattern:
///  1. Send user message → stream standard tokens
///  2. If model emits a tool call → dispatch to skill → feed result back
///  3. Repeat until model gives a final text answer or max loops reached
class AgentChat implements BaseChat {
  /// All registered skills (can be [GemmaDartSkill] or [GemmaJsSkill]).
  final List<Object> skills;

  /// Maximum number of tool-call → response loops before giving up.
  final int maxLoops;

  /// Session configuration for the underlying chat engine.
  late final SessionConfig config;

  /// Fraction of the context window that triggers an "approaching limit" warning.
  /// Default is 0.8 (80%). Passed through to [GemmaChat].
  final double contextFillThreshold;

  /// Timeout in milliseconds for JS skill execution. Default is 30000 (30s).
  /// Pass 0 to disable timeout.
  final int jsSkillTimeoutMs;

  /// Callback fired when a skill returns a result. Useful for displaying images
  /// without polluting the context history.
  final void Function(AgentToolCall)? onSkillResult;

  late final GemmaChat _chat;

  AgentChat({
    required this.skills,
    this.maxLoops = 10,
    this.jsSkillTimeoutMs = 30000,
    this.onSkillResult,
    SessionConfig? config,
    this.contextFillThreshold = 0.8,
  }) {
    final modelPath = (FlutterLocalGemma().currentModelPath ?? '').toLowerCase();
    final isGemma3 = modelPath.contains('gemma-3') || modelPath.contains('gemma_3') || modelPath.contains('gemma-2') || modelPath.contains('gemma_2');

    var baseConfig = config ?? SessionConfig();
    var nativeToolCalling = baseConfig.nativeToolCalling;
    var enableThinking = baseConfig.enableThinking;

    if (kIsWeb) {
      if (baseConfig.nativeToolCalling) {
        debugPrint('[AgentChat] Disabling nativeToolCalling because it relies on C++ LiteRT-LM APIs not exposed to Web WASM yet.');
      }
      nativeToolCalling = false;
    }

    if (isGemma3) {
      if (baseConfig.nativeToolCalling) {
        debugPrint('[AgentChat] Disabling nativeToolCalling because Gemma 3 relies on prompt-based / non-native tool calling mode.');
      }
      nativeToolCalling = false;
      
      if (baseConfig.enableThinking) {
        debugPrint('[AgentChat] Disabling enableThinking because Gemma 3 does not support thinking mode (only Gemma 4 does).');
      }
      enableThinking = false;
    }

    this.config = baseConfig.copyWith(
      nativeToolCalling: nativeToolCalling,
      enableThinking: enableThinking,
    );

    var effectivePrompt = this.config.systemPrompt ?? _kAgentSystemPrompt;

    if (this.config.systemPrompt == null) {
      if (this.config.nativeToolCalling == true && !kIsWeb) {
        final skillsSection = buildNativeSkillsSection(skills);
        effectivePrompt = effectivePrompt.replaceAll(
          '___SKILLS___',
          skillsSection,
        );
      } else {
        final skillsSection = buildPromptSkillsSection(skills);
        effectivePrompt = effectivePrompt.replaceAll(
          '___SKILLS___',
          skillsSection,
        );
      }
    } else {
      if (this.config.nativeToolCalling == true && !kIsWeb) {
        effectivePrompt += '\n\n' + buildNativeSkillsSection(skills);
      } else {
        effectivePrompt += '\n\n' + buildPromptSkillsSection(skills);
      }
    }

    if (this.config.enableThinking) {
      effectivePrompt = '<|think|>\n' + effectivePrompt;
    }

    _chat = GemmaChat(
      systemPrompt: effectivePrompt,
      config: this.config,
      contextFillThreshold: contextFillThreshold,
    );
  }

  /// Initialises the underlying chat engine.
  Future<void> init() async {
    await _chat.init();

    if (!kIsWeb) {
      final allSkills = skills
          .whereType<GemmaSkill>()
          .map(
            (s) => {
              'name': s.name,
              'description': s.description,
              'instructions': s is GemmaJsSkill ? s.instructions : '',
              'scriptHtml': s is GemmaJsSkill ? s.scriptHtml : null,
              'timeoutMs': jsSkillTimeoutMs,
            },
          )
          .toList();
      if (allSkills.isNotEmpty) {
        const ch = MethodChannel('flutter_local_gemma/js_skill');
        await ch.invokeMethod('registerSkills', {'skills': allSkills});
      }
    }

    if (config.nativeToolCalling == true && !kIsWeb) {
      FlutterLocalGemma().setNativeToolHandler(_handleNativeToolCall);
    }
  }

  Future<dynamic> _handleNativeToolCall(Map<String, dynamic> data) async {
    final skillName = data['skillName'] as String?;
    final toolArgs =
        (data['args'] as Map?)?.cast<String, dynamic>() ?? <String, dynamic>{};
    if (skillName == null) return {'error': 'Missing skillName'};
    final resultJson = await _dispatchSkill(skillName, toolArgs);
    return {'result': resultJson};
  }

  /// Releases all resources held by this agent.
  Future<void> dispose() async {
    if (config.nativeToolCalling == true && !kIsWeb) {
      FlutterLocalGemma().setNativeToolHandler(null);
    }
    await _chat.dispose();
  }

  /// Aborts any ongoing generation.
  ///
  /// Returns a [Future] that completes when the stop signal has been sent.
  Future<void> stop() async => _chat.stop();

  /// Clears conversation history and resets the context window.
  Future<void> clearHistory() async => _chat.clearHistory();

  /// The conversation history (read-only).
  List<ChatMessage> get history => _chat.history;

  /// Maximum context-window capacity (tokens).
  int get maxContextTokens => _chat.maxContextTokens;

  /// Estimated tokens consumed so far.
  int get currentTokenCount => _chat.currentTokenCount;

  /// Estimated tokens remaining before the context window is full.
  int get remainingTokens => _chat.remainingTokens;

  /// Whether the conversation is approaching the context limit (>= 80%).
  bool get isNearContextLimit => _chat.isNearContextLimit;

  /// A snapshot of the current token usage and capacity.
  TokenStats get tokenStats => _chat.tokenStats;

  @override
  Future<int> estimateNextTurnTokens(
    String prompt, {
    List<Uint8List>? images,
    List<Uint8List>? audios,
  }) {
    return _chat.estimateNextTurnTokens(prompt, images: images, audios: audios);
  }

  // ── History persistence ────────────────────────────────────────────────────

  /// Exports the conversation history to a JSON string.
  ///
  /// Binary payloads (images, audio) are base64-encoded.
  /// ```dart
  /// final json = await agent.exportHistory();
  /// await prefs.setString('agent_history', json);
  /// ```
  Future<String> exportHistory() => _chat.exportHistory();

  /// Imports a previously exported history string, replacing the current
  /// in-memory history.
  ///
  /// ```dart
  /// final saved = prefs.getString('agent_history') ?? '';
  /// if (saved.isNotEmpty) await agent.importHistory(saved);
  /// ```
  Future<void> importHistory(String json) => _chat.importHistory(json);

  /// Replaces the message at [index] with [msg] and rebuilds the context.
  Future<void> editHistory(int index, ChatMessage msg) =>
      _chat.editHistory(index, msg);

  /// Removes the message at [index] and rebuilds the context.
  @override
  Future<void> removeHistory(int index) async {
    await _chat.removeHistory(index);
  }

  @override
  Future<void> appendMessage(ChatMessage msg) async {
    await _chat.appendMessage(msg);
  }

  // ── Token tracking ───────────────────────────────────────────────────────────

  /// Creates an [AgentChat] with skills loaded from assets and/or URLs.
  ///
  /// ```dart
  /// final agent = await AgentChat.create(
  ///   assetSkills: ['assets/skills/calculator'],
  ///   urlSkills:   ['https://example.com/skills/qr-code'],
  ///   dartSkills:  [myDartSkill],
  /// );
  /// await agent.init();
  /// ```
  static Future<AgentChat> create({
    List<String> assetSkills = const [],
    List<String> urlSkills = const [],
    List<Object> dartSkills = const [],
    int maxLoops = 10,
    SessionConfig? config,
    double contextFillThreshold = 0.8,
  }) async {
    final loaded = <Object>[...dartSkills];
    for (final path in assetSkills) {
      loaded.add(await GemmaJsSkill.fromAsset(path));
    }
    for (final url in urlSkills) {
      loaded.add(await GemmaJsSkill.fromUrl(url));
    }
    return AgentChat(
      skills: loaded,
      maxLoops: maxLoops,
      config: config,
      contextFillThreshold: contextFillThreshold,
    );
  }

  // ── Main agent loop ────────────────────────────────────────────────────────

  Stream<AgentTurn> run(String message) async* {
    final bool isNativeMode = config.nativeToolCalling == true && !kIsWeb;

    String currentMessage = message;
    final thinkingBlocks = <String>[];
    final toolCalls = <AgentToolCall>[];
    final toolResultsForCallback = <AgentToolCall>[];
    bool isHidden = false;
    int loop = 0;

    // Prevent infinite loop from identical consecutive tool calls
    String? lastToolCallName;
    String? lastToolCallArgsStr;

    int maxToolLoops = maxLoops;
    int maxTotalLoops = maxToolLoops + 1; // 1 extra loop to force final answer

    for (; loop < maxTotalLoops; loop++) {
      if (loop == maxToolLoops && toolCalls.isNotEmpty) {
        currentMessage =
            'You have reached the maximum number of tool calls allowed. Please provide your final answer to the user based on the information you have so far. Do NOT output any more <|tool_call|> blocks.';
        isHidden = true;
      }

      GemmaDebug.log(
        GemmaDebug.tagAgent,
        'Agent loop $loop sending prompt:\n$currentMessage',
      );

      final rawStream = _chat.sendMessageStream(
        text: currentMessage,
        isHidden: isHidden,
        hideModelResponse: true,
      );
      final parsedStream = ThinkingParser.filterThinkingStream(rawStream);

      final textBuffer = StringBuffer();
      bool toolCalled = false;
      String currentThinking = '';

      await for (final response in parsedStream) {
        if (response is GemmaThinkingResponse) {
          currentThinking += response.content;

          final currentRawText = [
            ...thinkingBlocks.map((t) => '<|channel>thought\\n$t<channel|>'),
            if (currentThinking.isNotEmpty)
              '<|channel>thought\\n$currentThinking',
            textBuffer.toString(),
          ].where((s) => s.isNotEmpty).join('\\n');

          yield AgentTurn(
            userMessage: message,
            modelAnswer: textBuffer.toString(),
            rawText: currentRawText,
            thinkingBlocks: List.unmodifiable([
              ...thinkingBlocks,
              currentThinking,
            ]),
            toolCalls: List.unmodifiable(toolCalls),
            loopCount: loop,
            isComplete: false,
          );
        } else if (response is GemmaTextResponse) {
          if (currentThinking.isNotEmpty) {
            thinkingBlocks.add(currentThinking);
            currentThinking = '';
          }
          textBuffer.write(response.token);

          // Hide tool call wrappers from UI
          String displayAnswer = textBuffer.toString();
          final toolCallMatchUI = RegExp(
            r'<\|tool_call\|>|<tool_call\|>|</tool_call\|>|<\|tool\|>',
          ).firstMatch(displayAnswer);
          if (toolCallMatchUI != null) {
            displayAnswer = displayAnswer
                .substring(0, toolCallMatchUI.start)
                .trim();
          }

          if (kIsWeb) {
            final eotIdx = displayAnswer.indexOf('<end_of_turn>');
            if (eotIdx != -1) {
              displayAnswer = displayAnswer.substring(0, eotIdx);
            }
            displayAnswer = displayAnswer
                .replaceAll('<end_of_turn>', '')
                .replaceAll('[Stopped]', '')
                .trim();

            if (textBuffer.toString().contains('<end_of_turn>') ||
                textBuffer.toString().contains('<|tool_response>') ||
                textBuffer.toString().contains('<tool_call|>')) {
              await _chat.stop();
            }
          }

          final currentRawText = [
            ...thinkingBlocks.map((t) => '<|channel>thought\\n$t<channel|>'),
            textBuffer.toString(),
          ].where((s) => s.isNotEmpty).join('\\n');

          yield AgentTurn(
            userMessage: message,
            modelAnswer: displayAnswer,
            rawText: currentRawText,
            thinkingBlocks: List.unmodifiable(thinkingBlocks),
            toolCalls: List.unmodifiable(toolCalls),
            loopCount: loop,
            isComplete: false,
          );
        }
      }

      if (currentThinking.isNotEmpty) {
        thinkingBlocks.add(currentThinking);
      }

      String finalText = textBuffer.toString();
      if (kIsWeb) {
        final eotIndex = finalText.indexOf('<end_of_turn>');
        if (eotIndex != -1) {
          finalText = finalText.substring(0, eotIndex);
        }
        finalText = finalText
            .replaceAll('<end_of_turn>', '')
            .replaceAll('[Stopped]', '')
            .trim();
      }
      
      String combinedRawText = finalText;
      if (thinkingBlocks.isNotEmpty) {
        final thinkingString = thinkingBlocks
            .map((t) => '<|channel>thought\n$t<channel|>')
            .join();
        combinedRawText = thinkingString + '\n' + finalText;
      }

      // ── Tool-call detection ──────────────────────────────────────────────
      final toolResult = ToolCallParser.parse(finalText);

      if (toolResult != null && toolResult.name.isNotEmpty) {
        if (loop >= maxToolLoops) {
          GemmaDebug.log(
            GemmaDebug.tagAgent,
            'Max tool loops reached, ignoring tool call ${toolResult.name}',
          );

          // Strip the tool call syntax from the text
          final match = RegExp(
            r'<\|tool_call\|>|<tool_call\|>',
          ).firstMatch(finalText);
          if (match != null) {
            finalText = finalText.substring(0, match.start).trim();
          }

          combinedRawText = finalText;
          if (thinkingBlocks.isNotEmpty) {
            final thinkingString = thinkingBlocks
                .map((t) => '<|channel>thought\n$t<channel|>')
                .join();
            combinedRawText = thinkingString + '\n' + finalText;
          }

          // Force fallback to standard answer handling
          toolCalled = false;
        } else {
          toolCalled = true;

          // Strip hallucinated trailing output from history
          if (_chat.history.isNotEmpty) {
            final lastMsg = _chat.history.last;
            if (lastMsg.role == 'model') {
              if (kIsWeb) {
                final rawToolResult = ToolCallParser.parse(lastMsg.text);
                if (rawToolResult != null && lastMsg.text.length > rawToolResult.endIndex) {
                  await _chat.editHistory(
                    _chat.history.length - 1,
                    ChatMessage(
                      role: lastMsg.role,
                      text: lastMsg.text.substring(0, rawToolResult.endIndex),
                      images: lastMsg.images,
                      audios: lastMsg.audios,
                      isHidden: lastMsg.isHidden,
                      isUiOnly: lastMsg.isUiOnly,
                    ),
                    rebuildContext: !isNativeMode, // Don't rebuild native context to avoid destroying session
                  );
                }
              } else {
                // Preserve original behavior for Android non-native mode
                if (lastMsg.text.length > toolResult.endIndex) {
                  await _chat.editHistory(
                    _chat.history.length - 1,
                    ChatMessage(
                      role: lastMsg.role,
                      text: lastMsg.text.substring(0, toolResult.endIndex),
                      images: lastMsg.images,
                      audios: lastMsg.audios,
                      isHidden: lastMsg.isHidden,
                      isUiOnly: lastMsg.isUiOnly,
                    ),
                    rebuildContext: !isNativeMode,
                  );
                }
              }
            }
          }

          final argsStr = jsonEncode(toolResult.args);
          if (toolResult.name == lastToolCallName &&
              argsStr == lastToolCallArgsStr) {
            // Model is stuck in a loop trying to call the same tool with the same args. Break out.
            GemmaDebug.log(
              GemmaDebug.tagAgent,
              'Infinite tool loop detected for ${toolResult.name}. Breaking.',
            );
            break;
          }
          lastToolCallName = toolResult.name;
          lastToolCallArgsStr = argsStr;

          final resultJson = await _dispatchSkill(
            toolResult.name,
            toolResult.args,
          );

          // Strip large base64 strings before feeding back to model context
          String contextResultJson = resultJson;
          try {
            final decoded = jsonDecode(resultJson);
            if (decoded is Map &&
                decoded['image'] is Map &&
                decoded['image']['base64'] != null) {
              decoded['image']['base64'] =
                  '[IMAGE_GENERATED_SUCCESSFULLY_AND_DISPLAYED_TO_USER]';
              contextResultJson = jsonEncode(decoded);
            }
          } catch (_) {}

          toolCalls.add(
            AgentToolCall(
              skillName: toolResult.name,
              args: toolResult.args,
              resultJson: contextResultJson, // Strip from UI to prevent freeze
            ),
          );

          toolResultsForCallback.add(
            AgentToolCall(
              skillName: toolResult.name,
              args: toolResult.args,
              resultJson: resultJson, // Full JSON for callback
            ),
          );

          if (isNativeMode) {
            currentMessage =
                'Tool execution for skill "${toolResult.name}" returned:\n$contextResultJson\n\nProvide the final answer. Remember: to use another skill, you MUST call the `execute_skill` tool.';
          } else {
            currentMessage =
                'Tool ${toolResult.name} executed successfully. Result:\n$contextResultJson\n\nProvide the final answer based on the tool result. Do NOT output any more <|tool_call|> blocks.';
          }
          isHidden = true;
          if (kIsWeb) await Future.delayed(const Duration(milliseconds: 500));
          continue; // next agent loop
        }
      } // End of tool call block

      // No tool call — this is the final answer
      if (!toolCalled) {
        bool isComplete = true;
        final turn = AgentTurn(
          userMessage: message,
          modelAnswer: finalText,
          rawText: combinedRawText,
          thinkingBlocks: List.unmodifiable(thinkingBlocks),
          toolCalls: List.unmodifiable(toolCalls),
          loopCount: loop,
          isComplete: isComplete,
        );

        // Inject the combined UI message into history
        final turnJson = jsonEncode(turn.toJson());

        await _chat.appendMessage(
          ChatMessage(
            role: 'model',
            text: turnJson,
            isHidden: false,
            isUiOnly: true,
          ),
        );

        // Fire callbacks after appending the final answer so images appear below it
        for (final call in toolResultsForCallback) {
          onSkillResult?.call(call);
        }

        yield turn;
        return;
      }
    }

    // Fallback if loop exits without a final answer
    currentMessage =
        'You have reached the maximum number of tool calls allowed ($maxLoops). You must provide a final answer now without using any more tools.';
    isHidden = true;

    // Do one final loop for the answer
    final rawStream = _chat.sendMessageStream(
      text: currentMessage,
      isHidden: isHidden,
      hideModelResponse: true,
    );

    String finalText = '';
    await for (final response in rawStream) {
      finalText += response;
    }
    if (kIsWeb) {
      finalText = finalText
          .replaceAll('<end_of_turn>', '')
          .replaceAll('[Stopped]', '')
          .trim();
    }

    final turn = AgentTurn(
      userMessage: message,
      modelAnswer: finalText,
      rawText: finalText,
      thinkingBlocks: List.unmodifiable(thinkingBlocks),
      toolCalls: List.unmodifiable(toolCalls),
      loopCount: maxLoops,
      isComplete: true,
    );

    final turnJson = jsonEncode({
      'is_agent_turn': true,
      'thinkingBlocks': thinkingBlocks,
      'toolCalls': toolCalls
          .map(
            (t) => {
              'skillName': t.skillName,
              'args': t.args,
              'resultJson': t.resultJson,
            },
          )
          .toList(),
      'modelAnswer': finalText,
    });

    await _chat.appendMessage(
      ChatMessage(
        role: 'model',
        text: turnJson,
        isHidden: false,
        isUiOnly: true,
      ),
    );

    // Fire callbacks after appending the final answer so images appear below it
    for (final call in toolResultsForCallback) {
      onSkillResult?.call(call);
    }

    yield turn;
  }

  // ── Skill dispatch ────────────────────────────────────────────────────────

  Future<String> _dispatchSkill(String name, Map<String, dynamic> args) async {
    GemmaDebug.log(
      GemmaDebug.tagAgent,
      'Executing tool call: $name with args: $args',
    );

    String? result;

    // Try Dart skills first
    for (final s in skills) {
      if (s is GemmaDartSkill && s.name == name) {
        try {
          final res = await s.handler(args);
          result = jsonEncode(res.toGalleryJson());
          break;
        } catch (e) {
          result = jsonEncode({'error': e.toString(), 'status': 'failed'});
          break;
        }
      }
    }

    // Try JS skills if not yet handled
    if (result == null) {
      for (final s in skills) {
        if (s is GemmaJsSkill && s.name == name) {
          result = await _executeJsSkill(s, args);
          break;
        }
      }
    }

    if (result == null) {
      result = jsonEncode({
        'error': 'Skill not found or unsupported type',
        'status': 'failed',
      });
    }

    // Safely log the result without blowing up memory with huge base64 strings
    String strippedResultForLogs = result;
    if (strippedResultForLogs.length > 500) {
      strippedResultForLogs =
          strippedResultForLogs.substring(0, 500) + '... [TRUNCATED FOR LOGS]';
    }
    GemmaDebug.log(
      GemmaDebug.tagAgent,
      'Tool $name result: $strippedResultForLogs',
    );

    return result;
  }

  Future<String> _executeJsSkill(
    GemmaJsSkill skill,
    Map<String, dynamic> args,
  ) async {
    final String argsJsonStr = jsonEncode(args);
    final String secretStr = skill.secret ?? '';

    if (kIsWeb) {
      try {
        return await _executeJsSkillWeb(
          skill.scriptHtml,
          argsJsonStr,
          secretStr,
          jsSkillTimeoutMs > 0 ? jsSkillTimeoutMs : 999999999,
        );
      } catch (e) {
        GemmaDebug.logError(
          GemmaDebug.tagSkills,
          'JS skill web execution error',
          e,
        );
        return jsonEncode({'error': e.toString(), 'result': null});
      }
    } else {
      // Android: execute via MethodChannel → JsSkillExecutor.kt (method: "executeJsSkill")
      const ch = MethodChannel('flutter_local_gemma/js_skill');
      try {
        final result = await ch.invokeMethod<String>('executeJsSkill', {
          'skillName': skill.name,
          'scriptName': 'index.html',
          'argsJson': argsJsonStr,
          'secret': secretStr,
          'timeoutMs': jsSkillTimeoutMs,
        });
        return result ??
            jsonEncode({'error': 'Null result from plugin', 'result': null});
      } catch (e) {
        GemmaDebug.logError(GemmaDebug.tagSkills, 'JS skill native error', e);
        return jsonEncode({'error': e.toString(), 'result': null});
      }
    }
  }
}

// This top-level helper avoids requiring `dart:js_interop` in files that don't need it.
Future<String> _executeJsSkillWeb(
  String scriptHtml,
  String argsJson,
  String secret,
  int timeoutMs,
) async {
  if (kIsWeb) {
    return await FlutterLocalGemmaWeb().executeJsSkill(
      scriptHtml,
      argsJson,
      secret,
      timeoutMs,
    );
  }
  throw UnsupportedError('Only available on Web');
}

// ─── Supporting types ─────────────────────────────────────────────────────────

/// A snapshot of the agent's progress during a single user turn.
class AgentTurn {
  final String userMessage;
  final String modelAnswer;
  final String rawText;
  final List<String> thinkingBlocks;
  final List<AgentToolCall> toolCalls;
  final int loopCount;
  final bool isComplete;

  const AgentTurn({
    required this.userMessage,
    required this.modelAnswer,
    required this.rawText,
    required this.thinkingBlocks,
    required this.toolCalls,
    required this.loopCount,
    required this.isComplete,
  });

  /// Serializes this turn into a JSON map (used by ChatHistoryMixin).
  Map<String, dynamic> toJson() => {
    'is_agent_turn': true,
    'userMessage': userMessage,
    'modelAnswer': modelAnswer,
    'rawText': rawText,
    'thinkingBlocks': thinkingBlocks,
    'toolCalls': toolCalls.map((t) => t.toJson()).toList(),
    'loopCount': loopCount,
    'isComplete': isComplete,
  };

  /// Restores an AgentTurn from a JSON map (e.g. from exportHistory()).
  factory AgentTurn.fromJson(Map<String, dynamic> json) {
    return AgentTurn(
      userMessage: json['userMessage'] as String? ?? '',
      modelAnswer: json['modelAnswer'] as String? ?? '',
      rawText: json['rawText'] as String? ?? '',
      thinkingBlocks:
          (json['thinkingBlocks'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      toolCalls:
          (json['toolCalls'] as List<dynamic>?)
              ?.map((t) => AgentToolCall.fromJson(t as Map<String, dynamic>))
              .toList() ??
          [],
      loopCount: json['loopCount'] as int? ?? 1,
      isComplete: json['isComplete'] as bool? ?? true,
    );
  }

  @override
  String toString() =>
      'AgentTurn(loop=\$loopCount, complete=\$isComplete, '
      'thinking=\${thinkingBlocks.length}, tools=\${toolCalls.length}, '
      'answer=\${modelAnswer.length} chars)';
}

/// Record of a single tool call dispatched during an agent turn.
class AgentToolCall {
  final String skillName;
  final Map<String, dynamic> args;
  final String resultJson;

  const AgentToolCall({
    required this.skillName,
    required this.args,
    required this.resultJson,
  });

  /// Serializes this tool call into a JSON map.
  Map<String, dynamic> toJson() => {
    'skillName': skillName,
    'args': args,
    'resultJson': resultJson,
  };

  /// Restores an AgentToolCall from a JSON map.
  factory AgentToolCall.fromJson(Map<String, dynamic> json) {
    return AgentToolCall(
      skillName: json['skillName'] as String? ?? '',
      args: (json['args'] as Map?)?.cast<String, dynamic>() ?? {},
      resultJson: json['resultJson'] as String? ?? '',
    );
  }

  bool get succeeded =>
      !resultJson.contains('"error":') ||
      resultJson.contains('"status":"succeeded"');

  @override
  String toString() => 'AgentToolCall(\$skillName, succeeded=\$succeeded)';
}