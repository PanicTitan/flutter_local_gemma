import 'package:flutter/foundation.dart';

/// Base class for all blocks parsed from a raw Gemma response string.
sealed class GemmaContentBlock {
  final String content;
  const GemmaContentBlock(this.content);
}

/// A standard text segment intended to be displayed directly to the user.
class TextBlock extends GemmaContentBlock {
  const TextBlock(super.content);
  @override
  String toString() => 'TextBlock(${content.length} chars)';
}

/// A block representing the model's internal thinking process or chain of thought.
class ThinkingBlock extends GemmaContentBlock {
  const ThinkingBlock(super.content);
  @override
  String toString() => 'ThinkingBlock(${content.length} chars)';
}

/// A block containing a tool call invocation or tool syntax.
class ToolBlock extends GemmaContentBlock {
  const ToolBlock(super.content);
  @override
  String toString() => 'ToolBlock(${content.length} chars)';
}

/// A lightweight Abstract Syntax Tree (AST) parser for Gemma chat output.
///
/// Converts a raw text string containing `<think>`, `<|tool_call|>`, and
/// `<|channel>thought` tags into a structured list of blocks for easy UI rendering.
class GemmaTextParser {
  /// Parses raw text into a sequence of [GemmaContentBlock]s.
  ///
  /// Useful for chat UIs that want to wrap thinking blocks in an `ExpansionTile`
  /// or hide raw tool-calling JSON from the user, without having to write
  /// custom regular expressions.
  static List<GemmaContentBlock> parse(String text) {
    if (text.isEmpty) return [];

    final parts = <GemmaContentBlock>[];
    int currentIndex = 0;

    while (currentIndex < text.length) {
      // Find the next <think> or <|tool_call|> start tag
      final thinkStartMatch = RegExp(r'(?:<(?:think|thought|\s?思维步骤\d+)>|<\|channel>thought\n)')
          .firstMatch(text.substring(currentIndex));
      final toolStartMatch = RegExp(r'<\|?tool(?:_call)?\|?>')
          .firstMatch(text.substring(currentIndex));

      Match? firstMatch;
      bool isThink = false;

      if (thinkStartMatch != null && toolStartMatch != null) {
        if (thinkStartMatch.start < toolStartMatch.start) {
          firstMatch = thinkStartMatch;
          isThink = true;
        } else {
          firstMatch = toolStartMatch;
          isThink = false;
        }
      } else if (thinkStartMatch != null) {
        firstMatch = thinkStartMatch;
        isThink = true;
      } else if (toolStartMatch != null) {
        firstMatch = toolStartMatch;
        isThink = false;
      }

      // No more tags found, add the remainder as text
      if (firstMatch == null) {
        final remainder = text.substring(currentIndex).trim();
        if (remainder.isNotEmpty) {
          parts.add(TextBlock(remainder));
        }
        break;
      }

      // Add the preceding text as a TextBlock
      final before = text.substring(currentIndex, currentIndex + firstMatch.start).trim();
      if (before.isNotEmpty) {
        parts.add(TextBlock(before));
      }

      // Advance index past the opening tag
      currentIndex += firstMatch.start + firstMatch.group(0)!.length;

      if (isThink) {
        // Find the matching end tag for thinking
        final endMatch = RegExp(r'(?:</(?:think|thought|\s?思维步骤\d+)>|<channel\|>)')
            .firstMatch(text.substring(currentIndex));
        
        if (endMatch != null) {
          final content = text.substring(currentIndex, currentIndex + endMatch.start).trim();
          parts.add(ThinkingBlock(content));
          currentIndex += endMatch.start + endMatch.group(0)!.length;
        } else {
          // Unclosed tag, treat remainder as thinking block
          final content = text.substring(currentIndex).trim();
          parts.add(ThinkingBlock(content));
          currentIndex = text.length;
        }
      } else {
        // Parse the tool block
        // Capture everything until the closing tag, or end of string
        final toolMatch = RegExp(r'<\|?tool(?:_call)?\|?>([\s\S]*?)(?:</?\|?tool(?:_call)?\|?>|<tool\|>|$)')
            .firstMatch(text.substring(currentIndex - firstMatch.group(0)!.length));
        
        if (toolMatch != null) {
          final fullMatch = toolMatch.group(0)!;
          final content = toolMatch.group(1)!.trim();
          parts.add(ToolBlock(content));
          
          // Advance index past the whole matched block (including the opening tag we backtracked for)
          currentIndex = (currentIndex - firstMatch.group(0)!.length) + toolMatch.start + fullMatch.length;
        } else {
          // Fallback if the regex fails for some reason
          final content = text.substring(currentIndex).trim();
          parts.add(ToolBlock(content));
          currentIndex = text.length;
        }
      }
    }

    return parts;
  }
}