import 'dart:async';
import 'package:flutter_local_gemma/gemma/gemma_response.dart';

class ThinkingParser {
  /// Filters a raw token stream into [GemmaResponse] chunks.
  /// Handles Gemma 4 (`<|channel>thought\n...<channel|>`) and `<think>` tags,
  /// preserving boundaries across partial token deliveries.
  static Stream<GemmaResponse> filterThinkingStream(Stream<String> originalStream) async* {
    const startMarker = '<|channel>thought';
    const endMarker = '<channel|>';
    const startTag = '<think>';
    const endTag = '</think>';

    bool insideThinking = false;
    String buffer = '';

    await for (final token in originalStream) {
      buffer += token;

      while (buffer.isNotEmpty) {
        if (insideThinking) {
          // Check for end of <|channel>thought
          int endIdx = buffer.indexOf(endMarker);
          if (endIdx >= 0) {
            final thinkingContent = buffer.substring(0, endIdx);
            if (thinkingContent.isNotEmpty) yield GemmaThinkingResponse(thinkingContent);
            buffer = buffer.substring(endIdx + endMarker.length);
            insideThinking = false;
            continue;
          }

          // Check for end of <think>
          endIdx = buffer.indexOf(endTag);
          if (endIdx >= 0) {
            final thinkingContent = buffer.substring(0, endIdx);
            if (thinkingContent.isNotEmpty) yield GemmaThinkingResponse(thinkingContent);
            buffer = buffer.substring(endIdx + endTag.length);
            insideThinking = false;
            continue;
          }

          // Check for partial end markers at the tail
          final partialEnd1 = _findPartialSuffix(buffer, endMarker);
          final partialEnd2 = _findPartialSuffix(buffer, endTag);
          final maxPartial = partialEnd1 > partialEnd2 ? partialEnd1 : partialEnd2;

          final safe = buffer.substring(0, buffer.length - maxPartial);
          if (safe.isNotEmpty) yield GemmaThinkingResponse(safe);
          buffer = buffer.substring(buffer.length - maxPartial);
          break; // Need more tokens to resolve the partial match
        } else {
          // Check for start of <|channel>thought
          int startIdx = buffer.indexOf(startMarker);
          if (startIdx >= 0) {
            final textBefore = buffer.substring(0, startIdx);
            if (textBefore.isNotEmpty) yield GemmaTextResponse(textBefore);
            buffer = buffer.substring(startIdx + startMarker.length);
            insideThinking = true;
            continue;
          }

          // Check for start of <think>
          startIdx = buffer.indexOf(startTag);
          if (startIdx >= 0) {
            final textBefore = buffer.substring(0, startIdx);
            if (textBefore.isNotEmpty) yield GemmaTextResponse(textBefore);
            buffer = buffer.substring(startIdx + startTag.length);
            insideThinking = true;
            continue;
          }

          // Check for partial start markers at the tail
          final partialStart1 = _findPartialSuffix(buffer, startMarker);
          final partialStart2 = _findPartialSuffix(buffer, startTag);
          final maxPartial = partialStart1 > partialStart2 ? partialStart1 : partialStart2;

          final safe = buffer.substring(0, buffer.length - maxPartial);
          if (safe.isNotEmpty) yield GemmaTextResponse(safe);
          buffer = buffer.substring(buffer.length - maxPartial);
          break; // Need more tokens to resolve the partial match
        }
      }
    }

    // Flush remaining buffer
    if (buffer.isNotEmpty) {
      if (insideThinking) {
        yield GemmaThinkingResponse(buffer);
      } else {
        yield GemmaTextResponse(buffer);
      }
    }
  }

  /// Returns length of the longest suffix of [text] that is a prefix of [marker].
  static int _findPartialSuffix(String text, String marker) {
    for (int i = marker.length.clamp(0, text.length); i >= 1; i--) {
      if (text.endsWith(marker.substring(0, i))) {
        return i;
      }
    }
    return 0;
  }
}
