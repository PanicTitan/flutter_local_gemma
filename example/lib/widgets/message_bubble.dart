import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// A single chat message bubble used by [ChatScreen] and [SmartChatScreen].
class MessageBubble extends StatelessWidget {
  final ChatMessage message;
  final bool highlight;
  final VoidCallback? onEdit;
  final VoidCallback? onDelete;

  const MessageBubble(
    this.message, {
    super.key,
    this.highlight = false,
    this.onEdit,
    this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final isTool  = message.role == 'tool';
    final isUser  = message.role == 'user';
    final cs      = Theme.of(context).colorScheme;

    final Color bg = highlight
        ? cs.tertiaryContainer
        : isUser
            ? cs.primaryContainer
            : cs.surfaceContainerHighest;

    final Color fg = highlight
        ? cs.onTertiaryContainer
        : isUser
            ? cs.onPrimaryContainer
            : cs.onSurfaceVariant;

    if (isTool) {
      String toolName = 'Tool';
      try {
        final decoded = jsonDecode(message.text);
        if (decoded is Map && decoded['tool_name'] != null) {
          toolName = decoded['tool_name'].toString();
        }
      } catch (_) {}

      return Center(
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.build_circle, size: 16, color: cs.onSurfaceVariant),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  'Tool executed: $toolName',
                  style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(16).copyWith(
            bottomRight: isUser ? Radius.zero : null,
            bottomLeft:  isUser ? null : Radius.zero,
          ),
          border: highlight ? Border.all(color: cs.tertiary, width: 1.5) : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Image attachments
            if (message.images.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: message.images
                      .map((b) => ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: Image.memory(b, height: 120, fit: BoxFit.cover),
                          ))
                      .toList(),
                ),
              ),

            // Audio indicator
            if (message.audios.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.audiotrack, size: 16, color: fg),
                  const SizedBox(width: 4),
                  Text('${message.audios.length} audio clip(s)', style: TextStyle(fontSize: 12, color: fg)),
                ]),
              ),

            // Message text
            Builder(builder: (context) {
              final text = message.text;
              final parts = GemmaTextParser.parse(text);

              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: parts.map((part) {
                  if (part is TextBlock) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: SelectableText(part.content, style: TextStyle(color: fg)),
                    );
                  } else if (part is ThinkingBlock) {
                    return Theme(
                      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                      child: Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: ExpansionTile(
                          collapsedBackgroundColor: cs.onSurface.withValues(alpha: 0.05),
                          backgroundColor: cs.onSurface.withValues(alpha: 0.05),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8), side: BorderSide(color: cs.onSurface.withValues(alpha: 0.1))),
                          collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8), side: BorderSide(color: cs.onSurface.withValues(alpha: 0.1))),
                          tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                          leading: Icon(Icons.psychology_outlined, size: 20, color: fg.withValues(alpha: 0.7)),
                          title: Text('Thinking Process', style: TextStyle(color: fg.withValues(alpha: 0.8), fontStyle: FontStyle.italic, fontSize: 13)),
                          children: [
                            Align(
                              alignment: Alignment.centerLeft,
                              child: SelectableText(
                                part.content.isEmpty ? 'Thinking...' : part.content,
                                style: TextStyle(color: fg.withValues(alpha: 0.8), fontStyle: FontStyle.italic, fontSize: 13),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  } else if (part is ToolBlock) {
                    return Theme(
                      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                      child: Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: ExpansionTile(
                          collapsedBackgroundColor: cs.secondaryContainer,
                          backgroundColor: cs.secondaryContainer,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                          leading: Icon(Icons.build_circle_outlined, size: 20, color: cs.onSecondaryContainer),
                          title: Text('Tool Call Executed', style: TextStyle(color: cs.onSecondaryContainer, fontWeight: FontWeight.w500, fontSize: 13)),
                          children: [
                            Align(
                              alignment: Alignment.centerLeft,
                              child: SelectableText(
                                part.content,
                                style: TextStyle(color: cs.onSecondaryContainer.withValues(alpha: 0.8), fontFamily: 'monospace', fontSize: 12),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }
                  return const SizedBox();
                }).toList(),
              );
            }),

            // Edit / delete actions
            if (onEdit != null || onDelete != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (onEdit != null)
                      _SmallIconButton(icon: Icons.edit_outlined, color: fg, onTap: onEdit!),
                    if (onDelete != null)
                      _SmallIconButton(icon: Icons.delete_outline, color: fg, onTap: onDelete!),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SmallIconButton extends StatelessWidget {
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _SmallIconButton({required this.icon, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) => InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(4),
        child: Padding(
          padding: const EdgeInsets.all(4),
          child: Icon(icon, size: 15, color: color.withValues(alpha: 0.7)),
        ),
      );
}

class AgentTurnBubble extends StatelessWidget {
  final AgentTurn turn;

  const AgentTurnBubble(this.turn, {super.key});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final fg = cs.onSurfaceVariant;
   
    // Merge ALL thinking blocks into one string to avoid the duplicate-bubble
    // problem that occurs when the agent loop accumulates blocks across iterations.
    final allThinking = turn.thinkingBlocks.join('\n\n---\n\n').trim();

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.85),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(16).copyWith(bottomLeft: Radius.zero),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ONE thinking bubble (merged from all blocks)
          if (allThinking.isNotEmpty)
            Theme(
              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: ExpansionTile(
                  collapsedBackgroundColor: cs.onSurface.withValues(alpha: 0.05),
                  backgroundColor: cs.onSurface.withValues(alpha: 0.05),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8), side: BorderSide(color: cs.onSurface.withValues(alpha: 0.1))),
                  collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8), side: BorderSide(color: cs.onSurface.withValues(alpha: 0.1))),
                  tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                  childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                  leading: Icon(Icons.psychology_outlined, size: 20, color: fg.withValues(alpha: 0.7)),
                  title: Text(
                    turn.thinkingBlocks.length > 1
                        ? 'Thinking Process (${turn.thinkingBlocks.length} steps)'
                        : 'Thinking Process',
                    style: TextStyle(color: fg.withValues(alpha: 0.8), fontStyle: FontStyle.italic, fontSize: 13),
                  ),
                  children: [
                    Align(
                      alignment: Alignment.centerLeft,
                      child: SelectableText(
                        allThinking.isEmpty ? 'Thinking...' : allThinking,
                        style: TextStyle(color: fg.withValues(alpha: 0.8), fontStyle: FontStyle.italic, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
            ),
         
          // Render tool calls (one chip per call — correct)
          ...turn.toolCalls.map((call) => Theme(
            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
            child: Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: ExpansionTile(
                collapsedBackgroundColor: cs.secondaryContainer,
                backgroundColor: cs.secondaryContainer,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                leading: Icon(Icons.build_circle_outlined, size: 20, color: cs.onSecondaryContainer),
                title: Row(
                  children: [
                    Text('Used skill: ${call.skillName}', style: TextStyle(color: cs.onSecondaryContainer, fontWeight: FontWeight.bold, fontSize: 13)),
                    const Spacer(),
                    Icon(call.succeeded ? Icons.check_circle : Icons.error, size: 16, color: call.succeeded ? Colors.green : Colors.red),
                  ],
                ),
                children: [
                  if (call.args.isNotEmpty)
                    Align(
                      alignment: Alignment.centerLeft,
                      child: SelectableText(
                        'Args: ${call.args}',
                        style: TextStyle(color: cs.onSecondaryContainer.withValues(alpha: 0.8), fontSize: 11, fontFamily: 'monospace')
                      ),
                    ),
                  if (call.resultJson.isNotEmpty)
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Padding(
                        padding: const EdgeInsets.only(top: 4.0),
                        child: SelectableText(
                          'Result: ${call.resultJson}',
                          style: TextStyle(color: cs.onSecondaryContainer.withValues(alpha: 0.8), fontSize: 11, fontFamily: 'monospace')
                        ),
                      ),
                    ),
                ],
              ),
            ),
          )),

          // Render final answer or loading state
          if (turn.modelAnswer.isNotEmpty)
            SelectableText(turn.modelAnswer, style: TextStyle(color: fg))
          else if (!turn.isComplete)
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2, color: fg.withValues(alpha: 0.7))),
                const SizedBox(width: 8),
                Text('Agent working...', style: TextStyle(color: fg.withValues(alpha: 0.7), fontStyle: FontStyle.italic)),
              ],
            ),
        ],
      ),
    );
  }
}