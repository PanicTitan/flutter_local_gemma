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
            SelectableText(message.text, style: TextStyle(color: fg)),

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