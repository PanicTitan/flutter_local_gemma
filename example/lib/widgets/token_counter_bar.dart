import 'package:flutter/material.dart';

/// A thin coloured bar + label showing how much of the context window is used.
class TokenCounterBar extends StatelessWidget {
  final int usedTokens;
  final int maxTokens;
  final int pendingTokens;

  const TokenCounterBar({
    super.key,
    required this.usedTokens,
    required this.maxTokens,
    this.pendingTokens = 0,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final total   = usedTokens + pendingTokens;
    final ratio   = maxTokens > 0 ? (total / maxTokens).clamp(0.0, 1.0) : 0.0;
    final warn    = ratio > 0.8;
    final barColor = warn ? cs.error : cs.primary;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: ratio,
            minHeight: 3,
            backgroundColor: cs.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation(barColor),
          ),
        ),
        const SizedBox(height: 2),
        Text(
          'Tokens: $usedTokens history${pendingTokens > 0 ? " + $pendingTokens input" : ""} / $maxTokens',
          style: TextStyle(
            fontSize: 10,
            color: warn ? cs.error : cs.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}