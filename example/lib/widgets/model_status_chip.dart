import 'package:flutter/material.dart';
import '../app_state.dart';

/// A small chip showing the current [ModelStatus] with an icon and colour.
class ModelStatusChip extends StatelessWidget {
  final ModelStatus status;
  final String? errorText;

  const ModelStatusChip(this.status, {super.key, this.errorText});

  @override
  Widget build(BuildContext context) {
    final (icon, color) = _iconAndColor(context, status);
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: Chip(
        key: ValueKey(status),
        avatar: status.isBusy
            ? SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2, color: color),
              )
            : Icon(icon, size: 16, color: color),
        label: Text(
          errorText != null && status == ModelStatus.error
              ? 'Error'
              : status.label,
          style: TextStyle(fontSize: 12, color: color),
        ),
        backgroundColor:
            color.withValues(alpha: 0.12),
        side: BorderSide(color: color.withValues(alpha: 0.3)),
        padding: const EdgeInsets.symmetric(horizontal: 4),
        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
        visualDensity: VisualDensity.compact,
      ),
    );
  }

  (IconData, Color) _iconAndColor(BuildContext ctx, ModelStatus s) {
    final cs = Theme.of(ctx).colorScheme;
    return switch (s) {
      ModelStatus.unloaded    => (Icons.power_off_outlined, cs.outline),
      ModelStatus.downloading => (Icons.download_outlined, cs.primary),
      ModelStatus.loading     => (Icons.memory_outlined, cs.tertiary),
      ModelStatus.ready       => (Icons.check_circle_outline, Colors.green),
      ModelStatus.generating  => (Icons.stream, cs.primary),
      ModelStatus.rebooting   => (Icons.restart_alt, cs.tertiary),
      ModelStatus.error       => (Icons.error_outline, cs.error),
    };
  }
}