import '../gemma/gemma.dart';
import 'gemma_debug.dart';

/// Enforces the multimodality mutex resolution order at session initialization.
///
/// Resolution order:
/// 1. skills.isNotEmpty → effectiveMultimodal = false (debug warning logged)
/// 2. enableMultimodality == false → overrides and disables multimodal capabilities
/// 3. enableMultimodality == true → respects existing configuration
class MultimodalityGuard {
  static bool resolveEffectiveMultimodality(SessionConfig config) {
    if (config.skills.isNotEmpty) {
      GemmaDebug.log(
        GemmaDebug.tagMultimodal,
        'skills enabled — multimodality suppressed (vision/audio + tools are mutually exclusive on Gemma 4)',
      );
      return false;
    }

    if (!config.enableMultimodality) {
      return false;
    }

    return true;
  }
}
