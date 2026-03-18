import 'package:flutter/foundation.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// Lifecycle state for a single model engine (LLM or Embedding).
enum ModelStatus {
  unloaded,
  downloading,
  loading,
  ready,
  generating,
  rebooting,
  error,
}

extension ModelStatusLabel on ModelStatus {
  String get label => switch (this) {
        ModelStatus.unloaded    => 'Unloaded',
        ModelStatus.downloading => 'Downloading…',
        ModelStatus.loading     => 'Loading…',
        ModelStatus.ready       => 'Ready',
        ModelStatus.generating  => 'Generating…',
        ModelStatus.rebooting   => 'Rebooting…',
        ModelStatus.error       => 'Error',
      };

  bool get isBusy =>
      this == ModelStatus.downloading ||
      this == ModelStatus.loading     ||
      this == ModelStatus.rebooting   ||
      this == ModelStatus.generating;
}

/// Application-wide singleton that tracks both engine statuses and all
/// session configuration. All screens ListenableBuilder-listen to it.
class ModelManager extends ChangeNotifier {
  ModelManager._();
  static final instance = ModelManager._();

  // ── LLM ──────────────────────────────────────────────────────────────────

  ModelStatus llmStatus = ModelStatus.unloaded;
  String?     llmError;
  String?     llmModelPath;

  // ── Embedding ─────────────────────────────────────────────────────────────

  ModelStatus embeddingStatus = ModelStatus.unloaded;
  String?     embeddingError;

  // ── Engine settings (require model reload) ────────────────────────────────

  int  maxTokens    = 4096;
  bool useGpu       = true;
  bool supportAudio = true;

  // ── Session settings (apply without reload via rebuildChat) ───────────────

  double temperature  = 0.8;
  double topP         = 0.95;
  int    topK         = 40;
  int?   randomSeed;
  String systemPrompt = 'You are a helpful assistant.';

  SessionConfig get sessionConfig => SessionConfig(
        temperature:    temperature,
        topP:           topP,
        topK:           topK,
        randomSeed:     randomSeed,
        systemPrompt:   systemPrompt,
        autoStopConfig: const AutoStopConfig(),
      );

  // ── Status mutators ───────────────────────────────────────────────────────

  void setLlmStatus(ModelStatus s, {String? error}) {
    llmStatus = s;
    llmError  = error;
    notifyListeners();
  }

  void setEmbeddingStatus(ModelStatus s, {String? error}) {
    embeddingStatus = s;
    embeddingError  = error;
    notifyListeners();
  }

  void setGenerating(bool generating) {
    if (llmStatus == ModelStatus.ready && generating) {
      setLlmStatus(ModelStatus.generating);
    } else if (llmStatus == ModelStatus.generating && !generating) {
      setLlmStatus(ModelStatus.ready);
    }
  }

  void updateSessionConfig({
    double? temperature,
    double? topP,
    int?    topK,
    int?    randomSeed,
    String? systemPrompt,
  }) {
    if (temperature  != null) this.temperature  = temperature;
    if (topP         != null) this.topP         = topP;
    if (topK         != null) this.topK         = topK;
    if (systemPrompt != null) this.systemPrompt = systemPrompt;
    this.randomSeed = randomSeed ?? this.randomSeed;
    notifyListeners();
  }

  // ── Cache management ──────────────────────────────────────────────────────

  /// Purges all cached model files on the current platform.
  ///
  /// Android: deletes `filesDir/models/` (SAF copies) + downloaded model
  ///          files from `applicationDocumentsDirectory`.
  /// Web:     calls `purgeOpfsCache()` in JS to remove all OPFS entries.
  ///
  /// Returns a human-readable summary like "Freed 3.2 GB".
  Future<String> purgeCache() async {
    try {
      final bytesFreed = await ModelInstaller.purgeCache();
      notifyListeners();
      return 'Freed ${_formatBytes(bytesFreed)}';
    } catch (e) {
      return 'Purge failed: $e';
    }
  }

  String _formatBytes(int bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
    }
    if (bytes >= 1024 * 1024) return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    if (bytes >= 1024)        return '${(bytes / 1024).toStringAsFixed(0)} KB';
    return '$bytes bytes';
  }
}