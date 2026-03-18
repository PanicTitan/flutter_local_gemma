// lib/utils/model_loader.dart
//
// Thin adapter between the example-app's [ModelManager] state machine and the
// stateless [GemmaLoader] / [GemmaModelPicker] helpers from the plugin.
//
// All heavy lifting (URL selection, installer calls, EmbeddingConfig wiring)
// now lives in the plugin; this file only bridges status updates.

import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:flutter_local_gemma/helpers/model_loader.dart';
import 'package:flutter_local_gemma/model_picker/model_picker.dart';
import '../app_state.dart';

const _hfToken = 'hf_YOUR_TOKEN_HERE';

/// Downloads (if needed) and initialises the LLM engine.
///
/// Pass [local] to open the platform file picker instead of downloading.
/// [onProgress] receives [0, 100] during the download phase.
Future<void> loadLlm({
  bool local = false,
  void Function(double)? onProgress,
}) async {
  final mgr = ModelManager.instance;
  try {
    String? localPath;

    if (local) {
      mgr.setLlmStatus(ModelStatus.loading, error: null);
      localPath = await GemmaModelPicker.pick();
      if (localPath == null) {
        mgr.setLlmStatus(ModelStatus.unloaded);
        return;
      }
    } else {
      mgr.setLlmStatus(ModelStatus.downloading, error: null);
    }

    mgr.llmModelPath = localPath; // null when downloading (resolved after)

    await GemmaLoader.loadLlm(
      localPath: localPath,
      token: _hfToken,
      maxTokens: mgr.maxTokens + 1024,
      useGpu: mgr.useGpu,
      supportAudio: mgr.supportAudio,
      onProgress: (p) {
        onProgress?.call(p);
        // Switch status label once download finishes
        if (p >= 100) mgr.setLlmStatus(ModelStatus.loading);
      },
    );

    mgr.setLlmStatus(ModelStatus.ready);
  } catch (e) {
    mgr.setLlmStatus(ModelStatus.error, error: e.toString());
    rethrow;
  }
}

/// Unloads the LLM engine.
Future<void> unloadLlm() async {
  final mgr = ModelManager.instance;
  try {
    await GemmaLoader.unloadLlm();
    mgr.setLlmStatus(ModelStatus.unloaded);
  } catch (e) {
    mgr.setLlmStatus(ModelStatus.error, error: e.toString());
  }
}

/// Downloads (if needed) and initialises the embedding engine.
Future<void> loadEmbedding({
  bool local = false,
  void Function(double)? onProgress,
}) async {
  final mgr = ModelManager.instance;
  try {
    String? localPath;

    if (local) {
      mgr.setEmbeddingStatus(ModelStatus.loading, error: null);
      localPath = await GemmaModelPicker.pick();
      if (localPath == null) {
        mgr.setEmbeddingStatus(ModelStatus.unloaded);
        return;
      }
      await GemmaLoader.initEmbedding(
        modelPath: localPath,
        useGpu: mgr.useGpu,
        token: _hfToken,
      );
    } else {
      mgr.setEmbeddingStatus(ModelStatus.downloading, error: null);
      await GemmaLoader.loadEmbedding(
        token: _hfToken,
        useGpu: mgr.useGpu,
        onProgress: (p) {
          onProgress?.call(p);
          if (p >= 100) mgr.setEmbeddingStatus(ModelStatus.loading);
        },
      );
    }

    mgr.setEmbeddingStatus(ModelStatus.ready);
  } catch (e) {
    mgr.setEmbeddingStatus(ModelStatus.error, error: e.toString());
    rethrow;
  }
}

/// Unloads the embedding engine.
Future<void> unloadEmbedding() async {
  final mgr = ModelManager.instance;
  try {
    await GemmaLoader.unloadEmbedding();
    mgr.setEmbeddingStatus(ModelStatus.unloaded);
  } catch (e) {
    mgr.setEmbeddingStatus(ModelStatus.error, error: e.toString());
  }
}