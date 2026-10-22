// lib/utils/model_loader.dart
//
// Thin adapter between the example-app's [ModelManager] state machine and the
// stateless [PluginModelLoader] / [GemmaModelPicker] helpers from the plugin.
//
// All heavy lifting (URL selection, installer calls, EmbeddingConfig wiring)
// now lives in the plugin; this file only bridges status updates.

import 'package:flutter/foundation.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import '../app_state.dart';

const _hfToken = 'hf_YOUR_TOKEN_HERE';

String? _getEffectiveToken() {
  final mgrToken = ModelManager.instance.hfToken;
  if (mgrToken != null && mgrToken.isNotEmpty) return mgrToken;
  return _hfToken.contains('YOUR_TOKEN') ? null : _hfToken;
}

/// Downloads (if needed) and initialises the LLM engine.
///
/// Pass [local] to open the platform file picker instead of downloading.
/// [onProgress] receives [0, 100] during the download phase.
Future<void> loadLlm({
  bool local = false,
  GemmaModel? model,
  bool enableMtp = true,
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

    mgr.llmModelPath = await PluginModelLoader.loadLlm(
      localPath: localPath,
      networkUrl: model != null
          ? (kIsWeb ? model.webUrl : model.androidUrl)
          : null,
      token: _getEffectiveToken(),
      maxTokens: mgr.maxTokens + 1024,
      useGpu: mgr.useGpu,
      enableMtp: enableMtp,
      supportAudio: mgr.supportAudio,
      modelName: model != null
          ? (kIsWeb ? model.webFile : model.androidFile)
          : null,
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
    await PluginModelLoader.unloadLlm();
    mgr.setLlmStatus(ModelStatus.unloaded);
  } catch (e) {
    mgr.setLlmStatus(ModelStatus.error, error: e.toString());
  }
}

/// Downloads (if needed) and initialises the embedding engine.
Future<void> loadEmbedding({
  bool local = false,
  GemmaModel? model,
  bool preferChipsetVariant = false,
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
      mgr.embeddingModelPath = localPath;
      if (model?.isEmbedding == true || localPath.endsWith('.litertlm')) {
        await PluginModelLoader.initEmbeddingGemma2(
          modelPathOrId: localPath,
          token: _getEffectiveToken(),
        );
      } else {
        await PluginModelLoader.initEmbedding(
          modelPath: localPath,
          useGpu: mgr.useGpu,
          token: _getEffectiveToken(),
        );
      }
    } else {
      mgr.setEmbeddingStatus(ModelStatus.downloading, error: null);
      if (model == null || model == GemmaModel.embeddingGemma2) {
        final path = await PluginModelLoader.loadEmbeddingGemma2(
          token: _getEffectiveToken(),
          preferChipsetVariant: preferChipsetVariant,
          onProgress: (p) {
            onProgress?.call(p);
            if (p >= 100) mgr.setEmbeddingStatus(ModelStatus.loading);
          },
        );
        mgr.embeddingModelPath = path;
      } else {
        await PluginModelLoader.loadEmbedding(
          token: _getEffectiveToken(),
          useGpu: mgr.useGpu,
          onProgress: (p) {
            onProgress?.call(p);
            if (p >= 100) mgr.setEmbeddingStatus(ModelStatus.loading);
          },
        );
      }
    }

    mgr.setEmbeddingStatus(ModelStatus.ready);
  } catch (e) {
    mgr.setEmbeddingStatus(ModelStatus.error, error: e.toString());
    rethrow;
  }
}

/// Downloads (if needed) and initialises the EmbeddingGemma 2 engine.
Future<void> loadEmbeddingGemma2({
  bool local = false,
  bool preferChipsetVariant = false,
  void Function(double)? onProgress,
}) =>
    loadEmbedding(
      local: local,
      model: GemmaModel.embeddingGemma2,
      preferChipsetVariant: preferChipsetVariant,
      onProgress: onProgress,
    );

/// Unloads the embedding engine.
Future<void> unloadEmbedding() async {
  final mgr = ModelManager.instance;
  try {
    await PluginModelLoader.unloadEmbedding();
    mgr.embeddingModelPath = null;
    mgr.setEmbeddingStatus(ModelStatus.unloaded);
  } catch (e) {
    mgr.setEmbeddingStatus(ModelStatus.error, error: e.toString());
  }
}
