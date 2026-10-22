import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// One downloadable EmbeddingGemma 2 `.litertlm` build.
class EmbeddingGemma2Variant {
  /// File name inside [EmbeddingGemma2Catalog.repo].
  final String fileName;

  /// Human readable label (e.g. `Qualcomm SM8550`).
  final String label;

  /// Upper-case SoC tokens that identify the chipset this build targets.
  /// Empty for the generic (GPU/CPU) build.
  final List<String> socTokens;

  /// Approximate download size in MB (from the HuggingFace file listing).
  final int sizeMb;

  const EmbeddingGemma2Variant({
    required this.fileName,
    required this.label,
    this.socTokens = const [],
    required this.sizeMb,
  });

  bool get isGeneric => socTokens.isEmpty;

  /// Direct download URL on HuggingFace.
  String get url => EmbeddingGemma2Catalog.urlFor(fileName);
}

/// Device chipset identifiers, as reported by `android.os.Build`.
class DeviceSocInfo {
  final String socModel;
  final String socManufacturer;
  final String hardware;
  final String board;
  final String device;
  final String model;

  const DeviceSocInfo({
    this.socModel = '',
    this.socManufacturer = '',
    this.hardware = '',
    this.board = '',
    this.device = '',
    this.model = '',
  });

  factory DeviceSocInfo.fromMap(Map<dynamic, dynamic> m) => DeviceSocInfo(
        socModel: '${m['socModel'] ?? ''}',
        socManufacturer: '${m['socManufacturer'] ?? ''}',
        hardware: '${m['hardware'] ?? ''}',
        board: '${m['board'] ?? ''}',
        device: '${m['device'] ?? ''}',
        model: '${m['model'] ?? ''}',
      );

  /// Single normalised haystack used for token matching.
  String get haystack =>
      '$socManufacturer $socModel $hardware $board'.toUpperCase();

  @override
  String toString() =>
      'DeviceSocInfo(soc: $socModel, maker: $socManufacturer, hw: $hardware, board: $board)';
}

/// EmbeddingGemma 2 model locations and chipset-aware variant selection.
///
/// The files mirror
/// <https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm>,
/// the same repository used by Google AI Edge Gallery's `model_allowlist`.
abstract final class EmbeddingGemma2Catalog {
  static const repo = 'litert-community/embeddinggemma-2-740m-litert-lm';

  /// Web model (Transformers.js / WebGPU), as used by the Embed Anything demo.
  static const webModelId = 'onnx-community/embeddinggemma-2-ONNX';

  /// Generic GPU/CPU build — the one Gallery ships by default. Works everywhere.
  static const generic = EmbeddingGemma2Variant(
    fileName: 'embeddinggemma-2-740m.litertlm',
    label: 'Generic (GPU/CPU)',
    sizeMb: 405,
  );

  /// Chipset-specific builds.
  static const chipsetVariants = <EmbeddingGemma2Variant>[
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Google_Tensor_G5.litertlm',
      label: 'Google Tensor G5',
      socTokens: ['TENSOR G5', 'GS501', 'LAGUNA'],
      sizeMb: 658,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Google_Tensor_G6.litertlm',
      label: 'Google Tensor G6',
      socTokens: ['TENSOR G6', 'MALIBU'],
      sizeMb: 616,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_MediaTek_MT6991.litertlm',
      label: 'MediaTek MT6991',
      socTokens: ['MT6991'],
      sizeMb: 895,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_MediaTek_MT6993.litertlm',
      label: 'MediaTek MT6993',
      socTokens: ['MT6993'],
      sizeMb: 865,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Qualcomm_QCS8275.litertlm',
      label: 'Qualcomm QCS8275',
      socTokens: ['QCS8275'],
      sizeMb: 563,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Qualcomm_SM8550.litertlm',
      label: 'Qualcomm SM8550',
      socTokens: ['SM8550'],
      sizeMb: 563,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Qualcomm_SM8650.litertlm',
      label: 'Qualcomm SM8650',
      socTokens: ['SM8650'],
      sizeMb: 563,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Qualcomm_SM8750.litertlm',
      label: 'Qualcomm SM8750',
      socTokens: ['SM8750'],
      sizeMb: 544,
    ),
    EmbeddingGemma2Variant(
      fileName: 'embeddinggemma-2-740m_Qualcomm_SM8850.litertlm',
      label: 'Qualcomm SM8850',
      socTokens: ['SM8850'],
      sizeMb: 559,
    ),
  ];

  static const _channel = MethodChannel('embedding_plugin');

  static String urlFor(String fileName) =>
      'https://huggingface.co/$repo/resolve/main/$fileName?download=true';

  /// Reads chipset identifiers from the native side (Android only).
  /// Returns an empty [DeviceSocInfo] on web / unsupported platforms.
  static Future<DeviceSocInfo> detectSoc() async {
    if (kIsWeb) return const DeviceSocInfo();
    try {
      final map = await _channel.invokeMethod<Map<dynamic, dynamic>>('getSocInfo');
      return map == null ? const DeviceSocInfo() : DeviceSocInfo.fromMap(map);
    } catch (_) {
      return const DeviceSocInfo();
    }
  }

  /// The chipset build matching [soc], or `null` when none matches.
  static EmbeddingGemma2Variant? matchChipset(DeviceSocInfo soc) {
    final hay = soc.haystack;
    for (final v in chipsetVariants) {
      if (v.socTokens.any(hay.contains)) return v;
    }
    return null;
  }

  /// Download candidates in priority order. The chipset build (when one
  /// matches and [preferChipset] is true) comes first; the generic build is
  /// always last so a failed/404 download can fall back to it.
  static List<EmbeddingGemma2Variant> candidates(
    DeviceSocInfo soc, {
    bool preferChipset = false,
  }) {
    final chipset = preferChipset ? matchChipset(soc) : null;
    return [?chipset, generic];
  }
}

