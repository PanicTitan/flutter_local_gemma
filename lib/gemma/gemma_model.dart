/// Defines the available Gemma model variants with their platform-specific
/// download URLs and capability flags.
enum GemmaModel {
  gemma3nE2B(
    androidRepo: 'google/gemma-3n-E2B-it-litert-lm',
    androidFile: 'gemma-3n-E2B-it-int4.litertlm',
    webRepo: 'google/gemma-3n-E2B-it-litert-lm',
    webFile: 'gemma-3n-E2B-it-int4-Web.litertlm',
    supportsVision: true,
    supportsAudio: true,
    supportsTools: true,
    supportsThinking: false,
    supportsMtp: false,
    isGated: true,
    sizeDescription: '~3 GB  •  Gated',
  ),
  gemma3nE4B(
    androidRepo: 'google/gemma-3n-E4B-it-litert-lm',
    androidFile: 'gemma-3n-E4B-it-int4.litertlm',
    webRepo: 'google/gemma-3n-E4B-it-litert-lm',
    webFile: 'gemma-3n-E4B-it-int4-Web.litertlm',
    supportsVision: true,
    supportsAudio: true,
    supportsTools: true,
    supportsThinking: false,
    supportsMtp: false,
    isGated: true,
    sizeDescription: '~4 GB  •  Gated',
  ),
  gemma4E2B(
    androidRepo: 'litert-community/gemma-4-E2B-it-litert-lm',
    androidFile: 'gemma-4-E2B-it.litertlm',
    webRepo: 'litert-community/gemma-4-E2B-it-litert-lm',
    webFile: 'gemma-4-E2B-it-web.task',
    supportsVision: true,
    supportsAudio: true,
    supportsTools: true,
    supportsThinking: true,
    supportsMtp: true,
    isGated: false,
    sizeDescription: '2.6 GB  •  Open',
  ),
  gemma4E4B(
    androidRepo: 'litert-community/gemma-4-E4B-it-litert-lm',
    androidFile: 'gemma-4-E4B-it.litertlm',
    webRepo: 'litert-community/gemma-4-E4B-it-litert-lm',
    webFile: 'gemma-4-E4B-it-web.task',
    supportsVision: true,
    supportsAudio: false,
    supportsTools: true,
    supportsThinking: true,
    supportsMtp: true,
    isGated: false,
    sizeDescription: '3.7 GB  •  Open',
  );

  const GemmaModel({
    required this.androidRepo,
    required this.androidFile,
    required this.webRepo,
    required this.webFile,
    required this.supportsVision,
    required this.supportsAudio,
    required this.supportsTools,
    required this.supportsThinking,
    required this.supportsMtp,
    required this.isGated,
    required this.sizeDescription,
  });

  final String androidRepo, androidFile, webRepo, webFile;
  final bool supportsVision, supportsAudio, supportsTools;
  final bool supportsThinking, supportsMtp, isGated;
  final String sizeDescription;

  String get androidUrl =>
      'https://huggingface.co/$androidRepo/resolve/main/$androidFile';
  String get webUrl =>
      'https://huggingface.co/$webRepo/resolve/main/$webFile';
  bool get isGemma4 => supportsMtp;
  bool get isGemma3n => !isGemma4;
  String get displayName => switch (this) {
        GemmaModel.gemma3nE2B => 'Gemma 3n E2B',
        GemmaModel.gemma3nE4B => 'Gemma 3n E4B',
        GemmaModel.gemma4E2B => 'Gemma 4 E2B',
        GemmaModel.gemma4E4B => 'Gemma 4 E4B',
      };
}
