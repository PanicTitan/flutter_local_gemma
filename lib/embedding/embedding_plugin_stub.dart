import 'dart:typed_data';

/// Stub implementation of [EmbeddingPluginWeb] for Android / iOS.
///
/// Throws [UnimplementedError] on every method because the web-only
/// LiteRT embedding engine is not available on native platforms.
class EmbeddingPluginWeb {
  static final EmbeddingPluginWeb _instance = EmbeddingPluginWeb._internal();
  factory EmbeddingPluginWeb() => _instance;
  EmbeddingPluginWeb._internal();

  bool get isInitialized => false;

  Future<void> init(
    String modelUrl, {
    String? token,
    void Function(double)? onProgress,
  }) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<void> initGemma2(
    String modelPathOrId, {
    String? token,
    int? targetDimension,
    void Function(double)? onProgress,
  }) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<List<double>> getEmbedding(String text) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<List<double>> getEmbeddingFromImages(List<Uint8List> imageDataList) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<List<double>> getEmbeddingFromParts(List<dynamic> parts) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<bool> supportsAudio() async => false;

  void unload() =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<({int durationMs, int width, int height})?> extractVideoMetadata(String videoUrl) async => null;

  Future<Uint8List?> extractVideoFrame(String videoUrl, int timeMs) async => null;
}