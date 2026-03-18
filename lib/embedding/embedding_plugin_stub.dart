/// Stub implementation of [EmbeddingPluginWeb] for Android / iOS.
///
/// Throws [UnimplementedError] on every method because the web-only
/// LiteRT embedding engine is not available on native platforms.
class EmbeddingPluginWeb {
  static final EmbeddingPluginWeb _instance = EmbeddingPluginWeb._internal();
  factory EmbeddingPluginWeb() => _instance;
  EmbeddingPluginWeb._internal();

  bool get isInitialized => false;

  Future<void> init(String modelUrl, {String? token}) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  Future<List<double>> getEmbedding(String text) async =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');

  void unload() =>
      throw UnimplementedError('Web embedding engine unavailable on this platform.');
}