import 'dart:async';
import '../types/content_parts.dart';
import 'gemma.dart';

/// Stub implementation of [FlutterLocalGemmaWeb] for Android / iOS.
///
/// Every method throws [UnimplementedError] because the web engine is not
/// available on native platforms.
class FlutterLocalGemmaWeb {
  static final FlutterLocalGemmaWeb _instance = FlutterLocalGemmaWeb._internal();
  factory FlutterLocalGemmaWeb() => _instance;
  FlutterLocalGemmaWeb._internal();

  Future<void> init(InferenceConfig config) async =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  void unload() =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  void addToBuffer(List<ContentPart> parts) =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  void clearBuffer() =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  Future<int> countTokensWeb(String text) async =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  Stream<Map<String, dynamic>> generateResponse({
    required AutoStopConfig autoStopConfig,
  }) =>
      throw UnimplementedError('Web engine unavailable on this platform.');

  Future<void> cancelProcessing() async =>
      throw UnimplementedError('Web engine unavailable on this platform.');
}