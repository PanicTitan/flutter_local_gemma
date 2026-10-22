import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:flutter_local_gemma/flutter_local_gemma_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterLocalGemmaPlatform
    with MockPlatformInterfaceMixin
    implements FlutterLocalGemmaPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  group('GemmaModel ONNX configuration', () {
    test('gemma4E2BOnnx is properly configured', () {
      const model = GemmaModel.gemma4E2BOnnx;
      expect(model.isOnnx, isTrue);
      expect(model.isGemma4, isTrue);
      expect(model.supportsVision, isTrue);
      expect(model.supportsAudio, isTrue);
      expect(model.webRepo, 'onnx-community/gemma-4-E2B-it-ONNX');
      expect(model.webFile, 'gemma-4-E2B-it.onnx');
      expect(model.displayName, 'Gemma 4 E2B (ONNX)');
    });

    test('gemma4E4BOnnx is properly configured', () {
      const model = GemmaModel.gemma4E4BOnnx;
      expect(model.isOnnx, isTrue);
      expect(model.isGemma4, isTrue);
      expect(model.supportsVision, isTrue);
      expect(model.supportsAudio, isTrue);
      expect(model.webRepo, 'onnx-community/gemma-4-E4B-it-ONNX');
      expect(model.webFile, 'gemma-4-E4B-it.onnx');
      expect(model.displayName, 'Gemma 4 E4B (ONNX)');
    });

    test('non-ONNX models have isOnnx false', () {
      expect(GemmaModel.gemma4E2B.isOnnx, isFalse);
      expect(GemmaModel.gemma4E4B.isOnnx, isFalse);
      expect(GemmaModel.gemma3nE2B.isOnnx, isFalse);
    });
  });
}

