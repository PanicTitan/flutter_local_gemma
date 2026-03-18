import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:flutter_local_gemma/flutter_local_gemma_platform_interface.dart';
import 'package:flutter_local_gemma/flutter_local_gemma_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterLocalGemmaPlatform
    with MockPlatformInterfaceMixin
    implements FlutterLocalGemmaPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  // final FlutterLocalGemmaPlatform initialPlatform = FlutterLocalGemmaPlatform.instance;

  // test('$MethodChannelFlutterLocalGemma is the default instance', () {
  //   expect(initialPlatform, isInstanceOf<MethodChannelFlutterLocalGemma>());
  // });

  // test('getPlatformVersion', () async {
  //   FlutterLocalGemma flutterGemmaPlugin = FlutterLocalGemma();
  //   MockFlutterLocalGemmaPlatform fakePlatform = MockFlutterLocalGemmaPlatform();
  //   FlutterLocalGemmaPlatform.instance = fakePlatform;

  //   expect(await flutterGemmaPlugin.getPlatformVersion(), '42');
  // });
}
