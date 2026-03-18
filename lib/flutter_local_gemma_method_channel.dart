import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_local_gemma_platform_interface.dart';

/// An implementation of [FlutterLocalGemmaPlatform] that uses method channels.
class MethodChannelFlutterLocalGemma extends FlutterLocalGemmaPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_local_gemma');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }
}
