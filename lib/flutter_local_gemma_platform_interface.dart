import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_local_gemma_method_channel.dart';

abstract class FlutterLocalGemmaPlatform extends PlatformInterface {
  /// Constructs a FlutterLocalGemmaPlatform.
  FlutterLocalGemmaPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterLocalGemmaPlatform _instance = MethodChannelFlutterLocalGemma();

  /// The default instance of [FlutterLocalGemmaPlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterLocalGemma].
  static FlutterLocalGemmaPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterLocalGemmaPlatform] when
  /// they register themselves.
  static set instance(FlutterLocalGemmaPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
