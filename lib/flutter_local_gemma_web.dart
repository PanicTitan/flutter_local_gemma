// // In order to *not* need this ignore, consider extracting the "web" version
// // of your plugin as a separate package, instead of inlining it in the same
// // package as the core of your plugin.
// // ignore: avoid_web_libraries_in_flutter

// import 'package:flutter_web_plugins/flutter_web_plugins.dart';
// import 'package:web/web.dart' as web;

// import 'flutter_local_gemma_platform_interface.dart';

// /// A web implementation of the FlutterLocalGemmaPlatform of the FlutterLocalGemma plugin.
// class FlutterLocalGemmaWeb extends FlutterLocalGemmaPlatform {
//   /// Constructs a FlutterLocalGemmaWeb
//   FlutterLocalGemmaWeb();

//   static void registerWith(Registrar registrar) {
//     FlutterLocalGemmaPlatform.instance = FlutterLocalGemmaWeb();
//   }

//   /// Returns a [String] containing the version of the platform.
//   @override
//   Future<String?> getPlatformVersion() async {
//     final version = web.window.navigator.userAgent;
//     return version;
//   }
// }

import 'package:flutter_web_plugins/flutter_web_plugins.dart';
import 'package:web/web.dart' as web;
import 'flutter_local_gemma_platform_interface.dart';

/// The official entry point for the Flutter Web plugin.
class FlutterLocalGemmaWeb extends FlutterLocalGemmaPlatform {
  FlutterLocalGemmaWeb();

  // This method is required by the Flutter toolchain
  static void registerWith(Registrar registrar) {
    FlutterLocalGemmaPlatform.instance = FlutterLocalGemmaWeb();
  }

  @override
  Future<String?> getPlatformVersion() async {
    return web.window.navigator.userAgent;
  }
}
