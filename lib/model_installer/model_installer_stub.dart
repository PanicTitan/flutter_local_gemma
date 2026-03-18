import 'package:cross_file/cross_file.dart';
import 'model_installer.dart';

ModelDownloaderBuilder createDownloader(
  String url,
  String fileName, {
  String? token,
  String? directoryPath,
}) =>
    throw UnimplementedError('Platform not supported');

Future<String> createXFileInstaller(XFile xFile) async =>
    throw UnimplementedError('Platform not supported');

Future<String> createBlobInstaller(Object blob) =>
    throw UnimplementedError('Platform not supported');

Future<int> purgeModelCacheImpl() async => 0;

// Web-only – not available on this platform.
Future<List<Map<String, dynamic>>> listOpfsCacheImpl() async => [];