import 'dart:convert';
import 'dart:io';

/// Fetches text content from [url] using dart:io HttpClient.
Future<String> fetchText(String url) async {
  final client = HttpClient();
  try {
    client.connectionTimeout = const Duration(seconds: 15);
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();
    if (response.statusCode != 200) {
      throw HttpException(
        'HTTP ${response.statusCode} for $url',
        uri: Uri.parse(url),
      );
    }
    return await response.transform(utf8.decoder).join();
  } finally {
    client.close();
  }
}
