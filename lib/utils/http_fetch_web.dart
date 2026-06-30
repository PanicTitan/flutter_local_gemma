import 'dart:js_interop';
import 'package:web/web.dart' as web;

/// Fetches text content from [url] using the browser's fetch() API.
Future<String> fetchText(String url) async {
  final response = await web.window.fetch(url.toJS).toDart;
  if (!response.ok) {
    throw Exception('HTTP ${response.status} for $url');
  }
  final text = await response.text().toDart;
  return text.toDart;
}
