import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../gemma/gemma.dart';
import '../embedding/embedding_plugin.dart';
import '../types/content_parts.dart';
import 'pdf_processor_web.dart' if (dart.library.io) 'pdf_processor_stub.dart';

/// Controls which content is extracted from each PDF page.
///
/// | Mode | Text extracted | Images extracted |
/// |------|----------------|-----------------|
/// | [auto] | ✓ | Only if no text found on the page |
/// | [textOnly] | ✓ | ✗ |
/// | [imagesOnly] | ✗ | ✓ |
/// | [fullRender] | ✗ | ✓ (each page rendered as a raster image) |
/// | [textAndImages] | ✓ | ✓ |
enum PdfExtractionMode { textOnly, textAndImages, imagesOnly, fullRender, auto }

/// Selects which pages of the PDF are processed.
///
/// Use [range] together with [PdfExtractionConfig.startPage] and
/// [PdfExtractionConfig.endPage] for a specific page range.
enum PdfPageFilter { all, odd, even, range }

/// Configuration for [PdfProcessor.extract].
///
/// ```dart
/// // Extract text from all pages, with images as fallback
/// const config = PdfExtractionConfig();
///
/// // Extract only the first 10 pages, text + images at 3× resolution
/// const config = PdfExtractionConfig(
///   mode: PdfExtractionMode.textAndImages,
///   filter: PdfPageFilter.range,
///   startPage: 1,
///   endPage: 10,
///   renderScale: 3.0,
/// );
/// ```
class PdfExtractionConfig {
  /// What content to extract from each page. Defaults to [PdfExtractionMode.auto].
  final PdfExtractionMode mode;

  /// Which pages to include. Defaults to [PdfPageFilter.all].
  final PdfPageFilter filter;

  /// First page to include when [filter] is [PdfPageFilter.range] (1-indexed).
  final int? startPage;

  /// Last page to include when [filter] is [PdfPageFilter.range] (1-indexed, inclusive).
  final int? endPage;

  /// DPI multiplier for rendered page images. `2.0` gives 144 DPI on a
  /// standard 72 DPI PDF, which is a good balance between quality and size.
  /// Only relevant for modes that produce images ([imagesOnly], [fullRender],
  /// [textAndImages], and [auto] fallback).
  final double renderScale;

  const PdfExtractionConfig({
    this.mode = PdfExtractionMode.auto,
    this.filter = PdfPageFilter.all,
    this.startPage,
    this.endPage,
    this.renderScale = 2.0,
  });
}

/// Cross-platform PDF text and image extractor.
///
/// On **web** it delegates to PDF.js running in a web worker.
/// On **Android** it uses `PdfRenderer.Page.textContents` (API 35+) or
/// OpenPDF for older devices, and `PdfRenderer` for image rendering.
///
/// ## Usage
/// ```dart
/// final pdfBytes = await File('document.pdf').readAsBytes();
/// final parts = await PdfProcessor.extract(
///   pdfBytes,
///   const PdfExtractionConfig(mode: PdfExtractionMode.auto),
/// );
/// for (final part in parts) {
///   if (part is TextPart)  print(part.text);
///   if (part is ImagePart) { /* PNG bytes in part.bytes */ }
/// }
/// ```
class PdfProcessor {
  static const MethodChannel _channel = MethodChannel('pdf_plugin');

  static Future<List<ContentPart>> extract(
    Uint8List pdfBytes,
    PdfExtractionConfig config,
  ) async {
    final List<ContentPart> parts = [];

    if (kIsWeb) {
      await PdfProcessorWeb().initPdfWorker();

      final webParts = await PdfProcessorWeb().extractPdf(
        pdfBytes,
        config.mode.name,
        config.filter.name,
        config.startPage,
        config.endPage,
        config.renderScale,
      );
      _mapToContentParts(webParts, parts);
    } else {
      final List<dynamic>? nativeParts = await _channel
          .invokeMethod('extractPdf', {
            'bytes': pdfBytes,
            'mode': config.mode.name,
            'filter': config.filter.name,
            'startPage': config.startPage,
            'endPage': config.endPage,
            'renderScale': config.renderScale,
          });

      if (nativeParts != null) {
        _mapToContentParts(nativeParts, parts);
      }
    }

    return parts;
  }

  // Safely map dynamic lists to avoid Dart casting crashes
  static void _mapToContentParts(
    List<dynamic> rawParts,
    List<ContentPart> parts,
  ) {
    for (final part in rawParts) {
      if (part is Map) {
        final type = part['type']?.toString();
        final data = part['data'];

        if (type == 'text' && data is String) {
          parts.add(TextPart(data));
        } else if (type == 'image' && data is Uint8List) {
          parts.add(ImagePart(data));
        }
      }
    }
  }
}
