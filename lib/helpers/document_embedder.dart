import 'dart:typed_data';
import '../gemma/gemma.dart';
import '../pdf/pdf_processor.dart';
import '../embedding/embedding_plugin.dart';
import '../types/content_parts.dart';

/// Combines PDF extraction and text embedding in a single pipeline step.
///
/// Designed for **Retrieval-Augmented Generation (RAG)** workflows where you
/// want to index a PDF document into a vector store.
///
/// Each text block extracted from the PDF is embedded directly.
/// Each image page is first described by the LLM (via [FlutterLocalGemma.computeSingle])
/// and the resulting description is then embedded — ensuring that charts,
/// diagrams, and scanned pages are included in the index.
///
/// ## Prerequisites
/// Both [FlutterLocalGemma] and [EmbeddingPlugin] must be initialised before
/// calling [embedPdf].
///
/// ## Usage
/// ```dart
/// await GemmaLoader.loadLlm();
/// await GemmaLoader.loadEmbedding();
///
/// final pdfBytes = await File('document.pdf').readAsBytes();
/// final embeddings = await DocumentEmbedder.embedPdf(
///   pdfBytes,
///   const PdfExtractionConfig(mode: PdfExtractionMode.textAndImages),
/// );
/// // embeddings[i] is a 768-dim float64 vector for the i-th content block
/// ```
class DocumentEmbedder {
  /// Extracts content from [pdfBytes] and returns one embedding vector per
  /// text block or image found in the PDF.
  ///
  /// [config] controls which pages and content types are processed.
  ///
  /// [imageInterrogationPrompt] is the prompt sent to the LLM when an image
  /// part is encountered. Override it to focus the description on specific
  /// details relevant to your domain.
  static Future<List<List<double>>> embedPdf(
    Uint8List pdfBytes,
    PdfExtractionConfig config, {
    String imageInterrogationPrompt =
        "Describe all the details, text, and charts visible in this image document.",
  }) async {
    final parts = await PdfProcessor.extract(pdfBytes, config);
    final List<List<double>> embeddings = [];

    for (final part in parts) {
      if (part is TextPart && part.text.trim().isNotEmpty) {
        final vec = await EmbeddingPlugin().getEmbedding(part.text);
        embeddings.add(vec);
      } else if (part is ImagePart) {
        final description = await FlutterLocalGemma().computeSingle([
          part,
          TextPart(imageInterrogationPrompt),
        ], config: SessionConfig(temperature: 0.2));
        final vec = await EmbeddingPlugin().getEmbedding(description);
        embeddings.add(vec);
      }
    }
    return embeddings;
  }
}
