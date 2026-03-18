import 'dart:typed_data';

class PdfProcessorWeb {
  Future<void> initPdfWorker() async {}

  Future<List<Map<String, dynamic>>> extractPdf(
    Uint8List bytes,
    String mode,
    String filter,
    int? startPage,
    int? endPage,
    double renderScale,
  ) async {
    throw UnimplementedError(
      "Web PDF extraction is not available on this platform",
    );
  }
}
