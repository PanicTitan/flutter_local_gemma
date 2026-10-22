import 'dart:math' as math;
import 'dart:typed_data';

/// Modality sample rate expected by the EmbeddingGemma 2 audio frontend.
const int kAudioSampleRateHz = 16000;

/// Default embedding dimension produced by EmbeddingGemma 2.
const int kDefaultEmbeddingDimension = 768;

/// Supported Matryoshka representation dimensions for EmbeddingGemma 2.
const List<int> kSupportedMatryoshkaDimensions = [768, 512, 256, 128];

/// Task instruction prefixes for asymmetric and symmetric EmbeddingGemma 2 tasks.
enum EmbeddingTaskType {
  /// Asymmetric search query: "task: search result | query: {query}"
  retrievalQuery,

  /// Asymmetric corpus item: "title: {title} | text: {content}"
  retrievalDocument,

  /// Asymmetric question answering: "task: question answering | query: {question}"
  questionAnswering,

  /// Asymmetric fact checking: "task: fact checking | query: {claim}"
  factChecking,

  /// Asymmetric code retrieval: "task: code retrieval | query: {query}"
  codeRetrieval,

  /// Symmetric text classification: "task: classification | query: {content}"
  classification,

  /// Symmetric clustering: "task: clustering | query: {content}"
  clustering,

  /// Symmetric sentence similarity: "task: sentence similarity | query: {content}"
  sentenceSimilarity,

  /// Raw input with no added instruction prefix.
  none,
}

/// Formats text with the recommended EmbeddingGemma 2 task instruction prefix.
String formatEmbeddingText(
  String text, {
  EmbeddingTaskType taskType = EmbeddingTaskType.retrievalQuery,
  String? title,
}) {
  final cleanText = text.trim();
  switch (taskType) {
    case EmbeddingTaskType.retrievalQuery:
      return 'task: search result | query: $cleanText';
    case EmbeddingTaskType.retrievalDocument:
      final cleanTitle = (title != null && title.trim().isNotEmpty) ? title.trim() : 'none';
      return 'title: $cleanTitle | text: $cleanText';
    case EmbeddingTaskType.questionAnswering:
      return 'task: question answering | query: $cleanText';
    case EmbeddingTaskType.factChecking:
      return 'task: fact checking | query: $cleanText';
    case EmbeddingTaskType.codeRetrieval:
      final cleanTitle = (title != null && title.trim().isNotEmpty) ? title.trim() : 'none';
      return 'task: code retrieval | query: $cleanText | title: $cleanTitle';
    case EmbeddingTaskType.classification:
      return 'task: classification | query: $cleanText';
    case EmbeddingTaskType.clustering:
      return 'task: clustering | query: $cleanText';
    case EmbeddingTaskType.sentenceSimilarity:
      return 'task: sentence similarity | query: $cleanText';
    case EmbeddingTaskType.none:
      return cleanText;
  }
}

/// One element of an ordered multimodal embedding request for EmbeddingGemma 2.
sealed class EmbedPart {
  const EmbedPart();

  /// Creates a text part (e.g. timestamp "00:02" or title).
  const factory EmbedPart.text(String text) = TextEmbedPart;

  /// Creates an image part from encoded image bytes (JPEG, PNG, WebP, TGA).
  const factory EmbedPart.image(Uint8List bytes) = ImageEmbedPart;

  /// Creates a 16 kHz mono 16-bit PCM audio clip part.
  const factory EmbedPart.audio(Int16List pcm, {int sampleRateHz}) = AudioEmbedPart;

  /// Serializes this part into a map for method channel or JS interop.
  Map<String, dynamic> toMap();
}

/// A text component of an interleaved multimodal sequence.
final class TextEmbedPart extends EmbedPart {
  final String text;
  const TextEmbedPart(this.text);

  @override
  Map<String, dynamic> toMap() => {'type': 'text', 'text': text};

  @override
  String toString() => 'TextEmbedPart("$text")';
}

/// An image component of an interleaved multimodal sequence.
final class ImageEmbedPart extends EmbedPart {
  final Uint8List bytes;
  const ImageEmbedPart(this.bytes);

  @override
  Map<String, dynamic> toMap() => {'type': 'image', 'bytes': bytes};

  @override
  String toString() => 'ImageEmbedPart(${bytes.length} bytes)';
}

/// An audio component of an interleaved multimodal sequence.
final class AudioEmbedPart extends EmbedPart {
  final Int16List pcm;
  final int sampleRateHz;

  const AudioEmbedPart(this.pcm, {this.sampleRateHz = kAudioSampleRateHz});

  @override
  Map<String, dynamic> toMap() => {
        'type': 'audio',
        'pcm': pcm,
        'sampleRateHz': sampleRateHz,
      };

  @override
  String toString() => 'AudioEmbedPart(${pcm.length} samples @ ${sampleRateHz}Hz)';
}

// ── Vector Math Utilities ───────────────────────────────────────────────────

/// Computes the cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0. Returns 0.0 if vectors differ in size
/// or either has zero magnitude.
double cosineSimilarity(List<double> v1, List<double> v2) {
  if (v1.isEmpty || v2.isEmpty || v1.length != v2.length) return 0.0;
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;

  for (int i = 0; i < v1.length; i++) {
    final a = v1[i];
    final b = v2[i];
    dot += a * b;
    norm1 += a * a;
    norm2 += b * b;
  }

  final denom = math.sqrt(norm1) * math.sqrt(norm2);
  if (denom <= 0.0) return 0.0;
  return (dot / denom).clamp(-1.0, 1.0);
}

/// Normalizes a vector in-place or returns an L2-normalized copy.
List<double> l2Normalize(List<double> vector) {
  if (vector.isEmpty) return vector;
  double sumSq = 0.0;
  for (final v in vector) {
    sumSq += v * v;
  }
  final norm = math.sqrt(sumSq);
  if (norm <= 0.0) return List<double>.from(vector);
  return vector.map((v) => v / norm).toList();
}

/// Truncates an EmbeddingGemma 2 vector to a smaller Matryoshka dimension
/// (e.g., 768 -> 512, 256, or 128) and re-normalizes with L2 norm.
List<double> truncateMatryoshka(List<double> vector, int targetDimension) {
  if (vector.length <= targetDimension) {
    return l2Normalize(vector);
  }
  final sliced = vector.sublist(0, targetDimension);
  return l2Normalize(sliced);
}
