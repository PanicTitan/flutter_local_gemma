import 'dart:typed_data' show Uint8List;

/// Base class for all content parts that can be passed to the inference engine.
///
/// Use the concrete subtypes:
/// - [TextPart] for text prompts and conversational turns.
/// - [ImagePart] for PNG/JPEG image bytes (multimodal input).
/// - [AudioPart] for raw PCM audio bytes (Android only).
///
/// Content parts are consumed by [ChatSession.addToContext],
/// [ChatSession.generateResponseStream], and [FlutterLocalGemma.computeSingle].
sealed class ContentPart {}

/// A plain-text content part.
///
/// ```dart
/// final parts = [TextPart('What is the capital of France?')];
/// final response = await session.generateResponseFuture(parts);
/// ```
class TextPart extends ContentPart {
  /// The text payload.
  final String text;
  TextPart(this.text);
}

/// A raster image content part (multimodal input).
///
/// [bytes] must be a valid PNG or JPEG byte buffer.
/// Each image consumes approximately 257 tokens in the KV-cache.
///
/// ```dart
/// final imageBytes = await File('photo.png').readAsBytes();
/// final parts = [ImagePart(imageBytes), TextPart('Describe this image.')];
/// ```
class ImagePart extends ContentPart {
  /// Raw PNG or JPEG bytes of the image.
  final Uint8List bytes;
  ImagePart(this.bytes);
}

/// A raw PCM audio content part.
///
/// [bytes] should be 16 kHz, 16-bit, mono PCM (standard WAV PCM format).
/// Token cost is approximately 1 token per 150 ms of audio
/// (i.e. `bytes.length / 32 / 150` tokens).
///
/// ```dart
/// final audioBytes = await File('clip.wav').readAsBytes();
/// final parts = [AudioPart(audioBytes), TextPart('Transcribe this audio.')];
/// ```
class AudioPart extends ContentPart {
  /// Raw PCM audio bytes (16 kHz, 16-bit, mono).
  final Uint8List bytes;
  AudioPart(this.bytes);
}
