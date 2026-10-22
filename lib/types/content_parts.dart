import 'dart:typed_data' show Uint8List;

/// Base class for all content parts that can be passed to the inference engine.
///
/// Use the concrete subtypes:
/// - [TextPart] for text prompts and conversational turns.
/// - [ImagePart] for PNG/JPEG image bytes (multimodal input).
/// - [ImageFilePart] for an on-disk image path (Android only, zero-copy).
/// - [AudioPart] for raw PCM audio bytes (Android only).
/// - [AudioFilePart] for an on-disk audio path (Android only, zero-copy).
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

/// An on-disk image content part — the zero-copy alternative to [ImagePart].
///
/// **Android only** (falls back to reading the file into an [ImagePart]-style
/// byte transfer on Web, see [FlutterLocalGemmaWeb]). Prefer this over
/// [ImagePart] whenever the image already exists as a file (camera capture,
/// file picker result, cached download): the bytes never have to be read
/// into a Dart [Uint8List] or copied across the platform MethodChannel,
/// which matters a lot for large photos.
///
/// ```dart
/// final parts = [ImageFilePart('/storage/emulated/0/DCIM/photo.jpg')];
/// ```
class ImageFilePart extends ContentPart {
  /// Absolute on-disk path to a PNG or JPEG file.
  final String path;
  ImageFilePart(this.path);
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

/// An on-disk audio content part — the zero-copy alternative to [AudioPart].
///
/// **Android only.** Avoids reading the whole clip into memory as a
/// [Uint8List] and marshalling it across the MethodChannel; LiteRT-LM reads
/// the file directly.
///
/// Since token estimation ([ChatSession.countTokens]) needs a duration,
/// pass [durationMs] if you know it (e.g. from a recorder or player API) for
/// an accurate estimate. If omitted, a conservative fixed estimate is used.
///
/// ```dart
/// final parts = [AudioFilePart('/data/.../clip.wav', durationMs: 4200)];
/// ```
class AudioFilePart extends ContentPart {
  /// Absolute on-disk path to a 16 kHz/16-bit/mono PCM (WAV) file.
  final String path;

  /// Known clip duration in milliseconds, used for token estimation.
  /// If null, a fixed default estimate is used instead.
  final int? durationMs;

  AudioFilePart(this.path, {this.durationMs});
}