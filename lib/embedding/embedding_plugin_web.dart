import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:web/web.dart' as web;
import '../utils/web_script_loader.dart';
import 'embed_part.dart';

// ─── JS Interop declarations ──────────────────────────────────────────────────

@JS('initEmbeddingModel')
external JSPromise<JSBoolean> _jsInitEmbedding(
  JSString url,
  JSString base, [
  JSString? token,
  JSFunction? onProgress,
]);

@JS('initEmbeddingGemma2')
external JSPromise<JSBoolean> _jsInitEmbeddingGemma2(
  JSString modelId, [
  JSString? token,
  JSNumber? targetDimension,
  JSFunction? onProgress,
]);

@JS('getEmbedding')
external JSPromise<JSAny> _jsGetEmbedding(JSString text);

@JS('getEmbeddingFromImages')
external JSPromise<JSAny> _jsGetEmbeddingFromImages(JSAny images);

@JS('getEmbeddingFromParts')
external JSPromise<JSAny> _jsGetEmbeddingFromParts(JSAny parts);

@JS('supportsAudio')
external JSPromise<JSBoolean> _jsSupportsAudio();

@JS('unloadEmbeddingModel')
external void _jsUnloadEmbeddingModel();

// ─────────────────────────────────────────────────────────────────────────────

/// Web implementation of the embedding plugin.
///
/// Wraps the LiteRT and Transformers.js based embedding engines exposed by `main.ts`.
class EmbeddingPluginWeb {
  static final EmbeddingPluginWeb _instance = EmbeddingPluginWeb._internal();
  factory EmbeddingPluginWeb() => _instance;
  EmbeddingPluginWeb._internal();

  bool _initialized = false;

  /// Whether the web embedding engine is ready to generate embeddings.
  bool get isInitialized => _initialized;

  /// Initialises the legacy LiteRT model and tokenizer in the browser.
  Future<void> init(
    String modelUrl, {
    String? token,
    void Function(double)? onProgress,
  }) async {
    await WebScriptLoader.ensureJsLoaded();
    final JSFunction? jsOnProgress = onProgress != null
        ? ((JSNumber p) {
            onProgress(p.toDartDouble);
          }).toJS
        : null;
    await _jsInitEmbedding(
      modelUrl.toJS,
      WebScriptLoader.assetBase.toJS,
      token?.toJS,
      jsOnProgress,
    ).toDart;
    _initialized = true;
  }

  /// Initialises the EmbeddingGemma 2 model using Transformers.js with WebGPU.
  Future<void> initGemma2(
    String modelId, {
    String? token,
    int? targetDimension,
    void Function(double)? onProgress,
  }) async {
    await WebScriptLoader.ensureJsLoaded();
    final JSFunction? jsOnProgress = onProgress != null
        ? ((JSNumber p) {
            onProgress(p.toDartDouble);
          }).toJS
        : null;
    await _jsInitEmbeddingGemma2(
      modelId.toJS,
      token?.toJS,
      targetDimension?.toJS,
      jsOnProgress,
    ).toDart;
    _initialized = true;
  }

  /// Computes a semantic embedding vector for [text].
  Future<List<double>> getEmbedding(String text) async {
    if (!_initialized) throw StateError('EmbeddingPluginWeb not initialised.');
    final result = await _jsGetEmbedding(text.toJS).toDart;
    return (result as JSFloat32Array).toDart.map((e) => e.toDouble()).toList();
  }

  /// Computes an embedding vector for encoded images.
  Future<List<double>> getEmbeddingFromImages(List<Uint8List> imageDataList) async {
    if (!_initialized) throw StateError('EmbeddingPluginWeb not initialised.');
    final jsArray = imageDataList.map((bytes) => bytes.toJS).toList().toJS;
    final result = await _jsGetEmbeddingFromImages(jsArray).toDart;
    return (result as JSFloat32Array).toDart.map((e) => e.toDouble()).toList();
  }

  /// Computes an embedding vector for multimodal parts.
  Future<List<double>> getEmbeddingFromParts(List<EmbedPart> parts) async {
    if (!_initialized) throw StateError('EmbeddingPluginWeb not initialised.');
    final list = <JSObject>[];
    for (final part in parts) {
      final obj = JSObject();
      if (part is TextEmbedPart) {
        obj.setProperty('type'.toJS, 'text'.toJS);
        obj.setProperty('text'.toJS, part.text.toJS);
      } else if (part is ImageEmbedPart) {
        obj.setProperty('type'.toJS, 'image'.toJS);
        obj.setProperty('bytes'.toJS, part.bytes.toJS);
      } else if (part is AudioEmbedPart) {
        obj.setProperty('type'.toJS, 'audio'.toJS);
        obj.setProperty(
          'bytes'.toJS,
          part.pcm.buffer.asUint8List(part.pcm.offsetInBytes, part.pcm.lengthInBytes).toJS,
        );
        obj.setProperty('sampleRateHz'.toJS, part.sampleRateHz.toJS);
      }
      list.add(obj);
    }
    final jsArray = list.toJS;
    final result = await _jsGetEmbeddingFromParts(jsArray).toDart;
    return (result as JSFloat32Array).toDart.map((e) => e.toDouble()).toList();
  }

  /// Checks if audio embedding is supported on web.
  Future<bool> supportsAudio() async {
    if (!_initialized) return false;
    try {
      final result = await _jsSupportsAudio().toDart;
      return result.toDart;
    } catch (_) {
      return false;
    }
  }

  /// Releases the compiled model and tokenizer from browser memory.
  void unload() {
    _jsUnloadEmbeddingModel();
    _initialized = false;
  }

  // ── Video Frame & Metadata Extraction Helpers (Native Browser Canvas) ─────

  web.HTMLVideoElement? _cachedVideoEl;
  String? _cachedVideoSrc;

  Future<web.HTMLVideoElement> _getOrLoadVideo(String url) async {
    final cleanUrl = url.split('#')[0];
    if (_cachedVideoEl != null && _cachedVideoSrc == cleanUrl && _cachedVideoEl!.readyState >= 1) {
      return _cachedVideoEl!;
    }
    if (_cachedVideoEl != null) {
      try {
        _cachedVideoEl!.pause();
        _cachedVideoEl!.removeAttribute('src');
        _cachedVideoEl!.load();
      } catch (_) {}
    }

    final video = web.document.createElement('video') as web.HTMLVideoElement;
    if (!cleanUrl.startsWith('blob:') && !cleanUrl.startsWith('data:')) {
      video.crossOrigin = 'anonymous';
    }
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';

    final completer = Completer<void>();
    late web.EventListener onMeta;
    late web.EventListener onErr;
    Timer? timer;

    void cleanup() {
      timer?.cancel();
      video.removeEventListener('loadedmetadata', onMeta);
      video.removeEventListener('error', onErr);
    }

    onMeta = ((web.Event _) {
      cleanup();
      if (!completer.isCompleted) completer.complete();
    }).toJS;

    onErr = ((web.Event _) {
      cleanup();
      if (!completer.isCompleted) completer.complete();
    }).toJS;

    timer = Timer(const Duration(seconds: 8), () {
      cleanup();
      if (!completer.isCompleted) completer.complete();
    });

    video.addEventListener('loadedmetadata', onMeta);
    video.addEventListener('error', onErr);
    video.src = cleanUrl;
    video.load();

    if (video.readyState >= 1) {
      cleanup();
      if (!completer.isCompleted) completer.complete();
    }

    await completer.future;
    _cachedVideoEl = video;
    _cachedVideoSrc = cleanUrl;
    return video;
  }

  Future<void> _seekVideo(web.HTMLVideoElement video, double targetSec) async {
    if ((video.currentTime - targetSec).abs() <= 0.05) return;
    final completer = Completer<void>();
    late web.EventListener listener;
    Timer? timer;
    listener = ((web.Event _) {
      timer?.cancel();
      video.removeEventListener('seeked', listener);
      if (!completer.isCompleted) completer.complete();
    }).toJS;
    timer = Timer(const Duration(seconds: 3), () {
      video.removeEventListener('seeked', listener);
      if (!completer.isCompleted) completer.complete();
    });
    video.addEventListener('seeked', listener);
    video.currentTime = targetSec;
    await completer.future;
  }

  /// Extracts video duration and dimensions from a video URL on Web using HTML5 video element.
  Future<({int durationMs, int width, int height})?> extractVideoMetadata(String videoUrl) async {
    try {
      final video = await _getOrLoadVideo(videoUrl);
      final dur = (video.duration.isFinite && video.duration > 0) ? video.duration : 0.0;
      final w = video.videoWidth > 0 ? video.videoWidth : 640;
      final h = video.videoHeight > 0 ? video.videoHeight : 360;
      return (
        durationMs: (dur * 1000).round(),
        width: w,
        height: h,
      );
    } catch (e) {
      debugPrint('[EmbeddingPluginWeb] extractVideoMetadata error: $e');
      return null;
    }
  }

  /// Extracts a video frame JPEG as Uint8List at [timeMs] on Web via HTML5 2D canvas.
  Future<Uint8List?> extractVideoFrame(String videoUrl, int timeMs) async {
    try {
      final video = await _getOrLoadVideo(videoUrl);
      final dur = (video.duration.isFinite && video.duration > 0) ? video.duration : (timeMs / 1000.0);
      final targetSec = (timeMs / 1000.0).clamp(0.0, dur);

      await _seekVideo(video, targetSec);

      final vw = video.videoWidth > 0 ? video.videoWidth : 640;
      final vh = video.videoHeight > 0 ? video.videoHeight : 360;
      final targetW = math.min(640, vw);
      final targetH = math.max(1, (targetW * (vh / vw)).round());

      final canvas = web.document.createElement('canvas') as web.HTMLCanvasElement;
      canvas.width = targetW;
      canvas.height = targetH;
      final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
      ctx.drawImage(video, 0, 0, targetW, targetH);

      final dataUrl = canvas.toDataURL('image/jpeg', 0.85.toJS);
      final commaIdx = dataUrl.indexOf(',');
      if (commaIdx == -1) return null;
      return base64Decode(dataUrl.substring(commaIdx + 1));
    } catch (e) {
      debugPrint('[EmbeddingPluginWeb] extractVideoFrame error: $e');
      return null;
    }
  }
}