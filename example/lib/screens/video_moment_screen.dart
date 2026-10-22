import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:video_player/video_player.dart';

/// Screen implementing Google AI Edge Gallery's Realtime Video Moment Finder
/// powered by EmbeddingGemma 2, sliding window chunking, audio interleaving,
/// and live video playback with extracted preview frame thumbnails.
class VideoMomentScreen extends StatefulWidget {
  const VideoMomentScreen({super.key});

  @override
  State<VideoMomentScreen> createState() => _VideoMomentScreenState();
}

class _VideoMomentScreenState extends State<VideoMomentScreen> {
  final _searchController = TextEditingController();

  late final SemanticRetrievalService _retrievalService;
  late final VideoMomentService _momentService;

  VideoPlayerController? _videoPlayerController;
  bool _isVideoControllerReady = false;
  bool _isPlaying = false;
  int _currentScrubberPositionMs = 0;
  Timer? _playbackTimer;
  Timer? _debounceTimer;

  // Frame thumbnail cache keyed by timestamp (ms)
  final Map<int, Uint8List> _frameThumbnailCache = {};

  // Sliding Window Settings (aligned with Google AI Edge Gallery)
  int _windowDurationSec = 3;
  int _framesPerWindow = 2;
  final int _overlapDurationSec = 1;
  bool _includeAudio = true;
  int _topK = 5;

  // Active moment clip playback auto-stop boundary
  int? _activeMomentEndMs;

  @override
  void initState() {
    super.initState();
    _retrievalService = SemanticRetrievalService(databaseName: 'video_moments_gemma.db');
    _momentService = VideoMomentService(retrievalService: _retrievalService);
    _initEngine();
  }

  @override
  void dispose() {
    _debounceTimer?.cancel();
    _playbackTimer?.cancel();
    _videoPlayerController?.dispose();
    _searchController.dispose();
    _momentService.dispose();
    _retrievalService.dispose();
    super.dispose();
  }

  Future<void> _initEngine() async {
    await _retrievalService.initialize();
  }

  // ── Video Player Initialization ──────────────────────────────────────────

  Future<void> _initVideoPlayer(String pathOrUri) async {
    try {
      final old = _videoPlayerController;
      _videoPlayerController = null;
      _isVideoControllerReady = false;
      await old?.dispose();

      VideoPlayerController controller;
      if (kIsWeb || pathOrUri.startsWith('http://') || pathOrUri.startsWith('https://')) {
        controller = VideoPlayerController.networkUrl(Uri.parse(pathOrUri));
      } else if (pathOrUri.startsWith('content://')) {
        controller = VideoPlayerController.contentUri(Uri.parse(pathOrUri));
      } else {
        final f = File(pathOrUri);
        if (await f.exists()) {
          controller = VideoPlayerController.file(f);
        } else {
          return;
        }
      }

      await controller.initialize();
      controller.setLooping(true);
      controller.addListener(_onVideoPlayerUpdate);

      if (mounted) {
        setState(() {
          _videoPlayerController = controller;
          _isVideoControllerReady = true;
        });
      }
    } catch (e) {
      debugPrint('[VideoMomentScreen] Video player init notice ($e), using simulated renderer.');
      if (mounted) {
        setState(() {
          _isVideoControllerReady = false;
        });
      }
    }
  }

  void _onVideoPlayerUpdate() {
    final controller = _videoPlayerController;
    if (controller == null || !controller.value.isInitialized) return;

    final posMs = controller.value.position.inMilliseconds;
    final isPlayingNow = controller.value.isPlaying;

    // Check if auto-stop moment clip boundary reached
    if (_activeMomentEndMs != null && posMs >= _activeMomentEndMs!) {
      controller.pause();
      _activeMomentEndMs = null;
      setState(() {
        _isPlaying = false;
        _currentScrubberPositionMs = posMs;
      });
      return;
    }

    if (mounted) {
      setState(() {
        _currentScrubberPositionMs = posMs;
        _isPlaying = isPlayingNow;
      });
    }
  }



  // ── Realistic Video Frame Generator (High Fidelity Scenery) ─────────────

  // ── Video Frame Generator Fallback ──────────────────────────────────────────

  Future<Uint8List> _generateFallbackVideoFrame(int timestampMs, int totalMs) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, const Rect.fromLTWH(0, 0, 640, 360));
    const width = 640.0;
    const height = 360.0;

    // Elegant deep slate gradient
    final bgPaint = Paint()
      ..shader = ui.Gradient.linear(
        const Offset(0, 0),
        const Offset(0, height),
        [const Color(0xFF1E2433), const Color(0xFF0F1219)],
      );
    canvas.drawRect(const Rect.fromLTWH(0, 0, width, height), bgPaint);

    // Subtle filmstrip grid line accents
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.05)
      ..strokeWidth = 1;
    for (double y = 40; y < height; y += 40) {
      canvas.drawLine(Offset(0, y), Offset(width, y), gridPaint);
    }

    // Centered camera / play icon watermark
    final circlePaint = Paint()
      ..color = Colors.deepOrange.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(const Offset(width / 2, height / 2 - 15), 44, circlePaint);

    // Timecode
    final sec = (timestampMs / 1000).floor();
    final min = sec ~/ 60;
    final remSec = sec % 60;
    final ms = (timestampMs % 1000) ~/ 100;
    final timeStr = '${min.toString().padLeft(2, '0')}:${remSec.toString().padLeft(2, '0')}.$ms';

    final textPainter = TextPainter(
      text: TextSpan(
        children: [
          const TextSpan(
            text: 'Video Snapshot\n',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          TextSpan(
            text: 'Timestamp: $timeStr',
            style: const TextStyle(
              color: Colors.deepOrangeAccent,
              fontSize: 13,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    );
    textPainter.layout(maxWidth: width - 40);
    textPainter.paint(canvas, Offset((width - textPainter.width) / 2, height / 2 + 35));

    final picture = recorder.endRecording();
    final img = await picture.toImage(width.toInt(), height.toInt());
    final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }

  // ── Video Picker ─────────────────────────────────────────────────────────

  Future<void> _pickVideoFile() async {
    try {
      final videoPath = await GemmaModelPicker.pickVideo();
      if (videoPath == null) return;

      final displayName = videoPath.split('/').last.split('\\').last.split('#').last;

      // Query native video metadata for exact duration
      final meta = await _retrievalService.extractVideoMetadata(videoPath);
      final realDurationMs = (meta != null && meta.durationMs > 0)
          ? meta.durationMs
          : 45000;

      final project = VideoMomentProject(
        id: 'video_${DateTime.now().millisecondsSinceEpoch}',
        videoPath: videoPath,
        displayName: displayName,
        durationMs: realDurationMs,
        windowDurationSec: _windowDurationSec,
        framesPerWindow: _framesPerWindow,
        overlapDurationSec: _overlapDurationSec,
        includeAudio: _includeAudio,
        topK: _topK,
      );

      _momentService.addProject(project);
      _momentService.selectProject(project);

      await _initVideoPlayer(videoPath);

      // Pre-extract real video frame thumbnails at intervals
      _frameThumbnailCache.clear();
      final effectiveDurationMs = (_videoPlayerController?.value.isInitialized == true &&
              (_videoPlayerController!.value.duration.inMilliseconds) > 0)
          ? _videoPlayerController!.value.duration.inMilliseconds
          : realDurationMs;

      for (int t = 0; t <= effectiveDurationMs; t += 2000) {
        Uint8List? frame = await _retrievalService.extractVideoFrame(videoPath, t);
        frame ??= await _generateFallbackVideoFrame(t, effectiveDurationMs);
        _frameThumbnailCache[t] = frame;
      }

      // Index with real frame extraction
      await _momentService.indexProject(
        project: project.copyWith(durationMs: effectiveDurationMs),
        extractFrames: (timestamps) async {
          final frames = <Uint8List>[];
          for (final t in timestamps) {
            Uint8List? f = _frameThumbnailCache[t];
            f ??= await _retrievalService.extractVideoFrame(videoPath, t);
            f ??= await _generateFallbackVideoFrame(t, effectiveDurationMs);
            _frameThumbnailCache[t] = f;
            frames.add(f);
          }
          return frames;
        },
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Indexed video: $displayName'),
            backgroundColor: Colors.deepOrange,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Video picker error: $e')),
        );
      }
    }
  }

  // ── Playback & Scrubber Controls ─────────────────────────────────────────

  void _seekTo(int timeMs, {int? endBoundaryMs}) {
    final dur = _videoPlayerController?.value.duration.inMilliseconds ??
        _momentService.selectedProject?.durationMs ??
        30000;
    final clampedMs = timeMs.clamp(0, dur);

    setState(() {
      _currentScrubberPositionMs = clampedMs;
      _activeMomentEndMs = endBoundaryMs;
      if (endBoundaryMs != null) {
        _isPlaying = true;
      }
    });

    if (_videoPlayerController != null && _isVideoControllerReady) {
      _videoPlayerController!.seekTo(Duration(milliseconds: clampedMs));
      if (endBoundaryMs != null) {
        _videoPlayerController!.play();
      }
    }
  }

  void _togglePlayback() {
    if (_videoPlayerController != null && _isVideoControllerReady) {
      if (_videoPlayerController!.value.isPlaying) {
        _videoPlayerController!.pause();
        setState(() => _isPlaying = false);
      } else {
        _activeMomentEndMs = null;
        _videoPlayerController!.play();
        setState(() => _isPlaying = true);
      }
      return;
    }

    // Simulated playback loop if native video player is unavailable
    setState(() {
      _isPlaying = !_isPlaying;
      if (_isPlaying) {
        _playbackTimer = Timer.periodic(const Duration(milliseconds: 200), (timer) {
          final dur = _momentService.selectedProject?.durationMs ?? 30000;
          if (_currentScrubberPositionMs >= dur) {
            _currentScrubberPositionMs = 0;
          } else {
            _currentScrubberPositionMs += 200;
          }
          setState(() {});
        });
      } else {
        _playbackTimer?.cancel();
      }
    });
  }

  void _onSearchQueryChanged(String query) {
    _debounceTimer?.cancel();
    _debounceTimer = Timer(const Duration(milliseconds: 250), () async {
      await _momentService.searchMoments(
        query,
        extractPreviewFrame: (timeMs) async {
          final proj = _momentService.selectedProject;
          if (proj != null) {
            final realFrame = await _retrievalService.extractVideoFrame(proj.videoPath, timeMs);
            if (realFrame != null && realFrame.isNotEmpty) {
              return realFrame;
            }
          }
          final bestMatchKey = _frameThumbnailCache.keys.isEmpty
              ? null
              : _frameThumbnailCache.keys.reduce(
                  (a, b) => (a - timeMs).abs() < (b - timeMs).abs() ? a : b,
                );
          if (bestMatchKey != null && (bestMatchKey - timeMs).abs() <= 3000) {
            return _frameThumbnailCache[bestMatchKey];
          }
          final dur = _momentService.selectedProject?.durationMs ?? 30000;
          return await _generateFallbackVideoFrame(timeMs, dur);
        },
      );
    });
  }

  String _formatMs(int ms) {
    final totalSec = ms ~/ 1000;
    final min = totalSec ~/ 60;
    final sec = totalSec % 60;
    return '${min.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
  }

  // ── UI Builder ───────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      backgroundColor: const Color(0xFF101318),
      appBar: AppBar(
        backgroundColor: const Color(0xFF141820),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Colors.deepOrange, Colors.orangeAccent],
                ),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.movie_filter_rounded, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Video Moments',
                  style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                Text(
                  'Google AI Edge Video Moment Finder',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Open Video File',
            icon: const Icon(Icons.video_file_outlined, color: Colors.deepOrangeAccent),
            onPressed: _pickVideoFile,
          ),
          IconButton(
            tooltip: 'Sliding Window & Audio Settings',
            icon: const Icon(Icons.tune_rounded, color: Colors.white70),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: _momentService,
        builder: (context, _) {
          final project = _momentService.selectedProject;
          final isIndexing = _momentService.isIndexing;
          final isSearching = _momentService.isSearching;
          final searchResults = _momentService.searchResults;
          final mergedIntervals = _momentService.mergedIntervals;

          return Column(
            children: [
              // ── 1. Video Player Container ─────────────────────────────────
              _buildVideoPlayerSection(project),

              // ── 2. Interactive Scrubber Timeline with Highlight Bars ─────
              _buildTimelineScrubber(project, mergedIntervals),

              // ── 3. Search Bar with Quick Suggestion Chips ────────────────
              _buildSearchBar(colorScheme),

              // ── 4. Indexing Progress Banner ──────────────────────────────
              if (isIndexing) _buildIndexingBanner(),

              if (isSearching)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: LinearProgressIndicator(
                    color: Colors.deepOrange,
                    backgroundColor: Colors.white10,
                  ),
                ),

              // ── 5. Matched Moments with Frame Thumbnails ─────────────────
              Expanded(
                child: searchResults.isEmpty
                    ? _buildEmptyState(colorScheme)
                    : _buildMomentsGrid(searchResults, colorScheme),
              ),
            ],
          );
        },
      ),
    );
  }

  // ── Video Player View ────────────────────────────────────────────────────

  Widget _buildVideoPlayerSection(VideoMomentProject? project) {
    if (project == null) {
      return Container(
        height: 220,
        margin: const EdgeInsets.fromLTRB(16, 10, 16, 6),
        decoration: BoxDecoration(
          color: const Color(0xFF161B24),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: Colors.white12),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.deepOrange.withValues(alpha: 0.15),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.video_library_rounded,
                  size: 40,
                  color: Colors.deepOrangeAccent,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'No Video Loaded',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 4),
              const Text(
                'Pick a video file from your device to index and search moments',
                style: TextStyle(color: Colors.white60, fontSize: 12),
              ),
              const SizedBox(height: 14),
              ElevatedButton.icon(
                icon: const Icon(Icons.file_upload_outlined, size: 18),
                label: const Text('Pick Video File'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepOrange,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                ),
                onPressed: _pickVideoFile,
              ),
            ],
          ),
        ),
      );
    }

    final dur = _videoPlayerController?.value.duration.inMilliseconds ??
        project.durationMs;

    return Container(
      height: 250,
      margin: const EdgeInsets.fromLTRB(16, 10, 16, 6),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.6),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Actual Video Player or Real Keyframe
          Positioned.fill(
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: _togglePlayback,
              child: _videoPlayerController != null && _isVideoControllerReady
                  ? Center(
                      child: AspectRatio(
                        aspectRatio: _videoPlayerController!.value.aspectRatio,
                        child: VideoPlayer(_videoPlayerController!),
                      ),
                    )
                  : _buildSimulatedVideoFrame(dur),
            ),
          ),

          // Top Gradient Scrim with Video Title
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: IgnorePointer(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [Colors.black.withValues(alpha: 0.8), Colors.transparent],
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.videocam, color: Colors.deepOrangeAccent, size: 18),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        project.displayName,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.white24),
                      ),
                      child: Text(
                        '${_formatMs(_currentScrubberPositionMs)} / ${_formatMs(dur)}',
                        style: const TextStyle(
                          color: Colors.amberAccent,
                          fontSize: 12,
                          fontFamily: 'monospace',
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Central Play/Pause Tap Target Overlay
          Center(
            child: IgnorePointer(
              child: AnimatedOpacity(
                opacity: _isPlaying ? 0.0 : 1.0,
                duration: const Duration(milliseconds: 200),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.65),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white30, width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.5),
                        blurRadius: 12,
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.play_arrow_rounded,
                    color: Colors.white,
                    size: 44,
                  ),
                ),
              ),
            ),
          ),

          // Bottom Controls Bar
          Positioned(
            bottom: 8,
            left: 12,
            right: 12,
            child: Row(
              children: [
                IconButton(
                  icon: Icon(
                    _isPlaying ? Icons.pause_circle_filled : Icons.play_circle_filled,
                    color: Colors.deepOrangeAccent,
                    size: 30,
                  ),
                  onPressed: _togglePlayback,
                ),
                IconButton(
                  icon: Icon(
                    (_videoPlayerController?.value.volume ?? 1.0) == 0.0
                        ? Icons.volume_off_rounded
                        : Icons.volume_up_rounded,
                    color: Colors.white70,
                    size: 22,
                  ),
                  onPressed: () {
                    final ctrl = _videoPlayerController;
                    if (ctrl != null) {
                      final vol = ctrl.value.volume;
                      ctrl.setVolume(vol == 0.0 ? 1.0 : 0.0);
                      setState(() {});
                    }
                  },
                ),
                const Spacer(),
                if (_activeMomentEndMs != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.deepOrange.withValues(alpha: 0.8),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.repeat_one, size: 14, color: Colors.white),
                        const SizedBox(width: 4),
                        Text(
                          'Playing Clip (${_formatMs(_activeMomentEndMs!)})',
                          style: const TextStyle(fontSize: 11, color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSimulatedVideoFrame(int dur) {
    // Find closest cached frame for simulated display
    final closestKey = _frameThumbnailCache.keys.isEmpty
        ? null
        : _frameThumbnailCache.keys.reduce(
            (a, b) => (a - _currentScrubberPositionMs).abs() < (b - _currentScrubberPositionMs).abs() ? a : b,
          );
    final cachedBytes = closestKey != null ? _frameThumbnailCache[closestKey] : null;

    if (cachedBytes != null) {
      return Image.memory(
        cachedBytes,
        fit: BoxFit.cover,
        width: double.infinity,
        height: double.infinity,
      );
    }

    return Container(
      color: const Color(0xFF1A1F2C),
      child: const Center(
        child: Icon(Icons.movie_creation_outlined, size: 48, color: Colors.white24),
      ),
    );
  }

  // ── Scrubber Timeline with Highlight Zones ───────────────────────────────

  Widget _buildTimelineScrubber(
    VideoMomentProject? project,
    List<MergedMomentInterval> intervals,
  ) {
    if (project == null) return const SizedBox.shrink();
    final dur = project.durationMs;
    final progress = dur > 0 ? (_currentScrubberPositionMs / dur).clamp(0.0, 1.0) : 0.0;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
      child: Column(
        children: [
          // Visual Multi-interval Track
          Container(
            height: 18,
            decoration: BoxDecoration(
              color: const Color(0xFF222834),
              borderRadius: BorderRadius.circular(9),
              border: Border.all(color: Colors.white12),
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final trackWidth = constraints.maxWidth;
                return Stack(
                  clipBehavior: Clip.none,
                  children: [
                    // Highlighted moment match intervals
                    for (final interval in intervals)
                      Positioned(
                        left: ((interval.startTimeMs / dur) * trackWidth).clamp(0.0, trackWidth),
                        width: (((interval.endTimeMs - interval.startTimeMs) / dur) * trackWidth)
                            .clamp(6.0, trackWidth),
                        top: 0,
                        bottom: 0,
                        child: Container(
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Color(0xFFFF5722), Color(0xFFFF9800)],
                            ),
                            borderRadius: BorderRadius.circular(6),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.deepOrange.withValues(alpha: 0.6),
                                blurRadius: 6,
                              ),
                            ],
                          ),
                        ),
                      ),

                    // Scrubber Cursor
                    Positioned(
                      left: (progress * (trackWidth - 14)).clamp(0.0, trackWidth - 14),
                      top: 0,
                      bottom: 0,
                      child: Container(
                        width: 14,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.deepOrange, width: 3),
                          boxShadow: const [
                            BoxShadow(color: Colors.black54, blurRadius: 4),
                          ],
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),

          // Seek Slider
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              thumbColor: Colors.deepOrange,
              activeTrackColor: Colors.deepOrange.withValues(alpha: 0.8),
              inactiveTrackColor: Colors.white12,
              trackHeight: 3,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
            ),
            child: Slider(
              value: _currentScrubberPositionMs.toDouble().clamp(0.0, dur.toDouble()),
              max: dur.toDouble(),
              onChanged: (v) => _seekTo(v.toInt()),
            ),
          ),
        ],
      ),
    );
  }

  // ── Search Bar & Suggestion Chips ────────────────────────────────────────

  Widget _buildSearchBar(ColorScheme colorScheme) {
    const quickQueries = ['kitchen', 'living room', 'balcony', 'window', 'walking'];

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextField(
            controller: _searchController,
            onChanged: _onSearchQueryChanged,
            style: const TextStyle(color: Colors.white, fontSize: 14),
            decoration: InputDecoration(
              hintText: 'Search video moments ("kitchen", "balcony", "window")...',
              hintStyle: const TextStyle(color: Colors.white38, fontSize: 13),
              prefixIcon: const Icon(Icons.search_rounded, color: Colors.deepOrange),
              suffixIcon: _searchController.text.isNotEmpty
                  ? IconButton(
                      icon: const Icon(Icons.clear_rounded, color: Colors.white54),
                      onPressed: () {
                        _searchController.clear();
                        _momentService.clearSearch();
                      },
                    )
                  : null,
              filled: true,
              fillColor: const Color(0xFF1E2430),
              contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide.none,
              ),
            ),
          ),
          const SizedBox(height: 6),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: quickQueries.map((q) {
                return Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: ActionChip(
                    label: Text(
                      q,
                      style: const TextStyle(fontSize: 11, color: Colors.white70),
                    ),
                    backgroundColor: const Color(0xFF222938),
                    side: const BorderSide(color: Colors.white12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    onPressed: () {
                      _searchController.text = q;
                      _onSearchQueryChanged(q);
                    },
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  // ── Indexing Progress Banner ─────────────────────────────────────────────

  Widget _buildIndexingBanner() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.deepOrange.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.deepOrange.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.deepOrange),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Analyzing sliding windows with EmbeddingGemma 2: ${(_momentService.indexingProgress * 100).toStringAsFixed(0)}%',
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.deepOrange),
            ),
          ),
        ],
      ),
    );
  }

  // ── Moments Grid with Frame Thumbnails ───────────────────────────────────

  Widget _buildMomentsGrid(List<MomentSearchResult> moments, ColorScheme colorScheme) {
    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 6, 16, 20),
      itemCount: moments.length,
      itemBuilder: (context, index) {
        final m = moments[index];
        final isCurrentlyPlaying = _currentScrubberPositionMs >= m.startTimeMs &&
            _currentScrubberPositionMs <= m.endTimeMs;

        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF1A1F2C),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: isCurrentlyPlaying ? Colors.deepOrange : Colors.white10,
              width: isCurrentlyPlaying ? 2 : 1,
            ),
            boxShadow: [
              if (isCurrentlyPlaying)
                BoxShadow(
                  color: Colors.deepOrange.withValues(alpha: 0.3),
                  blurRadius: 10,
                  offset: const Offset(0, 2),
                ),
            ],
          ),
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: () => _seekTo(m.startTimeMs, endBoundaryMs: m.endTimeMs),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ── Frame Thumbnail Image Preview ────────────────────────────
                Stack(
                  alignment: Alignment.center,
                  children: [
                    AspectRatio(
                      aspectRatio: 16 / 9,
                      child: m.frameBytes != null
                          ? Image.memory(
                              m.frameBytes!,
                              fit: BoxFit.cover,
                            )
                          : Container(
                              color: const Color(0xFF282E3E),
                              child: const Center(
                                child: Icon(Icons.image_outlined, size: 40, color: Colors.white24),
                              ),
                            ),
                    ),

                    // Play Overlay Icon
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.6),
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white38),
                      ),
                      child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 26),
                    ),

                    // Timestamp Badge Overlay (Top Left)
                    Positioned(
                      top: 10,
                      left: 10,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.75),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.white24),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.access_time_filled, size: 12, color: Colors.amberAccent),
                            const SizedBox(width: 4),
                            Text(
                              m.formattedTimeRange,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    // Match Similarity Badge (Top Right)
                    Positioned(
                      top: 10,
                      right: 10,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFFFF5722), Color(0xFFFF9800)],
                          ),
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.deepOrange.withValues(alpha: 0.5),
                              blurRadius: 6,
                            ),
                          ],
                        ),
                        child: Text(
                          '${m.formattedScore} Match',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),

                // ── Card Details & Action Row ────────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Moment at ${_formatMs(m.timeMs)} (${((m.endTimeMs - m.startTimeMs) / 1000).toStringAsFixed(1)}s window)',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 13,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              'Matched by multimodal query "${_searchController.text.trim()}"',
                              style: const TextStyle(color: Colors.white54, fontSize: 11),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        tooltip: 'Save Clip',
                        icon: const Icon(Icons.bookmark_add_outlined, color: Colors.deepOrangeAccent),
                        onPressed: () {
                          _momentService.saveMomentClip(m, searchQuery: _searchController.text);
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text('Saved clip: ${m.formattedTimeRange}'),
                              backgroundColor: Colors.deepOrange,
                            ),
                          );
                        },
                      ),
                      FilledButton.tonalIcon(
                        icon: const Icon(Icons.play_arrow, size: 16),
                        label: const Text('Play'),
                        style: FilledButton.styleFrom(
                          backgroundColor: Colors.deepOrange.withValues(alpha: 0.2),
                          foregroundColor: Colors.deepOrangeAccent,
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        ),
                        onPressed: () => _seekTo(m.startTimeMs, endBoundaryMs: m.endTimeMs),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ── Empty State ──────────────────────────────────────────────────────────

  Widget _buildEmptyState(ColorScheme colorScheme) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF1E2430),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white10),
            ),
            child: const Icon(Icons.movie_filter_outlined, size: 48, color: Colors.deepOrangeAccent),
          ),
          const SizedBox(height: 16),
          const Text(
            'Type a query to find video moments',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          const SizedBox(height: 6),
          const Text(
            'EmbeddingGemma 2 indexes text, keyframes, and audio tracks',
            style: TextStyle(fontSize: 12, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  // ── Settings Dialog ──────────────────────────────────────────────────────

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              backgroundColor: const Color(0xFF1A1F2C),
              title: const Text('Sliding Window & Audio Settings', style: TextStyle(color: Colors.white)),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ListTile(
                    title: const Text('Window Duration', style: TextStyle(color: Colors.white)),
                    trailing: DropdownButton<int>(
                      dropdownColor: const Color(0xFF222938),
                      value: _windowDurationSec,
                      items: const [
                        DropdownMenuItem(value: 2, child: Text('2 seconds', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 3, child: Text('3 seconds', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 5, child: Text('5 seconds', style: TextStyle(color: Colors.white))),
                      ],
                      onChanged: (v) {
                        if (v != null) {
                          setDialogState(() => _windowDurationSec = v);
                          setState(() => _windowDurationSec = v);
                        }
                      },
                    ),
                  ),
                  ListTile(
                    title: const Text('Frames Per Window', style: TextStyle(color: Colors.white)),
                    trailing: DropdownButton<int>(
                      dropdownColor: const Color(0xFF222938),
                      value: _framesPerWindow,
                      items: const [
                        DropdownMenuItem(value: 1, child: Text('1 frame', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 2, child: Text('2 frames', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 4, child: Text('4 frames', style: TextStyle(color: Colors.white))),
                      ],
                      onChanged: (v) {
                        if (v != null) {
                          setDialogState(() => _framesPerWindow = v);
                          setState(() => _framesPerWindow = v);
                        }
                      },
                    ),
                  ),
                  ListTile(
                    title: const Text('Top Matches (Top-K)', style: TextStyle(color: Colors.white)),
                    trailing: DropdownButton<int>(
                      dropdownColor: const Color(0xFF222938),
                      value: _topK,
                      items: const [
                        DropdownMenuItem(value: 3, child: Text('3 results', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 5, child: Text('5 results', style: TextStyle(color: Colors.white))),
                        DropdownMenuItem(value: 10, child: Text('10 results', style: TextStyle(color: Colors.white))),
                      ],
                      onChanged: (v) {
                        if (v != null) {
                          setDialogState(() => _topK = v);
                          setState(() => _topK = v);
                        }
                      },
                    ),
                  ),
                  SwitchListTile(
                    title: const Text('Interleave 16 kHz Audio', style: TextStyle(color: Colors.white)),
                    subtitle: const Text('Sync mono PCM audio chunks with frames', style: TextStyle(color: Colors.white54)),
                    value: _includeAudio,
                    activeThumbColor: Colors.deepOrange,
                    activeTrackColor: Colors.deepOrange.withValues(alpha: 0.5),
                    onChanged: (v) {
                      setDialogState(() => _includeAudio = v);
                      setState(() => _includeAudio = v);
                    },
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Done', style: TextStyle(color: Colors.deepOrangeAccent)),
                ),
              ],
            );
          },
        );
      },
    );
  }
}
