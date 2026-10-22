import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';

import 'embed_part.dart';
import 'semantic_retrieval_service.dart';

/// Configuration and state for a video moment indexing project.
class VideoMomentProject {
  final String id;
  final String videoPath;
  final String displayName;
  final int durationMs;
  final int windowDurationSec;
  final int framesPerWindow;
  final int overlapDurationSec;
  final bool includeAudio;
  final int topK;
  final bool isProcessed;
  final Uint8List? thumbnailBytes;

  const VideoMomentProject({
    required this.id,
    required this.videoPath,
    required this.displayName,
    required this.durationMs,
    this.windowDurationSec = 2,
    this.framesPerWindow = 2,
    this.overlapDurationSec = 0,
    this.includeAudio = true,
    this.topK = 5,
    this.isProcessed = false,
    this.thumbnailBytes,
  });

  VideoMomentProject copyWith({
    String? id,
    String? videoPath,
    String? displayName,
    int? durationMs,
    int? windowDurationSec,
    int? framesPerWindow,
    int? overlapDurationSec,
    bool? includeAudio,
    int? topK,
    bool? isProcessed,
    Uint8List? thumbnailBytes,
  }) {
    return VideoMomentProject(
      id: id ?? this.id,
      videoPath: videoPath ?? this.videoPath,
      displayName: displayName ?? this.displayName,
      durationMs: durationMs ?? this.durationMs,
      windowDurationSec: windowDurationSec ?? this.windowDurationSec,
      framesPerWindow: framesPerWindow ?? this.framesPerWindow,
      overlapDurationSec: overlapDurationSec ?? this.overlapDurationSec,
      includeAudio: includeAudio ?? this.includeAudio,
      topK: topK ?? this.topK,
      isProcessed: isProcessed ?? this.isProcessed,
      thumbnailBytes: thumbnailBytes ?? this.thumbnailBytes,
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'videoPath': videoPath,
        'displayName': displayName,
        'durationMs': durationMs,
        'windowDurationSec': windowDurationSec,
        'framesPerWindow': framesPerWindow,
        'overlapDurationSec': overlapDurationSec,
        'includeAudio': includeAudio,
        'topK': topK,
        'isProcessed': isProcessed,
      };
}

/// A matched video moment ranked by semantic similarity.
class MomentSearchResult {
  final String id;
  final int startTimeMs;
  final int endTimeMs;
  final int timeMs;
  final double similarityScore;
  final Uint8List? frameBytes;

  const MomentSearchResult({
    required this.id,
    required this.startTimeMs,
    required this.endTimeMs,
    required this.timeMs,
    required this.similarityScore,
    this.frameBytes,
  });

  String get formattedTimeRange {
    final start = _formatTimestamp(startTimeMs);
    final end = _formatTimestamp(endTimeMs);
    return '$start - $end';
  }

  String get formattedScore => '${(similarityScore * 100).round()}%';

  static String _formatTimestamp(int ms) {
    final totalSec = ms ~/ 1000;
    final min = totalSec ~/ 60;
    final sec = totalSec % 60;
    final minStr = min.toString().padLeft(2, '0');
    final secStr = sec.toString().padLeft(2, '0');
    return '$minStr:$secStr';
  }
}

/// Continuous interval formed by merging overlapping or adjacent moments.
/// Used to highlight match zones on the video timeline scrubber.
class MergedMomentInterval {
  final int startTimeMs;
  final int endTimeMs;
  final Set<String> resultIds;

  const MergedMomentInterval({
    required this.startTimeMs,
    required this.endTimeMs,
    this.resultIds = const {},
  });

  bool containsTime(int timeMs) => timeMs >= startTimeMs && timeMs <= endTimeMs;
}

/// A saved or exported video moment clip.
class SavedMomentClip {
  final String id;
  final String projectId;
  final String videoDisplayName;
  final int startTimeMs;
  final int endTimeMs;
  final String searchQuery;
  final Uint8List? thumbnailBytes;
  final DateTime savedAt;

  const SavedMomentClip({
    required this.id,
    required this.projectId,
    required this.videoDisplayName,
    required this.startTimeMs,
    required this.endTimeMs,
    required this.searchQuery,
    this.thumbnailBytes,
    required this.savedAt,
  });

  String get formattedDuration {
    final durSec = ((endTimeMs - startTimeMs) / 1000.0).toStringAsFixed(1);
    return '${durSec}s';
  }
}

/// Service managing video slicing, audio-video multimodal indexing, and
/// natural-language video moment search using EmbeddingGemma 2.
class VideoMomentService extends ChangeNotifier {
  final SemanticRetrievalService retrievalService;

  final List<VideoMomentProject> _projects = [];
  VideoMomentProject? _selectedProject;

  final List<MomentSearchResult> _searchResults = [];
  final List<MergedMomentInterval> _mergedIntervals = [];
  final List<SavedMomentClip> _savedClips = [];

  String _currentQuery = '';
  bool _isIndexing = false;
  double _indexingProgress = 0.0;
  bool _isSearching = false;
  String? _selectedResultId;

  VideoMomentService({required this.retrievalService});

  List<VideoMomentProject> get projects => List.unmodifiable(_projects);
  VideoMomentProject? get selectedProject => _selectedProject;
  List<MomentSearchResult> get searchResults => List.unmodifiable(_searchResults);
  List<MergedMomentInterval> get mergedIntervals => List.unmodifiable(_mergedIntervals);
  List<SavedMomentClip> get savedClips => List.unmodifiable(_savedClips);

  String get currentQuery => _currentQuery;
  bool get isIndexing => _isIndexing;
  double get indexingProgress => _indexingProgress;
  bool get isSearching => _isSearching;
  String? get selectedResultId => _selectedResultId;

  /// Registers a new video project.
  void addProject(VideoMomentProject project) {
    _projects.removeWhere((p) => p.id == project.id);
    _projects.add(project);
    _selectedProject ??= project;
    notifyListeners();
  }

  /// Selects the active video project.
  void selectProject(VideoMomentProject? project) {
    _selectedProject = project;
    _searchResults.clear();
    _mergedIntervals.clear();
    _currentQuery = '';
    _selectedResultId = null;
    notifyListeners();
  }

  /// Removes a project and clears all its indexed vectors.
  Future<void> deleteProject(String projectId) async {
    _projects.removeWhere((p) => p.id == projectId);
    if (_selectedProject?.id == projectId) {
      _selectedProject = _projects.isNotEmpty ? _projects.first : null;
    }
    _savedClips.removeWhere((c) => c.projectId == projectId);
    await retrievalService.deleteRecordsWithIdPrefix('$projectId/');
    notifyListeners();
  }

  /// Computes sliding time windows for a video of duration [durationMs].
  static List<({int startTimeMs, int endTimeMs, List<int> sampleTimesMs})> computeWindows({
    required int durationMs,
    int windowDurationSec = 2,
    int framesPerWindow = 2,
    int overlapDurationSec = 0,
  }) {
    if (durationMs <= 0) return [];
    final windowDurationMs = windowDurationSec * 1000;
    final overlapMs = overlapDurationSec * 1000;
    final stepMs = math.max(1000, windowDurationMs - overlapMs);

    final windows = <({int startTimeMs, int endTimeMs, List<int> sampleTimesMs})>[];
    int windowStartMs = 0;

    while (windowStartMs < durationMs) {
      final windowEndMs = math.min(windowStartMs + windowDurationMs, durationMs);
      final curWindowDuration = windowEndMs - windowStartMs;
      final sampleTimestamps = <int>[];

      for (int i = 0; i < framesPerWindow; i++) {
        final t = framesPerWindow == 1
            ? windowStartMs + (curWindowDuration ~/ 2)
            : windowStartMs + (i * curWindowDuration ~/ (framesPerWindow - 1));
        sampleTimestamps.add(t.clamp(0, durationMs));
      }

      windows.add((
        startTimeMs: windowStartMs,
        endTimeMs: windowEndMs,
        sampleTimesMs: sampleTimestamps,
      ));
      windowStartMs += stepMs;
    }

    return windows;
  }

  /// Indexes the selected video project with frame and audio extraction.
  Future<void> indexProject({
    required VideoMomentProject project,
    required Future<List<Uint8List>> Function(List<int> timestampsMs) extractFrames,
    Future<Int16List?> Function(int startMs, int endMs)? extractAudioPcm,
  }) async {
    _isIndexing = true;
    _indexingProgress = 0.0;
    notifyListeners();

    try {
      // Clear previously indexed records for this project
      await retrievalService.deleteRecordsWithIdPrefix('${project.id}/');

      final windows = computeWindows(
        durationMs: project.durationMs,
        windowDurationSec: project.windowDurationSec,
        framesPerWindow: project.framesPerWindow,
        overlapDurationSec: project.overlapDurationSec,
      );

      final total = windows.length;
      for (int idx = 0; idx < total; idx++) {
        final win = windows[idx];
        final recordId = '${project.id}/${win.startTimeMs}-${win.endTimeMs}';

        final frames = await extractFrames(win.sampleTimesMs);
        if (frames.isEmpty) continue;

        Int16List? audioPcm;
        if (project.includeAudio && extractAudioPcm != null) {
          try {
            audioPcm = await extractAudioPcm(win.startTimeMs, win.endTimeMs);
          } catch (_) {}
        }

        final parts = <EmbedPart>[];
        for (int f = 0; f < frames.length; f++) {
          final timeStr = formatTimestamp(win.sampleTimesMs[f]);
          if (audioPcm != null && audioPcm.isNotEmpty) {
            // Slice audio roughly per frame
            final sliceLen = audioPcm.length ~/ frames.length;
            final sliceStart = f * sliceLen;
            final sliceEnd = math.min(audioPcm.length, sliceStart + sliceLen);
            if (sliceEnd > sliceStart) {
              final slice = audioPcm.sublist(sliceStart, sliceEnd);
              parts.add(EmbedPart.text(timeStr));
              parts.add(EmbedPart.audio(slice));
            }
          }
          parts.add(EmbedPart.text(timeStr));
          parts.add(EmbedPart.image(frames[f]));
        }

        await retrievalService.addMultimodalRecord(
          recordId,
          parts,
          metadata: {
            'parent_id': project.id,
            'start_time_ms': win.startTimeMs.toString(),
            'end_time_ms': win.endTimeMs.toString(),
          },
        );

        _indexingProgress = (idx + 1) / total;
        notifyListeners();
      }

      // Mark project as processed
      final updated = project.copyWith(isProcessed: true);
      addProject(updated);
      _selectedProject = updated;
    } catch (e) {
      debugPrint('[VideoMomentService] Error indexing video project: $e');
    } finally {
      _isIndexing = false;
      notifyListeners();
    }
  }

  /// Searches the current video project for moments matching [query].
  Future<void> searchMoments(
    String query, {
    Future<Uint8List?> Function(int timeMs)? extractPreviewFrame,
  }) async {
    final proj = _selectedProject;
    if (proj == null) return;

    _currentQuery = query.trim();
    if (_currentQuery.isEmpty) {
      clearSearch();
      return;
    }

    _isSearching = true;
    _searchResults.clear();
    _mergedIntervals.clear();
    notifyListeners();

    try {
      final records = await retrievalService.retrieveEntities(
        _currentQuery,
        limit: proj.topK,
        metadataFilter: {'parent_id': proj.id},
      );

      final results = <MomentSearchResult>[];
      for (final rec in records) {
        final startMs = int.tryParse(rec.metadata['start_time_ms'] ?? '');
        final endMs = int.tryParse(rec.metadata['end_time_ms'] ?? '');
        if (startMs == null || endMs == null) continue;

        final midMs = (startMs + endMs) ~/ 2;
        Uint8List? frameBytes;
        if (extractPreviewFrame != null) {
          try {
            frameBytes = await extractPreviewFrame(midMs);
          } catch (_) {}
        }

        results.add(MomentSearchResult(
          id: rec.id,
          startTimeMs: startMs,
          endTimeMs: endMs,
          timeMs: midMs,
          similarityScore: rec.similarityScore ?? 0.0,
          frameBytes: frameBytes,
        ));
      }

      results.sort((a, b) => b.similarityScore.compareTo(a.similarityScore));
      _searchResults.addAll(results);
      _mergedIntervals.addAll(computeMergedIntervals(results));
      if (_searchResults.isNotEmpty) {
        _selectedResultId = _searchResults.first.id;
      }
    } catch (e) {
      debugPrint('[VideoMomentService] Search error: $e');
    } finally {
      _isSearching = false;
      notifyListeners();
    }
  }

  /// Clears active moment search results and selection.
  void clearSearch() {
    _currentQuery = '';
    _searchResults.clear();
    _mergedIntervals.clear();
    _isSearching = false;
    _selectedResultId = null;
    notifyListeners();
  }

  /// Selects a specific moment search result by ID.
  void selectResult(String id) {
    _selectedResultId = id;
    notifyListeners();
  }

  /// Saves a moment as a favorite / exported clip.
  void saveClip({
    required MomentSearchResult moment,
    String? customQuery,
  }) {
    final proj = _selectedProject;
    if (proj == null) return;

    final clip = SavedMomentClip(
      id: '${proj.id}_${moment.startTimeMs}_${DateTime.now().millisecondsSinceEpoch}',
      projectId: proj.id,
      videoDisplayName: proj.displayName,
      startTimeMs: moment.startTimeMs,
      endTimeMs: moment.endTimeMs,
      searchQuery: customQuery ?? _currentQuery,
      thumbnailBytes: moment.frameBytes,
      savedAt: DateTime.now(),
    );

    _savedClips.add(clip);
    notifyListeners();
  }

  /// Convenience alias for [saveClip].
  void saveMomentClip(
    MomentSearchResult moment, {
    String? searchQuery,
  }) =>
      saveClip(moment: moment, customQuery: searchQuery);

  /// Deletes a saved moment clip.
  void deleteSavedClip(String clipId) {
    _savedClips.removeWhere((c) => c.id == clipId);
    notifyListeners();
  }

  /// Merges overlapping or adjacent moments within 500 ms for timeline highlight bars.
  static List<MergedMomentInterval> computeMergedIntervals(
    List<MomentSearchResult> results, {
    int overlapToleranceMs = 500,
  }) {
    if (results.isEmpty) return [];
    final sorted = List<MomentSearchResult>.from(results)
      ..sort((a, b) => a.startTimeMs.compareTo(b.startTimeMs));

    final merged = <MergedMomentInterval>[];

    for (final res in sorted) {
      if (merged.isNotEmpty && res.startTimeMs <= merged.last.endTimeMs + overlapToleranceMs) {
        final last = merged.last;
        merged[merged.length - 1] = MergedMomentInterval(
          startTimeMs: last.startTimeMs,
          endTimeMs: math.max(last.endTimeMs, res.endTimeMs),
          resultIds: {...last.resultIds, res.id},
        );
      } else {
        merged.add(MergedMomentInterval(
          startTimeMs: res.startTimeMs,
          endTimeMs: res.endTimeMs,
          resultIds: {res.id},
        ));
      }
    }

    return merged;
  }

  /// Formats milliseconds into mm:ss timestamp.
  static String formatTimestamp(int ms) {
    final totalSec = ms ~/ 1000;
    final min = totalSec ~/ 60;
    final sec = totalSec % 60;
    return '${min.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
  }
}
