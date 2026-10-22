import 'dart:async';
import 'package:flutter/foundation.dart';

import 'embedding_plugin.dart';
import 'semantic_retrieval_service.dart';

/// Representation of a media item (photo or video) in the gallery.
class MediaAsset {
  final String id;
  final String? path;
  final Uint8List? bytes;
  final String displayName;
  final int dateTaken;
  final bool isVideo;
  final int? durationMs;
  final String? locationName;

  const MediaAsset({
    required this.id,
    this.path,
    this.bytes,
    required this.displayName,
    this.dateTaken = 0,
    this.isVideo = false,
    this.durationMs,
    this.locationName,
  });

  Map<String, dynamic> toMap() => {
        'id': id,
        'path': path,
        'displayName': displayName,
        'dateTaken': dateTaken,
        'isVideo': isVideo,
        'durationMs': durationMs,
        'locationName': locationName,
      };

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is MediaAsset && other.id == id);

  @override
  int get hashCode => id.hashCode;
}

/// A matched media asset with its similarity score.
class MatchedAssetResult {
  final MediaAsset asset;
  final double similarity;

  const MatchedAssetResult({
    required this.asset,
    required this.similarity,
  });

  /// Formatted similarity percentage (e.g. "87%").
  String get formattedScore => '${(similarity * 100).round()}%';
}

/// Progress state of gallery media indexing.
class GalleryIndexingProgress {
  final int processedCount;
  final int totalCount;
  final bool isIndexing;
  final List<String> recentIndexedIds;

  const GalleryIndexingProgress({
    this.processedCount = 0,
    this.totalCount = 0,
    this.isIndexing = false,
    this.recentIndexedIds = const [],
  });

  double get progress => totalCount > 0 ? (processedCount / totalCount).clamp(0.0, 1.0) : 0.0;
  int get percentage => (progress * 100).round();
}

/// High-level service managing media gallery indexing, text-to-media search,
/// similar-image search, and live camera search with EmbeddingGemma 2.
class GallerySearchService extends ChangeNotifier {
  final SemanticRetrievalService retrievalService;

  final List<MediaAsset> _assets = [];
  final List<MatchedAssetResult> _searchResults = [];
  final List<MatchedAssetResult> _cameraSearchResults = [];

  String _currentQuery = '';
  MediaAsset? _selectedSimilarAsset;
  bool _isSearching = false;
  bool _isCameraSearching = false;

  GalleryIndexingProgress _indexingProgress = const GalleryIndexingProgress();
  bool _isDisposed = false;

  GallerySearchService({required this.retrievalService});

  /// All registered media assets.
  List<MediaAsset> get assets => List.unmodifiable(_assets);

  /// Current search results (ranked by descending similarity).
  List<MatchedAssetResult> get searchResults => List.unmodifiable(_searchResults);

  /// Realtime camera search results.
  List<MatchedAssetResult> get cameraSearchResults => List.unmodifiable(_cameraSearchResults);

  /// Current search query string.
  String get currentQuery => _currentQuery;

  /// Currently selected seed asset for similar-image search.
  MediaAsset? get selectedSimilarAsset => _selectedSimilarAsset;

  /// Whether a search is currently executing.
  bool get isSearching => _isSearching;

  /// Whether a search or visual similarity filter is actively being displayed.
  bool get isSearchActive => _currentQuery.isNotEmpty || _selectedSimilarAsset != null;

  /// Whether a live camera frame search is currently executing.
  bool get isCameraSearching => _isCameraSearching;

  /// Indexing progress.
  GalleryIndexingProgress get indexingProgress => _indexingProgress;

  /// Adds a list of media assets to the gallery catalog.
  void addAssets(List<MediaAsset> newAssets) {
    for (final a in newAssets) {
      if (!_assets.any((existing) => existing.id == a.id)) {
        _assets.add(a);
      }
    }
    _updateIndexingProgress(isIndexing: _indexingProgress.isIndexing);
    notifyListeners();
  }

  /// Removes an asset by ID from the catalog and the vector database.
  Future<void> removeAsset(String id) async {
    _assets.removeWhere((a) => a.id == id);
    _searchResults.removeWhere((r) => r.asset.id == id);
    _cameraSearchResults.removeWhere((r) => r.asset.id == id);
    await retrievalService.deleteRecord(id);
    _updateIndexingProgress(isIndexing: false);
    notifyListeners();
  }

  /// Removes all assets from the catalog and wipes the vector store.
  Future<void> clearAll() async {
    _assets.clear();
    _searchResults.clear();
    _cameraSearchResults.clear();
    _currentQuery = '';
    _selectedSimilarAsset = null;
    await retrievalService.deleteAllRecords();
    _updateIndexingProgress(isIndexing: false);
    notifyListeners();
  }

  /// Indexes [newAssets] by adding them to the catalog and updating the vector store.
  Future<void> indexAssets(List<MediaAsset> newAssets) async {
    addAssets(newAssets);
    await indexPendingAssets(loadThumbnailBytes: (a) async => a.bytes);
  }

  /// Forces a complete re-indexing of all gallery assets into the vector store.
  Future<void> reindexAllAssets({
    required Future<Uint8List?> Function(MediaAsset asset) loadThumbnailBytes,
  }) async {
    if (_assets.isEmpty) return;
    await retrievalService.deleteAllRecords();
    await indexPendingAssets(loadThumbnailBytes: loadThumbnailBytes);
  }

  /// Indexes any assets that are not yet present in the vector store.
  Future<void> indexPendingAssets({
    required Future<Uint8List?> Function(MediaAsset asset) loadThumbnailBytes,
  }) async {
    if (!EmbeddingPlugin().isInitialized) {
      debugPrint('[GallerySearchService] Model not initialized yet. Deferring indexing until model is loaded.');
      _updateIndexingProgress(
        processedCount: retrievalService.indexedCount,
        totalCount: _assets.length,
        isIndexing: false,
      );
      return;
    }

    final unindexed = _assets.where((a) => !retrievalService.isRecordIndexed(a.id)).toList();
    if (unindexed.isEmpty) {
      _updateIndexingProgress(
        processedCount: retrievalService.indexedCount,
        totalCount: _assets.length,
        isIndexing: false,
      );
      return;
    }

    _updateIndexingProgress(
      processedCount: retrievalService.indexedCount,
      totalCount: _assets.length,
      isIndexing: true,
    );

    int count = retrievalService.indexedCount;
    for (final asset in unindexed) {
      if (_isDisposed) break;
      try {
        final bytes = asset.bytes ?? await loadThumbnailBytes(asset);
        if (bytes != null && bytes.isNotEmpty) {
          await retrievalService.addRecord(
            asset.id,
            bytes,
            metadata: {
              'displayName': asset.displayName,
              'isVideo': asset.isVideo.toString(),
              'locationName': asset.locationName ?? '',
            },
          );
        }
      } catch (e) {
        debugPrint('[GallerySearchService] Failed to index asset ${asset.id}: $e');
      }
      count++;
      _updateIndexingProgress(
        processedCount: count,
        totalCount: _assets.length,
        isIndexing: true,
      );
    }

    _updateIndexingProgress(
      processedCount: retrievalService.indexedCount,
      totalCount: _assets.length,
      isIndexing: false,
    );
  }

  /// Executes natural-language search over the gallery.
  Future<void> search(
    String query, {
    double minScore = 0.15,
    int limit = 100,
  }) async {
    _currentQuery = query.trim();
    _selectedSimilarAsset = null;

    if (_currentQuery.isEmpty) {
      _searchResults.clear();
      _isSearching = false;
      notifyListeners();
      return;
    }

    _isSearching = true;
    notifyListeners();

    try {
      final records = await retrievalService.retrieveEntities(_currentQuery, limit: limit);
      final assetMap = {for (final a in _assets) a.id: a};

      _searchResults.clear();
      for (final rec in records) {
        final score = rec.similarityScore ?? 0.0;
        if (score < minScore) continue;
        final asset = assetMap[rec.id];
        if (asset != null) {
          _searchResults.add(MatchedAssetResult(asset: asset, similarity: score));
        }
      }
      _searchResults.sort((a, b) => b.similarity.compareTo(a.similarity));
    } catch (e) {
      debugPrint('[GallerySearchService] Search error: $e');
    } finally {
      _isSearching = false;
      notifyListeners();
    }
  }

  /// Searches for assets visually/semantically similar to [seedAsset].
  Future<void> searchSimilar(
    MediaAsset seedAsset, {
    required Uint8List seedImageBytes,
    double minScore = 0.20,
    int limit = 100,
  }) async {
    _selectedSimilarAsset = seedAsset;
    _currentQuery = '';
    _isSearching = true;
    notifyListeners();

    try {
      final records = await retrievalService.retrieveEntitiesByImage(
        seedImageBytes,
        seedId: seedAsset.id,
        limit: limit,
      );
      final assetMap = {for (final a in _assets) a.id: a};

      _searchResults.clear();
      for (final rec in records) {
        final score = rec.similarityScore ?? 0.0;
        if (score < minScore) continue;
        final asset = assetMap[rec.id];
        if (asset != null) {
          _searchResults.add(MatchedAssetResult(asset: asset, similarity: score));
        }
      }
      _searchResults.sort((a, b) => b.similarity.compareTo(a.similarity));
    } catch (e) {
      debugPrint('[GallerySearchService] Similar search error: $e');
    } finally {
      _isSearching = false;
      notifyListeners();
    }
  }

  /// Processes a single live camera frame to find matching assets.
  Future<void> processCameraFrame(
    Uint8List frameBytes, {
    double minScore = 0.15,
    int limit = 10,
  }) async {
    if (_isCameraSearching) return;
    _isCameraSearching = true;

    try {
      final records = await retrievalService.retrieveEntitiesByImage(frameBytes, limit: limit);
      final assetMap = {for (final a in _assets) a.id: a};

      final matched = <MatchedAssetResult>[];
      for (final rec in records) {
        final score = rec.similarityScore ?? 0.0;
        if (score < minScore) continue;
        final asset = assetMap[rec.id];
        if (asset != null) {
          matched.add(MatchedAssetResult(asset: asset, similarity: score));
        }
      }
      matched.sort((a, b) => b.similarity.compareTo(a.similarity));

      _cameraSearchResults.clear();
      _cameraSearchResults.addAll(matched);
      notifyListeners();
    } catch (e) {
      debugPrint('[GallerySearchService] Camera search error: $e');
    } finally {
      _isCameraSearching = false;
    }
  }

  /// Convenience method: searches for assets similar to [asset].
  Future<void> searchSimilarToAsset(MediaAsset asset) async {
    if (asset.bytes != null) {
      await searchSimilar(asset, seedImageBytes: asset.bytes!);
    } else {
      _selectedSimilarAsset = asset;
      _currentQuery = asset.displayName;
      await search(asset.displayName);
    }
  }

  /// Convenience alias for [processCameraFrame].
  Future<void> searchByCameraFrame(Uint8List frameBytes) => processCameraFrame(frameBytes);

  /// Clears active search results.
  void clearSearch() {
    _currentQuery = '';
    _selectedSimilarAsset = null;
    _searchResults.clear();
    _cameraSearchResults.clear();
    _isSearching = false;
    notifyListeners();
  }

  void _updateIndexingProgress({
    int? processedCount,
    int? totalCount,
    required bool isIndexing,
  }) {
    final processed = processedCount ?? retrievalService.indexedCount;
    final total = totalCount ?? _assets.length;
    _indexingProgress = GalleryIndexingProgress(
      processedCount: processed,
      totalCount: total,
      isIndexing: isIndexing,
      recentIndexedIds: retrievalService.indexedIds.take(12).toList(),
    );
    if (!_isDisposed) {
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    super.dispose();
  }
}
