import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'embed_part.dart';
import 'embedding_plugin.dart';
import 'embedding_plugin_web.dart'
    if (dart.library.io) 'embedding_plugin_stub.dart';

/// Single item retrieved from the semantic retrieval vector database.
class SemanticRetrievalResult {
  /// Unique identifier of the indexed entity (e.g. photo path, video window id).
  final String id;

  /// The embedding vector associated with this record.
  final List<double> embeddings;

  /// Metadata associated with this record (e.g. parent_id, timestamp, tags).
  final Map<String, String> metadata;

  /// Cosine similarity score between the query and this record (0.0 to 1.0).
  final double? similarityScore;

  const SemanticRetrievalResult({
    required this.id,
    this.embeddings = const [],
    this.metadata = const {},
    this.similarityScore,
  });

  factory SemanticRetrievalResult.fromMap(Map<dynamic, dynamic> map) {
    final rawScores = map['embeddings'];
    final emb = rawScores is List ? rawScores.map((e) => (e as num).toDouble()).toList() : <double>[];
    final rawMeta = map['metadata'];
    final meta = rawMeta is Map ? rawMeta.map((k, v) => MapEntry(k.toString(), v.toString())) : <String, String>{};
    final rawSim = map['similarityScore'];
    final sim = rawSim != null ? (rawSim as num).toDouble() : null;

    return SemanticRetrievalResult(
      id: map['id']?.toString() ?? '',
      embeddings: emb,
      metadata: meta,
      similarityScore: sim,
    );
  }

  Map<String, dynamic> toMap() => {
        'id': id,
        'embeddings': embeddings,
        'metadata': metadata,
        'similarityScore': similarityScore,
      };

  @override
  String toString() => 'SemanticRetrievalResult(id: $id, score: ${similarityScore?.toStringAsFixed(3)})';
}

/// A stored vector record in the in-memory or fallback vector database.
class _StoredRecord {
  final String id;
  final List<double> embeddings;
  final Map<String, String> metadata;

  const _StoredRecord({
    required this.id,
    required this.embeddings,
    required this.metadata,
  });
}

/// Service providing on-device vector storage and multimodal semantic retrieval
/// powered by EmbeddingGemma 2 and MediaPipe Tasks Retrieval.
///
/// On Android, delegates to MediaPipe's `SqliteVectorStore` and `SemanticRetriever`.
/// On Web and desktop, uses an in-memory/OPFS-backed vector store with SIMD/Float32 dot product.
class SemanticRetrievalService implements ChangeNotifier {
  static const MethodChannel _channel = MethodChannel('semantic_retrieval_plugin');

  final String databaseName;
  final int dimension;
  final EmbeddingPlugin _embedder;

  final Map<String, _StoredRecord> _inMemoryStore = {};
  final Set<String> _indexedIds = <String>{};
  final List<VoidCallback> _listeners = [];

  bool _isDisposed = false;
  bool _useNativeBackend = false;

  SemanticRetrievalService({
    this.databaseName = 'semantic_retrieval_gemma.db',
    this.dimension = 768,
    EmbeddingPlugin? embedder,
  }) : _embedder = embedder ?? EmbeddingPlugin();

  /// Currently indexed record identifiers.
  Set<String> get indexedIds => Set.unmodifiable(_indexedIds);

  /// Number of indexed vector records.
  int get indexedCount => _indexedIds.length;

  /// Whether a record with [id] is present in the index.
  bool isRecordIndexed(String id) => _indexedIds.contains(id);

  /// Initializes the vector store and synchronizes indexed record IDs.
  Future<void> initialize() async {
    if (_isDisposed) return;

    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      try {
        final success = await _channel.invokeMethod<bool>('initRetriever', {
          'databaseName': databaseName,
          'dimension': dimension,
        });
        _useNativeBackend = success ?? false;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native backend unavailable ($e), using local vector store.');
        _useNativeBackend = false;
      }
    } else {
      _useNativeBackend = false;
    }

    await loadFromDatabase();
  }

  /// Reloads indexed IDs from the persistent store.
  Future<void> loadFromDatabase() async {
    if (_useNativeBackend) {
      try {
        final List<dynamic>? ids = await _channel.invokeMethod('getAllRecordIds');
        if (ids != null) {
          _indexedIds.clear();
          _indexedIds.addAll(ids.cast<String>());
          notifyListeners();
        }
        return;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Error fetching native ids: $e');
      }
    }
    _indexedIds.clear();
    _indexedIds.addAll(_inMemoryStore.keys);
    notifyListeners();
  }

  /// Indexes an image under [id] with optional [metadata].
  Future<void> addRecord(
    String id,
    Uint8List imageData, {
    Map<String, String> metadata = const {},
  }) async {
    await addRecords([
      (id: id, images: [imageData], metadata: metadata)
    ]);
  }

  /// Indexes a list of image keyframes under [id] with optional [metadata].
  Future<void> addImageRecordList(
    String id,
    List<Uint8List> imageDataList, {
    Map<String, String> metadata = const {},
  }) async {
    await addRecords([
      (id: id, images: imageDataList, metadata: metadata)
    ]);
  }

  /// Indexes an interleaved multimodal sequence of parts (frames, audio slices, timestamps).
  Future<void> addMultimodalRecord(
    String id,
    List<EmbedPart> parts, {
    Map<String, String> metadata = const {},
  }) async {
    if (parts.isEmpty) return;

    if (_useNativeBackend) {
      try {
        await _channel.invokeMethod('addMultimodalRecord', {
          'id': id,
          'parts': parts.map((p) => p.toMap()).toList(),
          'metadata': metadata,
        });
        _indexedIds.add(id);
        notifyListeners();
        return;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native addMultimodalRecord failed: $e. Falling back.');
      }
    }

    // Fallback: extract embedding via EmbeddingPlugin and store in memory
    final embedding = await _embedder.getEmbeddingFromParts(parts);
    _inMemoryStore[id] = _StoredRecord(id: id, embeddings: embedding, metadata: metadata);
    _indexedIds.add(id);
    notifyListeners();
  }

  /// Batch indexes multiple records.
  Future<void> addRecords(
    List<({String id, List<Uint8List> images, Map<String, String> metadata})> records,
  ) async {
    if (records.isEmpty) return;

    if (_useNativeBackend) {
      try {
        final payload = records
            .map((r) => {
                  'id': r.id,
                  'images': r.images,
                  'metadata': r.metadata,
                })
            .toList();
        await _channel.invokeMethod('addRecordsBatch', {'records': payload});
        for (final r in records) {
          _indexedIds.add(r.id);
        }
        notifyListeners();
        return;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native addRecordsBatch failed: $e. Falling back.');
      }
    }

    // Fallback in-memory
    for (final r in records) {
      final embedding = await _embedder.getEmbeddingFromImages(r.images);
      _inMemoryStore[r.id] = _StoredRecord(id: r.id, embeddings: embedding, metadata: r.metadata);
      _indexedIds.add(r.id);
    }
    notifyListeners();
  }

  /// Indexes raw text into the vector store.
  Future<void> addTextRecord(
    String id,
    String text, {
    String? title,
    EmbeddingTaskType taskType = EmbeddingTaskType.retrievalDocument,
    Map<String, String> metadata = const {},
  }) async {
    final embedding = await _embedder.getEmbedding(
      text,
      taskType: taskType,
      title: title,
    );
    if (_useNativeBackend) {
      try {
        await _channel.invokeMethod('insertVectorRecord', {
          'id': id,
          'embeddings': embedding,
          'metadata': metadata,
        });
        _indexedIds.add(id);
        notifyListeners();
        return;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native insertVectorRecord failed: $e');
      }
    }
    _inMemoryStore[id] = _StoredRecord(id: id, embeddings: embedding, metadata: metadata);
    _indexedIds.add(id);
    notifyListeners();
  }

  /// Indexes a precomputed embedding vector directly into the vector store.
  Future<void> addVectorRecord(
    String id,
    List<double> embedding, {
    Map<String, String> metadata = const {},
  }) async {
    if (_useNativeBackend) {
      try {
        await _channel.invokeMethod('insertVectorRecord', {
          'id': id,
          'embeddings': embedding,
          'metadata': metadata,
        });
        _indexedIds.add(id);
        notifyListeners();
        return;
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native insertVectorRecord failed: $e');
      }
    }
    _inMemoryStore[id] = _StoredRecord(id: id, embeddings: embedding, metadata: metadata);
    _indexedIds.add(id);
    notifyListeners();
  }

  /// Alias for [addVectorRecord].
  Future<void> insertVectorRecord(
    String id,
    List<double> embedding, {
    Map<String, String> metadata = const {},
  }) =>
      addVectorRecord(id, embedding, metadata: metadata);

  /// Searches the vector store using natural language text query.
  Future<List<SemanticRetrievalResult>> retrieveEntities(
    String queryText, {
    int limit = 100,
    Map<String, String> metadataFilter = const {},
  }) async {
    if (queryText.trim().isEmpty) return [];

    if (_useNativeBackend) {
      try {
        final List<dynamic>? results = await _channel.invokeMethod('retrieveEntities', {
          'queryText': queryText,
          'limit': limit,
          'metadataFilter': metadataFilter,
        });
        if (results != null) {
          return results.map((m) => SemanticRetrievalResult.fromMap(m as Map)).toList();
        }
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native retrieveEntities failed: $e. Using local search.');
      }
    }

    // Fallback: Embed query and search local store
    final queryVec = await _embedder.getEmbedding(
      queryText,
      taskType: EmbeddingTaskType.retrievalQuery,
    );
    return retrieveEntitiesByVector(queryVec, limit: limit, metadataFilter: metadataFilter);
  }

  /// Searches the vector store using an image (visual similarity search / search-as-you-see).
  Future<List<SemanticRetrievalResult>> retrieveEntitiesByImage(
    Uint8List imageData, {
    int limit = 100,
    String? seedId,
    Map<String, String> metadataFilter = const {},
  }) async {
    if (_useNativeBackend) {
      try {
        final List<dynamic>? results = await _channel.invokeMethod('retrieveEntitiesByImage', {
          'imageData': imageData,
          'limit': limit,
          'seedId': seedId,
          'metadataFilter': metadataFilter,
        });
        if (results != null) {
          return results.map((m) => SemanticRetrievalResult.fromMap(m as Map)).toList();
        }
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native retrieveEntitiesByImage failed: $e');
      }
    }

    final queryVec = await _embedder.getEmbeddingFromImage(imageData);
    return retrieveEntitiesByVector(
      queryVec,
      limit: limit,
      seedId: seedId,
      metadataFilter: metadataFilter,
    );
  }

  /// Searches the vector store against a precomputed normalized vector.
  Future<List<SemanticRetrievalResult>> retrieveEntitiesByVector(
    List<double> queryVector, {
    int limit = 100,
    String? seedId,
    Map<String, String> metadataFilter = const {},
  }) async {
    if (_inMemoryStore.isEmpty && _useNativeBackend) {
      try {
        final List<dynamic>? results = await _channel.invokeMethod('retrieveEntitiesByVector', {
          'queryVector': queryVector,
          'limit': limit,
          'seedId': seedId,
          'metadataFilter': metadataFilter,
        });
        if (results != null) {
          return results.map((m) => SemanticRetrievalResult.fromMap(m as Map)).toList();
        }
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native retrieveEntitiesByVector failed: $e');
      }
    }

    final results = <SemanticRetrievalResult>[];

    for (final entry in _inMemoryStore.values) {
      if (seedId != null && entry.id == seedId) continue;

      // Apply metadata filter
      if (metadataFilter.isNotEmpty) {
        bool matches = true;
        for (final filterEntry in metadataFilter.entries) {
          if (entry.metadata[filterEntry.key] != filterEntry.value) {
            matches = false;
            break;
          }
        }
        if (!matches) continue;
      }

      final score = cosineSimilarity(queryVector, entry.embeddings);
      results.add(SemanticRetrievalResult(
        id: entry.id,
        embeddings: entry.embeddings,
        metadata: entry.metadata,
        similarityScore: score,
      ));
    }

    results.sort((a, b) => (b.similarityScore ?? 0).compareTo(a.similarityScore ?? 0));
    if (results.length > limit) {
      return results.sublist(0, limit);
    }
    return results;
  }

  /// Deletes a record by ID.
  Future<void> deleteRecord(String id) async {
    await deleteRecords([id]);
  }

  /// Deletes multiple records by their IDs.
  Future<void> deleteRecords(List<String> ids) async {
    if (ids.isEmpty) return;

    if (_useNativeBackend) {
      try {
        await _channel.invokeMethod('deleteRecords', {'ids': ids});
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native deleteRecords error: $e');
      }
    }

    for (final id in ids) {
      _inMemoryStore.remove(id);
      _indexedIds.remove(id);
    }
    notifyListeners();
  }

  /// Deletes all records whose IDs start with [idPrefix].
  /// Useful for deleting all windows belonging to a specific video project (`${projectId}/`).
  Future<int> deleteRecordsWithIdPrefix(String idPrefix) async {
    if (_useNativeBackend) {
      try {
        final count = await _channel.invokeMethod<int>('deleteRecordsWithIdPrefix', {
          'idPrefix': idPrefix,
        });
        if (count != null) {
          _indexedIds.removeWhere((id) => id.startsWith(idPrefix));
          notifyListeners();
          return count;
        }
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native deleteRecordsWithIdPrefix error: $e');
      }
    }

    final toRemove = _indexedIds.where((id) => id.startsWith(idPrefix)).toList();
    for (final id in toRemove) {
      _inMemoryStore.remove(id);
      _indexedIds.remove(id);
    }
    notifyListeners();
    return toRemove.length;
  }

  /// Deletes all records in this database.
  Future<void> deleteAllRecords() async {
    if (_useNativeBackend) {
      try {
        await _channel.invokeMethod('deleteAllRecords');
      } catch (e) {
        debugPrint('[SemanticRetrievalService] Native deleteAllRecords error: $e');
      }
    }
    _inMemoryStore.clear();
    _indexedIds.clear();
    notifyListeners();
  }

  /// Convenience alias for [deleteAllRecords].
  Future<void> clearAll() => deleteAllRecords();

  /// Checks if the underlying model supports audio embedding.
  Future<bool> supportsAudioEmbedding() async {
    if (_useNativeBackend) {
      try {
        final supported = await _channel.invokeMethod<bool>('supportsAudioEmbedding');
        if (supported != null) return supported;
      } catch (_) {}
    }
    return true;
  }

  /// Extracts a video frame thumbnail at [timeMs] using platform MediaMetadataRetriever or Web video canvas.
  Future<Uint8List?> extractVideoFrame(String videoPath, int timeMs) async {
    if (kIsWeb) {
      return await EmbeddingPluginWeb().extractVideoFrame(videoPath, timeMs);
    }
    try {
      final dynamic result = await _channel.invokeMethod('extractVideoFrame', {
        'videoPath': videoPath,
        'timeMs': timeMs,
      });
      return result as Uint8List?;
    } catch (e) {
      debugPrint('[SemanticRetrievalService] extractVideoFrame failed: $e');
      return null;
    }
  }

  /// Extracts video duration and dimensions using platform MediaMetadataRetriever or Web video element.
  Future<({int durationMs, int width, int height})?> extractVideoMetadata(String videoPath) async {
    if (kIsWeb) {
      return await EmbeddingPluginWeb().extractVideoMetadata(videoPath);
    }
    try {
      final Map? result = await _channel.invokeMethod('extractVideoMetadata', {
        'videoPath': videoPath,
      });
      if (result != null) {
        return (
          durationMs: (result['durationMs'] as num?)?.toInt() ?? 0,
          width: (result['width'] as num?)?.toInt() ?? 0,
          height: (result['height'] as num?)?.toInt() ?? 0,
        );
      }
    } catch (e) {
      debugPrint('[SemanticRetrievalService] extractVideoMetadata failed: $e');
    }
    return null;
  }

  @override
  void addListener(VoidCallback listener) {
    _listeners.add(listener);
  }

  @override
  void removeListener(VoidCallback listener) {
    _listeners.remove(listener);
  }

  @override
  void notifyListeners() {
    if (_isDisposed) return;
    for (final listener in List.of(_listeners)) {
      listener();
    }
  }

  @override
  bool get hasListeners => _listeners.isNotEmpty;

  @override
  void dispose() {
    _isDisposed = true;
    _listeners.clear();
    if (_useNativeBackend) {
      _channel.invokeMethod('closeRetriever').catchError((_) => null);
    }
  }
}
