import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('EmbeddingGemma 2 - EmbedPart & Multimodal Structure', () {
    test('creates text embed part and serializes correctly', () {
      const part = EmbedPart.text('A photo of a golden retriever puppy');
      expect(part, isA<TextEmbedPart>());
      expect((part as TextEmbedPart).text, 'A photo of a golden retriever puppy');

      final map = part.toMap();
      expect(map['type'], 'text');
      expect(map['text'], 'A photo of a golden retriever puppy');
    });

    test('creates image embed part and serializes correctly', () {
      final dummyBytes = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A]);
      final part = EmbedPart.image(dummyBytes);
      expect(part, isA<ImageEmbedPart>());
      expect((part as ImageEmbedPart).bytes.length, 6);

      final map = part.toMap();
      expect(map['type'], 'image');
      expect(map['bytes'], dummyBytes);
    });

    test('creates audio embed part with 16kHz default', () {
      final pcm = Int16List.fromList([0, 100, -200, 300, 0]);
      final part = EmbedPart.audio(pcm);
      expect(part, isA<AudioEmbedPart>());
      final audioPart = part as AudioEmbedPart;
      expect(audioPart.sampleRateHz, 16000);
      expect(audioPart.pcm.length, 5);

      final map = part.toMap();
      expect(map['type'], 'audio');
      expect(map['pcm'], pcm);
      expect(map['sampleRateHz'], 16000);
    });
  });

  group('EmbeddingGemma 2 - Task Instruction Formatter', () {
    test('formats asymmetric retrieval queries', () {
      final query = formatEmbeddingText(
        'red sports car',
        taskType: EmbeddingTaskType.retrievalQuery,
      );
      expect(query, 'task: search result | query: red sports car');
    });

    test('formats asymmetric documents with title', () {
      final doc = formatEmbeddingText(
        'The car was parked outside the garage.',
        taskType: EmbeddingTaskType.retrievalDocument,
        title: 'Vehicle Report',
      );
      expect(doc, 'title: Vehicle Report | text: The car was parked outside the garage.');
    });

    test('formats question answering query', () {
      final qa = formatEmbeddingText(
        'Where was the car found?',
        taskType: EmbeddingTaskType.questionAnswering,
      );
      expect(qa, 'task: question answering | query: Where was the car found?');
    });

    test('formats classification and similarity queries', () {
      final cls = formatEmbeddingText(
        'Sentiment analysis sample',
        taskType: EmbeddingTaskType.classification,
      );
      expect(cls, 'task: classification | query: Sentiment analysis sample');

      final sim = formatEmbeddingText(
        'Pair of sentences',
        taskType: EmbeddingTaskType.sentenceSimilarity,
      );
      expect(sim, 'task: sentence similarity | query: Pair of sentences');
    });

    test('formats none task type without prefix', () {
      final raw = formatEmbeddingText(
        '   raw untagged prompt   ',
        taskType: EmbeddingTaskType.none,
      );
      expect(raw, 'raw untagged prompt');
    });
  });

  group('Embedding Vector Mathematics & Matryoshka Truncation', () {
    test('l2Normalize produces unit vector', () {
      final vec = [3.0, 4.0];
      final norm = l2Normalize(vec);
      expect(norm[0], closeTo(0.6, 1e-5));
      expect(norm[1], closeTo(0.8, 1e-5));

      final magnitude = math.sqrt(norm[0] * norm[0] + norm[1] * norm[1]);
      expect(magnitude, closeTo(1.0, 1e-5));
    });

    test('cosineSimilarity handles identical, opposite, and orthogonal vectors', () {
      final v1 = [1.0, 0.0, 0.0];
      final v2 = [1.0, 0.0, 0.0];
      final v3 = [-1.0, 0.0, 0.0];
      final v4 = [0.0, 1.0, 0.0];

      expect(cosineSimilarity(v1, v2), closeTo(1.0, 1e-5));
      expect(cosineSimilarity(v1, v3), closeTo(-1.0, 1e-5));
      expect(cosineSimilarity(v1, v4), closeTo(0.0, 1e-5));
    });

    test('truncateMatryoshka performs truncation and re-normalization', () {
      final full768 = List<double>.generate(768, (i) => (i + 1).toDouble());
      final dim256 = truncateMatryoshka(full768, 256);
      expect(dim256.length, 256);

      // Verify unit length after truncation
      final sqSum = dim256.fold<double>(0.0, (acc, v) => acc + v * v);
      expect(math.sqrt(sqSum), closeTo(1.0, 1e-5));
    });
  });

  group('SemanticRetrievalService & In-Memory Store Fallback', () {
    late SemanticRetrievalService service;

    setUp(() async {
      service = SemanticRetrievalService(databaseName: 'test_store.db');
      await service.initialize();
      await service.clearAll();
    });

    tearDown(() {
      service.dispose();
    });

    test('adds records and retrieves nearest neighbor correctly', () async {
      final sunsetEmb = l2Normalize([1.0, 0.8, 0.1]);
      final beachEmb = l2Normalize([0.9, 0.7, 0.2]);
      final winterSnowEmb = l2Normalize([-0.8, -0.9, 0.1]);

      await service.insertVectorRecord(
        'photo_sunset',
        sunsetEmb,
        metadata: {'title': 'Golden hour sunset over mountains'},
      );

      await service.insertVectorRecord(
        'photo_beach',
        beachEmb,
        metadata: {'title': 'Sunny sandy beach'},
      );

      await service.insertVectorRecord(
        'photo_snow',
        winterSnowEmb,
        metadata: {'title': 'Heavy winter blizzard'},
      );

      expect(service.indexedCount, 3);

      // Query close to sunset
      final queryEmb = l2Normalize([0.95, 0.75, 0.15]);
      final results = await service.retrieveEntitiesByVector(
        queryEmb,
        limit: 2,
      );

      expect(results.length, 2);
      expect(results[0].id, 'photo_sunset');
      expect(results[0].similarityScore, greaterThan(0.9));
      expect(results[1].id, 'photo_beach');
    });

    test('filters results using metadata filter', () async {
      final v1 = l2Normalize([1.0, 0.0, 0.0]);
      await service.insertVectorRecord('cat_1', v1, metadata: {'category': 'cat'});
      await service.insertVectorRecord('dog_1', v1, metadata: {'category': 'dog'});

      final results = await service.retrieveEntitiesByVector(
        v1,
        metadataFilter: {'category': 'dog'},
      );

      expect(results.length, 1);
      expect(results.first.id, 'dog_1');
    });

    test('deletes records by id', () async {
      await service.insertVectorRecord('temp_doc', [0.5, 0.5]);
      expect(service.indexedCount, 1);

      await service.deleteRecord('temp_doc');
      expect(service.indexedCount, 0);
    });
  });

  group('VideoMomentService & Interval Clustering', () {
    test('formats milliseconds as mm:ss timestamp', () {
      expect(VideoMomentService.formatTimestamp(0), '00:00');
      expect(VideoMomentService.formatTimestamp(5000), '00:05');
      expect(VideoMomentService.formatTimestamp(65000), '01:05');
    });

    test('merges adjacent and overlapping moment intervals', () {
      const rawMoments = [
        MomentSearchResult(
          id: 'w1',
          startTimeMs: 1000,
          endTimeMs: 4000,
          timeMs: 2500,
          similarityScore: 0.85,
        ),
        MomentSearchResult(
          id: 'w2',
          startTimeMs: 3000,
          endTimeMs: 6000,
          timeMs: 4500,
          similarityScore: 0.92,
        ),
        MomentSearchResult(
          id: 'w3',
          startTimeMs: 12000,
          endTimeMs: 15000,
          timeMs: 13500,
          similarityScore: 0.78,
        ),
      ];

      final merged = VideoMomentService.computeMergedIntervals(
        rawMoments,
        overlapToleranceMs: 500,
      );

      expect(merged.length, 2);
      expect(merged[0].startTimeMs, 1000);
      expect(merged[0].endTimeMs, 6000);
      expect(merged[0].resultIds, containsAll({'w1', 'w2'}));

      expect(merged[1].startTimeMs, 12000);
      expect(merged[1].endTimeMs, 15000);
      expect(merged[1].resultIds, contains('w3'));
    });
  });
}
