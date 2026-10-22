import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:image_picker/image_picker.dart';
import '../app_state.dart';

/// Screen implementing Google AI Edge Gallery's Smart Album / Instant Media Gallery Search
/// powered by EmbeddingGemma 2 and on-device multimodal vector retrieval.
class GallerySearchScreen extends StatefulWidget {
  const GallerySearchScreen({super.key});

  @override
  State<GallerySearchScreen> createState() => _GallerySearchScreenState();
}

class _GallerySearchScreenState extends State<GallerySearchScreen> {
  final _searchController = TextEditingController();
  final _picker = ImagePicker();

  late final SemanticRetrievalService _retrievalService;
  late final GallerySearchService _galleryService;

  bool _isModelReady = false;
  String _statusMessage = 'Smart Album Demo (768d Vector Store)';
  final int _targetDimension = 768; // Matryoshka dimension
  int? _lastSearchLatencyMs;

  Timer? _debounceTimer;

  // Active filter category
  String _selectedCategory = 'All';

  @override
  void initState() {
    super.initState();
    _retrievalService = SemanticRetrievalService(databaseName: 'gallery_album_gemma.db');
    _galleryService = GallerySearchService(retrievalService: _retrievalService);
    EmbeddingPlugin.isInitializedNotifier.addListener(_onModelInitChanged);
    _initEngine();
  }

  @override
  void dispose() {
    EmbeddingPlugin.isInitializedNotifier.removeListener(_onModelInitChanged);
    _debounceTimer?.cancel();
    _searchController.dispose();
    _galleryService.dispose();
    _retrievalService.dispose();
    super.dispose();
  }

  void _onModelInitChanged() {
    final ready = EmbeddingPlugin().isInitialized;
    if (mounted) {
      setState(() {
        _isModelReady = ready;
        _statusMessage = ready
            ? 'EmbeddingGemma 2 Active (768d)'
            : 'Smart Album (768d Vector Store)';
      });
      if (ready && _galleryService.assets.isNotEmpty) {
        _galleryService.indexPendingAssets(loadThumbnailBytes: (a) async => a.bytes);
      }
    }
  }

  Future<void> _initEngine() async {
    try {
      await _retrievalService.initialize();

      setState(() {
        _isModelReady = EmbeddingPlugin().isInitialized;
        _statusMessage = _isModelReady
            ? 'EmbeddingGemma 2 Active (768d)'
            : 'Smart Album (768d Vector Store)';
      });

      if (_isModelReady && _galleryService.assets.isNotEmpty) {
        await _galleryService.indexPendingAssets(loadThumbnailBytes: (a) async => a.bytes);
      }
    } catch (e) {
      debugPrint('[GallerySearchScreen] Engine initialization notice: $e');
    }
  }

  // ── User Photo Import ────────────────────────────────────────────────────

  Future<void> _pickUserPhotos() async {
    try {
      final pickedFiles = await _picker.pickMultiImage();
      if (pickedFiles.isEmpty) return;

      final newAssets = <MediaAsset>[];
      for (final xfile in pickedFiles) {
        final bytes = await xfile.readAsBytes();
        final name = xfile.name.isNotEmpty ? xfile.name : 'Photo_${DateTime.now().millisecondsSinceEpoch}';
        newAssets.add(
          MediaAsset(
            id: 'user_${DateTime.now().millisecondsSinceEpoch}_${xfile.name}',
            displayName: name,
            bytes: bytes,
            dateTaken: DateTime.now().millisecondsSinceEpoch,
            locationName: 'My Device Gallery',
          ),
        );
      }

      await _galleryService.indexAssets(newAssets);
      if (mounted) {
        final msg = _isModelReady
            ? 'Added ${newAssets.length} photos to Smart Album'
            : 'Added ${newAssets.length} photos. They will be indexed when EmbeddingGemma 2 is loaded.';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(msg),
            backgroundColor: Colors.teal,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error adding photos: $e')),
        );
      }
    }
  }

  // ── Search & Retrieval ───────────────────────────────────────────────────

  void _onSearchQueryChanged(String query) {
    _debounceTimer?.cancel();
    _debounceTimer = Timer(const Duration(milliseconds: 250), () async {
      final sw = Stopwatch()..start();
      await _galleryService.search(query);
      sw.stop();
      if (mounted) {
        setState(() => _lastSearchLatencyMs = sw.elapsedMilliseconds);
      }
    });
  }

  // ── UI Builder ───────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      backgroundColor: const Color(0xFF0F1318),
      appBar: AppBar(
        backgroundColor: const Color(0xFF141922),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [Colors.teal, Colors.tealAccent]),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.photo_library_rounded, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Smart Album',
                  style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                Text(
                  'EmbeddingGemma 2 Multimodal Search',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Sync / Re-index Assets',
            icon: const Icon(Icons.sync_rounded, color: Colors.tealAccent),
            onPressed: () async {
              await _galleryService.reindexAllAssets(loadThumbnailBytes: (a) async => a.bytes);
              if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Index sync complete: ${_retrievalService.indexedCount} photos indexed.'),
                    backgroundColor: Colors.teal,
                  ),
                );
              }
            },
          ),
          IconButton(
            tooltip: 'Add Photos from Device',
            icon: const Icon(Icons.add_photo_alternate_outlined, color: Colors.white70),
            onPressed: _pickUserPhotos,
          ),
          IconButton(
            tooltip: 'Clear Album',
            icon: const Icon(Icons.delete_sweep_outlined, color: Colors.white70),
            onPressed: () async {
              await _galleryService.clearAll();
              if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Smart Album cleared.')),
                );
              }
            },
          ),
          IconButton(
            tooltip: 'Engine & Model Settings',
            icon: const Icon(Icons.settings_suggest_rounded, color: Colors.white70),
            onPressed: _showModelSetupDialog,
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: _galleryService,
        builder: (context, _) {
          final isSearching = _galleryService.isSearching;
          final isIndexing = _galleryService.indexingProgress.isIndexing &&
              _galleryService.indexingProgress.totalCount > 0 &&
              _galleryService.indexingProgress.processedCount < _galleryService.indexingProgress.totalCount;
          final query = _galleryService.currentQuery;
          final allAssets = _galleryService.assets;
          final rankedResults = _galleryService.searchResults;

          // Filter by category chip
          final displayedAssets = allAssets.where((a) {
            final loc = a.locationName ?? '';
            if (_selectedCategory == 'All') return true;
            if (_selectedCategory == 'Nature') return loc.contains('nature') || loc.contains('beach') || loc.contains('forest');
            if (_selectedCategory == 'Pets') return loc.contains('cat') || loc.contains('dog');
            if (_selectedCategory == 'City') return loc.contains('city');
            if (_selectedCategory == 'Food') return loc.contains('coffee');
            if (_selectedCategory == 'Vehicles') return loc.contains('car');
            return true;
          }).toList();

          return Column(
            children: [
              // ── Engine Status & Dimension Pill Banner ─────────────────────
              _buildStatusPillHeader(),

              // ── Modern Google Photos Search Bar ───────────────────────────
              _buildSearchBar(colorScheme),

              // ── Filter Chips ──────────────────────────────────────────────
              _buildFilterChips(),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 4),
                child: Row(
                  children: [
                    Text(
                      '${displayedAssets.length} items',
                      style: const TextStyle(fontSize: 12, color: Colors.white60, fontWeight: FontWeight.w600),
                    ),
                    if (query.isNotEmpty && _lastSearchLatencyMs != null) ...[
                      const SizedBox(width: 8),
                      Text(
                        '(${_lastSearchLatencyMs}ms search)',
                        style: const TextStyle(fontSize: 11, color: Colors.tealAccent, fontWeight: FontWeight.bold),
                      ),
                    ],
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.white10,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        '${_targetDimension}d vectors',
                        style: const TextStyle(fontSize: 10, color: Colors.white54, fontWeight: FontWeight.w500),
                      ),
                    ),
                  ],
                ),
              ),

              if (isIndexing)

                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: Row(
                    children: [
                      const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.tealAccent),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'Indexing photo vectors: ${_galleryService.indexingProgress.percentage}%',
                        style: const TextStyle(fontSize: 12, color: Colors.tealAccent, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),

              if (isSearching)
                const LinearProgressIndicator(
                  color: Colors.tealAccent,
                  backgroundColor: Colors.white10,
                ),

              // ── Photos Gallery Grid ───────────────────────────────────────
              Expanded(
                child: displayedAssets.isEmpty
                    ? _buildEmptyState()
                    : _buildPhotoGrid(displayedAssets, rankedResults, query),
              ),
            ],
          );
        },
      ),
    );
  }

  // ── Engine Status Banner ─────────────────────────────────────────────────

  Widget _buildStatusPillHeader() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 6),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF181E28),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _isModelReady ? Colors.tealAccent : const Color(0xFF4FC3F7),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _statusMessage,
              style: const TextStyle(fontSize: 12, color: Colors.white70, fontWeight: FontWeight.w600),
            ),
          ),
          InkWell(
            onTap: _showModelSetupDialog,
            borderRadius: BorderRadius.circular(8),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.tune, size: 12, color: Colors.tealAccent),
                  SizedBox(width: 4),
                  Text('Setup', style: TextStyle(fontSize: 11, color: Colors.tealAccent, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Search Bar ───────────────────────────────────────────────────────────

  Widget _buildSearchBar(ColorScheme colorScheme) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 6),
      child: TextField(
        controller: _searchController,
        onChanged: _onSearchQueryChanged,
        style: const TextStyle(color: Colors.white, fontSize: 14),
        decoration: InputDecoration(
          hintText: 'Search photos ("sunset", "cute cat", "beach", "coffee")...',
          hintStyle: const TextStyle(color: Colors.white38, fontSize: 13),
          prefixIcon: const Icon(Icons.search_rounded, color: Colors.tealAccent),
          suffixIcon: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (_searchController.text.isNotEmpty)
                IconButton(
                  icon: const Icon(Icons.clear_rounded, color: Colors.white54, size: 20),
                  onPressed: () {
                    _searchController.clear();
                    _galleryService.clearSearch();
                  },
                ),
              IconButton(
                tooltip: 'Search by Photo Similarity',
                icon: const Icon(Icons.camera_alt_outlined, color: Colors.tealAccent, size: 20),
                onPressed: _searchByImagePicker,
              ),
            ],
          ),
          filled: true,
          fillColor: const Color(0xFF1E2430),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
        ),
      ),
    );
  }

  Future<void> _searchByImagePicker() async {
    try {
      final xfile = await _picker.pickImage(source: ImageSource.gallery);
      if (xfile == null) return;
      final bytes = await xfile.readAsBytes();
      await _galleryService.processCameraFrame(bytes);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Visual similarity search completed'),
            backgroundColor: Colors.teal,
          ),
        );
      }
    } catch (e) {
      debugPrint('[GallerySearchScreen] Image search error: $e');
    }
  }

  // ── Filter Chips ─────────────────────────────────────────────────────────

  Widget _buildFilterChips() {
    const categories = ['All', 'Nature', 'Pets', 'City', 'Food', 'Vehicles'];

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Row(
        children: categories.map((cat) {
          final isSelected = _selectedCategory == cat;
          return Padding(
            padding: const EdgeInsets.only(right: 6),
            child: ChoiceChip(
              label: Text(
                cat,
                style: TextStyle(
                  fontSize: 12,
                  color: isSelected ? Colors.black : Colors.white70,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                ),
              ),
              selected: isSelected,
              selectedColor: Colors.tealAccent,
              backgroundColor: const Color(0xFF1A1F2A),
              side: BorderSide(color: isSelected ? Colors.tealAccent : Colors.white10),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              onSelected: (val) {
                if (val) setState(() => _selectedCategory = cat);
              },
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Photo Grid ───────────────────────────────────────────────────────────

  Widget _buildPhotoGrid(
    List<MediaAsset> assets,
    List<MatchedAssetResult> rankedResults,
    String activeQuery,
  ) {
    // Map scores if query active
    final scoreMap = <String, double>{};
    for (final r in rankedResults) {
      scoreMap[r.asset.id] = r.similarity;
    }

    // Sort items by similarity score when search is active
    final sorted = List<MediaAsset>.from(assets);
    if (activeQuery.isNotEmpty && scoreMap.isNotEmpty) {
      sorted.sort((a, b) {
        final sa = scoreMap[a.id] ?? 0.0;
        final sb = scoreMap[b.id] ?? 0.0;
        return sb.compareTo(sa);
      });
    }

    return GridView.builder(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 0.95,
      ),
      itemCount: sorted.length,
      itemBuilder: (context, index) {
        final asset = sorted[index];
        final score = scoreMap[asset.id];
        final scorePct = score != null ? (score * 100).round() : null;

        return GestureDetector(
          onTap: () => _showPhotoDetailModal(asset),
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF191E28),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: scorePct != null && scorePct > 70
                    ? Colors.tealAccent.withValues(alpha: 0.8)
                    : Colors.white10,
                width: scorePct != null && scorePct > 70 ? 2 : 1,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.4),
                  blurRadius: 8,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
            clipBehavior: Clip.antiAlias,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Real Image Bytes
                if (asset.bytes != null)
                  Image.memory(
                    asset.bytes!,
                    fit: BoxFit.cover,
                  )
                else
                  Container(color: const Color(0xFF222834)),

                // Bottom Gradient Scrim
                Positioned(
                  left: 0,
                  right: 0,
                  bottom: 0,
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter,
                        end: Alignment.topCenter,
                        colors: [
                          Colors.black.withValues(alpha: 0.85),
                          Colors.transparent,
                        ],
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          asset.displayName,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            shadows: [Shadow(color: Colors.black, blurRadius: 4)],
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 2),
                        Text(
                          (asset.locationName ?? '').split(' ').take(3).join(', '),
                          style: const TextStyle(
                            color: Colors.white60,
                            fontSize: 10,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                ),

                // Similarity Score Pill (when search is active)
                if (scorePct != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF00B0FF), Color(0xFF00E676)],
                        ),
                        borderRadius: BorderRadius.circular(10),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.teal.withValues(alpha: 0.5),
                            blurRadius: 6,
                          ),
                        ],
                      ),
                      child: Text(
                        '$scorePct%',
                        style: const TextStyle(
                          color: Colors.black,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ── Photo Detail Modal ───────────────────────────────────────────────────

  void _showPhotoDetailModal(MediaAsset asset) {
    showDialog(
      context: context,
      builder: (ctx) {
        return Dialog(
          backgroundColor: const Color(0xFF141922),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          clipBehavior: Clip.antiAlias,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AspectRatio(
                aspectRatio: 4 / 3,
                child: asset.bytes != null
                    ? Image.memory(
                        asset.bytes!,
                        fit: BoxFit.cover,
                      )
                    : Container(color: Colors.white12),
              ),
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      asset.displayName,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Semantic Keywords: ${asset.locationName ?? "None"}',
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        FilledButton.tonalIcon(
                          icon: const Icon(Icons.travel_explore_rounded, size: 16),
                          label: const Text('Find Similar'),
                          style: FilledButton.styleFrom(
                            backgroundColor: Colors.teal.withValues(alpha: 0.2),
                            foregroundColor: Colors.tealAccent,
                          ),
                          onPressed: () {
                            Navigator.pop(ctx);
                            if (asset.bytes != null) {
                              _galleryService.searchSimilar(asset, seedImageBytes: asset.bytes!);
                            }
                          },
                        ),
                        const Spacer(),
                        TextButton(
                          onPressed: () => Navigator.pop(ctx),
                          child: const Text('Close', style: TextStyle(color: Colors.white60)),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // ── Empty State ──────────────────────────────────────────────────────────

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.photo_library_outlined, size: 56, color: Colors.white24),
          const SizedBox(height: 14),
          const Text(
            'Smart Album is ready',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          const SizedBox(height: 6),
          const Text(
            'Add photos from your device to index and search with EmbeddingGemma 2',
            style: TextStyle(fontSize: 12, color: Colors.grey),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 18),
          ElevatedButton.icon(
            icon: const Icon(Icons.add_photo_alternate_rounded),
            label: const Text('Pick Photos from Device'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.teal,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
            ),
            onPressed: _pickUserPhotos,
          ),
        ],
      ),
    );
  }

  // ── Model & Engine Setup Dialog ──────────────────────────────────────────

  void _showModelSetupDialog() {
    final tokenController = TextEditingController(text: ModelManager.instance.hfToken ?? '');
    double downloadProgress = 0.0;
    bool isDownloading = false;
    String? dialogStatus;
    bool preferChipset = false;
    String chipsetLabel = 'detecting chipset…';
    EmbeddingGemma2Catalog.detectSoc().then((soc) {
      final match = EmbeddingGemma2Catalog.matchChipset(soc);
      chipsetLabel = match != null
          ? 'matched: ${match.label} (${match.sizeMb} MB)'
          : 'no chipset build for ${soc.socModel.isEmpty ? "this device" : soc.socModel}; generic 405 MB build is used';
    });

    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              backgroundColor: const Color(0xFF161B24),
              title: Row(
                children: [
                  const Icon(Icons.memory_rounded, color: Colors.tealAccent),
                  const SizedBox(width: 10),
                  const Text('EmbeddingGemma 2 Engine', style: TextStyle(color: Colors.white, fontSize: 17)),
                ],
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E2430),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Active Mode: ${_isModelReady ? "EmbeddingGemma 2 Native" : "Instant Demo (768d MRL)"}',
                          style: TextStyle(
                            color: _isModelReady ? Colors.tealAccent : const Color(0xFF4FC3F7),
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          'EmbeddingGemma 2 generates 768-dim multimodal vectors for text and images.',
                          style: TextStyle(fontSize: 11, color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text('HuggingFace Access Token (Optional):', style: TextStyle(color: Colors.white70, fontSize: 12)),
                  const SizedBox(height: 4),
                  TextField(
                    controller: tokenController,
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                    decoration: InputDecoration(
                      hintText: 'hf_xxxxxxxxxxxxxxxxxxxx',
                      hintStyle: const TextStyle(color: Colors.white24, fontSize: 12),
                      filled: true,
                      fillColor: const Color(0xFF1E2430),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
                    ),
                  ),
                  const SizedBox(height: 8),
                  SwitchListTile(
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Chipset-optimised build', style: TextStyle(color: Colors.white, fontSize: 12)),
                    subtitle: Text(chipsetLabel, style: const TextStyle(color: Colors.white38, fontSize: 10)),
                    value: preferChipset,
                    activeThumbColor: Colors.tealAccent,
                    onChanged: isDownloading ? null : (v) => setDialogState(() => preferChipset = v),
                  ),
                  if (dialogStatus != null) ...[
                    const SizedBox(height: 10),
                    Text(dialogStatus!, style: const TextStyle(color: Colors.tealAccent, fontSize: 11)),
                  ],
                  if (isDownloading) ...[
                    const SizedBox(height: 8),
                    LinearProgressIndicator(value: downloadProgress / 100.0, color: Colors.tealAccent),
                  ],
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      OutlinedButton.icon(
                        icon: const Icon(Icons.folder_open, size: 16),
                        label: const Text('Pick .litertlm'),
                        style: OutlinedButton.styleFrom(foregroundColor: Colors.white70),
                        onPressed: () async {
                          final path = await GemmaModelPicker.pick();
                          if (path != null) {
                            try {
                              await PluginModelLoader.initEmbeddingGemma2(modelPathOrId: path);
                              setState(() {
                                _isModelReady = true;
                                _statusMessage = 'EmbeddingGemma 2 Active (768d)';
                              });
                              if (ctx.mounted) Navigator.pop(ctx);
                              if (_galleryService.assets.isNotEmpty) {
                                await _galleryService.indexPendingAssets(loadThumbnailBytes: (a) async => a.bytes);
                              }
                            } catch (e) {
                              setDialogState(() => dialogStatus = 'Init error: $e');
                            }
                          }
                        },
                      ),
                      const Spacer(),
                      FilledButton.icon(
                        icon: const Icon(Icons.download, size: 16),
                        label: const Text('Download'),
                        style: FilledButton.styleFrom(backgroundColor: Colors.teal),
                        onPressed: isDownloading
                            ? null
                            : () async {
                                final tok = tokenController.text.trim();
                                if (tok.isNotEmpty) {
                                  ModelManager.instance.setHfToken(tok);
                                }
                                setDialogState(() {
                                  isDownloading = true;
                                  dialogStatus = 'Downloading model...';
                                });
                                try {
                                  final path = await PluginModelLoader.loadEmbeddingGemma2(
                                    token: tok.isNotEmpty ? tok : null,
                                    preferChipsetVariant: preferChipset,
                                    onProgress: (p) {
                                      setDialogState(() {
                                        downloadProgress = p.clamp(0.0, 100.0);
                                        dialogStatus = 'Downloading: ${downloadProgress.toStringAsFixed(0)}%';
                                      });
                                    },
                                  );
                                  ModelManager.instance.embeddingStatus = ModelStatus.ready;
                                  ModelManager.instance.embeddingModelPath = path;
                                  ModelManager.instance.update();
                                  setState(() {
                                    _isModelReady = true;
                                    _statusMessage = 'EmbeddingGemma 2 Active (768d)';
                                  });
                                  if (ctx.mounted) Navigator.pop(ctx);
                                  await _galleryService.indexPendingAssets(loadThumbnailBytes: (a) async => a.bytes);
                                } catch (e) {
                                  setDialogState(() {
                                    isDownloading = false;
                                    dialogStatus = 'Download notice: $e\nDemo mode remains active.';
                                  });
                                }
                              },
                      ),

                    ],
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Close', style: TextStyle(color: Colors.white60)),
                ),
              ],
            );
          },
        );
      },
    );
  }
}
