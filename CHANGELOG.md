## 0.3.1

### Gemma 4 ONNX Web Multimodal Models & Web Stability Fixes
- **Gemma 4 ONNX Web Support**:
  - Added `GemmaModel.gemma4E2BOnnx` (`onnx-community/gemma-4-E2B-it-ONNX`) and `GemmaModel.gemma4E4BOnnx` (`onnx-community/gemma-4-E4B-it-ONNX`) to the model catalog.
  - Enabled full multimodal inference on Web (interleaved text, images, and audio) using `@huggingface/transformers` v4+ accelerated by WebGPU with Cache Storage persistence.
  - Added dedicated cache verification (`ModelInstaller.isTransformersModelCached`) and model eviction (`ModelInstaller.deleteTransformersModel`).
- **Capability Detection & Multimodal Guarding**:
  - Updated `currentModelSupportsVision` and `currentModelSupportsAudio` on `FlutterLocalGemma` to accurately recognize multimodal capabilities: on Web, only ONNX models and Gemma 3n models support vision/audio, whereas standard LiteRT `.task` models run in text-only mode.
  - Updated `currentModelIsGemma4` to robustly identify all Gemma 4 variants across different model naming schemes (`gemma4`, `gemma-4`, `gemma_4`).
  - Added automatic skip guards (`_SkipException`) to the test suite (`llm_with_image`, `llm_with_audio`, `llm_with_pdf`), preventing initialization errors and test stalls when running on models without image/audio support.
- **Async Prediction Error Handling & Stream Hanging Fix**:
  - Fixed an issue where unhandled asynchronous MediaPipe prediction errors in `main.ts` left the Dart generation stream open indefinitely. Added error termination callbacks (`callback('', true)`) to guarantee streams close cleanly on engine failures.
- **PDF.js Worker Version Alignment**:
  - Synchronized static PDF.js worker files in `web/pdfjs-dist/` with `pdfjs-dist` version 6.4.299 and pinned the version in `package.json`, resolving `UnknownErrorException: The API version does not match the Worker version`.

---

## 0.3.0

### EmbeddingGemma 2 & Google Edge AI Gallery Features
- **EmbeddingGemma 2 Support**: Full integration of Google Edge AI Gallery's EmbeddingGemma 2 model across Android and Web.
  - **Android**: Powered by `com.google.mediapipe:tasks-retrieval:1.1.0` and LiteRT-LM runtime with GPU/CPU hardware acceleration.
  - **Web**: Powered by Transformers.js with WebGPU acceleration (with automatic WASM fallback) and OPFS model caching.
- **Multimodal Embedding Sequences**:
  - `EmbedPart` sealed class hierarchy supporting `EmbedPart.text(...)`, `EmbedPart.image(...)`, and `EmbedPart.audio(...)` (16 kHz mono 16-bit PCM).
  - Interleaved multimodal representations combining video frames, audio slices, and timestamp tokens into unified embedding vectors.
- **Task Instruction Formatting**:
  - `formatEmbeddingText()` supporting asymmetric retrieval (`task: search result | query: ...`, `title: ... | text: ...`), question answering, fact checking, code retrieval, and symmetric classification/clustering.
- **Matryoshka Representation Truncation**:
  - Support for Matryoshka dimension scaling (768, 512, 256, 128 dimensions) with automatic L2 re-normalization.
- **On-Device Vector Database & Retrieval Engine**:
  - `SemanticRetrievalService` providing persistent vector indexing with MediaPipe's `SqliteVectorStore` on Android and high-performance in-memory/OPFS vector store on Web and desktop.
  - Real-time cosine similarity search, metadata filtering, top-K selection, and batch indexing.
- **Smart Album / Instant Media Gallery Search (`GallerySearchService`)**:
  - Natural language search-as-you-type querying across local photo libraries.
  - Visual similarity search ("search similar to this photo").
  - Live camera search ("search-as-you-see") using real-time frame embeddings.
  - Native directory picker with Android persistable folder URI permissions (`ACTION_OPEN_DOCUMENT_TREE`).
- **Realtime Video Moment Finder (`VideoMomentService`)**:
  - Natural language search inside videos without audio transcription.
  - Configurable sliding window chunking (window duration, sample frames per window, overlap).
  - 16 kHz mono audio interleaving synchronized with video frame chunks.
  - Interactive timeline scrubber with highlighted match zones and merged continuous intervals.
  - Moment snippet bookmarking and clip management.
- **Model Loader & Picker Enhancements**:
  - Added `ModelLoader.downloadEmbeddingGemma2()` and `ModelLoader.loadEmbeddingGemma2()`.
  - Added `GemmaModelPicker.pickVideo()`, `pickFolder()`, `listFilesInFolder()`, and `pickMultipleMedia()`.
- **Example App Showcase**:
  - Added **Smart Album** interactive gallery screen with live camera search, folder selection, and demo sample library.
  - Added **Video Moment Finder** interactive screen with custom video canvas player, timeline scrubber, and moment jump buttons.

---

## 0.2.0

### Gemma 4, Agentic Tools & Thinking Mode
- Full support for **Gemma 4** (E2B and E4B) on Android (LiteRT-LM) and Web.
- **Thinking Mode**: Support for reasoning models (`<think>` blocks) with typed `GemmaThinkingResponse` streams.
- **Agentic Tool Calling (`AgentChat`)**: Autonomous multi-turn agent loop with tool selection, execution, and response synthesis.
- **Hybrid & Native Skills**:
  - Dart skills (`GemmaDartSkill`) executed in-process.
  - JS skills (`GemmaJsSkill`) running securely in sandboxed WebViews/iframes with Edge AI Gallery contract compatibility (`window.ai_edge_gallery_get_result`).
- **Speculative Decoding (MTP)**: Automatic detection and activation of Multi-Token Prediction on compatible Android builds.
- **Polymorphic API (`BaseChat`)**: Unified interface shared between conversational `GemmaChat` and agentic `AgentChat`.
- **Multimodality Mutex Guard**: Master `enableMultimodality` toggle preventing conflicting tool-calling and multimodal configurations.
- **GemmaTextParser**: Extracted AST parser for user-facing, thinking, and tool call blocks.
- **GemmaState & GemmaDebug**: Centralized reactive engine lifecycle listener and debug logger.

---

## 0.1.0

Initial public release.

### Features
- On-device Gemma 3 text generation for Android (LiteRT-LM) and Web (MediaPipe GenAI + WASM)
- `GemmaChat` — stateful multi-turn chat with conversation history, context-window overflow strategies (`slidingWindow`, `summarize`), system prompts, and history export/import
- `SingleTurnChat` — stateless single-call wrapper for batch / RAG pipelines
- Structured JSON output via a Zod-like `Schema` builder and streaming `repairJson` parser
- `EmbeddingPlugin` — on-device 768-dim text embeddings (Gemma 300M) for both Android and Web (LiteRT + Transformers.js)
- `PdfProcessor` — cross-platform PDF text extraction and page rendering using PDF.js (web) and Android PdfRenderer / OpenPDF (mobile)
- `DocumentEmbedder` — convenience helper that combines PDF extraction and embedding with LLM-powered image description for RAG workflows
- `GemmaLoader` — stateless helpers for downloading, caching (OPFS / local FS), and initialising the LLM and embedding models
- `GemmaModelPicker` — platform-native file picker (SAF on Android, `<input file>` on web)
- `ModelInstaller` — fluent builder for installing models from network URLs, `XFile` picks, or raw web `Blob` objects; zero-copy where possible
- Token counting and context-window usage tracking on all platforms
- Streaming and blocking generation with mid-stream cancellation (`stopGeneration`)
- Copy-progress `EventChannel` for large Android model file copies
- Automatic repetition-loop detection and generation stop (`AutoStopConfig`)