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