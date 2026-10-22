# flutter_local_gemma — Web Layer

This directory contains the TypeScript bridge that connects Flutter's `dart:js_interop` layer to the browser-side AI APIs.

---

## Structure

```
web/
├── flutter_local_gemma_web/          # TypeScript source (Vite project)
│   ├── src/main.ts             # All JS globals exposed to Flutter
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── dist/                       # Compiled output (committed, bundled as Flutter assets)
│   ├── gemma_web.js            # Main bundle — exposes window.initLLM, etc.
│   └── ort-wasm-simd-threaded.asyncify.wasm
├── @mediapipe/tasks-genai/wasm/    # MediaPipe GenAI WASM binaries
├── @litertjs/core/wasm/            # LiteRT WASM binaries (embedding)
└── pdfjs-dist/                     # PDF.js module files
```

---

## Globals exposed to Flutter

All functions are registered on `window` so Flutter's `@JS(...)` annotations can reach them.

### LLM (MediaPipe GenAI + LiteRT-LM + Transformers.js ONNX)

| Function | Description |
|----------|-------------|
| `initLLM(options)` | Loads the Gemma LLM. Automatically selects the runtime based on model type: **Transformers.js WebGPU** for ONNX models (`isOnnx: true`), **LiteRT-LM Web** for Gemma 4 `.task` models, or **MediaPipe GenAI** for Gemma 3 models. |
| `generateResponse(parts, callback)` | Starts token streaming. `callback(text, isDone)` is called once per token. For ONNX models, accepts interleaved text, image data URLs (`imageSource`), and audio data URLs (`audioSource`). Ensures `isDone` is signaled even on async engine errors. |
| `cancelProcessing()` | Aborts the active generation cleanly across all engine runtimes. |
| `unloadLLM()` | Releases the LLM from GPU/WASM memory (MediaPipe instance, LiteRT engine, or Transformers.js ONNX pipelines). |
| `countTokens(text)` | Returns a `Promise<number>` with the estimated token count. |

### Embeddings & Multimodal Retrieval (LiteRT + Transformers.js)

| Function | Description |
|----------|-------------|
| `initEmbeddingModel(url, base, token?)` | Downloads (or loads from OPFS) and compiles the `.tflite` model (Gemma 300M). |
| `getEmbedding(text)` | Returns a `Promise<Float32Array>` with the 768-dim embedding vector. |
| `unloadEmbeddingModel()` | Frees the compiled model graph (WASM runtime stays alive). |
| `initEmbeddingGemma2(options)` | Loads EmbeddingGemma 2 via Transformers.js (`@huggingface/transformers`) with WebGPU acceleration. |
| `getEmbeddingGemma2(parts, targetDim)` | Computes multimodal embedding from interleaved text, image, and 16 kHz audio PCM slices. |
| `purgeEmbeddingCache(patterns[])` | Clears matching embedding model entries from the browser Cache Storage. |

### PDF (PDF.js)

| Function | Description |
|----------|-------------|
| `initPdfWorker(assetBase)` | Points PDF.js at the bundled worker script (`web/pdfjs-dist/pdf.worker.mjs`, version 6.4.299). |
| `extractPdf(bytes, mode, filter, start?, end?, scale)` | Extracts text and/or page images from a PDF byte array. |

### Model Installer & Cache Management (OPFS & Cache Storage)

| Function | Description |
|----------|-------------|
| `downloadModelWithProgress(url, token, onProgress)` | Streams the model binary into OPFS; returns an object-URL. |
| `purgeOpfsCache(keep[])` | Deletes all OPFS files except those named in `keep`. |
| `listOpfsCache()` | Returns `{name, size}[]` for all cached OPFS files. |
| `isTransformersModelCached(pattern)` | Checks if an ONNX model (Transformers.js) is cached in Cache Storage. |
| `deleteTransformersModel(pattern)` | Deletes a specific ONNX model and its shards from Cache Storage. |

---

## Building

```sh
cd flutter_local_gemma_web
npm install
npm run build    # outputs to ../dist/
```

The `dist/gemma_web.js` bundle is committed to the repository because it is declared as a Flutter asset in `pubspec.yaml`. Consumers do **not** need Node.js.

---

## Asset path resolution

Flutter serves plugin assets at:

```
/assets/packages/flutter_local_gemma/web/<path>
```

`WebScriptLoader` (in `lib/utils/web_script_loader.dart`) resolves this prefix at runtime using `ui_web.BrowserPlatformLocation().getBaseHref()` so the app works correctly whether it is deployed at the domain root or a subdirectory.