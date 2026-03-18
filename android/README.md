# flutter_local_gemma — Android Native Layer

This document explains the design, architecture, and usage of the Android Kotlin files that power the `flutter_local_gemma` plugin.

---

## Overview

The plugin enables on-device AI inference on Android by wrapping three Google libraries:

| Library | Gradle coordinate | Purpose |
|---------|------------------|---------|
| **LiteRT-LM** | `com.google.ai.edge.litertlm:litertlm-android` | Gemma 3 text / multimodal generation |
| **Local Agents RAG** | `com.google.ai.edge.localagents:localagents-rag` | Gemma 300M embedding vectors |
| **Android PdfRenderer** / **OpenPDF** | built-in / `com.github.librepdf:openpdf` | PDF text extraction & page rendering |

---

## File map

```
android/src/main/kotlin/com/example/flutter_local_gemma/
├── FlutterLocalGemmaPlugin.kt       ← Entry point; coordinates sub-plugins
├── GemmaPlugin.kt              ← LiteRT-LM inference engine wrapper
├── EmbeddingPlugin.kt          ← Gemma embedding model wrapper
├── PdfPlugin.kt                ← PDF extraction & rendering (stateless)
└── NativeFilePickerPlugin.kt   ← System file picker (content URIs)
```

---

## Architecture

### Sub-plugin pattern

`FlutterLocalGemmaPlugin` is the Flutter entry point registered in `AndroidManifest.xml`. It owns four sub-plugin instances but does **not** handle their method calls itself — each sub-plugin registers its own `MethodChannel` (and `EventChannel` where streaming is needed).

```
FlutterLocalGemmaPlugin (coordinator)
 ├── GemmaPlugin         → "gemma_bundled" + "gemma_stream"
 ├── EmbeddingPlugin     → "embedding_plugin"
 ├── PdfPlugin           → "pdf_plugin"
 └── NativeFilePickerPlugin → "native_file_picker"
```

**Benefits:**
- A crash inside one sub-plugin cannot affect the others.
- Each plugin can be unloaded individually without disturbing the rest.
- Channels are decoupled, making it trivial to add a new capability without touching the coordinator.

---

### Threading model

Every sub-plugin uses exactly **one coroutine scope**:

```kotlin
val supervisorJob = SupervisorJob()
val ioScope = CoroutineScope(supervisorJob + Dispatchers.IO)
```

| Component | Role |
|-----------|------|
| `SupervisorJob` | Child failures are isolated — one bad coroutine does not cancel the whole scope |
| `Dispatchers.IO` | All blocking work (model inference, file I/O, PDF rendering) runs here, never on the Flutter/UI thread |
| `withContext(Dispatchers.Main)` | Any callback into `MethodChannel.Result` or `EventChannel.EventSink` is switched back to the Main thread |

There is intentionally **no** fancy thread pool, actor, or channel. One dispatcher, one job. Simple and predictable.

---

### Memory lifecycle

The plugin is designed to support two usage patterns:

**Pattern A — Load → Use → Discard**
```
createModel  →  createSession  →  generate  →  closeModel
```
Call `closeModel` / `closeEmbeddingModel` when you are done. All native heap and GPU buffers are released immediately. The plugin returns to its initial blank state and can be reloaded at any time.

**Pattern B — Long-lived (whole-app lifetime)**
```
createModel  →  [createSession  →  generate  →  clearContext]  × N
```
The `Engine` stays loaded. Only the `Conversation` is recycled on each `createSession` call, which is cheap. Call `closeModel` only when the app exits or you genuinely need the RAM back.

#### What gets freed by `closeModel`

| Resource | Action |
|----------|--------|
| `Conversation` | `close()` called, reference nulled |
| `Engine` | `close()` called, reference nulled |
| Pending content buffer | cleared |
| Response buffer | cleared |
| In-flight generation job | cancelled |

The GC can then reclaim the native JNI heap immediately.

---

### Error handling / stateless crashes

Every method call flows through a `safeResultCall` wrapper:

```kotlin
private suspend fun  safeResultCall(
    result: MethodChannel.Result,
    errorCode: String,
    block: suspend () -> T
) {
    try {
        val value = block()
        withContext(Dispatchers.Main) { result.success(value) }
    } catch (e: Exception) {
        Log.e(TAG, "$errorCode: ${e.message}", e)
        withContext(Dispatchers.Main) { result.error(errorCode, e.message, null) }
    }
}
```

**Key guarantee:** No exception ever reaches Android's uncaught-exception handler. Every error is converted to a `result.error(...)` call so Flutter receives it as a `PlatformException` — not a crash.

---

## GemmaPlugin

Wraps the LiteRT-LM `Engine` + `Conversation` API.

### Channel: `gemma_bundled` (MethodChannel)

| Method | Required args | Optional args | Returns |
|--------|---------------|---------------|---------|
| `createModel` | `modelPath: String` | `maxTokens: Int`, `preferredBackend: Int` (0=CPU, 1=GPU), `supportAudio: Bool` | `null` |
| `createSession` | — | `temperature: Double`, `topP: Double`, `topK: Int`, `systemPrompt: String`, `autoStopEnabled: Bool`, `maxRepetitions: Int` | `null` |
| `addQueryChunk` | `prompt: String` | — | `null` |
| `addImage` | `imageBytes: ByteArray` | — | `null` |
| `addAudio` | `audioBytes: ByteArray` | — | `null` |
| `generateResponseAsync` | — | — | `null` (tokens arrive on event channel) |
| `generateResponseSync` | — | — | `String` |
| `stopGeneration` | — | — | `null` |
| `clearContext` | — | — | `null` |
| `countTokens` | `text: String` | `imageCount: Int`, `audioDurationMs: Int` | `Int` |
| `closeModel` | — | — | `null` |

### Channel: `gemma_stream` (EventChannel)

Streaming events are `Map<String, Any>` with these shapes:

```dart
// Normal token
{ "partialResult": "Hello", "done": false }

// Generation complete
{ "partialResult": "", "done": true }

// Error (non-fatal — generation is aborted but session remains valid)
{ "error": "Session not initialised." }
```

### Model validation

`createModel` validates the file before passing it to LiteRT-LM:
- File must exist and be readable.
- File must be **> 50 MB** (a common corrupted download is a tiny HTML error page).
- First 16 bytes must not look like HTML or JSON (another symptom of a bad HuggingFace token).

### Repetition / loop detection

If `autoStopEnabled = true`, the plugin watches the rolling output buffer for repeated patterns using a regex. When a pattern of ≥ `minRepeatCharLen` characters repeats `maxRepetitions` or more times in the last 200 characters, generation is stopped and a notice is appended to the stream.

---

## EmbeddingPlugin

Wraps `GemmaEmbeddingModel` from the `localagents-rag` library.

### Channel: `embedding_plugin`

| Method | Required args | Optional args | Returns |
|--------|---------------|---------------|---------|
| `initEmbeddingModel` | `modelPath: String`, `tokenizerPath: String` | `useGpu: Bool` | `null` |
| `getEmbedding` | `text: String` | — | `List<Double>` |
| `closeEmbeddingModel` | — | — | `null` |

**Notes:**
- GPU is automatically disabled on emulators (detected via `Build.HARDWARE`).
- `getEmbedding` uses `TaskType.SEMANTIC_SIMILARITY` which is optimal for RAG retrieval.
- Float values from the model are promoted to `Double` for Dart compatibility.

---

## PdfPlugin

**Stateless** — no objects are held between calls, so there is nothing to leak.

### Channel: `pdf_plugin`

| Method | Required args | Optional args | Returns |
|--------|---------------|---------------|---------|
| `extractPdf` | `bytes: ByteArray` | `mode`, `filter`, `startPage`, `endPage`, `renderScale` | `List<Map<String,Any>>` |
| `clearCache` | — | — | `null` |

#### Extraction modes

| Mode | Text | Images |
|------|------|--------|
| `"auto"` | ✓ | fallback if no text found |
| `"textOnly"` | ✓ | ✗ |
| `"imagesOnly"` | ✗ | ✓ |
| `"fullRender"` | ✗ | ✓ always |
| `"textAndImages"` | ✓ | ✓ |

#### Text extraction strategy

| Android API | Library |
|-------------|---------|
| 35+ | `PdfRenderer.Page.textContents` (built-in, no extra dep) |
| < 35 | `org.openpdf.text.pdf.parser.PdfTextExtractor` |

Text passes through `cleanExtractedText()` to repair kerning artefacts (space-separated characters, e.g. `"H e l l o"` → `"Hello"`).

#### Memory notes

- PDF bytes are written to a temp file, processed, then deleted inside a `finally` block.
- Each rendered `Bitmap` is recycled immediately after PNG encoding.
- `clearCache` provides a manual cleanup for temp files if a previous call crashed.

---

## NativeFilePickerPlugin

Launches `Intent.ACTION_OPEN_DOCUMENT` and returns the selected content URI to Flutter.

### Channel: `native_file_picker`

| Method | Args | Returns |
|--------|------|---------|
| `pickFile` | — | `String?` (content URI, or null if cancelled) |

**Why native?** Flutter's `file_picker` package cannot call `takePersistableUriPermission`, which is required for reliably reading large model files via content URIs across app restarts without copying them to internal storage.

---

## Dependency reference

```groovy
// build.gradle
implementation("com.google.ai.edge.litertlm:litertlm-android:0.9.0-alpha01")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
implementation("com.google.mediapipe:tasks-genai:latest.release")
implementation("com.google.mediapipe:tasks-vision:latest.release")
implementation("com.google.ai.edge.localagents:localagents-rag:0.3.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.10.2")
implementation("com.github.librepdf:openpdf:3.0.0")
```

---

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `INIT_ERROR: File too small` | HuggingFace token invalid, downloaded HTML instead of binary | Verify your token in the Flutter download logic |
| `SIGABRT` / native crash on old devices | GPU backend on unsupported hardware | Pass `preferredBackend: 0` (CPU) |
| Embedding returns empty list | Model/tokenizer path mismatch | Check both paths exist before calling `initEmbeddingModel` |
| PDF extraction returns only images | API < 35, OpenPDF failed to initialise | Check the `PdfPlugin` `WARN` logs; the PDF may be encrypted |
| `ALREADY_ACTIVE` from file picker | User triggered the picker twice before first resolved | Ignore the second tap on the Flutter side |