# flutter_local_gemma

A Flutter plugin for **on-device AI inference** powered by Google's Gemma 3 family of models.  
Supports **Android** (via LiteRT-LM) and **Web** (via MediaPipe GenAI + LiteRT WASM).

---

## Features

| Feature | Android | Web |
|---------|---------|-----|
| Text generation (streaming & blocking) | ✅ | ✅ |
| Multi-turn chat with history management | ✅ | ✅ |
| Structured JSON output (with schema) | ✅ | ✅ |
| Image input (multimodal) | ✅ | ✅ |
| Audio input | ✅ | ✅ |
| Text embeddings | ✅ | ✅ |
| PDF extraction (text + images) | ✅ | ✅ |
| Model download + OPFS/local caching | ✅ | ✅ |
| Native file picker | ✅ | ✅ |
| Token counting | ✅ | ✅ |
| Context window management | ✅ | ✅ |

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Android Setup](#android-setup)
- [Web Setup](#web-setup)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Loading a Model](#loading-a-model)
  - [GemmaChat — Multi-turn Chat](#gemmachat--multi-turn-chat)
  - [SingleTurnChat — Stateless Inference](#singleturnchat--stateless-inference)
  - [Structured JSON Output](#structured-json-output)
  - [Embeddings](#embeddings)
  - [PDF Processing](#pdf-processing)
  - [RAG Helper](#rag-helper)
  - [Model Picker](#model-picker)
- [Token Tracking](#token-tracking)
- [Example App](#example-app)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Requirements

| Platform | Minimum |
|----------|---------|
| Android | API 24 (Android 7.0) |
| Flutter SDK | 3.3.0 |
| Dart SDK | 3.10.7 |
| Web browser | Chrome 120+ / Edge 120+ (SharedArrayBuffer required) |

> **Web note:** The browser must serve the app with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers so SharedArrayBuffer (required by WASM threads) is available.

---

## Installation

Add the package to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_local_gemma: ^0.1.0
```

Then run:

```sh
flutter pub get
```

---

## Android Setup

### 1. Minimum SDK

In your app's `android/app/build.gradle.kts`, ensure:

```kotlin
android {
    defaultConfig {
        minSdk = 24
    }
}
```

### 2. Internet permission

Add to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

### 3. Memory

Gemma 3n E2B INT4 requires approximately **2–3 GB of RAM** at runtime. On low-memory devices consider using the CPU backend and closing the engine when it is not actively in use.

---

## Web Setup

### 1. COOP / COEP headers

Flutter's dev server adds these automatically. For production, configure your server:

**Nginx:**
```nginx
add_header Cross-Origin-Opener-Policy same-origin;
add_header Cross-Origin-Embedder-Policy require-corp;
```

**Firebase Hosting (`firebase.json`):**
```json
{
  "hosting": {
    "headers": [
      {
        "source": "**",
        "headers": [
          { "key": "Cross-Origin-Opener-Policy",  "value": "same-origin" },
          { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
        ]
      }
    ]
  }
}
```

### 2. No manual script tags needed

The plugin injects `gemma_web.js` (compiled from `web/flutter_local_gemma_web/src/main.ts`) at runtime via a dynamic `<script type="module">` tag. All WASM binaries are bundled as Flutter assets — no CDN dependencies.

---

## Quick Start

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// 1. Download and initialise the model (once per app launch)
await GemmaLoader.loadLlm(
  token: 'hf_YOUR_HUGGINGFACE_TOKEN',  // required for gated models
  onProgress: (p) => print('Downloading: ${p.toStringAsFixed(0)}%'),
);

// 2. Create a chat session
final chat = GemmaChat(systemPrompt: 'You are a helpful assistant.');
await chat.init();

// 3. Stream a response token-by-token
await for (final token in chat.sendMessageStream(text: 'Hello!')) {
  stdout.write(token);
}

// 4. Clean up
await chat.dispose();
await GemmaLoader.unloadLlm();
```

---

## Core Concepts

### Engine vs. Session

`FlutterLocalGemma` is a **singleton engine** that holds the loaded model in memory. It is expensive to initialise and should live for the lifetime of the feature that needs it.

`ChatSession` (returned by `FlutterLocalGemma().createSession()`) is a **lightweight handle** to a single conversation. You can create and destroy many sessions without reloading the engine.

### Platform differences

| Behaviour | Android | Web |
|-----------|---------|-----|
| Model storage | Local file system (`applicationDocumentsDirectory`) | Origin Private File System (OPFS) |
| Backend options | CPU, GPU | WebGPU / WASM (auto-selected by MediaPipe) |
| Session state | Native `Conversation` object | In-memory content buffer |
| Audio input | Supported (PCM WAV) | Not supported |

---

## API Reference

### Loading a Model

`GemmaLoader` provides stateless helpers for downloading and initialising both the LLM and the embedding engine.

```dart
// Download from HuggingFace and initialise in one call
await GemmaLoader.loadLlm(
  token: 'hf_…',
  maxTokens: 4096,       // KV-cache capacity
  useGpu: true,
  onProgress: (p) { /* p is 0.0–100.0 */ },
);

// Or supply a path you already have (skips download)
await GemmaLoader.initLlm(path: '/path/to/model.litertlm');

// Remove the model from memory when done
await GemmaLoader.unloadLlm();

// Delete all downloaded/cached model files
final freed = await GemmaLoader.purgeCache();
print('Freed $freed');
```

**Default model URLs** (Gemma 3n E2B INT4):
- **Android:** `hf.co/google/gemma-3n-E2B-it-litert-lm` — `gemma-3n-E2B-it-int4.litertlm`
- **Web:** `hf.co/google/gemma-3n-E2B-it-litert-lm` — `gemma-3n-E2B-it-int4-Web.litertlm`

---

### GemmaChat — Multi-turn Chat

`GemmaChat` maintains conversation history, handles context-window overflow, and supports system prompts.

```dart
final chat = GemmaChat(
  systemPrompt: 'You are a helpful pirate. Answer in character.',
  maxContextTokens: 4096,
  contextStrategy: ContextStrategy.slidingWindow, // auto-manages overflow
);
await chat.init();

// Streaming response
await for (final token in chat.sendMessageStream(text: 'Ahoy!')) {
  stdout.write(token);
}

// Blocking response
final reply = await chat.sendMessage(text: 'Tell me a joke.');

// Multimodal (image + text)
final imageBytes = await File('photo.png').readAsBytes();
await chat.sendMessage(
  text: 'What is in this image?',
  images: [imageBytes],
);

// Inspect or manipulate history
print(chat.history.length);
await chat.clearHistory();
await chat.removeHistory(index);
await chat.editHistory(index, newMessage);

// Persist / restore across sessions
final json = await chat.exportHistory();
await chat.importHistory(json);

// Stop mid-generation
await chat.stop();

await chat.dispose();
```

#### Context strategies

| Strategy | Behaviour |
|----------|-----------|
| `ContextStrategy.none` | Does nothing — let the engine error when the window fills. |
| `ContextStrategy.slidingWindow` | Drops the oldest messages until usage is back below 80 %. |
| `ContextStrategy.summarize` | Summarises the oldest half of the conversation into a single digest. |

---

### SingleTurnChat — Stateless Inference

Ideal for batch processing, RAG pipelines, or any use-case where history is not needed:

```dart
final chat = SingleTurnChat(
  config: SessionConfig(temperature: 0.3, systemPrompt: 'You are a tagger.'),
);

// Plain text
final answer = await chat.generate('Summarise this article: ...');

// Streaming
await for (final token in chat.generateStream('Explain quantum computing.')) {
  stdout.write(token);
}

// Check how many tokens the last call consumed
print(chat.lastCallStats); // TokenStats(340 / 4096 used, 3756 remaining)
```

---

### Structured JSON Output

Both `GemmaChat` and `SingleTurnChat` support constrained JSON output using a Zod-like schema builder.

```dart
final schema = Schema.object({
  'name':  Schema.string().description('Full name'),
  'age':   Schema.number(),
  'roles': Schema.array(items: Schema.stringEnum(['admin', 'user', 'guest'])),
  'notes': Schema.string().optional(),
});

// GemmaChat — stream partial JSON as the model fills in fields
await for (final partial in chat.sendMessageJsonStream(
  text: 'Extract: "Alice, 30, admin"',
  schema: schema,
)) {
  print(partial); // progressively completed Map
}

// SingleTurnChat — returns a fully parsed Dart value
final result = await chat.generateJson(
  'Extract: "Bob, 25, user"',
  schema: schema,
) as Map<String, dynamic>;
print(result['name']); // "Bob"
```

Pass `rawSchemaStr` instead of `schema` to supply a raw JSON Schema string directly.

---

### Embeddings

```dart
// Load the embedding engine (Gemma 300M embedding model, 768-dim vectors)
await GemmaLoader.loadEmbedding(
  token: 'hf_…',
  onProgress: (p) => print('Embedding model: ${p.toStringAsFixed(0)}%'),
);

final vector = await EmbeddingPlugin().getEmbedding('Hello world');
print(vector.length); // 768

await GemmaLoader.unloadEmbedding();
```

**Cosine similarity:**

```dart
import 'dart:math';

double cosineSimilarity(List<double> a, List<double> b) {
  double dot = 0, normA = 0, normB = 0;
  for (int i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (sqrt(normA) * sqrt(normB));
}
```

---

### PDF Processing

```dart
final pdfBytes = await File('document.pdf').readAsBytes();

final parts = await PdfProcessor.extract(
  pdfBytes,
  const PdfExtractionConfig(
    mode: PdfExtractionMode.auto,   // text-first, images as fallback
    filter: PdfPageFilter.all,
    renderScale: 2.0,               // image DPI multiplier
  ),
);

for (final part in parts) {
  if (part is TextPart)  print(part.text);
  if (part is ImagePart) { /* part.bytes is a PNG */ }
}
```

| Mode | Text | Images |
|------|------|--------|
| `auto` | ✓ | Only if no text found |
| `textOnly` | ✓ | ✗ |
| `imagesOnly` | ✗ | ✓ |
| `fullRender` | ✗ | ✓ (rendered page images) |
| `textAndImages` | ✓ | ✓ |

Page ranges:

```dart
PdfExtractionConfig(
  filter: PdfPageFilter.range,
  startPage: 1,
  endPage: 5,
)
```

---

### RAG Helper

`DocumentEmbedder` combines PDF extraction and embedding in a single call. Image pages are described by the LLM before embedding, so charts and diagrams are included in the vector index.

```dart
final embeddings = await DocumentEmbedder.embedPdf(
  pdfBytes,
  const PdfExtractionConfig(mode: PdfExtractionMode.textAndImages),
  imageInterrogationPrompt:
      'Describe all text, data, and visual elements in this image.',
);
// Returns List<List<double>> — one embedding vector per text block / image
```

---

### Model Picker

Opens the platform-native file picker and returns an installed model path ready for `GemmaLoader.initLlm`:

```dart
final path = await GemmaModelPicker.pick();
if (path != null) {
  await GemmaLoader.initLlm(path: path);
}
```

Platform behaviour:
- **Android:** `Intent.ACTION_OPEN_DOCUMENT` with `takePersistableUriPermission`. Zero-copy for files on local storage; transparent copy for cloud/MTP sources.
- **Web:** `<input type="file">` element; installs to OPFS via a streaming write.

---

## Token Tracking

```dart
// ChatSession
print(session.stats);               // TokenStats(340 / 4096 used, 3756 remaining)
print(session.usedTokens);
print(session.remainingTokens);
print(session.isNearContextLimit);  // true when ≥ 80 % used

// GemmaChat
print(chat.tokenStats);
print(chat.currentTokenCount);

// SingleTurnChat (after a call)
print(chat.lastCallStats);

// Accurate count via native tokenizer
final count = await session.countTokens('my text', imageCount: 1);
```

---

## Example App

The `example/` directory contains a full Flutter app demonstrating all features:

| Screen | What it shows |
|--------|---------------|
| Chat | Multi-turn streaming conversation |
| Smart Chat | Multimodal input (image + text) |
| Embedding | Semantic similarity between sentences |
| Benchmark | Tokens-per-second measurement |
| Test Runner | Automated integration tests |

```sh
cd example
flutter run -d chrome           # Web
flutter run -d <android-id>     # Android
```

See [example/README.md](example/README.md) for details.

---

## Known Limitations

- **iOS / macOS / Windows / Linux:** Not supported. The plugin stubs out cleanly — methods throw `UnimplementedError` on unsupported platforms.
- **Model size:** Gemma 3n E2B INT4 is ~2 GB. Download progress and OPFS/local caching are built in to make this manageable, but users need adequate storage for the first load.
- **GPU on old Android:** Some GPUs are unsupported by LiteRT-LM. The plugin detects emulators and forces CPU; on physical devices pass `PreferredBackend.cpu` if you see native crashes.

---

## License

[MIT](LICENSE)
