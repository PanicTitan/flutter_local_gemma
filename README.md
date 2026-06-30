# flutter_local_gemma

A Flutter plugin for **on-device AI inference** powered by Google's Gemma 3n and Gemma 4 family of models.
Supports **Android** (via LiteRT-LM) and **Web** (via MediaPipe GenAI + LiteRT WASM).

---

## What's New in v2 (Compared to v1)

Version 2 introduces a complete architectural rebuild from v1 to support **Gemma 4**, **Agentic Tool Calling**, and **Thinking Mode**, while preserving backward compatibility.

- **Gemma 4 Support**: Full support for Gemma 4 (E2B and E4B) on Android and Web.
- **Thinking Mode**: Support for Gemma 4's reasoning capabilities (`<think>` tags) with a new `GemmaThinkingResponse` stream.
- **Agentic Tool Calling**: High-level `AgentChat` API with an autonomous loop. Supports both Dart skills and Gallery-compatible JS skills running in secure WebViews/iframes.
- **MTP (Speculative Decoding)**: Auto-detected support for MTP on Android for significantly faster inference.
- **Polymorphic API**: New `BaseChat` interface to seamlessly switch between `GemmaChat` (conversational) and `AgentChat` (tool-calling).
- **Multimodality Mutex**: Master `enableMultimodality` toggle that safely guards against using Vision/Audio simultaneously with Tool Calling (which is unsupported by the underlying LiteRT engine).
- **Model Lifecycle**: Simplified one-liner `GemmaLoader.loadGemmaModel` and a robust `GemmaModel` enum catalog.
- **Debug System**: A unified `GemmaDebug` logging system across Dart and Native code.

---

## Features

| Feature | Android | Web |
|---------|---------|-----|
| Gemma 3n E2B (text + vision + audio) | ✅ | ✅ |
| Gemma 3n E4B (text + vision + audio) | ✅ | ✅ |
| Gemma 4 E2B (text + vision + tools + thinking) | ✅ | ⚠️ (No Vision/Audio)* |
| Gemma 4 E4B (text + vision + tools + thinking) | ✅ | ⚠️ (No Vision/Audio)* |
| `GemmaLoader` — one-liner model lifecycle helper | ✅ | ✅ |
| `BaseChat` — polymorphic interface for `GemmaChat`/`AgentChat` | ✅ | ✅ |
| `sealed GemmaSkill` + `GemmaSkillList` typedef | ✅ | ✅ |
| `ToolCallParser` — standalone extracted parser | ✅ | ✅ |
| `ChatHistoryMixin` — reusable history mixin | ✅ | ✅ |
| `sendMessageResponseStream` — typed GemmaResponse events | ✅ | ✅ |
| `AgentChat.create()` static factory | ✅ | ✅ |
| `contextFillThreshold` configurable per chat instance | ✅ | ✅ |
| `enableMtp` — MTP speculative decoding (auto-detected) | ✅ | ❌ |
| `enableThinking` — thinking mode toggle | ✅ | ✅ |
| `enableMultimodality` — master multimodal toggle | ✅ | ✅ |
| Tools — Dart skills | ✅ | ✅ |
| Tools — JS skills (Gallery-compatible) | ✅ | ✅ |
| `AgentChat` — high-level agentic loop | ✅ | ✅ |
| Embeddings (Gemma 300M, 768-dim) | ✅ | ✅ |
| PDF extraction (text + images) | ✅ | ✅ |
| RAG helper (PDF → embeddings) | ✅ | ✅ |

> **Gemma 4 Web Multimodality Note:** Gemma 4 currently lacks vision and audio support on the Web target due to missing multimodal adapters in LiteRT-LM Web. For more details, see [LiteRT-LM Issue #2150](https://github.com/google-ai-edge/LiteRT-LM/issues/2150).
> **Gemma 3 Agentic Note:** Gemma 3 now supports skills/tool calls via prompt-based (non-native) mode on all platforms. However, Gemma 3 does **not** support Thinking mode (only Gemma 4 supports thinking).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Android Setup](#android-setup)
- [Web Setup](#web-setup)
- [Quick Start](#quick-start)
- [Model Catalog](#model-catalog)
- [API Reference](#api-reference)
  - [GemmaLoader — Model Lifecycle Helper](#gemmaloader--model-lifecycle-helper)
  - [Loading a Model (Low Level)](#loading-a-model-low-level)
  - [GemmaChat — Multi-turn Chat](#gemmachat--multi-turn-chat)
  - [BaseChat — Polymorphic Interface](#basechat--polymorphic-interface)
  - [Thinking Mode](#thinking-mode)
  - [sendMessageResponseStream — Typed Events](#sendmessageresponsestream--typed-events)
  - [SingleTurnChat — Stateless Inference](#singleturnchat--stateless-inference)
  - [AgentChat & Skills — Tool Calling](#agentchat--skills--tool-calling)
  - [GemmaSkill — Sealed Base Class](#gemmaskill--sealed-base-class)
  - [GemmaResponse Types](#gemmaresponse-types)
  - [ToolCallParser](#toolcallparser)
  - [ChatHistoryMixin](#chathistorymixin)
  - [Structured JSON Output](#structured-json-output)
  - [Embeddings](#embeddings)
  - [PDF Processing](#pdf-processing)
  - [RAG Helper](#rag-helper)
  - [Model Installer](#model-installer)
- [Token Tracking](#token-tracking)
- [SessionConfig Reference](#sessionconfig-reference)
- [GemmaDebug — Logging](#gemmadebug--logging)
- [Known Limitations](#known-limitations)
- [Example App](#example-app)
- [License](#license)

---

## Requirements

| Platform | Minimum |
|----------|---------|
| Android | API 24 (Android 7.0) |
| Flutter SDK | 3.3.0 |
| Dart SDK | 3.10.7 |
| Web browser | Chrome 120+ / Edge 120+ (SharedArrayBuffer required) |

> **Web note:** The browser must serve the app with `Cross-Origin-Opener-Policy: same-origin` and
> `Cross-Origin-Embedder-Policy: require-corp` headers so SharedArrayBuffer (required by WASM threads) is available.

---

## Installation

Add the package to your `pubspec.yaml` using the Git repository:

```yaml
dependencies:
  flutter_local_gemma:
    git:
      url: https://github.com/PanicTitan/flutter_local_gemma.git
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

Gemma 3n E2B INT4 requires approximately **2–3 GB of RAM** at runtime. On low-memory devices consider using the CPU backend and closing the engine when not in use.

---

## Web Setup

### 1. COOP/COEP headers (Your may need this headers for production)

Add these headers to your server (required for SharedArrayBuffer / WASM threads):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### 2. No manual script tags needed

The plugin injects `gemma_web.js` at runtime via a dynamic `<script type="module">` tag. All WASM binaries are bundled as Flutter assets — no CDN dependencies.

---

## Quick Start

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// 1. Initialise the engine + download the model
final gemma = FlutterLocalGemma();
await gemma.init(
  InferenceConfig(
    modelPath: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
    maxTokens: 4096,
    preferredBackend: PreferredBackend.gpu,
  ),
  onProgress: (p) => print('Downloading: ${p.toStringAsFixed(0)}%'),
);

// 2. Create a chat session
final session = await gemma.createSession(
  config: const SessionConfig(temperature: 0.8),
);

// 3. Stream a response token-by-token
await for (final token in session.generateResponseStream([TextPart('Hello!')])) {
  stdout.write(token);
}

// 4. Clean up
await session.dispose();
await gemma.dispose();
```

Or use the higher-level `GemmaChat` wrapper (recommended):

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// (after engine is initialised as above)
final chat = GemmaChat(
  systemPrompt: 'You are a helpful assistant.',
  maxContextTokens: 4096,
  contextStrategy: ContextStrategy.slidingWindow,
);
await chat.init();

await for (final token in chat.sendMessageStream(text: 'Hello!')) {
  stdout.write(token);
}

await chat.dispose();
```

---

## Model Catalog

### All Model Variants

| ID | Platform | HuggingFace Repo | File | Size | Gated |
|---|---|---|---|---|---|
| `gemma3nE2B` | Android | `google/gemma-3n-E2B-it-litert-lm` | `gemma-3n-E2B-it-int4.litertlm` | ~3 GB | ✅ |
| `gemma3nE2B` | Web | `google/gemma-3n-E2B-it-litert-lm` | `gemma-3n-E2B-it-int4-Web.litertlm` | ~2 GB | ✅ |
| `gemma3nE4B` | Android | `google/gemma-3n-E4B-it-litert-lm` | `gemma-3n-E4B-it-int4.litertlm` | ~4 GB | ✅ |
| `gemma3nE4B` | Web | `google/gemma-3n-E4B-it-litert-lm` | `gemma-3n-E4B-it-int4-Web.litertlm` | ~3 GB | ✅ |
| `gemma4E2B` | Android | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma-4-E2B-it.litertlm` | 2.59 GB | ❌ |
| `gemma4E2B` | Web | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma-4-E2B-it-web.task` | ~2 GB | ❌ |
| `gemma4E4B` | Android | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma-4-E4B-it.litertlm` | 3.66 GB | ❌ |
| `gemma4E4B` | Web | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma-4-E4B-it-web.task` | ~3 GB | ❌ |

Use the `GemmaModel` enum to reference models and get their platform-appropriate URLs:

```dart
final model = GemmaModel.gemma4E2B;
print(model.androidUrl);       // full HuggingFace Android download URL
print(model.webUrl);           // full HuggingFace Web download URL
print(model.supportsThinking); // true
print(model.supportsMtp);      // true (MTP drafter in shard #9)
print(model.isGated);          // false (no HuggingFace token needed)
print(model.displayName);      // 'Gemma 4 E2B'
print(model.sizeDescription);  // '2.6 GB  •  Open'
```

### Capability Matrix

| Model | Text | Vision | Audio | Tools | Thinking | MTP (Android) |
|-------|------|--------|-------|-------|----------|---------------|
| Gemma 3n E2B | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Gemma 3n E4B | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Gemma 4 E2B | ✅ | ✅ ¹ | ✅ ¹ | ✅ ¹ | ✅ | ✅ |
| Gemma 4 E4B | ✅ | ✅ ¹ | ❌ | ✅ ¹ | ✅ | ✅ |

> ¹ **Gemma 4 vision/audio and tools are mutually exclusive.** When `skills` are set in `SessionConfig`,
> multimodality is automatically suppressed (debug warning logged, no exception). Setting
> `enableMultimodality: false` also suppresses image/audio input.

---

## API Reference

### GemmaLoader — Model Lifecycle Helper

`GemmaLoader` wraps `ModelInstaller` and `FlutterLocalGemma` to provide a single-call model lifecycle API:

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// Download + initialise in one call
await GemmaLoader.loadGemmaModel(
  GemmaModel.gemma4E2B,
  token: 'hf_...',     // only needed for gated (✅) models
  maxTokens: 4096,
  useGpu: true,
  supportAudio: false,
  onProgress: (p) => print('${p.toStringAsFixed(0)}%'),
);

// Check availability before loading
final isCached = await GemmaLoader.isModelCached(GemmaModel.gemma4E2B);
final isReady  = await GemmaLoader.isModelReady(GemmaModel.gemma4E2B);

// Cache metadata
final cachedAt = await GemmaLoader.modelCachedAt(GemmaModel.gemma4E2B); // DateTime?

// Purge all cached models (Android: documents dir; Web: OPFS)
final freedBytes = await GemmaLoader.purgeCache();

// Purge a specific model
await GemmaLoader.purgeModel(GemmaModel.gemma4E2B);
```

---

### Loading a Model (Low Level)

Use `ModelInstaller` to download and cache model files, then pass the path to `FlutterLocalGemma.init()`:

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// Download from HuggingFace (streaming, progress callback)
final installer = ModelInstaller();
final path = await installer
    .fromNetwork(
      GemmaModel.gemma4E2B.androidUrl,
      token: 'hf_YOUR_TOKEN',  // only needed for gated (✅) models
    )
    .withProgress((p) => print('${p.toStringAsFixed(0)}%'))
    .install();

// Install from a local file (e.g. file picker result)
import 'package:cross_file/cross_file.dart';
final path2 = await installer.fromXFile(xFile);

// Web: install from a JS Blob (zero-copy, no RAM spike)
final path3 = await installer.fromWebBlob(jsBlob);

// Delete all cached files (Android: documents dir; Web: OPFS root)
final freedBytes = await ModelInstaller.purgeCache();
print('Freed $freedBytes bytes');

// Web: list files in OPFS cache
final files = await ModelInstaller.listWebCache();
// [{'name': 'gemma-4-E2B-it-web.task', 'size': 2720000000}, ...]

// Initialise the engine with the installed path
final gemma = FlutterLocalGemma();
await gemma.init(InferenceConfig(
  modelPath: path,
  maxTokens: 4096,
  preferredBackend: PreferredBackend.gpu,
  supportAudio: true,
));
```

> **MTP (Speculative Decoding):** For Gemma 4 Android models, MTP is **automatically enabled** when
> the model file contains the MTP drafter shard (shard #9, ~43 MB). This is auto-detected at engine
> init time — no flag needed from your code.

---

### GemmaChat — Multi-turn Chat

`GemmaChat` maintains conversation history, handles context-window overflow, and supports system prompts.

```dart
final chat = GemmaChat(
  systemPrompt: 'You are a helpful pirate. Answer in character.',
  maxContextTokens: 4096,
  contextStrategy: ContextStrategy.slidingWindow, // auto-manages overflow
  config: const SessionConfig(
    temperature: 0.8,
    topK: 40,
    enableThinking: false,
    enableMultimodality: true,
  ),
);
await chat.init();

// Streaming text response
await for (final token in chat.sendMessageStream(text: 'Ahoy!')) {
  stdout.write(token);
}

// Blocking response
final reply = await chat.sendMessage(text: 'Tell me a joke.');

// Multimodal — image + text
final imageBytes = await File('photo.png').readAsBytes();
await chat.sendMessage(
  text: 'What is in this image?',
  images: [imageBytes],
);

// Multimodal — audio (Android only)
final audioBytes = await File('clip.wav').readAsBytes();
await chat.sendMessage(
  text: 'Transcribe this audio.',
  audios: [audioBytes],
);

// Token tracking
print(chat.currentTokenCount);       // estimated tokens used
print(chat.isNearContextLimit);      // true when >= 80 % used
print(chat.remainingTokens);
print(chat.tokenStats);              // TokenStats object

// History management
print(chat.history.length);          // List<ChatMessage>
await chat.clearHistory();
await chat.removeHistory(index);
await chat.editHistory(index, ChatMessage(role: 'user', text: 'New text'));

// Persist and restore history across app sessions
final json = await chat.exportHistory();  // JSON string
await chat.importHistory(json);

// Stop mid-generation
chat.stop();

await chat.dispose();
```

#### Context Strategies

| Strategy | Behaviour |
|----------|-----------|
| `ContextStrategy.none` | Does nothing — let the engine error when the window fills. |
| `ContextStrategy.slidingWindow` | Drops the oldest messages until usage is below 80 %. |
| `ContextStrategy.summarize` | Summarises the oldest half of the conversation into a digest. |

---

### BaseChat — Polymorphic Interface

`BaseChat` is an abstract interface implemented by both `GemmaChat` and `AgentChat`.
Declare your field as `BaseChat?` to write UI code that works with either:

```dart
BaseChat? _chat;

_chat = GemmaChat(maxContextTokens: 4096);
// or:
_chat = AgentChat(skills: [mySkill]);

// Uniform API:
await _chat!.init();
await _chat!.clearHistory();
final json = await _chat!.exportHistory();
await _chat!.importHistory(json);
print(_chat!.tokenStats);
print(_chat!.isNearContextLimit);
await _chat!.dispose();
```

---

### Thinking Mode

When `SessionConfig.enableThinking = true`, Gemma 4 interleaves `<think>…</think>` reasoning blocks
mid-stream. The plugin's state-machine `ThinkingParser` handles these transparently.

**Via `sendMessageStream` — thinking stripped, only final answer returned:**

```dart
// sendMessageStream always returns only final-answer tokens.
// <think> blocks are automatically stripped — no API change needed.
final chat = GemmaChat(
  config: const SessionConfig(enableThinking: true),
);
await chat.init();

await for (final token in chat.sendMessageStream(text: 'Solve 17 x 19 step by step')) {
  stdout.write(token);  // only the final answer, no <think> tags
}
```

**Via `AgentChat.run()` — typed stream with thinking blocks exposed:**

```dart
final agent = AgentChat(
  skills: [],  // empty = pure agentic chat, no tools
  config: const SessionConfig(enableThinking: true),
);
await agent.init();

await for (final turn in agent.run('Solve 17 x 19 step by step')) {
  if (turn.thinkingBlocks.isNotEmpty) {
    print('Reasoning: ${turn.thinkingBlocks.last}');
  }
  print('Answer so far: ${turn.modelAnswer}');
  if (turn.isComplete) break;
}
```

---

### sendMessageResponseStream — Typed Events

Instead of raw string tokens, `sendMessageResponseStream` returns a `Stream<GemmaResponse>`
where each event is a typed sealed class:

```dart
final chat = GemmaChat(
  config: const SessionConfig(enableThinking: true),
);
await chat.init();

final thinkingBuf = StringBuffer();
final answerBuf   = StringBuffer();

await for (final resp in chat.sendMessageResponseStream(text: 'Solve 17 × 19')) {
  switch (resp) {
    case GemmaThinkingResponse(:final content):
      thinkingBuf.write(content);
    case GemmaTextResponse(:final token):
      answerBuf.write(token);
    case GemmaCancelledResponse():
      print('Stopped.');
    default:
      break;
  }
}

print('Reasoning: $thinkingBuf');
print('Answer: $answerBuf');
```

---

### SingleTurnChat — Stateless Inference

Ideal for batch processing, RAG pipelines, or any use-case where conversation history is not needed:

```dart
final chat = SingleTurnChat(
  config: const SessionConfig(temperature: 0.3),
);

// Plain text — blocking
final answer = await chat.generate('Summarise this article: ...');

// Streaming
await for (final token in chat.generateStream('Explain quantum computing.')) {
  stdout.write(token);
}

// JSON output (see Structured JSON Output section for schema builder)
final schema = Schema.object({'name': Schema.string(), 'age': Schema.number()});
final result = await chat.generateJson(
  'Extract: "Alice is 30 years old."',
  schema: schema,
) as Map<String, dynamic>;
print(result['name']); // "Alice"

// Stream partial JSON as the model fills in fields
await for (final partial in chat.generateJsonStream('Extract: "Bob, 25"', schema: schema)) {
  print(partial);  // progressively completed Map
}
```

---

### AgentChat & Skills — Tool Calling

`AgentChat` lets the model call skills to perform actions or fetch data before answering.
Skills can be written in Dart or JS (Gallery-compatible format).

#### Dart Skills

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// A simple skill without parameters
final timeSkill = GemmaDartSkill(
  name: 'get_time',
  description: 'Returns the current date and time in ISO 8601 format.',
  parametersSchema: {},
  handler: (_) async => GemmaSkillResult.text(DateTime.now().toIso8601String()),
);

// A skill with a JSON schema for its arguments
final morseSkill = GemmaDartSkill(
  name: 'morse_code',
  description: 'Converts text to Morse code. Pass {"text": "HELLO"}.',
  parametersSchema: {
    'text': {'type': 'string', 'description': 'Plain text to encode'},
  },
  handler: (args) async {
    final text = (args['text'] as String).toUpperCase();
    // (mock encoding logic...)
    return GemmaSkillResult.text('Morse: .... . .-.. .-.. ---');
  },
);
```

#### JS Skills (Gallery-Compatible)

> **Reference Guide:** You can explore a wide variety of official pre-built skills in the [Google Edge AI Gallery Skills Folder](https://github.com/google-ai-edge/gallery/tree/main/skills). You can load these remotely via URL or bundle them in your assets.
>
> **Compatibility Note:** This package is designed to be fully compatible with the standard Edge AI Gallery skill format. However, because on-device models (like Gemma 3 and 4) are much smaller than cloud models, we require an additional `parametersSchema` field in the skill definition (which the standard Gallery does not use). This explicitly injects the required parameter structure into the system prompt, vastly improving the reliability of the model's tool-call formatting.


```dart
// Load from a Flutter asset directory (bundled with your app):
final calculatorSkill = await GemmaJsSkill.fromAsset('assets/skills/calculator');

// Load from a remote URL:
final qrCodeSkill = await GemmaJsSkill.fromUrl(
  'https://raw.githubusercontent.com/google-ai-edge/ai-edge-gallery/main/skills/qr-code',
);

// Define inline (for simple one-off scripts):
final unitConverterSkill = GemmaJsSkill(
  name: 'convert_units',
  description: 'Converts a value between common units. Pass {"value": <number>, "from": "<unit>", "to": "<unit>"}.',
  parametersSchema: {
    'type': 'object',
    'properties': {
      'value': {'type': 'number'},
      'from': {'type': 'string'},
      'to': {'type': 'string'},
    }
  },
  scriptHtml: '''
    <script>
      window.ai_edge_gallery_get_result = async (args) => {
        // (conversion logic in JS...)
        return { result: convertedValue };
      };
    </script>
  ''',
);
```

#### Running the Agent

You can use the convenient `AgentChat.create()` factory to cleanly mix Dart skills, local asset JS skills, and remote URL JS skills in a single initialization:

```dart
final agent = await AgentChat.create(
  dartSkills:  [timeSkill, morseSkill],
  assetSkills: ['assets/skills/calculator'], // Automatically calls GemmaJsSkill.fromAsset
  urlSkills:   ['https://example.com/skills/my_skill'], // Automatically calls GemmaJsSkill.fromUrl
  maxLoops: 10,
  config: const SessionConfig(
    enableThinking: true,
    nativeToolCalling: false, // true = native constrained decoding on Android
  ),
);
await agent.init();

#### Under the Hood: Native vs Prompt-Based Tool Calling

The `nativeToolCalling` flag in `SessionConfig` dictates how the underlying engine interacts with skills:

**1. Native Constrained Decoding (`nativeToolCalling: true`)**
Supported on **Android with Gemma 4**. The LiteRT-LM C++ engine uses a strict grammar to constrain the model's output to valid tool signatures.
- The model outputs a proprietary token format: `call:skill_name{"arg": "value"}`.
- In Kotlin, the `GemmaAgentToolSet` uses the `@Tool` annotation to map `load_skill`, `run_js`, and `execute_skill` directly to the C++ runtime.
- JS skills are executed entirely natively in a hidden Android `WebView` pool (`JsSkillExecutor.kt`). Dart skills are routed back up to Flutter via a `MethodChannel`.

**2. Prompt-Based Tool Calling (`nativeToolCalling: false`)**
The fallback mode used for **Web targets** and **Gemma 3 models**, which lack C++ constrained decoding support.
- A custom system prompt is injected to instruct the model to emit JSON blocks: `<|tool_call|>{"name": "...", "arguments": {...}}<tool_call|>`.
- The token stream is continuously monitored by Dart's `ToolCallParser`. When a valid JSON block is matched, Dart intercepts the execution.
- JS skills run securely in a sandboxed, invisible `iframe` (on Web). The result is fed back into the conversation history as a `<|tool_response|>` block.

Because `ToolCallParser` seamlessly handles both the native `call:` syntax and the prompt-based JSON syntax, the `AgentChat` API behaves exactly the same in your Flutter code regardless of the underlying engine's architecture!

await for (final turn in agent.run('What time is it and what is the weather in Paris?')) {
  // Live updates during tool calls and streaming answer
  for (final call in turn.toolCalls) {
    print('Called: ${call.skillName} → ${call.resultJson}');
  }
  print('Answer: ${turn.modelAnswer}');
  if (turn.isComplete) {
    print('Done in ${turn.loopCount} loop(s)');
    break;
  }
}

// AgentTurn fields:
// turn.userMessage    — original user message
// turn.modelAnswer    — accumulated model text (updates mid-stream)
// turn.thinkingBlocks — List<String> of reasoning blocks (when enableThinking: true)
// turn.toolCalls      — List<AgentToolCall> of all executed calls so far
// turn.loopCount      — current iteration number
// turn.isComplete     — true only on the final turn (model gave text answer or max loops hit)
// turn.rawText        — raw token buffer (useful for debugging)

agent.clearHistory();     // reset conversation
await agent.stop();       // stop mid-generation (async)
await agent.dispose();
```

#### `AgentChat.create()` — Static Factory

Loads skills from assets and/or URLs before constructing the agent:

```dart
final agent = await AgentChat.create(
  assetSkills:          ['assets/skills/calculator'],
  urlSkills:            ['https://example.com/skills/qr-code'],
  dartSkills:           [myDartSkill],
  maxLoops:             10,
  contextFillThreshold: 0.85,
);
await agent.init();
```

#### History & Token Accessors

```dart
// History persistence
final json = await agent.exportHistory();
await agent.importHistory(json);
await agent.editHistory(0, ChatMessage(role: 'user', text: 'Updated'));
await agent.removeHistory(1);

// Token stats
print(agent.maxContextTokens);    // configured max
print(agent.currentTokenCount);   // estimated usage
print(agent.remainingTokens);     // maxContext - current
print(agent.isNearContextLimit);  // >= contextFillThreshold
print(agent.tokenStats);          // TokenStats struct
```

---

### GemmaSkill — Sealed Base Class

`GemmaSkill` is the sealed base for all skill types. Use `List<GemmaSkill>` for type-safe skill lists:

```dart
// Using the convenience typedef:
GemmaSkillList skills = [getTimeSkill, calculatorSkill];

// Pattern matching:
for (final skill in skills) {
  switch (skill) {
    case GemmaDartSkill(:final name):
      print('Dart skill: $name');
    case GemmaJsSkill(:final name):
      print('JS skill: $name');
  }
}
```

> `List<Object>` is still accepted everywhere for backwards compatibility.

---

### GemmaResponse Types

`GemmaResponse` is a sealed class. Currently **emitted** subtypes:

| Type | When |
|------|------|
| `GemmaTextResponse(token)` | Each text token from the model |
| `GemmaThinkingResponse(content)` | `<think>…</think>` blocks (thinking mode only) |
| `GemmaToolResultResponse(name, args, result)` | After a skill is called and returns |
| `GemmaCancelledResponse()` | When `stop()` is called mid-generation |

Two types are **reserved** (not yet emitted):
- `GemmaFunctionCallResponse` — future streaming tool-call API
- `GemmaParallelFunctionCallResponse` — future parallel tool-call API

---

### ToolCallParser

Stand-alone tool-call parser extracted from `AgentChat`. Useful for unit testing or custom dispatch:

```dart
final result = ToolCallParser.parse(modelRawText);
if (result != null) {
  print(result.name);      // skill name
  print(result.args);      // Map<String, dynamic>
  print(result.endIndex);  // offset of last token char
  print(result.isNative);  // true for Gemma 4 native format
}
```

Formats parsed automatically:
1. **JSON:** `<|tool_call|>{"name":"weather","arguments":{"city":"London"}}<tool_call|>`
2. **Native:** `call:weather{city:<|"|>London<|"|>}` (Gemma 4 constrained decoding)
3. **Hybrid:** `call:{"name":"weather","arguments":{"city":"London"}}`

### GemmaTextParser (UI rendering AST)

A pure Dart utility that converts raw Gemma text outputs (containing `<think>` tags and `<|tool_call|>` JSON) into a clean list of AST blocks. This saves you from writing complex regular expressions in your UI.

```dart
final parts = GemmaTextParser.parse(turn.rawText);

for (final part in parts) {
  if (part is TextBlock) {
    print(part.content); // Standard response text
  } else if (part is ThinkingBlock) {
    // Put inside an ExpansionTile
    print('Thinking: ${part.content}');
  } else if (part is ToolBlock) {
    print('Tool call syntax hidden from user: ${part.content}');
  }
}
```

---

### ChatHistoryMixin

Reusable mixin for custom chat wrappers:

```dart
class MyChat with ChatHistoryMixin {
  @override
  final List<ChatMessage> history = [];

  @override
  Future<void> onHistoryChanged() async { /* rebuild context */ }
}

final chat = MyChat();
await chat.addMessage(ChatMessage(role: 'user', text: 'Hi'));
await chat.editMessage(0, ChatMessage(role: 'user', text: 'Hello'));
await chat.removeMessage(0);
print(chat.isEmpty);          // true
print(chat.lastUserMessage);  // null
final json = chat.exportMessages();
await chat.importMessages(json);
```

#### Saving AgentChat History

Unlike standard chat, `AgentChat` yields `AgentTurn` objects representing intermediate multi-loop steps. When these are added to history, they are serialized as JSON. You can parse them back into `AgentTurn` instances using native deserialization:

```dart
// Native fromJson allows seamless UI reloading of exported Agent sessions
final data = jsonDecode(message.text);
if (data is Map<String, dynamic> && data['is_agent_turn'] == true) {
  final agentTurn = AgentTurn.fromJson(data);
  return AgentTurnBubble(agentTurn);
}
```

#### Native Tool Calling (Android)

Set `nativeToolCalling: true` to use the native Gemma 4 tool calling syntax (`call:NAME{...}`).
Instead of injecting JSON schemas, this mode modifies the system prompt to instruct the model to use the native syntax. The text stream is then parsed dynamically by `ToolCallParser` in Dart.

> **Note:** The underlying LiteRT-LM C++ constrained decoding is intentionally disabled in this plugin. This ensures the model directly calls the skills by their actual names rather than being forced through multiplexer functions (`execute_skill`, `run_js`), resulting in better reliability and flexibility.

```dart
final agent = AgentChat(
  skills: [assetSkill, timeSkill],
  config: const SessionConfig(nativeToolCalling: true),
);
await agent.init();

// Usage is identical — run() works the same either way
await for (final turn in agent.run('Make a QR code for https://flutter.dev')) {
  print(turn.modelAnswer);
  if (turn.isComplete) break;
}
```

#### JS Skill File Format

```
my-skill/
├── SKILL.md             ← Required: YAML frontmatter + LLM instructions
└── scripts/
    └── index.html       ← Required: exposes window.ai_edge_gallery_get_result
```

**`SKILL.md`:**
```markdown
---
name: search-wikipedia
description: Search Wikipedia for factual information about any topic.
parameters: {"type": "object", "properties": {"query": {"type": "string", "description": "the search query"}}} # Optional: JSON Schema to enforce typed arguments instead of a generic `data` string
---

## Instructions
Call this skill with JSON data containing:
- query: String — the search query.
```

**`scripts/index.html`** (Gallery result contract):
```html
<!DOCTYPE html><html><body><script>
window['ai_edge_gallery_get_result'] = async (data) => {
  const { query } = JSON.parse(data);
  const res = await fetch(`https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(query)}&prop=extracts&exintro&format=json&origin=*`);
  const json = await res.json();
  const page = Object.values(json.query.pages)[0];
  return JSON.stringify({
    result: page.extract?.replace(/<[^>]+>/g, '').slice(0, 500) ?? 'Not found.',
    // Optional extras:
    // image: { base64: '...' }
    // webview: { url: '...', iframe: true, aspectRatio: 1.33 }
    error: null
  });
};
</script></body></html>
```

#### Skill Result Types

```dart
// Text result — most common
return GemmaSkillResult.text('The answer is 42.');

// Text + image (image shown in UI, text fed back to model)
return GemmaSkillResult.withImage('QR code generated.', base64EncodedPng);

// Text + webview panel rendered in chat UI
return GemmaSkillResult.withWebview(
  'Map loaded.',
  GemmaWebviewResult(url: 'https://example.com/map', iframe: true, aspectRatio: 1.33),
);
```

---

### Structured JSON Output

Both `GemmaChat` and `SingleTurnChat` support constrained JSON generation using a schema builder.

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// Build a schema
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
  print(partial); // progressively completed Map<String, dynamic>
}

// SingleTurnChat — fully parsed result
final data = await singleTurnChat.generateJson(
  'Extract: "Bob, 25, user"',
  schema: schema,
) as Map<String, dynamic>;
print(data['name']); // "Bob"

// Pass a raw JSON Schema string directly
await chat.sendMessageJsonStream(
  text: 'Tag this article.',
  rawSchemaStr: '{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}}}',
);
```

---

### Embeddings

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// (Initialise engine with an embedding model first)
final plugin = EmbeddingPlugin();
final vector = await plugin.getEmbedding('Hello world');
print(vector.length); // 768

// Cosine similarity helper
import 'dart:math';
double cosineSimilarity(List<double> a, List<double> b) {
  double dot = 0, normA = 0, normB = 0;
  for (int i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (sqrt(normA) * sqrt(normB));
}

final sim = cosineSimilarity(
  await plugin.getEmbedding('What is the capital of France?'),
  await plugin.getEmbedding('Paris is the capital city of France.'),
);
print(sim); // ~0.92
```

---

### PDF Processing

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

final pdfBytes = await File('document.pdf').readAsBytes();

final parts = await PdfProcessor.extract(
  pdfBytes,
  const PdfExtractionConfig(
    mode: PdfExtractionMode.textAndImages,
    filter: PdfPageFilter.all,
    renderScale: 2.0,     // image DPI multiplier
  ),
);

for (final part in parts) {
  if (part is TextPart)  print(part.text);
  if (part is ImagePart) { /* part.bytes is PNG data */ }
}
```

| Mode | Text | Images |
|------|------|--------|
| `auto` | ✅ | Only if no text found |
| `textOnly` | ✅ | ❌ |
| `imagesOnly` | ❌ | ✅ |
| `fullRender` | ❌ | ✅ (rendered page) |
| `textAndImages` | ✅ | ✅ |

Page range extraction:

```dart
PdfExtractionConfig(
  filter: PdfPageFilter.range,
  startPage: 1,
  endPage: 5,
)
```

---

### RAG Helper

`DocumentEmbedder` combines PDF extraction and embedding in a single call. Image pages are described by
the LLM before embedding, so charts and diagrams are included in the vector index.

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

final embeddings = await DocumentEmbedder.embedPdf(
  pdfBytes,
  const PdfExtractionConfig(mode: PdfExtractionMode.textAndImages),
  imageInterrogationPrompt:
      'Describe all text, data, and visual elements in this image in detail.',
);
// Returns List<List<double>> — one 768-dim vector per text block / image page
```

---

### Model Installer

`ModelInstaller` provides platform-aware model installation across all supported download sources.

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// Download from a URL with progress
final installer = ModelInstaller(fileName: 'gemma-4-E2B-it.litertlm');
final path = await installer
    .fromNetwork(
      GemmaModel.gemma4E2B.androidUrl,
      token: 'hf_xxx',  // omit for open models
    )
    .withProgress((p) => setState(() => _progress = p))
    .install();

// Install from file picker (XFile — works on Android and Web)
import 'package:cross_file/cross_file.dart';
final path2 = await installer.fromXFile(pickedFile);

// Web only: install from a JS Blob (streaming copy to OPFS, zero extra RAM)
final path3 = await installer.fromWebBlob(jsBlob);

// Delete all cached files
final freedBytes = await ModelInstaller.purgeCache();

// Web only: list OPFS cached files
final files = await ModelInstaller.listWebCache();
// [{'name': 'gemma-4-E2B-it-web.task', 'size': 2720000000}]
```

---

## Token Tracking

```dart
// GemmaChat
print(chat.currentTokenCount);      // estimated tokens used
print(chat.remainingTokens);        // maxContextTokens - currentTokenCount
print(chat.isNearContextLimit);     // true when >= 80 % used
print(chat.tokenStats);             // TokenStats(estimatedUsed, maxContext)

// Estimate UI input size before sending
final projectedSize = await chat.estimateNextTurnTokens(
  'What is in this image?',
  images: [imageBytes],
);
print('Tokens if I hit send: $projectedSize');

// Low-level ChatSession
print(session.stats);
print(session.usedTokens);
print(session.remainingTokens);
print(session.isNearContextLimit);

// Accurate native tokenizer count
final count = await session.countTokens('my text', imageCount: 1);
```

---

## SessionConfig Reference

```dart
const SessionConfig({
  double  temperature    = 0.8,    // ⚠️ Android: not applied (LiteRT-LM limitation)
  int     topK           = 40,     // ⚠️ Android: not applied (LiteRT-LM limitation)
  double? topP,                    // ⚠️ Android: not applied (LiteRT-LM limitation)
  int?    randomSeed,
  String? systemPrompt,
  bool    enableThinking      = false, // Gemma 4 only; strips <think> from sendMessageStream
  bool    enableMultimodality = true,  // false: suppresses all image/audio input
  bool    nativeToolCalling   = false, // true: LiteRT-LM native engine path (Android only)
  List<Object> skills         = const [],  // GemmaDartSkill | GemmaJsSkill
})
```

> **Android sampling note:** `temperature`, `topP`, and `topK` are sent from Dart but the LiteRT-LM
> SDK does not expose per-generation sampling on its conversation API. These values have no effect on
> Android. They are fully respected on Web.

> **Multimodality mutex:** When `skills.isNotEmpty`, `enableMultimodality` is automatically forced to
> `false` (vision and tools are mutually exclusive on Gemma 4). A `debugPrint` warning is logged.
> No exception is thrown.

---

## GemmaDebug — Logging

`GemmaDebug` is a conditional logging utility that is a no-op in release builds:

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';

// Enable globally (call once, e.g. during app init)
GemmaDebug.enabled = true;

// Log with a module tag
GemmaDebug.log(GemmaDebug.tagChat,   'Session created');
GemmaDebug.log(GemmaDebug.tagAgent,  'Running loop 1');
GemmaDebug.log(GemmaDebug.tagLoader, 'Model ready');

// Available tags
GemmaDebug.tagChat;    // 'GemmaChat'
GemmaDebug.tagAgent;   // 'AgentChat'
GemmaDebug.tagLoader;  // 'GemmaLoader'
GemmaDebug.tagGuard;   // 'MultimodalityGuard'
```

> In release builds, `GemmaDebug.log(...)` compiles to nothing — it is completely stripped.
> Debug mode respects `GemmaDebug.enabled` (defaults to `false`).

---

## Known Limitations

- **iOS / macOS / Windows / Linux:** Not supported. Methods throw `UnimplementedError` on unsupported platforms.
- **Model size:** Gemma 4 E2B is ~2.6 GB. Download progress and OPFS/local caching are built in.
- **GPU on old Android:** Some GPUs are unsupported. Pass `PreferredBackend.cpu` if you see native crashes.
- **Temperature/topK/topP on Android:** No-ops. The LiteRT-LM SDK does not expose per-generation sampling.
- **MTP on Web:** Not available in the current MediaPipe build. `enableMtp` in `InferenceConfig` is ignored on Web.
- **Gemma 4 vision + tools:** Mutually exclusive. Enable one or the other per session.
- **Audio input:** Android only (PCM WAV). Not supported on Web.
- **`GemmaFunctionCallResponse` / `GemmaParallelFunctionCallResponse`:** Defined but not yet emitted by the engine. Reserved for a future streaming tool-call API.

---

## Example App

The `example/` directory contains a full reference application demonstrating all existing and new features across multiple dedicated tabs:

- **Chat Tab:** A unified chat interface showcasing image, audio, and PDF upload capabilities. It includes comprehensive settings to enable/disable Thinking mode, Tool Calling (Agentic loop), Skills, Gemma 4 MTP, and other model settings on the fly. It dynamically switches between `GemmaChat` and `AgentChat`.
- **Models Tab:** Complete model management. Download, load, unload, delete models, and pick local `.litertlm`/`.task` files.
- **Embedding Tab:** Semantic search and similarity features using the Gemma 300M embedding model.
- **Tests Tab:** Integrated runner to execute automated integration tests directly on the device, validating both new and existing features (vision, tools, thinking, etc).
- **Bench Tab:** Benchmarking tools to measure tokens-per-second, comparing performance with/without MTP, Gemma 3 vs Gemma 4, and skill execution overhead.

To run:

```sh
cd example
flutter run -d chrome          # Web
flutter run -d <android-id>    # Android
```

---

## License

[MIT](LICENSE)