# flutter_local_gemma — Example App

A full-featured Flutter application that demonstrates all capabilities of the `flutter_local_gemma` plugin on **Android** and **Web**.

---

## Screens

| Screen | File | What it shows |
|--------|------|---------------|
| **Chat** | `lib/screens/chat_screen.dart` | Multi-turn streaming conversation with `GemmaChat`. Includes a token-usage bar and stop-generation button. |
| **Smart Chat** | `lib/screens/smart_chat_screen.dart` | Multimodal input — attach images or audio alongside text messages. |
| **Embedding** | `lib/screens/embedding_screen.dart` | Compute semantic embeddings and display cosine-similarity scores between sentences. |
| **Benchmark** | `lib/screens/benchmark_screen.dart` | Measure tokens-per-second throughput using `BenchmarkRunner`. |
| **Test Runner** | `lib/screens/test_runner_screen.dart` | Run the built-in integration test suite against the live model. |

---

## Running the example

### Prerequisites

1. A HuggingFace account with access to the [Gemma 3n gated model](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).
2. A HuggingFace access token (`hf_…`).

### Android

```sh
flutter run -d <android-device-id>
```

Minimum Android API: 24.  
Recommended: a device with ≥ 6 GB RAM for the default INT4 model.

### Web

```sh
flutter run -d chrome
```

Flutter's dev server automatically adds the required `COOP`/`COEP` headers.  
Recommended: Chrome 120+ on a machine with ≥ 8 GB RAM.

---

## Model loading

On first launch the app prompts for a HuggingFace token and downloads the model. Progress is displayed on the home screen. The downloaded model is cached in:

- **Android:** `applicationDocumentsDirectory/`
- **Web:** Origin Private File System (OPFS) — survives page reloads

Subsequent launches load from cache (no re-download).

You can also load a model you already have on disk using the **file picker** button on the home screen.

---

## Running tests

### Unit tests

```sh
flutter test
```

### Integration tests (requires a connected device with model loaded)

```sh
flutter test integration_test/plugin_integration_test.dart -d <device-id>
```

See [TESTING.MD](TESTING.MD) for the full test suite description.

---

## Project structure

```
example/
├── lib/
│   ├── main.dart               # App entry point and model loader UI
│   ├── app_state.dart          # Top-level ChangeNotifier for model state
│   ├── benchmark_runner.dart   # Tokens/sec benchmark logic
│   ├── screens/                # One file per screen
│   ├── widgets/                # Reusable UI components
│   │   ├── message_bubble.dart
│   │   ├── model_status_chip.dart
│   │   └── token_counter_bar.dart
│   ├── testing/
│   │   └── test_suite.dart     # In-app integration test definitions
│   └── utils/
│       └── model_loader.dart   # Model download / init helpers
├── integration_test/           # Flutter integration test runner
└── test/                       # Unit tests + test assets
    ├── test.pdf
    ├── test.png
    └── test.wav
```