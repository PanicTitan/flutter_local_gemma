# flutter_local_gemma — Testing Guide

## Overview

Three layers of tests, each with a different purpose:

| Layer | File | Needs device? | Duration |
|-------|------|---------------|----------|
| **Unit tests** | `test/unit_test.dart` | ❌ No | ~1 s |
| **Integration tests** (automated) | `integration_test/plugin_test.dart` | ✅ Yes | ~10–30 min (first run, downloads models) |
| **Visual test runner** (interactive) | Tests tab in the app | ✅ Yes | Same as above |

---

## 1. Unit tests (pure Dart, no device)

Tests BenchmarkRunner, TokenStats, ChatMessage serialisation, SessionConfig defaults — everything that doesn't touch the native plugin.

```bash
flutter test test/unit_test.dart
```

---

## 2. Visual test runner (interactive)

Build and run the example app normally. Navigate to the **Tests** tab (🧪 icon). Tap **Run All Tests**. Each test row updates live as it runs.

```bash
# Android
flutter run -d 

# Web
flutter run -d chrome
```

Individual failing tests can be retried by tapping the **Retry** button on the row.

---

## 3. Integration tests (fully automated, CI-friendly)

The same test suite runs headlessly via `integration_test`, driven by the test harness.

### Android

```bash
# Connect a device or start an emulator first.
flutter test integration_test/plugin_test.dart \
  --timeout=none \
  -d 
```

### Web (requires ChromeDriver)

```bash
# 1. Download ChromeDriver matching your Chrome version:
#    https://chromedriver.chromium.org/downloads
# 2. Start it in a separate terminal:
chromedriver --port=4444

# 3. Run the tests:
flutter drive \
  --driver=test_driver/integration_test_driver.dart \
  --target=integration_test/plugin_test.dart \
  --timeout=none \
  -d chrome
```

### pubspec.yaml additions needed

```yaml
dev_dependencies:
  integration_test:
    sdk: flutter
  flutter_test:
    sdk: flutter
```

---

## What is tested (integration suite)

### Phase 1 — LLM lifecycle
| ID | Test |
|----|------|
| `llm_download` | Downloads the Gemma model from HuggingFace |
| `llm_init` | Initialises the engine from the cached file |

### Phase 2 — LLM inference
| ID | Test |
|----|------|
| `llm_single_turn` | `SingleTurnChat.generate()` returns a reply |
| `llm_streaming` | `GemmaChat.sendMessageStream()` yields tokens incrementally |
| `llm_stop_mid_stream` | `chat.stop()` halts generation after ≥5 tokens |
| `llm_with_image` | Passes a 1×1 JPEG alongside the prompt *(Android only)* |
| `llm_with_audio` | Passes PCM audio bytes alongside the prompt *(Android only)* |
| `llm_with_pdf` | Extracts text+images from a PDF and sends to the LLM |
| `llm_cache_clear` | `clearHistory()` resets context; next message still works |
| `llm_settings_update` | Low/high temperature both produce non-empty replies |

### Phase 3 — JSON schema
| ID | Test |
|----|------|
| `llm_json_schema` | `sendMessageJsonStream()` produces valid JSON matching schema |

### Phase 4 — Multi-turn chat
| ID | Test |
|----|------|
| `chat_multi_turn` | Context preserved across 3 turns |
| `chat_history_export_import` | `exportHistory` / `importHistory` round-trip |
| `chat_sliding_window` | Sliding-window strategy trims old messages transparently |
| `chat_edit_delete_history` | `editHistory` / `removeHistory` mutate history list |

### Phase 5 — LLM unload / reload
| ID | Test |
|----|------|
| `llm_unload` | `dispose()` → `isInitialized == false` |
| `llm_reload` | Second `init()` from cache path works; inference succeeds |

### Phase 6 — Embedding
| ID | Test |
|----|------|
| `embed_download` | Downloads embedding model (+ tokenizer on Android) |
| `embed_init` | Initialises `EmbeddingPlugin` |
| `embed_vector` | `getEmbedding()` returns a non-zero float vector |
| `embed_similarity` | Semantically similar sentences score higher than dissimilar ones |
| `embed_search` | Top-1 semantic search selects the correct corpus entry |
| `embed_unload` | `dispose()` → `isInitialized == false` |

### Phase 7 — Cache
| ID | Test |
|----|------|
| `cache_purge` | `ModelInstaller.purgeCache()` completes without error |

---

## Tips

- **First run is slow** — models are ~2–4 GB. Subsequent runs reuse the cache and only re-run inference tests.
- **`--timeout=none`** is important; the default Flutter test timeout (30 s) is too short for model downloads.
- **Skipped tests** (e.g. audio/image on web) show as ⏭ skipped and don't count as failures.
- **Retry failed tests** — the visual runner has a per-row Retry button; in CI, re-run the specific test ID via `--name <id>`.