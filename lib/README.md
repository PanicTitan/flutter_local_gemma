# flutter_local_gemma — Dart sources

The public API is entirely exported from **`lib/flutter_local_gemma.dart`**.  
Import only that one file in consumer code:

```dart
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
```

---

## Folder map

| Folder | Contents |
|--------|----------|
| `gemma/` | `FlutterLocalGemma` singleton engine, `ChatSession`, all config types (`InferenceConfig`, `SessionConfig`, `AutoStopConfig`, `TokenStats`). Platform-conditional web/stub/mobile files live here. |
| `chat/` | `GemmaChat` (stateful multi-turn) and `SingleTurnChat` (stateless). |
| `embedding/` | `EmbeddingPlugin` + `EmbeddingConfig`. Web and stub implementations. |
| `pdf/` | `PdfProcessor` + `PdfExtractionConfig`. Web and stub implementations. |
| `helpers/` | `GemmaLoader` (download + init helpers) and `DocumentEmbedder` (RAG pipeline). |
| `json_schema/` | `Schema` builder and `repairJson` streaming-tolerant JSON parser. |
| `model_installer/` | `ModelInstaller` and `ModelDownloaderBuilder` — platform-specific download, OPFS install, and cache management. |
| `model_picker/` | `GemmaModelPicker` — wraps native SAF picker (Android) and `<input file>` (web). |
| `types/` | `ContentPart` sealed hierarchy: `TextPart`, `ImagePart`, `AudioPart`. |
| `utils/` | `WebScriptLoader` — injects `gemma_web.js` at runtime and resolves the Flutter asset base URL. |

---

## Platform-conditional file pattern

Several subsystems have three files:

```
something.dart          ← public API + conditional import
something_web.dart      ← web implementation (dart:js_interop)
something_stub.dart     ← mobile stub (throws UnimplementedError)
something_mobile.dart   ← mobile implementation (MethodChannel / file I/O)
```

The `something.dart` file uses:

```dart
import 'something_web.dart' if (dart.library.io) 'something_stub.dart';
```

This keeps web-only imports (`dart:js_interop`, `package:web`) from polluting mobile builds.