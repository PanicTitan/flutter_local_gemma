# How Web Asset Bundling Works in Flutter

This project uses a "No-Cloud" approach for WebAssembly. Instead of fetching `.wasm` files from a CDN, we bundle them into the Flutter app.

### 1. The Asset Path Secret
In a Flutter Web app, assets from a plugin are not served at the root. They are nested under:
`http://your-app.com/assets/packages/<plugin_name>/<internal_path>`

### 2. Runtime Path Resolution
In Dart, we use `ui_web.BrowserPlatformLocation().getBaseHref()` to determine if the app is running at `domain.com/` or `domain.com/my_subfolder/`. We then use `Uri.parse(...).resolve(...)` to get an **Absolute URL**.

### 3. Passing the Base to JavaScript
Standard JS doesn't know about Flutter's asset structure. During the `init` phase, Dart passes this `assetBase` string to JavaScript.

### 4. JavaScript Local Loading
Once JS has the `assetBase`, it can fetch WASM files like this:
```javascript
const genAiWasmPath = `${assetBase}@mediapipe/tasks-genai/wasm`;
// MediaPipe now looks in: assets/packages/flutter_local_gemma/web/@mediapipe/...
await FilesetResolver.forGenAiTasks(genAiWasmPath);
```

### 5. The "Dummy Object" Trick (Race Conditions)
When multiple plugins (like `pdfx` and `flutter_local_gemma`) load, they might conflict.
- `pdfx` runs an `assert(window.pdfjsLib != null)` as soon as it starts.
- Our script injection is **asynchronous**.
- **Solution:** We synchronously set `window.pdfjsLib = {}` in Dart `registerWith`. This satisfies the other plugin's check instantly, while our real library loads in the background.

### 6. Pubspec Configuration
Every WASM, JS, and Worker file must be explicitly listed in the plugin's `pubspec.yaml` under `assets:`. If a file is missing there, the browser will return a `404` error when trying to initialize the engine.