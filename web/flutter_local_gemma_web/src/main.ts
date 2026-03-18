/**
 * main.ts — Gemma & LiteRT Web Bridge
 *
 * Exposes the following globals to Flutter's JS interop layer:
 *
 * LLM (MediaPipe GenAI)
 *   initLLM(options)              – load and initialise the Gemma LLM
 *   generateResponse(parts, cb)   – stream tokens via callback
 *   cancelProcessing()            – interrupt the current generation
 *   unloadLLM()                   – fully release the LLM from memory
 *   countTokens(text)             – estimate token count for a string
 *
 * Embeddings (LiteRT + Transformers.js)
 *   initEmbeddingModel(url, base, token?, onProgress?) – load model + tokenizer (OPFS cached)
 *   getEmbedding(text)            – compute a float32 embedding vector
 *   unloadEmbeddingModel()        – release model and tokenizer from memory
 *
 * PDF (PDF.js)
 *   initPdfWorker(assetBase)      – point PDF.js at the bundled worker
 *   extractPdf(bytes, ...)        – extract text / render pages to images
 *
 * Model Installer (OPFS)
 *   downloadModelWithProgress(url, token, onProgress) – stream to disk
 */

import type { TextItem } from 'pdfjs-dist/types/src/display/api';

console.info('[GemmaWeb] Bridge loaded.');

// ─── Shared constants ────────────────────────────────────────────────────────

/** Fixed sequence length for the embedding model input tensor. */
const EMBEDDING_SEQ_LEN = 256;

// ─── Module-level state ──────────────────────────────────────────────────────

// LLM
let llmInstance: any = null;
type ActiveGeneration = { cancel: () => void };
let activeGeneration: ActiveGeneration | null = null;

// Embeddings
let tfliteModel: any = null;
let tokenizer: any = null;
let currentEmbedModelUrl: string | null = null;
let isLiteRtRuntimeLoaded = false; // WASM binary; survive model unload

// PDF
let pdfWorkerReady = false;

// ─────────────────────────────────────────────────────────────────────────────
// 1. OPFS Model Downloader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Downloads a model file using the Origin Private File System (OPFS).
 *
 * OPFS streams the response body directly to a private on-disk file, which
 * means multi-gigabyte models never fully reside in JS heap memory.
 * On browsers that don't support OPFS the raw URL is returned as a fallback
 * so MediaPipe can attempt its own fetch.
 *
 * @param url        Full URL of the model binary.
 * @param token      Optional Bearer token for authenticated endpoints (e.g. HuggingFace).
 * @param onProgress Called with a value in [0, 100] as bytes arrive.
 * @returns          An object-URL string pointing to the cached file.
 */
(window as any).downloadModelWithProgress = async (
    url: string,
    token: string | null,
    onProgress: (progress: number) => void
): Promise<string> => {
    if (!navigator.storage?.getDirectory) {
        console.warn('[GemmaWeb] OPFS unavailable – returning raw URL.');
        return url;
    }

    try {
        const root = await navigator.storage.getDirectory();
        const fileName = url.split('/').pop()?.split('?')[0] || 'model_cache.bin';
        const fileHandle = await root.getFileHandle(fileName, { create: true });

        // If the file already looks complete (> 50 MB), skip the download.
        const existing = await fileHandle.getFile();
        if (existing.size > 50 * 1024 * 1024) {
            onProgress(100);
            return URL.createObjectURL(existing);
        }

        const headers: Record<string, string> = token
            ? { Authorization: `Bearer ${token}` }
            : {};

        const response = await fetch(url, { headers, mode: 'cors' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const totalBytes = parseInt(
            response.headers.get('content-length') ??
            response.headers.get('x-linked-size') ??
            '0',
            10,
        );

        let loadedBytes = 0;
        const writable = await fileHandle.createWritable();
        const reader = response.body!.getReader();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            await writable.write(value);
            loadedBytes += value.byteLength;
            if (totalBytes > 0) onProgress((loadedBytes / totalBytes) * 100);
        }
        await writable.close();

        return URL.createObjectURL(await fileHandle.getFile());
    } catch (err) {
        console.error('[GemmaWeb] OPFS download error:', err);
        return url; // Fallback so MediaPipe can try on its own
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. LiteRT Embedding Engine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialises the LiteRT embedding model and the Gemma tokenizer.
 *
 * The LiteRT WASM runtime is loaded only once per page lifetime; subsequent
 * calls that swap to a different model re-use the already-loaded runtime.
 *
 * ## OPFS caching
 * The model binary is streamed into OPFS on first use (identical to what
 * `downloadModelWithProgress` does for the LLM). On subsequent calls the
 * cached file is read back from disk — no network round-trip, no re-download.
 * The cached file name is derived from the URL so different models coexist.
 *
 * This means the embedding model shows up in DevTools → Application → Storage
 * alongside the LLM, and is correctly removed by `purgeOpfsCache`.
 *
 * If OPFS is unavailable the model is fetched directly into memory as before
 * (no persistence — falls back to the old behaviour gracefully).
 *
 * @param modelUrl    URL of the `.tflite` embedding model.
 * @param assetBase   Path prefix for bundled WASM assets.
 * @param token       Optional Bearer token for authenticated model fetches.
 * @param onProgress  Optional callback receiving [0, 100] during first download.
 */
(window as any).initEmbeddingModel = async (
    modelUrl: string,
    assetBase: string,
    token?: string,
    onProgress?: (progress: number) => void,
): Promise<boolean> => {
    try {
        const { loadLiteRt, loadAndCompile } = await import('@litertjs/core');

        // Load the WASM runtime only once.
        if (!isLiteRtRuntimeLoaded) {
            const wasmPath = `${assetBase}@litertjs/core/wasm/`;
            console.log('[GemmaWeb] Loading LiteRT WASM from', wasmPath);
            await loadLiteRt(wasmPath);
            isLiteRtRuntimeLoaded = true;
        }

        // If the same model is already compiled in memory, nothing to do.
        if (tfliteModel && currentEmbedModelUrl === modelUrl && tokenizer) {
            return true;
        }

        // Free any previously compiled model before loading the new one.
        _releaseCompiledModel();

        // ── Load model bytes — OPFS cache first, network fallback ───────────────
        const bytes = await _loadEmbeddingModelBytes(modelUrl, token, onProgress);

        tfliteModel = await loadAndCompile(bytes, { accelerator: 'webgpu' });
        currentEmbedModelUrl = modelUrl;

        // Load the tokenizer lazily (transformers.js caches it in Cache Storage).
        if (!tokenizer) {
            const { AutoTokenizer, env } = await import('@huggingface/transformers');
            env.allowLocalModels = false;
            env.useBrowserCache = true;
            tokenizer = await AutoTokenizer.from_pretrained('onnx-community/embeddinggemma-300m-ONNX');
        }

        console.log('[GemmaWeb] Embedding engine ready.');
        return true;
    } catch (err) {
        console.error('[GemmaWeb] initEmbeddingModel failed:', err);
        throw err;
    }
};

/**
 * Returns the embedding model as a Uint8Array, using OPFS as a persistent
 * cache so the binary is only downloaded once.
 *
 * Flow:
 *  1. Derive a stable file name from the URL (`embed_<filename>`).
 *  2. If a cached OPFS file exists and is > 1 MB, read and return it.
 *  3. Otherwise stream the URL to OPFS while reporting progress, then return
 *     the bytes from the newly-written file.
 *  4. If OPFS is unavailable at all, fall back to a plain fetch → arrayBuffer.
 *
 * The `embed_` prefix distinguishes embedding models from LLMs in OPFS so
 * that targeted purges (e.g. "delete only the LLM") remain possible.
 */
async function _loadEmbeddingModelBytes(
    url: string,
    token?: string,
    onProgress?: (progress: number) => void,
): Promise<Uint8Array> {
    const fileName = 'embed_' + (url.split('/').pop()?.split('?')[0] ?? 'model.tflite');

    // ── Try OPFS ──────────────────────────────────────────────────────────────
    if (navigator.storage?.getDirectory) {
        try {
            const root = await navigator.storage.getDirectory();
            const fileHandle = await root.getFileHandle(fileName, { create: true });
            const existing = await fileHandle.getFile();

            if (existing.size > 1024 * 1024) {
                // Cache hit — read straight from OPFS, no network needed.
                console.log(`[GemmaWeb] Embedding model loaded from OPFS cache: ${fileName} (${(existing.size / 1024 / 1024).toFixed(1)} MB)`);
                onProgress?.(100);
                return new Uint8Array(await existing.arrayBuffer());
            }

            // Cache miss — stream from network into OPFS.
            console.log(`[GemmaWeb] Downloading embedding model to OPFS: ${fileName}`);
            const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
            const response = await fetch(url, { headers, mode: 'cors' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const totalBytes = parseInt(
                response.headers.get('content-length') ??
                response.headers.get('x-linked-size') ?? '0',
                10,
            );

            let loadedBytes = 0;
            const writable = await fileHandle.createWritable();
            const reader = response.body!.getReader();
            const chunks: Uint8Array[] = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                await writable.write(value);
                chunks.push(value);
                loadedBytes += value.byteLength;
                if (totalBytes > 0) onProgress?.((loadedBytes / totalBytes) * 100);
            }
            await writable.close();

            console.log(`[GemmaWeb] Embedding model cached to OPFS: ${fileName} (${(loadedBytes / 1024 / 1024).toFixed(1)} MB)`);

            // Re-read from the closed file handle to get a single contiguous buffer.
            const saved = await (await root.getFileHandle(fileName)).getFile();
            return new Uint8Array(await saved.arrayBuffer());

        } catch (opfsErr) {
            console.warn('[GemmaWeb] OPFS unavailable for embedding model, falling back to direct fetch:', opfsErr);
            // Fall through to the plain fetch below.
        }
    }

    // ── Plain fetch fallback (no OPFS) ────────────────────────────────────────
    console.log('[GemmaWeb] Fetching embedding model directly into memory (no OPFS).');
    const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`Model fetch failed: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
}

/**
 * Computes a semantic embedding vector for [text].
 *
 * Pads / truncates to EMBEDDING_SEQ_LEN and applies mean-pooling when the
 * model returns a full sequence output ([1, seq, dim]) rather than a pooled
 * one ([1, dim]).
 */
(window as any).getEmbedding = async (text: string): Promise<Float32Array> => {
    if (!tfliteModel || !tokenizer) throw new Error('[GemmaWeb] Embedder not initialised.');

    const { Tensor } = await import('@litertjs/core');

    const encoded = await tokenizer(text, {
        padding: 'max_length',
        truncation: true,
        max_length: EMBEDDING_SEQ_LEN,
    });

    // LiteRT models expect Int32 input tensors.
    const inputInt32 = new Int32Array(EMBEDDING_SEQ_LEN);
    for (let i = 0; i < EMBEDDING_SEQ_LEN; i++) {
        inputInt32[i] = Number(encoded.input_ids.data[i]);
    }

    const inputTensor = new Tensor(inputInt32, [1, EMBEDDING_SEQ_LEN]);
    const inputs: Record<string, any> = {};
    inputs[tfliteModel.getInputDetails()[0].name] = inputTensor;

    try {
        const outputs = await tfliteModel.run(inputs);
        const rawData = await (Object.values<any>(outputs)[0].data()) as Iterable<number>;
        const rawOutput = Float32Array.from(rawData);

        // If the model returns the full sequence ([1, seqLen, dim]), apply mean-pooling.
        if (rawOutput.length > 4096) {
            const dim = rawOutput.length / EMBEDDING_SEQ_LEN;
            return _meanPool(rawOutput, encoded.attention_mask.data, dim);
        }
        return rawOutput;
    } finally {
        inputTensor?.delete?.();
    }
};

/**
 * Releases the compiled LiteRT model and tokenizer from memory.
 *
 * The WASM runtime itself is NOT unloaded because doing so would require a
 * full page reload. Only the compiled model graph and tokenizer are freed.
 */
(window as any).unloadEmbeddingModel = (): void => {
    _releaseCompiledModel();
    // Note: tokenizer is kept alive because it's cheap and reloading it
    // requires a network round-trip on first use. Set to null if you want
    // it fully released.
    tokenizer = null;
    console.log('[GemmaWeb] Embedding model unloaded.');
};

/** Internal helper: nulls the compiled model without touching the runtime. */
function _releaseCompiledModel(): void {
    try { tfliteModel?.close?.(); } catch { /* ignore */ }
    tfliteModel = null;
    currentEmbedModelUrl = null;
}

/** Attention-mask weighted mean pooling over a [seqLen × dim] output. */
function _meanPool(output: Float32Array, mask: any, dim: number): Float32Array {
    const embedding = new Float32Array(dim).fill(0);
    let count = 0;
    for (let i = 0; i < EMBEDDING_SEQ_LEN; i++) {
        if (mask[i] > 0) {
            for (let j = 0; j < dim; j++) embedding[j] += output[i * dim + j];
            count++;
        }
    }
    if (count > 0) for (let j = 0; j < dim; j++) embedding[j] /= count;
    return embedding;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MediaPipe GenAI LLM Engine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Loads and initialises the MediaPipe GenAI LLM engine.
 *
 * @param options.assetBase          Base URL for bundled WASM assets.
 * @param options.baseOptions.modelPath  URL / OPFS object-URL of the model binary.
 * @param options.maxTokens          KV-cache capacity.
 * @param options.supportAudio       Whether to enable the audio modality.
 */
(window as any).initLLM = async (options: any): Promise<boolean> => {
    try {
        const { FilesetResolver, LlmInference } = await import('@mediapipe/tasks-genai');

        const assetBase = options.assetBase;
        if (!assetBase) throw new Error('[GemmaWeb] initLLM: assetBase is missing.');

        // Unload any previously loaded instance before creating a new one.
        _releaseLlmInstance();

        const wasmPath = `${assetBase}@mediapipe/tasks-genai/wasm`;
        console.log('[GemmaWeb] Loading MediaPipe GenAI WASM from', wasmPath);

        const resolver = await FilesetResolver.forGenAiTasks(wasmPath);
        llmInstance = await LlmInference.createFromOptions(resolver, {
            baseOptions: {
                modelAssetPath: options?.baseOptions?.modelPath ?? options?.modelPath,
            },
            maxTokens: options.maxTokens ?? 4096,
            maxNumImages: options.maxNumImages ?? 1,
            supportAudio: !!options.supportAudio,
            randomSeed: options.randomSeed ?? 101,
        });

        console.log('[GemmaWeb] LLM engine ready.');
        return true;
    } catch (err) {
        console.error('[GemmaWeb] initLLM failed:', err);
        throw err;
    }
};

/**
 * Starts an async generation and streams tokens via [callback].
 *
 * @param parts    Array of content objects understood by MediaPipe (strings,
 *                 image / audio source objects).
 * @param callback Called with (partialText, isDone) for every token event.
 */
(window as any).generateResponse = (
    parts: any[],
    callback: (partial: string, done: boolean) => void,
): void => {
    if (!llmInstance) throw new Error('[GemmaWeb] LLM not initialised.');

    const promise = llmInstance.generateResponse(
        parts,
        (partial: string, done: any) => callback(partial, !!done),
    );

    activeGeneration = {
        cancel: () => {
            llmInstance?.cancelProcessing?.();
        },
    };

    // Swallow the promise here; errors surface via the callback's done=true path.
    promise.catch((err: any) => {
        if (!String(err).includes('cancel')) {
            console.error('[GemmaWeb] generateResponse error:', err);
        }
    }).finally(() => {
        activeGeneration = null;
    });
};

/** Cancels the in-flight generation. No-op if nothing is running. */
(window as any).cancelProcessing = (): void => {
    activeGeneration?.cancel();
    activeGeneration = null;
};

/**
 * Fully releases the LLM instance and all associated GPU/WebAssembly memory.
 * After this call, `initLLM` must be called before generating again.
 */
(window as any).unloadLLM = (): void => {
    // Stop any running generation first.
    activeGeneration?.cancel();
    activeGeneration = null;

    _releaseLlmInstance();
    console.log('[GemmaWeb] LLM unloaded.');
};

/** Internal helper: closes and nulls the llmInstance reference. */
function _releaseLlmInstance(): void {
    try { llmInstance?.close?.(); } catch { /* ignore */ }
    llmInstance = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Token Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns the number of tokens in [text] using the Gemma tokenizer.
 * Falls back to a word-count heuristic if the tokenizer is not ready.
 */
(window as any).countTokens = async (text: string): Promise<number> => {
    try {
        if (!tokenizer) {
            const { AutoTokenizer, env } = await import('@huggingface/transformers');
            env.allowLocalModels = false;
            env.useBrowserCache = true;
            tokenizer = await AutoTokenizer.from_pretrained('onnx-community/embeddinggemma-300m-ONNX');
        }
        const encoded = await tokenizer(text);
        return encoded.input_ids.data.length as number;
    } catch (err) {
        console.warn('[GemmaWeb] Tokenizer unavailable, using heuristic:', err);
        const words = text.split(/\s+/).length;
        return Math.floor(words * 1.3 + text.length * 0.15);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 5. PDF Extraction Engine (PDF.js)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialises the PDF.js Web Worker, pointing it at the worker script that is
 * bundled inside the Flutter asset package.
 *
 * Safe to call multiple times – subsequent calls are no-ops.
 */
(window as any).initPdfWorker = async (assetBase: string): Promise<boolean> => {
    if (pdfWorkerReady) return true;
    try {
        const pdfjsLib = await import('pdfjs-dist');
        pdfjsLib.GlobalWorkerOptions.workerSrc = `${assetBase}pdfjs-dist/pdf.worker.mjs`;
        pdfWorkerReady = true;
        console.log('[GemmaWeb] PDF.js worker ready.');
        return true;
    } catch (err) {
        console.error('[GemmaWeb] initPdfWorker failed:', err);
        throw err;
    }
};

/**
 * Extracts content from a PDF supplied as a raw byte array.
 *
 * @param pdfBytes    Raw PDF binary.
 * @param mode        Extraction mode: 'auto' | 'textOnly' | 'imagesOnly' | 'fullRender' | 'textAndImages'
 * @param filter      Page filter: 'all' | 'odd' | 'even' | 'range'
 * @param startPage   First page to process (1-based, inclusive). Used when filter='range'.
 * @param endPage     Last  page to process (1-based, inclusive). Used when filter='range'.
 * @param renderScale Bitmap scale multiplier for image rendering (default 2.0).
 * @returns           Array of { type: 'text' | 'image', data: string | Uint8Array }
 */
(window as any).extractPdf = async (
    pdfBytes: Uint8Array,
    mode: string,
    filter: string,
    startPage?: number,
    endPage?: number,
    renderScale: number = 2.0,
): Promise<Array<{ type: string; data: any }>> => {
    const pdfjsLib = await import('pdfjs-dist');

    const pdf = await pdfjsLib.getDocument({ data: pdfBytes }).promise;
    const numPages = pdf.numPages;
    const parts: Array<{ type: string; data: any }> = [];

    const needsText = mode === 'textOnly' || mode === 'textAndImages' || mode === 'auto';
    const needsImages = mode === 'imagesOnly' || mode === 'fullRender' || mode === 'textAndImages' || mode === 'auto';

    for (let pageNum = 1; pageNum <= numPages; pageNum++) {
        // ── Page filter ─────────────────────────────────────────────────────────
        if (filter === 'odd' && pageNum % 2 === 0) continue;
        if (filter === 'even' && pageNum % 2 !== 0) continue;
        if (filter === 'range') {
            if (startPage && pageNum < startPage) continue;
            if (endPage && pageNum > endPage) break;
        }

        const page = await pdf.getPage(pageNum);

        // ── Text extraction ──────────────────────────────────────────────────────
        let pageText = '';
        if (needsText) {
            const textContent = await page.getTextContent();
            const items = textContent.items as TextItem[];

            // Sort top-to-bottom, left-to-right using the PDF transform matrix.
            // transform[5] = Y position (PDF: Y increases upward, so descending = reading order).
            // transform[4] = X position.
            items.sort((a, b) => {
                const yDiff = b.transform[5] - a.transform[5];
                if (Math.abs(yDiff) > 2) return yDiff; // different lines (2-px tolerance)
                return a.transform[4] - b.transform[4]; // same line → left to right
            });

            let lastY = -1;
            const sb: string[] = [];
            for (const item of items) {
                if (lastY !== -1 && Math.abs(item.transform[5] - lastY) > 5) sb.push('\n');
                sb.push(item.str + ' ');
                lastY = item.transform[5];
            }
            pageText = sb.join('');
        }

        const doText = needsText && (mode !== 'auto' || pageText.trim().length > 0);
        const doRender = needsImages && (mode !== 'auto' || pageText.trim().length === 0);

        if (doText && pageText.trim().length > 0) {
            parts.push({ type: 'text', data: `--- Page ${pageNum} ---\n${pageText}\n` });
        }

        if (doRender) {
            const viewport = page.getViewport({ scale: renderScale });
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d')!;
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            await page.render({ canvasContext: ctx, viewport, canvas }).promise;

            const blob = await new Promise<Blob | null>(res => canvas.toBlob(res, 'image/png'));
            if (blob) {
                parts.push({ type: 'image', data: new Uint8Array(await blob.arrayBuffer()) });
            }

            // Release canvas memory on browsers that support it.
            canvas.width = 0;
            canvas.height = 0;
        }
    }

    return parts;
};

// ─────────────────────────────────────────────────────────────────────────────
// 7. OPFS Cache Management
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Lists all files in the Origin Private File System cache.
 * Returns [{ name: string, size: number }]. Returns [] if OPFS unavailable.
 */
(window as any).listOpfsCache = async (): Promise<Array<{ name: string; size: number }>> => {
    if (!navigator.storage?.getDirectory) return [];
    try {
        const root = await navigator.storage.getDirectory();
        const entries: Array<{ name: string; size: number }> = [];
        for await (const [name, handle] of (root as any).entries()) {
            if (handle.kind === 'file') {
                const file = await handle.getFile();
                entries.push({ name, size: file.size });
            }
        }
        return entries;
    } catch (err) {
        console.error('[GemmaWeb] listOpfsCache error:', err);
        return [];
    }
};

/**
 * Deletes files from the OPFS cache.
 *
 * @param fileNames  Array of file names to delete.
 *                   Pass an EMPTY array to delete every file in the OPFS root.
 *                   Embedding model files are prefixed with `embed_` (e.g. `embed_model.tflite`).
 *                   LLM files have no prefix.
 * @returns          Total bytes freed.
 */
(window as any).purgeOpfsCache = async (fileNames: string[]): Promise<number> => {
    if (!navigator.storage?.getDirectory) return 0;
    try {
        const root = await navigator.storage.getDirectory();
        let freed = 0;
        const deleteAll = fileNames.length === 0;

        for await (const [name, handle] of (root as any).entries()) {
            if (handle.kind !== 'file') continue;
            if (!deleteAll && !fileNames.includes(name)) continue;
            try {
                const file = await handle.getFile();
                freed += file.size;
                await root.removeEntry(name);
                console.log(`[GemmaWeb] Purged OPFS file: ${name} (${file.size} bytes)`);
            } catch (err) {
                console.warn(`[GemmaWeb] Could not delete ${name}:`, err);
            }
        }

        console.log(`[GemmaWeb] OPFS purge complete. Freed ${freed} bytes.`);
        return freed;
    } catch (err) {
        console.error('[GemmaWeb] purgeOpfsCache error:', err);
        return 0;
    }
};

/**
 * Purges the browser's Cache Storage entries used by the embedding model.
 *
 * Transformers.js caches tokenizer JSON files under URL-keyed entries inside
 * a bucket named `transformers-cache-<hash>`. This function:
 *
 *  1. Deletes any **whole bucket** whose name matches a known pattern
 *     (e.g. `transformers-cache`). This is the fast path that removes the
 *     bucket and all its files in one call.
 *
 *  2. Scans all remaining buckets and deletes individual entries whose
 *     request URL contains a known model pattern (e.g. `onnx-community`,
 *     `embeddinggemma`). After removing entries it also deletes the bucket
 *     if it is now empty.
 *
 * @param patterns  URL / bucket-name substrings to match.
 *                  e.g. `["embeddinggemma", "onnx-community", "litert-community"]`
 *                  Pass an empty array to purge ALL Cache Storage buckets.
 * @returns         Total bytes freed (best-effort; some browsers don't expose size).
 */
(window as any).purgeEmbeddingCache = async (patterns: string[]): Promise<number> => {
    if (!('caches' in window)) return 0;

    /** Known bucket-name prefixes written by transformers.js / ONNX runtime. */
    const BUCKET_PATTERNS = ['transformers-cache', 'onnx-cache', 'litert-cache'];

    let freed = 0;
    const matchAll = patterns.length === 0;

    try {
        const cacheNames = await caches.keys();

        for (const name of cacheNames) {
            // ── Pass 1: delete entire bucket if its name matches a known pattern ──
            const bucketHit =
                matchAll ||
                BUCKET_PATTERNS.some(bp => name.includes(bp)) ||
                patterns.some(p => name.includes(p));

            if (bucketHit) {
                // Tally the size of every entry before deleting the bucket.
                try {
                    const cache = await caches.open(name);
                    for (const req of await cache.keys()) {
                        try {
                            const resp = await cache.match(req);
                            if (resp) {
                                const len = parseInt(resp.headers.get('content-length') ?? '0', 10);
                                freed += isNaN(len) ? 0 : len;
                            }
                        } catch (_) { /* size unknown */ }
                    }
                } catch (_) { /* can't open — still delete */ }

                const deleted = await caches.delete(name);
                if (deleted) {
                    console.log(`[GemmaWeb] Deleted Cache Storage bucket: "${name}"`);
                }
                continue; // bucket gone — no need for entry-level pass
            }

            // ── Pass 2: scan remaining buckets for matching entry URLs ────────────
            try {
                const cache = await caches.open(name);
                const reqs = await cache.keys();
                let deleted = 0;

                for (const req of reqs) {
                    const hit = matchAll || patterns.some(p => req.url.includes(p));
                    if (!hit) continue;

                    try {
                        const resp = await cache.match(req);
                        if (resp) {
                            const len = parseInt(resp.headers.get('content-length') ?? '0', 10);
                            freed += isNaN(len) ? 0 : len;
                        }
                    } catch (_) { /* size unknown, still delete */ }

                    await cache.delete(req);
                    deleted++;
                }

                if (deleted > 0) {
                    console.log(`[GemmaWeb] Purged ${deleted} entries from bucket "${name}"`);

                    // If the bucket is now empty, delete it too.
                    const remaining = await cache.keys();
                    if (remaining.length === 0) {
                        await caches.delete(name);
                        console.log(`[GemmaWeb] Deleted now-empty bucket: "${name}"`);
                    }
                }
            } catch (err) {
                console.warn(`[GemmaWeb] Could not scan bucket "${name}":`, err);
            }
        }
    } catch (err) {
        console.error('[GemmaWeb] purgeEmbeddingCache error:', err);
    }

    console.log(`[GemmaWeb] Cache Storage purge freed ~${freed} bytes.`);
    return freed;
};