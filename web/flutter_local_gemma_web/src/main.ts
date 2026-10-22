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
type ActiveGeneration = { cancel: () => void };
let activeGeneration: ActiveGeneration | null = null;

// Embeddings (LiteRT)
let tfliteModel: any = null;
let tokenizer: any = null;
let currentEmbedModelUrl: string | null = null;
let isLiteRtRuntimeLoaded = false; // WASM binary; survive model unload

// EmbeddingGemma 2 (Transformers.js + WebGPU)
let gemma2Model: any = null;
let gemma2Processor: any = null;
let gemma2ModelId: string | null = null;
let gemma2TargetDim: number | null = null;

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

/** Normalizes a float vector to unit length (L2 norm). */
function _l2Normalize(vec: Float32Array): Float32Array {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
        sum += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sum);
    if (norm > 0) {
        for (let i = 0; i < vec.length; i++) {
            vec[i] /= norm;
        }
    }
    return vec;
}

/** Optionally truncates vector to targetDimension and re-normalizes. */
function _applyMatryoshka(vec: Float32Array, targetDim?: number | null): Float32Array {
    if (targetDim && targetDim < vec.length) {
        const truncated = vec.slice(0, targetDim);
        return _l2Normalize(truncated);
    }
    return _l2Normalize(vec);
}

/**
 * Initialises the EmbeddingGemma 2 model via Transformers.js with WebGPU acceleration.
 * Falls back gracefully to WASM if WebGPU is not supported by the client browser.
 */
(window as any).initEmbeddingGemma2 = async (
    modelId?: string,
    token?: string,
    targetDimension?: number,
    onProgress?: (progress: number) => void
): Promise<boolean> => {
    try {
        const id = modelId || 'onnx-community/embeddinggemma-2-ONNX';
        if (gemma2Model && gemma2Processor && gemma2ModelId === id) {
            gemma2TargetDim = targetDimension ?? null;
            onProgress?.(100);
            return true;
        }

        const { env, AutoProcessor, AutoModel } = await import('@huggingface/transformers');
        env.allowLocalModels = false;
        env.useBrowserCache = true;

        console.log(`[GemmaWeb] Loading EmbeddingGemma 2: ${id}`);
        let lastReportedProgress = 0;
        const fileProgressMap = new Map<string, { loaded: number; total: number }>();

        const handleProgress = (e: any) => {
            if (!onProgress) return;
            if (e.status === 'progress_total' && typeof e.progress === 'number') {
                if (e.progress > lastReportedProgress && e.progress < 99.5) {
                    lastReportedProgress = e.progress;
                    onProgress(e.progress);
                }
                return;
            }
            if (e.status === 'progress' && e.file && e.total) {
                fileProgressMap.set(e.file, { loaded: e.loaded, total: e.total });
                let sumLoaded = 0;
                let sumTotal = 0;
                for (const f of fileProgressMap.values()) {
                    sumLoaded += f.loaded;
                    sumTotal += f.total;
                }
                if (sumTotal > 0) {
                    const totalPct = (sumLoaded / sumTotal) * 100;
                    if (totalPct > lastReportedProgress && totalPct < 99.5) {
                        lastReportedProgress = totalPct;
                        onProgress(totalPct);
                    }
                }
            }
        };

        const modelOptions: any = {
            dtype: 'q4',
            device: 'webgpu',
            progress_callback: handleProgress,
        };
        if (token) {
            modelOptions.token = token;
        }

        const processorOptions: any = { progress_callback: handleProgress };
        if (token) {
            processorOptions.token = token;
        }

        const [processor, model] = await Promise.all([
            AutoProcessor.from_pretrained(id, processorOptions),
            AutoModel.from_pretrained(id, modelOptions).catch(async (webgpuErr: any) => {
                console.warn('[GemmaWeb] WebGPU load failed for EmbeddingGemma 2, falling back to WASM:', webgpuErr);
                const wasmOptions = { ...modelOptions, device: 'wasm' };
                return AutoModel.from_pretrained(id, wasmOptions);
            }),
        ]);

        gemma2Model = model;
        gemma2Processor = processor;
        gemma2ModelId = id;
        gemma2TargetDim = targetDimension ?? null;
        onProgress?.(100);
        console.log('[GemmaWeb] EmbeddingGemma 2 initialized successfully.');
        return true;
    } catch (err) {
        console.error('[GemmaWeb] initEmbeddingGemma2 failed:', err);
        throw err;
    }
};

/**
 * Computes a semantic embedding vector for [text].
 * Routes to EmbeddingGemma 2 if initialized, or LiteRT as fallback.
 */
(window as any).getEmbedding = async (text: string): Promise<Float32Array> => {
    if (gemma2Model && gemma2Processor) {
        const inputs = await gemma2Processor(text.trim());
        const outputs = await gemma2Model(inputs);
        const sentence_embedding = outputs?.sentence_embedding ?? Object.values<any>(outputs)[0];
        try {
            const raw = sentence_embedding.data instanceof Float32Array
                ? sentence_embedding.data
                : Float32Array.from(sentence_embedding.tolist()[0]);
            return _applyMatryoshka(raw, gemma2TargetDim);
        } finally {
            sentence_embedding?.dispose?.();
            Object.values<any>(inputs).forEach((t: any) => t?.dispose?.());
        }
    }

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
 * Computes an embedding vector for one or more encoded images using EmbeddingGemma 2.
 */
(window as any).getEmbeddingFromImages = async (images: any): Promise<Float32Array> => {
    if (!gemma2Model || !gemma2Processor) {
        throw new Error('[GemmaWeb] EmbeddingGemma 2 not initialized.');
    }
    const { RawImage } = await import('@huggingface/transformers');

    let imageList: any[] = [];
    if (Array.isArray(images)) {
        imageList = images;
    } else if (images instanceof Uint8Array) {
        imageList = [images];
    } else if (images && typeof images[Symbol.iterator] === 'function') {
        imageList = Array.from(images);
    } else if (images && typeof images.length === 'number') {
        imageList = Array.from(images);
    } else if (images && typeof images === 'object') {
        imageList = Object.values(images);
    } else {
        imageList = [images];
    }

    const rawImages = await Promise.all(
        imageList.map(async (bytes) => {
            const blob = new Blob([bytes as any]);
            return await RawImage.fromBlob(blob);
        })
    );

    // In EmbeddingGemma2Processor, positional signature is (text = null, images = rawImages)
    const inputs = await gemma2Processor(null, rawImages);

    const outputs = await gemma2Model(inputs);
    const sentence_embedding = outputs?.sentence_embedding ?? Object.values<any>(outputs)[0];
    try {
        const raw = sentence_embedding.data instanceof Float32Array
            ? sentence_embedding.data
            : Float32Array.from(sentence_embedding.tolist()[0]);
        return _applyMatryoshka(raw, gemma2TargetDim);
    } finally {
        sentence_embedding?.dispose?.();
        Object.values<any>(inputs).forEach((t: any) => t?.dispose?.());
    }
};

/**
 * Synchronizes placeholder tokens (<|image|> or <|audio|>) in prompt text to match targetCount.
 */
function _syncPlaceholders(text: string, token: string, targetCount: number): string {
    const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(escaped, 'g');
    const matches = text.match(regex);
    let count = matches ? matches.length : 0;

    if (count < targetCount) {
        const needed = targetCount - count;
        text = text + (text.length > 0 ? ' ' : '') + Array(needed).fill(token).join(' ');
    } else if (count > targetCount) {
        let excess = count - targetCount;
        let lastIdx = text.length;
        while (excess > 0 && lastIdx > 0) {
            const idx = text.lastIndexOf(token, lastIdx);
            if (idx === -1) break;
            text = text.substring(0, idx) + text.substring(idx + token.length);
            lastIdx = idx;
            excess--;
        }
    }
    return text.trim();
}

/**
 * Computes an embedding vector for interleaved multimodal parts (text, image, audio).
 */
(window as any).getEmbeddingFromParts = async (parts: any): Promise<Float32Array> => {
    if (!gemma2Model || !gemma2Processor) {
        throw new Error('[GemmaWeb] EmbeddingGemma 2 not initialized.');
    }
    const { RawImage } = await import('@huggingface/transformers');

    let partList: any[] = [];
    if (Array.isArray(parts)) {
        partList = parts;
    } else if (parts && typeof parts[Symbol.iterator] === 'function') {
        partList = Array.from(parts);
    } else if (parts && typeof parts.length === 'number') {
        partList = Array.from(parts);
    } else if (parts && typeof parts === 'object') {
        partList = Object.values(parts);
    } else {
        partList = [parts];
    }

    const textSegments: string[] = [];
    const imageParts: any[] = [];
    const audioSegments: Float32Array[] = [];
    let hasExplicitText = false;

    for (const part of partList) {
        if (part.type === 'text' && part.text) {
            textSegments.push(part.text);
            hasExplicitText = true;
        } else if (part.type === 'image' && part.bytes) {
            const blob = new Blob([part.bytes as any]);
            imageParts.push(await RawImage.fromBlob(blob));
            textSegments.push('<|image|>');
        } else if (part.type === 'audio' && part.bytes) {
            const int16 = new Int16Array(part.bytes.buffer, part.bytes.byteOffset, part.bytes.byteLength / 2);
            const slice = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                slice[i] = int16[i] / 32768.0;
            }
            audioSegments.push(slice);
            textSegments.push('<|audio|>');
        }
    }

    let promptText: string | null = textSegments.join(' ').trim();
    if (imageParts.length > 0) {
        promptText = _syncPlaceholders(promptText, '<|image|>', imageParts.length);
    }
    if (audioSegments.length > 0) {
        promptText = _syncPlaceholders(promptText, '<|audio|>', audioSegments.length);
    }

    // If there is no explicit text and no audio, pass null so processor builds clean image inputs
    if (!hasExplicitText && audioSegments.length === 0 && imageParts.length > 0) {
        promptText = null;
    }

    const audioParam = audioSegments.length === 0
        ? null
        : (audioSegments.length === 1 ? audioSegments[0] : [audioSegments]);

    const imageParam = imageParts.length > 0 ? imageParts : null;

    let inputs: any;
    if (imageParam && audioParam) {
        inputs = await gemma2Processor(promptText, imageParam, audioParam);
    } else if (imageParam) {
        inputs = await gemma2Processor(promptText, imageParam);
    } else if (audioParam) {
        inputs = await gemma2Processor(promptText, null, audioParam);
    } else if (promptText) {
        inputs = await gemma2Processor(promptText);
    } else {
        throw new Error('[GemmaWeb] No valid modal parts provided.');
    }

    const outputs = await gemma2Model(inputs);
    const sentence_embedding = outputs?.sentence_embedding ?? Object.values<any>(outputs)[0];
    try {
        const raw = sentence_embedding.data instanceof Float32Array
            ? sentence_embedding.data
            : Float32Array.from(sentence_embedding.tolist()[0]);
        return _applyMatryoshka(raw, gemma2TargetDim);
    } finally {
        sentence_embedding?.dispose?.();
        Object.values<any>(inputs).forEach((t: any) => t?.dispose?.());
    }
};

/**
 * Returns whether audio embedding is supported.
 */
(window as any).supportsAudio = async (): Promise<boolean> => {
    return true;
};

/**
 * Video frame and metadata extraction helpers for Web.
 */
let _extractVideoEl: HTMLVideoElement | null = null;
let _extractVideoSrc: string | null = null;

async function _getOrLoadVideo(url: string): Promise<HTMLVideoElement> {
    const cleanUrl = url.split('#')[0];
    if (_extractVideoEl && _extractVideoSrc === cleanUrl && _extractVideoEl.readyState >= 1) {
        return _extractVideoEl;
    }
    if (_extractVideoEl) {
        try {
            _extractVideoEl.pause();
            _extractVideoEl.removeAttribute('src');
            _extractVideoEl.load();
        } catch (_) {}
    }
    const video = document.createElement('video');
    if (!cleanUrl.startsWith('blob:') && !cleanUrl.startsWith('data:')) {
        video.crossOrigin = 'anonymous';
    }
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';

    await new Promise<void>((resolve) => {
        let done = false;
        const timer = setTimeout(() => {
            if (!done) {
                done = true;
                cleanup();
                resolve();
            }
        }, 8000);

        const onLoaded = () => {
            if (!done && video.readyState >= 1) {
                done = true;
                clearTimeout(timer);
                cleanup();
                resolve();
            }
        };
        const onError = (e: any) => {
            if (!done) {
                done = true;
                clearTimeout(timer);
                cleanup();
                console.warn('[GemmaWeb] Video load notice:', cleanUrl, e);
                resolve();
            }
        };
        const cleanup = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            video.removeEventListener('loadeddata', onLoaded);
            video.removeEventListener('canplay', onLoaded);
            video.removeEventListener('error', onError);
        };

        video.addEventListener('loadedmetadata', onLoaded);
        video.addEventListener('loadeddata', onLoaded);
        video.addEventListener('canplay', onLoaded);
        video.addEventListener('error', onError);
        video.src = cleanUrl;
        video.load();

        if (video.readyState >= 1) {
            done = true;
            clearTimeout(timer);
            cleanup();
            resolve();
        }
    });

    _extractVideoEl = video;
    _extractVideoSrc = cleanUrl;
    return video;
}

(window as any).extractVideoMetadata = async (
    videoUrl: string
): Promise<{ durationMs: number; width: number; height: number } | null> => {
    try {
        const video = await _getOrLoadVideo(videoUrl);
        if (video.readyState < 1) {
            await new Promise<void>((resolve) => {
                const onMeta = () => {
                    video.removeEventListener('loadedmetadata', onMeta);
                    resolve();
                };
                video.addEventListener('loadedmetadata', onMeta);
                setTimeout(resolve, 3000);
            });
        }
        const durationSec = isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
        return {
            durationMs: Math.round(durationSec * 1000),
            width: video.videoWidth || 640,
            height: video.videoHeight || 360,
        };
    } catch (err) {
        console.warn('[GemmaWeb] extractVideoMetadata error:', err);
        return null;
    }
};

(window as any).extractVideoFrame = async (
    videoUrl: string,
    timeMs: number
): Promise<Uint8Array | null> => {
    try {
        const video = await _getOrLoadVideo(videoUrl);
        if (video.readyState < 1) {
            await new Promise<void>((resolve) => {
                const onMeta = () => {
                    video.removeEventListener('loadedmetadata', onMeta);
                    resolve();
                };
                video.addEventListener('loadedmetadata', onMeta);
                setTimeout(resolve, 3000);
            });
        }
        const maxSec = isFinite(video.duration) && video.duration > 0 ? video.duration : (timeMs / 1000);
        const targetSec = Math.max(0, Math.min(timeMs / 1000, maxSec));

        if (Math.abs(video.currentTime - targetSec) > 0.05) {
            await new Promise<void>((resolve) => {
                let resolved = false;
                const timer = setTimeout(() => {
                    if (!resolved) {
                        resolved = true;
                        video.removeEventListener('seeked', onSeeked);
                        resolve();
                    }
                }, 3000);

                const onSeeked = () => {
                    if (!resolved) {
                        resolved = true;
                        clearTimeout(timer);
                        video.removeEventListener('seeked', onSeeked);
                        resolve();
                    }
                };
                video.addEventListener('seeked', onSeeked);
                video.currentTime = targetSec;
            });
        }

        const width = video.videoWidth > 0 ? video.videoWidth : 640;
        const height = video.videoHeight > 0 ? video.videoHeight : 360;
        const targetWidth = Math.min(640, width);
        const targetHeight = Math.max(1, Math.round(targetWidth * (height / width)));

        const canvas = document.createElement('canvas');
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        const commaIdx = dataUrl.indexOf(',');
        if (commaIdx === -1) return null;
        const base64 = dataUrl.substring(commaIdx + 1);
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    } catch (err) {
        console.warn('[GemmaWeb] extractVideoFrame error at', timeMs, 'ms:', err);
        return null;
    }
};

/**
 * Releases all embedding models (both LiteRT and EmbeddingGemma 2) and tokenizer from memory.
 */
(window as any).unloadEmbeddingModel = (): void => {
    _releaseCompiledModel();
    tokenizer = null;
    try { gemma2Model?.dispose?.(); } catch (_) {}
    gemma2Model = null;
    gemma2Processor = null;
    gemma2ModelId = null;
    gemma2TargetDim = null;
    console.log('[GemmaWeb] Embedding models unloaded.');
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
// ─────────────────────────────────────────────────────────────────────────────
// 3. Hybrid Web LLM Engine (Gemma 4 ONNX, LiteRT-LM & MediaPipe Tasks GenAI)
// ─────────────────────────────────────────────────────────────────────────────

let engine: any = null;
let conversation: any = null;
let llmInstance: any = null;

// Gemma 4 ONNX (Transformers.js + WebGPU)
let onnxModel: any = null;
let onnxProcessor: any = null;
let onnxModelId: string | null = null;
let onnxStoppingCriteria: any = null;

/**
 * Loads and initialises the appropriate LLM engine based on model type.
 *
 * @param options.assetBase          Base URL for bundled WASM assets.
 * @param options.baseOptions.modelPath  URL / OPFS object-URL of the model binary.
 * @param options.maxTokens          KV-cache capacity.
 * @param options.supportAudio       Passed to MediaPipe if supported.
 * @param options.isGemma4           If true, forces LiteRT-LM engine.
 */
(window as any).initLLM = async (options: any): Promise<boolean> => {
    try {
        // Unload any previously loaded instance before creating a new one.
        await _releaseLlmInstance();

        const modelPath = options?.baseOptions?.modelPath ?? options?.modelPath;
        const modelPathStr = String(modelPath || '').toLowerCase();
        const isOnnx = options?.isOnnx === true ||
                       options?.format === 'onnx' ||
                       modelPathStr.endsWith('.onnx') ||
                       modelPathStr.includes('.onnx') ||
                       modelPathStr.includes('onnx-community') ||
                       (modelPathStr.includes('onnx') && !modelPathStr.includes('litertlm') && !modelPathStr.includes('.task'));

        if (isOnnx) {
            const isE4B = modelPathStr.includes('e4b') || modelPathStr.includes('4b');
            let modelId = isE4B ? 'onnx-community/gemma-4-E4B-it-ONNX' : 'onnx-community/gemma-4-E2B-it-ONNX';
            if (modelPath && modelPath.includes('onnx-community/') && !modelPath.includes('http') && !modelPath.includes('blob:')) {
                modelId = modelPath;
            }

            if (onnxModel && onnxProcessor && onnxModelId === modelId) {
                options?.onProgress?.(100);
                return true;
            }

            console.log(`[GemmaWeb] Loading Gemma 4 ONNX Engine (${modelId})...`);
            const { AutoProcessor, Gemma4ForConditionalGeneration, env } = await import('@huggingface/transformers');
            env.allowLocalModels = false;
            env.useBrowserCache = true;

            let lastReportedProgress = 0;
            const fileProgressMap = new Map<string, { loaded: number; total: number }>();
            const handleProgress = (e: any) => {
                if (options?.onProgress && typeof options.onProgress === 'function') {
                    if (e.status === 'progress_total' && typeof e.progress === 'number') {
                        if (e.progress > lastReportedProgress && e.progress < 99.5) {
                            lastReportedProgress = e.progress;
                            options.onProgress(e.progress);
                        }
                    } else if (e.status === 'progress' && e.file && e.total) {
                        fileProgressMap.set(e.file, { loaded: e.loaded, total: e.total });
                        let sumLoaded = 0;
                        let sumTotal = 0;
                        for (const f of fileProgressMap.values()) {
                            sumLoaded += f.loaded;
                            sumTotal += f.total;
                        }
                        if (sumTotal > 0) {
                            const totalPct = (sumLoaded / sumTotal) * 100;
                            if (totalPct > lastReportedProgress && totalPct < 99.5) {
                                lastReportedProgress = totalPct;
                                options.onProgress(totalPct);
                            }
                        }
                    }
                }
            };

            const modelOptions: any = {
                dtype: 'q4f16',
                device: 'webgpu',
                progress_callback: handleProgress,
            };
            if (options?.token) {
                modelOptions.token = options.token;
            }

            const processorOptions: any = { progress_callback: handleProgress };
            if (options?.token) {
                processorOptions.token = options.token;
            }

            const [processor, model] = await Promise.all([
                AutoProcessor.from_pretrained(modelId, processorOptions),
                Gemma4ForConditionalGeneration.from_pretrained(modelId, modelOptions).catch(async (webgpuErr: any) => {
                    console.warn('[GemmaWeb] WebGPU load failed for Gemma 4 ONNX, falling back to WASM / q4:', webgpuErr);
                    const wasmOptions = { ...modelOptions, dtype: 'q4', device: 'wasm' };
                    return Gemma4ForConditionalGeneration.from_pretrained(modelId, wasmOptions);
                }),
            ]);

            onnxModel = model;
            onnxProcessor = processor;
            onnxModelId = modelId;
            options?.onProgress?.(100);
            console.log(`[GemmaWeb] Gemma 4 ONNX Engine ready (${modelId}).`);
            return true;
        }

        const isTaskModel = modelPath?.endsWith('.task') || modelPath?.endsWith('.bin');
        // Force MediaPipe if the file is a .task model, even for Gemma 4
        const isGemma4 = options?.isGemma4 === true && !isTaskModel;

        if (isGemma4) {
            console.log('[GemmaWeb] Loading LiteRT-LM Engine (Gemma 4 path)...');
            const { Engine } = await import('@litert-lm/core');
            engine = await Engine.create({
                model: modelPath,
                mainExecutorSettings: {
                    maxNumTokens: options.maxTokens ?? 4096,
                }
            });
            conversation = await engine.createConversation();
        } else {
            console.log('[GemmaWeb] Loading MediaPipe Tasks GenAI (Gemma 3 path)...');
            const { FilesetResolver, LlmInference } = await import('@mediapipe/tasks-genai');
            const assetBase = options.assetBase;
            if (!assetBase) throw new Error('[GemmaWeb] initLLM: assetBase is missing.');
            const wasmPath = `${assetBase}@mediapipe/tasks-genai/wasm`;
            const resolver = await FilesetResolver.forGenAiTasks(wasmPath);
            llmInstance = await LlmInference.createFromOptions(resolver, {
                baseOptions: { modelAssetPath: modelPath },
                maxTokens: options.maxTokens ?? 4096,
                maxNumImages: (options.maxNumImages && options.maxNumImages > 0) ? options.maxNumImages : undefined,
                supportAudio: !!options.supportAudio,
                randomSeed: options.randomSeed ?? 101,
            });
        }

        console.log('[GemmaWeb] LLM engine ready.');
        return true;
    } catch (err) {
        console.error('[GemmaWeb] initLLM failed:', err);
        throw err;
    }
};

/**
 * Starts an async generation and streams tokens via [callback].
 */
(window as any).generateResponse = (
    parts: any[],
    callback: (partial: string, done: boolean) => void,
): void => {
    if (onnxModel && onnxProcessor) {
        // Gemma 4 ONNX path (Transformers.js + WebGPU)
        let isCancelled = false;
        activeGeneration = {
            cancel: () => {
                isCancelled = true;
                onnxStoppingCriteria?.interrupt?.();
            },
        };

        (async () => {
            try {
                const { RawImage, read_audio, TextStreamer, InterruptableStoppingCriteria } = await import('@huggingface/transformers');
                const stoppingCriteria = new InterruptableStoppingCriteria();
                onnxStoppingCriteria = stoppingCriteria;

                const textSegments: string[] = [];
                const images: any[] = [];
                const audios: any[] = [];

                for (const p of parts) {
                    if (typeof p === 'string') {
                        textSegments.push(p);
                    } else if (p && p.imageSource) {
                        const src = p.imageSource;
                        if (typeof src === 'string' && src.startsWith('data:')) {
                            const res = await fetch(src);
                            const blob = await res.blob();
                            images.push(await RawImage.fromBlob(blob));
                        } else {
                            images.push(await RawImage.read(src));
                        }
                    } else if (p && p.audioSource) {
                        audios.push(await read_audio(p.audioSource, 16000));
                    } else if (p && p.bytes) {
                        if (p.type === 'image') {
                            images.push(await RawImage.fromBlob(new Blob([p.bytes])));
                        } else if (p.type === 'audio') {
                            const blob = new Blob([p.bytes], { type: 'audio/wav' });
                            const url = URL.createObjectURL(blob);
                            try {
                                audios.push(await read_audio(url, 16000));
                            } finally {
                                URL.revokeObjectURL(url);
                            }
                        }
                    } else if (p && p.text) {
                        textSegments.push(p.text);
                    } else if (p != null) {
                        textSegments.push(String(p));
                    }
                }

                const fullText = textSegments.join(' ').trim();
                let prompt = '';

                if (fullText.includes('<start_of_turn>') || fullText.includes('<|turn>')) {
                    prompt = fullText
                        .replaceAll('<start_of_turn>', '<|turn>')
                        .replaceAll('<end_of_turn>', '<turn|>');
                    if (!prompt.startsWith('<bos>')) {
                        prompt = '<bos>' + prompt;
                    }
                    if (images.length > 0 && !prompt.includes('<|image|>')) {
                        const imageTokens = Array(images.length).fill('<|image|>').join('');
                        const lastUserIdx = prompt.lastIndexOf('<|turn>user\n');
                        if (lastUserIdx !== -1) {
                            prompt = prompt.substring(0, lastUserIdx + 12) + imageTokens + prompt.substring(lastUserIdx + 12);
                        } else {
                            prompt = prompt.replace('<|turn>user\n', `<|turn>user\n${imageTokens}`);
                        }
                    }
                    if (audios.length > 0 && !prompt.includes('<|audio|>')) {
                        const audioTokens = Array(audios.length).fill('<|audio|>').join('');
                        const lastUserIdx = prompt.lastIndexOf('<|turn>user\n');
                        if (lastUserIdx !== -1) {
                            prompt = prompt.substring(0, lastUserIdx + 12) + audioTokens + prompt.substring(lastUserIdx + 12);
                        } else {
                            prompt = prompt.replace('<|turn>user\n', `<|turn>user\n${audioTokens}`);
                        }
                    }
                } else {
                    const content: any[] = [];
                    for (let i = 0; i < images.length; i++) content.push({ type: 'image' });
                    for (let i = 0; i < audios.length; i++) content.push({ type: 'audio' });
                    if (fullText) content.push({ type: 'text', text: fullText });

                    const messages = [{ role: 'user', content }];
                    prompt = onnxProcessor.apply_chat_template(messages, {
                        enable_thinking: fullText.includes('<|think|>'),
                        add_generation_prompt: true,
                    });
                }

                // Ensure multimodal tokens match count exactly
                if (images.length > 0) {
                    const count = (prompt.match(/<\|image\|>/g) || []).length;
                    if (count < images.length) {
                        const needed = images.length - count;
                        const imageTokens = Array(needed).fill('<|image|>').join('');
                        const lastUserIdx = prompt.lastIndexOf('<|turn>user\n');
                        if (lastUserIdx !== -1) {
                            prompt = prompt.substring(0, lastUserIdx + 12) + imageTokens + prompt.substring(lastUserIdx + 12);
                        } else {
                            prompt = imageTokens + prompt;
                        }
                    }
                }
                if (audios.length > 0) {
                    const count = (prompt.match(/<\|audio\|>/g) || []).length;
                    if (count < audios.length) {
                        const needed = audios.length - count;
                        const audioTokens = Array(needed).fill('<|audio|>').join('');
                        const lastUserIdx = prompt.lastIndexOf('<|turn>user\n');
                        if (lastUserIdx !== -1) {
                            prompt = prompt.substring(0, lastUserIdx + 12) + audioTokens + prompt.substring(lastUserIdx + 12);
                        } else {
                            prompt = audioTokens + prompt;
                        }
                    }
                }

                const imageParam = images.length === 0 ? null : (images.length === 1 ? images[0] : images);
                const audioParam = audios.length === 0 ? null : (audios.length === 1 ? audios[0] : audios);

                const inputs = await onnxProcessor(prompt, imageParam, audioParam, {
                    add_special_tokens: false,
                });

                if (isCancelled) return;

                const streamer = new TextStreamer(onnxProcessor.tokenizer, {
                    skip_prompt: true,
                    skip_special_tokens: false,
                    callback_function: (chunk: string) => {
                        if (!isCancelled) {
                            const cleanChunk = chunk.replaceAll('<turn|>', '').replaceAll('<end_of_turn>', '');
                            if (cleanChunk.length > 0) {
                                callback(cleanChunk, false);
                            }
                        }
                    },
                });

                await onnxModel.generate({
                    ...inputs,
                    max_new_tokens: 1024,
                    do_sample: false,
                    streamer,
                    stopping_criteria: stoppingCriteria,
                });

            } catch (err: any) {
                if (!String(err).includes('cancel') && !isCancelled) {
                    console.error('[GemmaWeb] generateResponse ONNX error:', err);
                }
            } finally {
                activeGeneration = null;
                onnxStoppingCriteria = null;
                if (!isCancelled) {
                    callback('', true);
                }
            }
        })();
        return;
    }

    if (engine && conversation) {
        // LiteRT-LM path
        const contentItems = parts.map(p => {
            if (typeof p === 'string') {
                return { type: 'text', text: p };
            } else if (p && p.imageSource) {
                // Pass base64 data URL as 'path' or 'text' depending on what LiteRT expects.
                // We'll pass it as 'text' since it's a data URL, but also 'path' just in case.
                return { type: 'image', text: p.imageSource, path: p.imageSource };
            } else if (p && p.audioSource) {
                return { type: 'audio', text: p.audioSource, path: p.audioSource };
            }
            return { type: 'text', text: String(p) };
        });

        let isCancelled = false;
        activeGeneration = {
            cancel: () => {
                isCancelled = true;
                conversation?.cancel?.();
            },
        };

        const message = { role: 'user', content: contentItems };
        const stream = conversation.sendMessageStreaming(message);

        (async () => {
            try {
                for await (const chunk of stream) {
                    if (isCancelled) break;
                    if (typeof chunk.content === 'string') {
                        callback(chunk.content, false);
                    } else if (Array.isArray(chunk.content)) {
                        for (const item of chunk.content) {
                            if (item.type === 'text' && item.text) {
                                callback(item.text, false);
                            }
                        }
                    }
                }
            } catch (err: any) {
                if (!String(err).includes('cancel') && !isCancelled) {
                    console.error('[GemmaWeb] generateResponse error:', err);
                }
            } finally {
                activeGeneration = null;
                if (!isCancelled) {
                    callback('', true);
                }
            }
        })();
    } else if (llmInstance) {
        // MediaPipe path
        const promise = llmInstance.generateResponse(
            parts,
            (partial: string, done: any) => callback(partial, !!done),
        );

        activeGeneration = {
            cancel: () => {
                llmInstance?.cancelProcessing?.();
            },
        };

        promise.catch((err: any) => {
            if (!String(err).includes('cancel')) {
                console.error('[GemmaWeb] generateResponse error:', err);
            }
            callback('', true);
        }).finally(() => {
            activeGeneration = null;
        });
    } else {
        throw new Error('[GemmaWeb] LLM not initialised.');
    }
};

/** Cancels the in-flight generation. No-op if nothing is running. */
(window as any).cancelProcessing = (): void => {
    activeGeneration?.cancel();
    activeGeneration = null;
};

/**
 * Fully releases the active LLM instance and all associated WebGPU/WASM memory.
 */
(window as any).unloadLLM = async (): Promise<void> => {
    // Stop any running generation first.
    activeGeneration?.cancel();
    activeGeneration = null;

    await _releaseLlmInstance();
    // Allow browser WebGPU context & garbage collection to cleanly reclaim pipeline resources
    await new Promise((resolve) => setTimeout(resolve, 150));
    console.log('[GemmaWeb] LLM unloaded.');
};

/** Internal helper: deletes active engine resources. */
async function _releaseLlmInstance(): Promise<void> {
    try {
        conversation = null;
        if (engine) {
            await engine.delete();
        }
        if (llmInstance) {
            llmInstance.close();
        }
        if (onnxModel) {
            await onnxModel.dispose?.();
        }
    } catch { /* ignore */ }
    engine = null;
    llmInstance = null;
    onnxModel = null;
    onnxProcessor = null;
    onnxModelId = null;
    onnxStoppingCriteria = null;
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
        if (onnxProcessor?.tokenizer) {
            const encoded = await onnxProcessor.tokenizer(text);
            return (encoded.input_ids?.data?.length ?? encoded.input_ids?.length) as number;
        }
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

// ─────────────────────────────────────────────────────────────────────────────
// 8. JS Skill Executor (Iframe-based)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Executes a JS skill's HTML code inside a sandboxed iframe.
 * Exposes a standard callback mechanism matching the Android implementation.
 *
 * @param scriptHtml  The full HTML code of the JS skill (typically load from SKILL.md/index.html).
 * @param argsJson    The JSON-serialized arguments to pass to the skill function.
 * @param timeoutMs   Timeout in milliseconds (defaults to 30000).
 */
(window as any).executeJsSkill = async (
    scriptHtml: string,
    argsJson: string,
    secret: string,
    timeoutMs: number = 30000
): Promise<string> => {
    return new Promise<string>((resolve) => {
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        const timer = setTimeout(() => {
            cleanup();
            resolve(JSON.stringify({ error: "Timeout executing skill", result: null }));
        }, timeoutMs);

        const cleanup = () => {
            clearTimeout(timer);
            if (iframe.parentNode) {
                iframe.parentNode.removeChild(iframe);
            }
        };

        // We can expose a callback function inside the iframe
        (iframe.contentWindow as any).AiEdgeGallery = {
            onResultReady: (result: string) => {
                cleanup();
                resolve(result);
            }
        };

        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc) {
            cleanup();
            resolve(JSON.stringify({ error: "Could not access iframe document", result: null }));
            return;
        }

        doc.open();
        doc.write(scriptHtml);
        doc.close();

        // Evaluate script in the context of the iframe window
        const win = iframe.contentWindow as any;

        // Wait for page to finish loading or for ai_edge_gallery_get_result to be defined
        const checkAndExecute = async () => {
            const startTs = Date.now();
            while (true) {
                if (typeof win.ai_edge_gallery_get_result === 'function') {
                    break;
                }
                await new Promise(r => setTimeout(r, 100));
                if (Date.now() - startTs > 10000) {
                    break;
                }
            }

            try {
                if (typeof win.ai_edge_gallery_get_result === 'function') {
                    // Standard Edge Gallery protocol expects a JSON string
                    const result = await win.ai_edge_gallery_get_result(argsJson, secret);
                    const resultStr = typeof result === 'string' ? result : JSON.stringify(result);
                    cleanup();
                    resolve(resultStr);
                } else {
                    cleanup();
                    resolve(JSON.stringify({ error: "ai_edge_gallery_get_result not defined", result: null }));
                }
            } catch (e: any) {
                cleanup();
                resolve(JSON.stringify({ error: e.message || String(e), result: null }));
            }
        };

        // Trigger after a short delay to ensure iframe is loaded
        if (doc.readyState === 'complete') {
            checkAndExecute();
        } else {
            iframe.onload = () => {
                checkAndExecute();
            };
        }
    });
};

/**
 * Checks whether an ONNX model matching [pattern] is present in browser Cache Storage.
 */
(window as any).isTransformersModelCached = async (pattern: string): Promise<boolean> => {
    if (!('caches' in window)) return false;
    try {
        const cacheNames = await caches.keys();
        for (const name of cacheNames) {
            const cache = await caches.open(name);
            const reqs = await cache.keys();
            for (const req of reqs) {
                if (req.url.toLowerCase().includes(pattern.toLowerCase())) {
                    return true;
                }
            }
        }
        return false;
    } catch (_) {
        return false;
    }
};

/**
 * Deletes cached model entries matching [pattern] from browser Cache Storage.
 */
(window as any).deleteTransformersModel = async (pattern: string): Promise<boolean> => {
    if (!('caches' in window)) return false;
    try {
        const cacheNames = await caches.keys();
        let deletedAny = false;
        for (const name of cacheNames) {
            const cache = await caches.open(name);
            const reqs = await cache.keys();
            for (const req of reqs) {
                if (req.url.toLowerCase().includes(pattern.toLowerCase())) {
                    const ok = await cache.delete(req);
                    if (ok) deletedAny = true;
                }
            }
        }
        return deletedAny;
    } catch (_) {
        return false;
    }
};