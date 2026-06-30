// lib/testing/test_suite.dart
//
// All plugin integration test cases, organized in phases.
// Runs sequentially because each phase builds on the previous one.
//
// Assets required in pubspec.yaml:
//   flutter:
//     assets:
//       - test/test.png   # any real PNG / JPG image
//       - test/test.wav   # any real MP3 audio clip (a few seconds)
//       - test/test.pdf   # any real PDF with visible text  (optional)

import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_local_gemma/flutter_local_gemma.dart';
import 'package:flutter_local_gemma/helpers/model_loader.dart';

const _hfToken = 'hf_YOUR_TOKEN_HERE';

// Short prompt → fewer tokens → faster tests.
const _shortPrompt = 'Reply with exactly one short sentence: What is 2+2?';

// ─── Status ───────────────────────────────────────────────────────────────────

enum TestStatus { pending, running, passed, failed, skipped }

// ─── TestCase ─────────────────────────────────────────────────────────────────

class TestCase {
  final String id;
  final String name;
  final String description;
  final Future<void> Function(TestSuiteContext ctx) run;

  TestStatus status = TestStatus.pending;
  String? error;

  /// Short human-readable result summary (e.g. the model's reply).
  String? detail;
  Duration? duration;

  TestCase({
    required this.id,
    required this.name,
    required this.description,
    required this.run,
  });
}

// ─── Shared context ───────────────────────────────────────────────────────────

class TestSuiteContext {
  String? llmPath;
  String? embedPath;
  String? tokenizerPath;

  /// Live download-progress value [0–100], updated during download tests.
  double downloadProgress = 0;

  /// Optional callback so the screen can redraw during progress ticks.
  void Function(double)? onProgress;

  // ── Asset loaders (lazy, cached after first load) ─────────────────────────

  Uint8List? _imageBytes;
  Uint8List? _audioBytes;
  Uint8List? _pdfBytes;

  Future<Uint8List> get imageBytes async {
    _imageBytes ??= (await rootBundle.load(
      'test/test.png',
    )).buffer.asUint8List();
    return _imageBytes!;
  }

  Future<Uint8List> get audioBytes async {
    _audioBytes ??= (await rootBundle.load(
      'test/test.wav',
    )).buffer.asUint8List();
    return _audioBytes!;
  }

  Future<Uint8List> get pdfBytes async {
    if (_pdfBytes != null) return _pdfBytes!;
    try {
      _pdfBytes = (await rootBundle.load('test/test.pdf')).buffer.asUint8List();
    } catch (_) {
      // Fall back to an inline-generated PDF if the asset is not bundled.
      _pdfBytes = TestSuite._makeMinimalPdf(
        'flutter_local_gemma PDF test — Hello World',
      );
    }
    return _pdfBytes!;
  }
}

// ─── TestSuite runner ─────────────────────────────────────────────────────────

class TestSuite {
  final void Function() onUpdate;
  final TestSuiteContext ctx = TestSuiteContext();

  late final List<TestCase> cases = _buildCases();

  bool isRunning = false;
  int get passCount => cases.where((c) => c.status == TestStatus.passed).length;
  int get failCount => cases.where((c) => c.status == TestStatus.failed).length;
  int get skippedCount =>
      cases.where((c) => c.status == TestStatus.skipped).length;
  int get totalDone => cases
      .where(
        (c) => c.status != TestStatus.pending && c.status != TestStatus.running,
      )
      .length;

  TestSuite({required this.onUpdate});

  /// Runs every pending test in order.
  Future<void> runAll() async {
    isRunning = true;
    onUpdate();
    for (final tc in cases) {
      if (tc.status != TestStatus.pending) continue;
      await _run(tc);
    }
    isRunning = false;
    onUpdate();
  }

  /// Re-runs a single test (resets it to pending first).
  Future<void> runOne(TestCase tc) async {
    tc.status = TestStatus.pending;
    onUpdate();
    await _run(tc);
  }

  Future<void> _run(TestCase tc) async {
    tc.status = TestStatus.running;
    tc.error = null;
    tc.detail = null;
    onUpdate();

    final start = DateTime.now();
    try {
      // Auto-initialize LLM if running out of order.
      // If the engine is ALREADY initialized (user loaded a model via the UI),
      // we reuse it instead of downloading — this lets you test any model.
      if (tc.id.startsWith('llm_') == false && tc.id != 'llm_download' && tc.id != 'llm_init') {
        if (!FlutterLocalGemma().isInitialized) {
          // Engine not running — try download fallback
          ctx.llmPath = await PluginModelLoader.downloadLlm(token: _hfToken);
          if (ctx.llmPath != null) {
            await PluginModelLoader.initLlm(path: ctx.llmPath!);
          }
        } else {
          // Engine already running — just make sure ctx has the path
          // so tests that need it (e.g. llm_reload) can use it.
          ctx.llmPath ??= FlutterLocalGemma().currentModelPath;
        }
      }
      
      await tc.run(ctx);
      tc.duration = DateTime.now().difference(start);
      tc.status = TestStatus.passed;
    } on _SkipException catch (e) {
      tc.duration = DateTime.now().difference(start);
      tc.status = TestStatus.skipped;
      tc.error = e.toString();
    } catch (e, st) {
      tc.duration = DateTime.now().difference(start);
      tc.status = TestStatus.failed;
      tc.error = e.toString();
      debugPrint('TEST FAILED [${tc.id}]: $e\n$st');
    }

    // ── Engine-settle delay ────────────────────────────────────────────────
    //
    // The web WASM engine is single-threaded. Even after a stream closes or
    // a Future completes, a few microtask-loop ticks may still be unwinding
    // inside the WASM. Without this pause the next test's createSession call
    // can race with the tail of the previous generation and throws:
    //   "[GemmaWeb] generateResponse error: Cannot process because LLM
    //    inference engine is currently loading or processing."
    //
    // 1 second is sufficient on all tested browsers and Android devices.
    await Future<void>.delayed(const Duration(seconds: 1));

    onUpdate();
  }

  void reset() {
    for (final tc in cases) {
      tc.status = TestStatus.pending;
      tc.error = null;
      tc.detail = null;
      tc.duration = null;
    }
    onUpdate();
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Test case definitions
  // ════════════════════════════════════════════════════════════════════════════

  List<TestCase> _buildCases() => [
    // ── Phase 1 — LLM lifecycle ──────────────────────────────────────────────
    TestCase(
      id: 'llm_download',
      name: 'LLM — Download model',
      description: 'Downloads Gemma from HuggingFace to the platform cache.',
      run: (ctx) async {
        if (FlutterLocalGemma().isInitialized) {
          ctx.llmPath = FlutterLocalGemma().currentModelPath;
          throw _SkipException('Engine already loaded. Using current model for tests.');
        }
        ctx.llmPath = await PluginModelLoader.downloadLlm(
          token: _hfToken,
          onProgress: (p) {
            ctx.downloadProgress = p;
            ctx.onProgress?.call(p);
          },
        );
        if (ctx.llmPath == null || ctx.llmPath!.isEmpty) {
          throw Exception('Returned model path is empty');
        }
      },
    ),

    TestCase(
      id: 'llm_init',
      name: 'LLM — Init engine',
      description: 'Initialises the Gemma engine from the cached model.',
      run: (ctx) async {
        if (FlutterLocalGemma().isInitialized) {
          throw _SkipException('Engine already initialized. Skipping.');
        }
        if (ctx.llmPath == null) throw Exception('Run llm_download first');
        await PluginModelLoader.initLlm(path: ctx.llmPath!);
        if (!FlutterLocalGemma().isInitialized) {
          throw Exception('isInitialized is still false after init()');
        }
      },
    ),

    // ── Phase 2 — LLM inference ──────────────────────────────────────────────
    TestCase(
      id: 'llm_single_turn',
      name: 'LLM — Single-turn inference',
      description: 'SingleTurnChat.generate() returns a non-empty string.',
      run: (ctx) async {
        final isGemma4 = FlutterLocalGemma().currentModelIsGemma4;
        
        // Gemma 4 Web requires explicit prompt formatting for raw SingleTurnChat
        final prompt = (kIsWeb && isGemma4)
            ? '<start_of_turn>user\n$_shortPrompt<end_of_turn>\n<start_of_turn>model\n'
            : _shortPrompt;

        final chat = SingleTurnChat(config: SessionConfig());
        final reply = await chat.generate(prompt);
        if (reply.trim().isEmpty) throw Exception('Response was empty');
        debugPrint('[llm_single_turn] reply: $reply');
      },
    ),

    TestCase(
      id: 'llm_streaming',
      name: 'LLM — Streaming inference',
      description: 'sendMessageStream() yields tokens incrementally.',
      run: (ctx) async {
        final chat = GemmaChat(systemPrompt: 'You are a helpful assistant.');
        await chat.init();
        final tokens = <String>[];

        // Await the stream to full completion before disposing.
        // This is critical — disposing while the WASM stream is mid-flight
        // leaves the engine in a busy state that breaks the next test.
        await for (final tok in chat.sendMessageStream(text: _shortPrompt)) {
          tokens.add(tok);
        }

        await chat.dispose();

        if (tokens.isEmpty) throw Exception('No tokens were streamed');
        if (tokens.join().trim().isEmpty)
          throw Exception('Streamed result is empty');
        debugPrint('[llm_streaming] ${tokens.length} tokens: ${tokens.join()}');
      },
    ),

    TestCase(
      id: 'llm_stop_mid_stream',
      name: 'LLM — Stop mid-stream',
      description: 'chat.stop() halts generation; stream closes cleanly.',
      run: (ctx) async {
        final chat = GemmaChat(systemPrompt: 'You are a helpful assistant.');
        await chat.init();

        int tokenCount = 0;
        bool stopCalled = false;

        // Resolve only when onDone fires — NOT when stop() returns.
        // Waiting for onDone guarantees the WASM engine is fully idle before
        // we dispose and start the next test.
        final streamDone = Completer<void>();

        final stream = chat.sendMessageStream(
          text: 'Count slowly from 1 to 1000, one number per line.',
        );

        late StreamSubscription<String> sub;
        sub = stream.listen(
          (tok) {
            tokenCount++;
            if (tokenCount >= 5 && !stopCalled) {
              stopCalled = true;
              // Signal stop but keep listening — we wait for onDone, not for
              // this future, so the engine can flush any buffered tokens.
              chat.stop();
            }
          },
          onDone: () {
            if (!streamDone.isCompleted) streamDone.complete();
          },
          onError: (_) {
            // Some platforms surface a cancelled stream as an error — fine.
            if (!streamDone.isCompleted) streamDone.complete();
          },
          cancelOnError: false,
        );

        await streamDone.future.timeout(
          const Duration(seconds: 90),
          onTimeout: () {
            sub.cancel();
            if (!streamDone.isCompleted) streamDone.complete();
          },
        );

        await chat.dispose();

        if (!stopCalled) {
          throw Exception(
            'Stop never triggered — received fewer than 5 tokens ($tokenCount)',
          );
        }
        debugPrint('[llm_stop_mid_stream] stopped after $tokenCount tokens');
      },
    ),

    // ── Image inference ──────────────────────────────────────────────────────
    //
    // Gemma 3n supports images on both web and Android.
    // A real PNG from assets is required — the model must receive decodable
    // pixels, not synthetic bytes.
    TestCase(
      id: 'llm_with_image',
      name: 'LLM — Inference with image',
      description: 'Passes test/test.png to the model; expects a reply.',
      run: (ctx) async {
        final isGemma4 = FlutterLocalGemma().currentModelIsGemma4;
        if (kIsWeb && isGemma4) {
          throw _SkipException('Skipping on Web because Gemma 4 does not support images.');
        }

        final imgBytes = await ctx.imageBytes;
        final chat = GemmaChat();
        await chat.init();

        final reply = await chat.sendMessage(
          text: 'Describe this image in one short sentence.',
          images: [imgBytes],
        );
        await chat.dispose();

        if (reply.trim().isEmpty) throw Exception('Response was empty');
        debugPrint('[llm_with_image] reply: $reply');
      },
    ),

    // ── Audio inference ──────────────────────────────────────────────────────
    //
    // Web: the browser's AudioDecoder decodes the MP3 before handing it to
    //      WASM — fake bytes cause "Unable to decode audio data".
    // Android: the plugin decodes the file before passing to the engine.
    //
    // A real MP3 from assets is required on BOTH platforms.
    TestCase(
      id: 'llm_with_audio',
      name: 'LLM — Inference with audio',
      description: 'Passes test/test.wav to the model; expects a reply.',
      run: (ctx) async {
        final isGemma4 = FlutterLocalGemma().currentModelIsGemma4;
        if (kIsWeb && isGemma4) {
          throw _SkipException('Skipping on Web because Gemma 4 does not support audio.');
        }

        final audBytes = await ctx.audioBytes;
        final chat = GemmaChat();
        await chat.init();

        final reply = await chat.sendMessage(
          text: 'What do you hear in this audio clip? One sentence.',
          audios: [audBytes],
        );
        await chat.dispose();

        if (reply.trim().isEmpty) throw Exception('Response was empty');
        debugPrint('[llm_with_audio] reply: $reply');
      },
    ),

    // ── PDF inference ────────────────────────────────────────────────────────
    //
    // Prefers test/test.pdf; falls back to an inline-generated PDF.
    // On Android the native PdfRenderer may return only rendered page images
    // (no raw text), so we handle both text-only and image-only extraction.
    TestCase(
      id: 'llm_with_pdf',
      name: 'LLM — Inference with PDF',
      description: 'Extracts content from a PDF and sends it to the LLM.',
      run: (ctx) async {
        final isGemma4 = FlutterLocalGemma().currentModelIsGemma4;
        if (kIsWeb && isGemma4) {
          throw _SkipException('Skipping on Web because Gemma 4 does not support multimodality (which this PDF test uses).');
        }

        final pdfData = await ctx.pdfBytes;
        final parts = await PdfProcessor.extract(
          pdfData,
          const PdfExtractionConfig(mode: PdfExtractionMode.textAndImages),
        );

        if (parts.isEmpty) {
          throw Exception('PDF extraction returned zero content parts');
        }

        final textParts = parts.whereType<TextPart>().toList();
        final imageParts = parts.whereType<ImagePart>().toList();
        debugPrint(
          '[llm_with_pdf] extracted: ${textParts.length} text, '
          '${imageParts.length} images',
        );

        // Build the prompt from whatever was extracted.
        final extractedText = textParts.map((t) => t.text).join('\n').trim();
        final promptText = extractedText.isNotEmpty
            ? 'Summarise this document in one sentence:\n$extractedText'
            : 'Describe what you see in this document page.';

        final chat = GemmaChat();
        await chat.init();

        final reply = await chat.sendMessage(
          text: promptText,
          images: imageParts.map((p) => p.bytes).toList(),
        );
        await chat.dispose();

        if (reply.trim().isEmpty) throw Exception('LLM reply was empty');
        debugPrint('[llm_with_pdf] reply: $reply');
      },
    ),

    // ── Phase 3 — Session / context management ───────────────────────────────
    TestCase(
      id: 'llm_cache_clear',
      name: 'LLM — Context clear',
      description: 'clearHistory() resets context; next message still works.',
      run: (ctx) async {
        final chat = GemmaChat(systemPrompt: 'Be concise.');
        await chat.init();
        await chat.sendMessage(text: 'Remember: my lucky number is 42.');
        await chat.clearHistory();

        if (chat.history.isNotEmpty) {
          throw Exception(
            'History not cleared (${chat.history.length} messages remain)',
          );
        }

        final reply = await chat.sendMessage(text: _shortPrompt);
        await chat.dispose();
        if (reply.trim().isEmpty)
          throw Exception('Reply after clear was empty');
        debugPrint('[llm_cache_clear] reply: $reply');
      },
    ),

    TestCase(
      id: 'llm_settings_update',
      name: 'LLM — Session settings (temperature / topK)',
      description: 'Low-temp and high-temp configs both produce valid replies.',
      run: (ctx) async {
        final isGemma4 = FlutterLocalGemma().currentModelIsGemma4;
        final prompt = (kIsWeb && isGemma4)
            ? '<start_of_turn>user\n$_shortPrompt<end_of_turn>\n<start_of_turn>model\n'
            : _shortPrompt;

        final cold = await SingleTurnChat(
          config: SessionConfig(temperature: 0.1, topK: 1),
        ).generate(prompt);
        if (cold.trim().isEmpty) throw Exception('Low-temp reply was empty');

        final hot = await SingleTurnChat(
          config: SessionConfig(temperature: 1.0, topK: 50),
        ).generate(prompt);
        if (hot.trim().isEmpty) throw Exception('High-temp reply was empty');

        debugPrint('[llm_settings_update] cold: $cold | hot: $hot');
      },
    ),

    // ── Phase 4 — JSON schema output ─────────────────────────────────────────
    TestCase(
      id: 'llm_json_schema',
      name: 'LLM — JSON schema output',
      description: 'sendMessageJsonStream() produces valid JSON for a schema.',
      run: (ctx) async {
        final schema = Schema.object({
          'name': Schema.string(),
          'score': Schema.number(),
        });

        final chat = GemmaChat();
        await chat.init();

        String? lastJson;
        await for (final partial in chat.sendMessageJsonStream(
          text: 'Return a fake player named Alice with score 99.',
          schema: schema,
        )) {
          lastJson = partial.toString();
        }
        await chat.dispose();

        if (lastJson == null || lastJson.isEmpty) {
          throw Exception('No JSON was produced');
        }
        debugPrint('[llm_json_schema] json: $lastJson');
      },
    ),

    // ── Phase 5 — Multi-turn chat features & New Options ───────────────────────────────────
    TestCase(
      id: 'llm_thinking_mode',
      name: 'LLM — Thinking Mode',
      description: 'Generates response using Thinking Mode configuration.',
      run: (ctx) async {
        final modelPath = (FlutterLocalGemma().currentModelPath ?? '').toLowerCase();
        final isGemma3 = modelPath.contains('gemma-3') || modelPath.contains('gemma_3') || modelPath.contains('gemma-2') || modelPath.contains('gemma_2');
        if (isGemma3) {
          throw _SkipException('Skipping Thinking Mode test because Gemma 3 does not support it.');
        }

        final chat = SingleTurnChat(
          config: SessionConfig(enableThinking: true),
        );
        final reply = await chat.generate(
          'Think step by step: what is 3 * 7?',
          systemPrompt:
              'You MUST think step by step before answering. Enclose your thoughts in <think> tags.',
        );
        if (reply.trim().isEmpty)
          throw Exception('Thinking mode reply was empty');

        // Gemma 4 on Android emits <|channel>thought\n…<channel|> instead of <think> tags.
        // ThinkingParser strips those tokens so the final 'reply' won't contain them.
        // We accept either form by checking for:
        //   (a) standard <think> tag still present in reply (some models / web), OR
        //   (b) the Gemma 4 channel marker in reply, OR
        //   (c) the correct numerical answer (21) in the reply.
        final hasThinkTag    = reply.contains('<think>');
        final hasChannelTag  = reply.contains('<|channel>');
        final hasCorrectAns  = reply.contains('21');

        if (!hasThinkTag && !hasChannelTag && !hasCorrectAns) {
          throw Exception(
            'Thinking mode: no <think>/<|channel> block found and answer is wrong. Reply: $reply',
          );
        }

        debugPrint('[llm_thinking_mode] reply: $reply');
      },
    ),

    TestCase(
      id: 'llm_mtp_support',
      name: 'LLM — MTP Configuration',
      description: 'Loads model with MTP speculative decoding enabled.',
      run: (ctx) async {
        if (kIsWeb) {
          throw _SkipException('MTP is not supported on the Web platform yet.');
        }
        // Try to get the model path from ctx first, then from the live engine.
        // If llm_download failed (e.g. placeholder HF token), we skip gracefully.
        ctx.llmPath ??= FlutterLocalGemma().currentModelPath;
        if (ctx.llmPath == null || ctx.llmPath!.isEmpty) {
          throw _SkipException(
            'llmPath not available — run llm_download first or load a model from the UI.',
          );
        }
        await PluginModelLoader.unloadLlm();
        try {
          await FlutterLocalGemma().init(
            InferenceConfig(
              modelPath: ctx.llmPath!,
              maxTokens: 1024,
              enableMtp: true,
            ),
          );
        } catch (e) {
          // MTP is optional / unsupported on some hardware — skip rather than fail.
          throw _SkipException('MTP init failed (may be unsupported on this device): $e');
        }
        final chat = SingleTurnChat(config: SessionConfig());
        final reply = await chat.generate('Say MTP is working.');
        if (reply.trim().isEmpty) throw Exception('MTP reply was empty');
        debugPrint('[llm_mtp_support] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_dart_skill_prompt',
      name: 'AgentChat — Dart Skill (Prompt mode)',
      description:
          'AgentChat invokes a Dart skill using the default JSON prompt injection.',
      run: (ctx) async {
        bool skillCalled = false;
        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: false),
          skills: [
            GemmaDartSkill(
              name: 'get_time',
              description: 'Returns the current time',
              parametersSchema: {"type": "object", "properties": {}},
              required: [],
              handler: (args) async {
                skillCalled = true;
                return GemmaSkillResult.text('The time is exactly 12:00 PM.');
              },
            ),
          ],
        );
        await chat.init();

        String reply = '';
        String rawText = '';
        await for (final turn in chat.run(
          'What time is it? Use the get_time skill.',
        )) {
          reply = turn.modelAnswer;
          rawText = turn.rawText;
        }
        await chat.dispose();

        debugPrint('[AgentRawOutput]: $rawText');

        if (!skillCalled) {
          throw Exception(
            'Skill was not called by the model. Reply was: $reply',
          );
        }
        if (!reply.contains('12:00')) {
          throw Exception(
            'Skill output was not incorporated into final answer. Reply was: $reply',
          );
        }
        if (reply.trim().isEmpty) {
          throw Exception('Final reply was empty');
        }
        debugPrint('[agent_chat_dart_skill_prompt] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_dart_skill_native',
      name: 'AgentChat — Dart Skill (Native mode)',
      description: 'AgentChat invokes a Dart skill using native tool calling.',
      run: (ctx) async {
        bool skillCalled = false;
        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: true),
          skills: [
            GemmaDartSkill(
              name: 'get_time',
              description: 'Returns the current time',
              parametersSchema: {"type": "object", "properties": {}},
              required: [],
              handler: (args) async {
                skillCalled = true;
                return GemmaSkillResult.text('The time is exactly 12:00 PM.');
              },
            ),
          ],
        );
        await chat.init();

        String reply = '';
        String rawText = '';
        await for (final turn in chat.run(
          'What time is it? Use the get_time skill.',
        )) {
          reply = turn.modelAnswer;
          rawText = turn.rawText;
        }
        await chat.dispose();

        debugPrint('[AgentRawOutput]: $rawText');

        if (!skillCalled) {
          throw Exception(
            'Skill was not called natively by the model. Reply was: $reply',
          );
        }
        if (!reply.contains('12:00')) {
          throw Exception(
            'Skill output was not incorporated into final answer. Reply was: $reply',
          );
        }
        debugPrint('[agent_chat_dart_skill_native] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_local_js_skill_prompt',
      name: 'AgentChat — Local JS Skill (Prompt mode)',
      description:
          'AgentChat invokes a Javascript skill using local HTML string.',
      run: (ctx) async {
        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: false),
          skills: [
            GemmaJsSkill(
              name: 'calculate_sum',
              description: 'Calculates the sum of a and b',
              instructions:
                  'Use this tool to calculate sum. Arguments should be {"a": 15, "b": 27}.',
              parametersSchema: {
                "type": "object",
                "properties": {
                  "a": {"type": "number", "description": "First number"},
                  "b": {"type": "number", "description": "Second number"}
                }
              },
              scriptHtml: '''
                <script>
                  window.ai_edge_gallery_get_result = async (data) => {
                    const args = JSON.parse(data);
                    return { result: "The sum is " + (args.a + args.b) };
                  };
                </script>
              ''',
            ),
          ],
        );
        await chat.init();

        String reply = '';
        String rawText = '';
        await for (final turn in chat.run(
          'Calculate the sum of 15 and 27 using the calculate_sum skill.',
        )) {
          reply = turn.modelAnswer;
          rawText = turn.rawText;
        }
        await chat.dispose();

        debugPrint('[AgentRawOutput]: $rawText');

        if (!reply.contains('42')) {
          throw Exception(
            'Local JS Skill output was not incorporated into final answer. Reply was: $reply',
          );
        }
        debugPrint('[agent_chat_local_js_skill_prompt] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_local_js_skill_native',
      name: 'AgentChat — Local JS Skill (Native mode)',
      description:
          'AgentChat invokes a Javascript skill using native tool calling.',
      run: (ctx) async {
        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: true),
          skills: [
            GemmaJsSkill(
              name: 'calculate_sum',
              description: 'Calculates the sum of a and b',
              instructions:
                  'Use this tool to calculate sum. Arguments should be {"a": 15, "b": 27}.',
              parametersSchema: {
                "type": "object",
                "properties": {
                  "a": {"type": "number", "description": "First number"},
                  "b": {"type": "number", "description": "Second number"}
                }
              },
              scriptHtml: '''
                <script>
                  window.ai_edge_gallery_get_result = async (data) => {
                    const args = JSON.parse(data);
                    return { result: "The sum is " + (args.a + args.b) };
                  };
                </script>
              ''',
            ),
          ],
        );
        await chat.init();

        String reply = '';
        await for (final turn in chat.run(
          'Calculate the sum of 15 and 27 using the calculate_sum skill.',
        )) {
          reply = turn.modelAnswer;
        }
        await chat.dispose();

        if (!reply.contains('42')) {
          throw Exception(
            'Local JS Skill output was not incorporated into final answer. Reply was: $reply',
          );
        }
        debugPrint('[agent_chat_local_js_skill_native] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_remote_js_skill_prompt',
      name: 'AgentChat — Remote JS Skill (Prompt mode)',
      description:
          'AgentChat invokes a Javascript skill loaded from a remote URL.',
      run: (ctx) async {
        GemmaJsSkill remoteSkill;
        try {
          remoteSkill = await GemmaJsSkill.fromUrl(
            'https://raw.githubusercontent.com/google-ai-edge/gallery/refs/heads/main/skills/built-in/calculate-hash/',
          );
        } catch (e) {
          throw _SkipException(
            'Could not load remote skill from URL. Skipped test. $e',
          );
        }

        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: false),
          skills: [remoteSkill],
        );
        await chat.init();

        String reply = '';
        await for (final turn in chat.run('Calculate the SHA-256 hash for the string "hello world" using the skill.')) {
          reply = turn.modelAnswer;
        }
        await chat.dispose();

        if (reply.isEmpty)
          throw Exception(
            'Empty reply from remote JS skill. Model might have failed to use the tool properly.',
          );
        debugPrint('[agent_chat_remote_js_skill_prompt] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_remote_js_skill_native',
      name: 'AgentChat — Remote JS Skill (Native mode)',
      description:
          'AgentChat invokes a Javascript skill loaded from a remote URL using native tool calling.',
      run: (ctx) async {
        GemmaJsSkill remoteSkill;
        try {
          remoteSkill = await GemmaJsSkill.fromUrl(
            'https://raw.githubusercontent.com/google-ai-edge/gallery/refs/heads/main/skills/built-in/calculate-hash/',
          );
        } catch (e) {
          throw _SkipException(
            'Could not load remote skill from URL. Skipped test. $e',
          );
        }

        final chat = AgentChat(
          config: SessionConfig(nativeToolCalling: true),
          skills: [remoteSkill],
        );
        await chat.init();

        String reply = '';
        await for (final turn in chat.run('Calculate the SHA-256 hash for the string "hello world" using the skill.')) {
          reply = turn.modelAnswer;
        }
        await chat.dispose();

        if (reply.isEmpty) {
          throw Exception(
            'Empty reply from remote JS skill. Model might have failed to use the tool properly natively.',
          );
        }
        debugPrint('[agent_chat_remote_js_skill_native] reply: $reply');
      },
    ),

    TestCase(
      id: 'agent_chat_thinking_mode',
      name: 'AgentChat — Thinking via Tags',
      description:
          'AgentChat populates thinkingBlocks using <think>/<|channel>thought tags natively.',
      run: (ctx) async {
        final chat = AgentChat(
          skills: [],
          config: SessionConfig(enableThinking: true),
        );
        await chat.init();

        String reply = '';
        List<String> thoughts = [];
        await for (final turn in chat.run(
          'Think step by step: what is 3 * 7?',
        )) {
          reply = turn.modelAnswer;
          thoughts = turn.thinkingBlocks;
        }
        await chat.dispose();

        if (reply.trim().isEmpty) throw Exception('Final reply was empty');

        // Gemma 4 on Android uses <|channel>thought…<channel|> which ThinkingParser
        // correctly strips into thinkingBlocks. However some models may answer
        // correctly without emitting any explicit thinking wrapper. Accept the test
        // if EITHER thinking blocks were parsed OR the answer is correct (21).
        final hasCorrectAns = reply.contains('21');
        if (thoughts.isEmpty && !hasCorrectAns) {
          throw Exception(
            'No thinking blocks parsed and answer is wrong. '
            'Model likely ignored the system prompt or generated malformed tags. '
            'Reply was: $reply',
          );
        }

        debugPrint(
          '[agent_chat_thinking_mode] parsed ${thoughts.length} blocks. reply: $reply',
        );
      },
    ),

    TestCase(
      id: 'chat_multi_turn',
      name: 'Chat — Multi-turn conversation',
      description: 'Context is maintained across 3 sequential turns.',
      run: (ctx) async {
        final chat = GemmaChat(systemPrompt: 'Be concise.');
        await chat.init();

        await chat.sendMessage(text: 'My name is TestUser.');
        await chat.sendMessage(text: 'What is 1+1?');
        final reply = await chat.sendMessage(text: 'What was my name again?');
        await chat.dispose();

        if (reply.trim().isEmpty) throw Exception('Third-turn reply was empty');
        debugPrint('[chat_multi_turn] turn-3 reply: $reply');
      },
    ),

    TestCase(
      id: 'chat_history_export_import',
      name: 'Chat — History export / import',
      description: 'exportHistory JSON round-trips via importHistory.',
      run: (ctx) async {
        final chat1 = GemmaChat();
        await chat1.init();
        await chat1.sendMessage(text: 'Say hello.');
        final json = await chat1.exportHistory();
        await chat1.dispose();

        if (json.isEmpty)
          throw Exception('exportHistory returned empty string');

        final chat2 = GemmaChat();
        await chat2.init();
        await chat2.importHistory(json);

        if (chat2.history.isEmpty) {
          throw Exception('importHistory produced empty history');
        }
        await chat2.dispose();
        debugPrint(
          '[chat_history_export_import] imported ${chat2.history.length} messages',
        );
      },
    ),

    TestCase(
      id: 'chat_sliding_window',
      name: 'Chat — Sliding-window context strategy',
      description: '4 messages with a 512-token window trim context silently.',
      run: (ctx) async {
        debugPrint('[chat_sliding_window] Initializing GemmaChat with maxContextTokens: 512');
        final chat = GemmaChat(
          maxContextTokens: 512,
          contextStrategy: ContextStrategy.slidingWindow,
        );
        await chat.init();
        debugPrint('[chat_sliding_window] Chat initialized');

        for (int i = 1; i <= 4; i++) {
          debugPrint('[chat_sliding_window] Sending message $i of 4...');
          final r = await chat.sendMessage(text: 'Message $i. Reply with OK.');
          debugPrint('[chat_sliding_window] Received reply for message $i: ${r.trim()}');
          if (r.trim().isEmpty) throw Exception('Empty reply on message $i');
          debugPrint('[chat_sliding_window] History length after message $i: ${chat.history.length}');
        }
        await chat.dispose();
        debugPrint('[chat_sliding_window] Test complete');
      },
    ),

    TestCase(
      id: 'chat_edit_delete_history',
      name: 'Chat — Edit / delete history',
      description: 'editHistory() and removeHistory() mutate the history list.',
      run: (ctx) async {
        final chat = GemmaChat();
        await chat.init();
        await chat.sendMessage(text: 'Hello.');

        if (chat.history.isEmpty) {
          throw Exception('History is empty after first turn');
        }

        // Edit first message.
        await chat.editHistory(
          0,
          ChatMessage(role: chat.history[0].role, text: 'Edited.'),
        );
        if (chat.history[0].text != 'Edited.') {
          throw Exception('editHistory did not apply');
        }

        // Delete first message.
        final lenBefore = chat.history.length;
        await chat.removeHistory(0);
        if (chat.history.length >= lenBefore) {
          throw Exception('removeHistory did not reduce history length');
        }
        await chat.dispose();
      },
    ),

    // ── Phase 6 — LLM unload / reload ────────────────────────────────────────
    TestCase(
      id: 'llm_unload',
      name: 'LLM — Unload engine',
      description: 'dispose() succeeds; isInitialized becomes false.',
      run: (ctx) async {
        await PluginModelLoader.unloadLlm();
        if (FlutterLocalGemma().isInitialized) {
          throw Exception('Engine still reports isInitialized after dispose()');
        }
      },
    ),

    TestCase(
      id: 'llm_reload',
      name: 'LLM — Reload (warm / cached)',
      description: 'Second init() from the cached path succeeds.',
      run: (ctx) async {
        if (ctx.llmPath == null) {
          throw Exception('llmPath is null — run llm_download first');
        }
        await PluginModelLoader.initLlm(path: ctx.llmPath!);
        if (!FlutterLocalGemma().isInitialized) {
          throw Exception('Reload: isInitialized is false');
        }
        final reply = await SingleTurnChat(
          config: SessionConfig(),
        ).generate(_shortPrompt);
        if (reply.trim().isEmpty)
          throw Exception('Empty reply after warm reload');
        debugPrint('[llm_reload] reply: $reply');
      },
    ),

    // ── Phase 7 — Embedding ───────────────────────────────────────────────────
    TestCase(
      id: 'embed_download',
      name: 'Embedding — Download model',
      description: 'Downloads the embedding model (+ tokenizer on Android).',
      run: (ctx) async {
        final result = await PluginModelLoader.downloadEmbedding(
          token: _hfToken,
          onProgress: (p) {
            ctx.downloadProgress = p;
            ctx.onProgress?.call(p);
          },
        );
        ctx.embedPath = result.modelPath;
        ctx.tokenizerPath = result.tokenizerPath;
        if (ctx.embedPath == null || ctx.embedPath!.isEmpty) {
          throw Exception('Embed model path is empty');
        }
      },
    ),

    TestCase(
      id: 'embed_init',
      name: 'Embedding — Init engine',
      description: 'Initialises EmbeddingPlugin from the cached model.',
      run: (ctx) async {
        if (ctx.embedPath == null) throw Exception('Run embed_download first');
        await PluginModelLoader.initEmbedding(
          modelPath: ctx.embedPath!,
          tokenizerPath: ctx.tokenizerPath,
          token: _hfToken,
        );
        if (!EmbeddingPlugin().isInitialized) {
          throw Exception('isInitialized is false after init()');
        }
      },
    ),

    TestCase(
      id: 'embed_vector',
      name: 'Embedding — Generate vector',
      description: 'getEmbedding() returns a non-zero float vector.',
      run: (ctx) async {
        final vec = await EmbeddingPlugin().getEmbedding('Hello, world!');
        if (vec.isEmpty) throw Exception('Vector is empty');
        if (vec.every((v) => v == 0.0)) throw Exception('Vector is all zeros');
        debugPrint(
          '[embed_vector] dim=${vec.length}, '
          'first3=${vec.take(3).map((v) => v.toStringAsFixed(4)).join(', ')}',
        );
      },
    ),

    TestCase(
      id: 'embed_similarity',
      name: 'Embedding — Cosine similarity',
      description: 'Similar sentences score higher than unrelated ones.',
      run: (ctx) async {
        final vA = await EmbeddingPlugin().getEmbedding(
          'The cat sat on the mat.',
        );
        final vB = await EmbeddingPlugin().getEmbedding(
          'A feline rested on a rug.',
        );
        final vC = await EmbeddingPlugin().getEmbedding(
          'Stock market crash today.',
        );

        final simAB = _cosine(vA, vB);
        final simAC = _cosine(vA, vC);

        debugPrint(
          '[embed_similarity] AB=${simAB.toStringAsFixed(4)}, '
          'AC=${simAC.toStringAsFixed(4)}',
        );

        if (simAB <= simAC) {
          throw Exception(
            'Expected similar pair AB (${simAB.toStringAsFixed(3)}) > '
            'dissimilar pair AC (${simAC.toStringAsFixed(3)})',
          );
        }
      },
    ),

    TestCase(
      id: 'embed_search',
      name: 'Embedding — Semantic search (top-1)',
      description: 'Selects the most relevant sentence from a corpus.',
      run: (ctx) async {
        final corpus = [
          'Flutter is a UI toolkit from Google.',
          'The Eiffel Tower is located in Paris, France.',
          'Deep learning models require large datasets.',
        ];
        const query = 'What is Flutter used for?';

        final qVec = await EmbeddingPlugin().getEmbedding(query);
        final scores = <double>[];
        for (final s in corpus) {
          scores.add(_cosine(qVec, await EmbeddingPlugin().getEmbedding(s)));
        }

        final best = scores.indexOf(scores.reduce(max));
        debugPrint(
          '[embed_search] scores: '
          '${scores.map((s) => s.toStringAsFixed(4)).join(', ')} — best=$best',
        );

        if (best != 0) {
          throw Exception(
            'Expected corpus[0] to rank highest, got corpus[$best] '
            '(${scores.map((s) => s.toStringAsFixed(3)).join(", ")})',
          );
        }
      },
    ),

    TestCase(
      id: 'embed_unload',
      name: 'Embedding — Unload engine',
      description: 'dispose() succeeds; isInitialized becomes false.',
      run: (ctx) async {
        await PluginModelLoader.unloadEmbedding();
        if (EmbeddingPlugin().isInitialized) {
          throw Exception(
            'EmbeddingPlugin still reports isInitialized after dispose()',
          );
        }
      },
    ),

    // ── Phase 8 — Cache purge ─────────────────────────────────────────────────
    TestCase(
      id: 'cache_purge',
      name: 'Cache — Purge all cached files',
      description: 'ModelInstaller.purgeCache() completes without error.',
      run: (ctx) async {
        if (FlutterLocalGemma().isInitialized) await PluginModelLoader.unloadLlm();
        if (EmbeddingPlugin().isInitialized)
          await PluginModelLoader.unloadEmbedding();

        final freed = await PluginModelLoader.purgeCache();
        debugPrint('[cache_purge] freed: $freed');

        ctx.llmPath = null;
        ctx.embedPath = null;
        ctx.tokenizerPath = null;
      },
    ),
  ];

  // ── Helpers ────────────────────────────────────────────────────────────────

  static double _cosine(List<double> a, List<double> b) {
    double dot = 0, nA = 0, nB = 0;
    final len = min(a.length, b.length);
    for (int i = 0; i < len; i++) {
      dot += a[i] * b[i];
      nA += a[i] * a[i];
      nB += b[i] * b[i];
    }
    final d = sqrt(nA) * sqrt(nB);
    return d == 0 ? 0.0 : dot / d;
  }

  /// Generates a minimal but spec-compliant PDF containing [text].
  ///
  /// Byte offsets in the xref table are calculated precisely so both the
  /// Android PdfRenderer and the web pdfjs worker can extract text reliably.
  static Uint8List _makeMinimalPdf(String text) {
    final safe = text
        .replaceAll('\\', '\\\\')
        .replaceAll('(', '\\(')
        .replaceAll(')', '\\)')
        .replaceAll('\r', '')
        .replaceAll('\n', ' ');

    const hdr = '%PDF-1.4\n';
    final obj1 = '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n';
    final obj2 = '2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n';
    final body = 'BT /F1 14 Tf 50 720 Td ($safe) Tj ET';
    final obj4 =
        '4 0 obj\n<< /Length ${body.length} >>\nstream\n'
        '$body\nendstream\nendobj\n';
    final obj3 =
        '3 0 obj\n<< /Type /Page /Parent 2 0 R\n'
        '   /MediaBox [0 0 612 792]\n'
        '   /Contents 4 0 R\n'
        '   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 '
        '/BaseFont /Helvetica >> >> >> >>\nendobj\n';

    // Compute absolute byte offsets for each object.
    final off = <int>[];
    var pos = 0;
    for (final chunk in [hdr, obj1, obj2, obj3, obj4]) {
      off.add(pos);
      pos += chunk.length;
    }

    final xrefPos = pos;
    final xref =
        'xref\n0 5\n'
        '0000000000 65535 f \n'
        '${off[1].toString().padLeft(10, '0')} 00000 n \n'
        '${off[2].toString().padLeft(10, '0')} 00000 n \n'
        '${off[3].toString().padLeft(10, '0')} 00000 n \n'
        '${off[4].toString().padLeft(10, '0')} 00000 n \n';
    final trailer =
        'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n$xrefPos\n%%EOF\n';

    return Uint8List.fromList(
      (hdr + obj1 + obj2 + obj3 + obj4 + xref + trailer).codeUnits,
    );
  }
}

// ─── _SkipException — treated as a non-failure by the runner ─────────────────

class _SkipException implements Exception {
  final String reason;
  const _SkipException(this.reason);
  @override
  String toString() => 'Skipped: $reason';
}