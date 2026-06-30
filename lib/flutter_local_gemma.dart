library flutter_local_gemma;

// ── Core types ────────────────────────────────────────────────────────────────
export 'types/content_parts.dart';

// ── LLM engine ────────────────────────────────────────────────────────────────
export 'gemma/gemma.dart';
export 'gemma/gemma_model.dart';
export 'gemma/gemma_response.dart';
export 'gemma/gemma_skill.dart';

/// High-level model download + lifecycle helper.
export 'gemma/gemma_loader.dart';

// ── Chat wrappers ─────────────────────────────────────────────────────────────

/// Abstract interface implemented by [GemmaChat] and [AgentChat].
export 'chat/base_chat.dart';

/// Stateful multi-turn chat with history, context management, and JSON output.
export 'chat/gemma_chat.dart';

/// History mutation + serialization mixin for custom chat wrappers.
export 'chat/chat_history_mixin.dart';

/// Stateful multi-loop agentic chat.
export 'gemma/agent_chat.dart';

/// Stateless single-turn wrapper — no history, ideal for batch / RAG pipelines.
export 'chat/single_turn_chat.dart';

// ── Thinking support ──────────────────────────────────────────────────────────

/// State-machine parser that strips and surfaces `<think>` reasoning blocks.
export 'gemma/thinking_parser.dart';

/// Extracted tool-call parser; supports JSON, native, and hybrid formats.
export 'chat/tool_call_parser.dart';

/// Lightweight AST parser for rendering `<think>` and `<|tool_call|>` UI blocks.
export 'chat/gemma_text_parser.dart';

// ── JSON Schema helpers ───────────────────────────────────────────────────────
export 'json_schema/schema.dart';
export 'json_schema/json_repair.dart';

// ── Embedding engine ──────────────────────────────────────────────────────────
export 'embedding/embedding_plugin.dart';

// ── PDF extraction ────────────────────────────────────────────────────────────
export 'pdf/pdf_processor.dart';

// ── Model installer ───────────────────────────────────────────────────────────
export 'model_installer/model_installer.dart';

// ── RAG helpers ───────────────────────────────────────────────────────────────
export 'helpers/document_embedder.dart';

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Centralised debug logger with per-module tags and native bridge.
export 'utils/gemma_debug.dart';

/// Multimodality mutex (vision/tools are mutually exclusive on Gemma 4).
export 'utils/multimodality_guard.dart';