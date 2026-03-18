library flutter_local_gemma;

// ── Core types ────────────────────────────────────────────────────────────────
export 'types/content_parts.dart';

// ── LLM engine ────────────────────────────────────────────────────────────────
export 'gemma/gemma.dart';

// ── Chat wrappers ─────────────────────────────────────────────────────────────

/// Stateful multi-turn chat with history, context management, and JSON output.
export 'chat/gemma_chat.dart';

/// Stateless single-turn wrapper — no history, ideal for batch / RAG pipelines.
export 'chat/single_turn_chat.dart';

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