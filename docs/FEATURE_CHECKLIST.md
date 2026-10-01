# Feature checklist — v3 (`amalbro_ds_1.py`) → v4

KEEP = same behaviour · IMPROVE = kept, works better · REPLACE = same goal, new
mechanism · REMOVE = dropped (with reason). Nothing user-visible was removed.

| v3 feature | Status | v4 implementation |
|---|---|---|
| PyMuPDF parsing | KEEP | `document_processor.parse_pdf` (`import pymupdf`, not deprecated `fitz`) |
| Section detection (regex headings) | IMPROVE | Font-size/bold aware headings with regex fallback; heading stack carried across pages (v3 reset per page) |
| Recursive chunking | IMPROVE | Same separators hierarchy, plus greedy packing (v3 produced one chunk per line at the newline level) and tiny-chunk merge |
| Chunk overlap | IMPROVE | One word-aligned overlap per chunk (v3 re-applied overlap at every recursion level) |
| Parent-context injection | KEEP | `with_context_prefix` — heading chain prefixed to embedded text |
| `chunk_idx`, `depth` metadata | KEEP | stored on `Chunk`, shown in source cards; depth penalty kept as bounded feature |
| Table extraction (full table node) | IMPROVE | Header fix for tables whose title sits above the grid; large tables split into parts; table text masked from body text (no duplicate hits) |
| Row / pin nodes | IMPROVE | "Header: value;" rows so every value keeps its column name and unit |
| Equation detection | IMPROVE | Stricter `is_equation_line` with neighbour context (v3 matched most spec lines) |
| Figure extraction | IMPROVE | Raster **and** vector plots (v3 missed drawn graphs); files namespaced per document (v3 overwrote across PDFs); geometric caption matching per figure (v3 reused the first caption) |
| Vision (llava) on demand | IMPROVE | Lazy, only for retrieved figures; result persisted and re-embedded (v3 lost it on restart); disabled with a notice where no vision model exists |
| Metadata extraction (voltage, current, temp, part no.) | KEEP + extended | `text_utils.extract_metadata`; used as retrieval feature and for value grounding |
| Semantic search (cosine) | REPLACE | FAISS per document, query embedded once (v3: 4×), nomic `search_query:`/`search_document:` prefixes |
| Fuzzy search (rapidfuzz) | IMPROVE | Only on short keys (symbols, part numbers, row labels) — whole-chunk `token_set_ratio` inflated scores |
| Section / context / metadata bonuses | IMPROVE | Kept as small bounded features after RRF fusion (scores no longer mixed across scales) |
| Per-engine thresholds | REPLACE | Single evidence gate (rerank score, dense score, lexical coverage) → NOT FOUND |
| Engine fan-out (text/table/row/equation/figure) | REPLACE | Modality-aware retrieval: query intent sets modality weights + extra recall for preferred modality |
| Mistral answer synthesis | KEEP | `llm.OllamaLLMEngine` (streamed); same model default |
| `[REF-N: TYPE pN]` citations | IMPROVE | `[REF-n]` validated after generation; invalid refs removed; grouped refs normalised |
| Hover citation popups | IMPROVE | Escaped HTML (v3 injected raw LLM output = XSS), works for any `[REF-n]` form, tap-to-open on mobile |
| Source cards / browse panel | KEEP | `ui.components.render_sources`, `render_browse` (text/tables/rows/equations/figures) |
| Metrics (faithfulness, coverage, diversity, avg score, completeness) | IMPROVE | Renamed "Response Quality Score" (not confidence); diversity over 6 types (v3 divided by 5); uncited sentences and unsupported values reported |
| Feedback (👍/👎 + correction override) | IMPROVE | JSONL store; override only for the same documents and the same content terms (v3 matched substrings across all documents) |
| Disk cache by md5 | IMPROVE | sha256 doc id + chunker signature + embedding namespace (v3 crashed on embed-model change); query-answer cache added |
| Sidebar settings | IMPROVE | No global mutation; admin-gated chunking changes trigger re-index |
| Dark theme CSS | KEEP | `ui/styles.py` |
| ThreadPool embedding | REPLACE | Batch embedding API (Ollama `embed`, fastembed batches) |
| Ollama localhost only | REPLACE | `OLLAMA_HOST` (+ auth header), OpenAI-compatible servers, or no LLM (extractive) |
| Single document | IMPROVE | Multi-document manager, scope selector, per-document balanced evidence |
| `feedback.json` / `.rag_cache` in cwd | REPLACE | `data/` and `cache/` via storage backend; optional private HF dataset mirror |
| Not-found handling | NEW | Evidence gate + exact "NOT FOUND IN DOCUMENT" |
| Security | NEW | Upload size/page limits, PDF header check, safe filenames, query sanitising, rate limiter, admin password, no secrets in code |
| Tests / benchmark | NEW | `tests/` (offline) and `benchmark.py` |
