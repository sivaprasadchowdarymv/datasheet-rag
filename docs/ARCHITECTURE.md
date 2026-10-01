# Architecture — Datasheet RAG Agent v4

## Data flow

```
                 ┌────────────── INGEST (once per PDF, cached by sha256) ─────────────┐
 PDF upload ─► security.validate_pdf ─► document_processor.parse_pdf
                 ├─ font-aware headings + heading stack across pages
                 ├─ chunker: recursive split → greedy packing → single overlap
                 ├─ tables (masked from text) → table parts + row/pin nodes
                 ├─ equations (context-checked)   ├─ figures (raster + vector) + captions
                 ▼
          storage/documents.DocumentRepository  (chunks.jsonl, record.json, figures/, pdf)
                 ▼
          embeddings.EmbeddingEngine (batched)  ─►  vector_store.FaissVectorStore
                                                     (one index per embedding model × doc)
                 └──── storage/backends: LocalStorage | HFHubStorage (mirror to private HF dataset)

                 ┌────────────── ANSWER (per question) ────────────────────────────────┐
 question ─► security.sanitize_query + RateLimiter
          ─► cache.FeedbackStore.find_override (human corrections, same docs only)
          ─► cache.QueryCache (key = question + docs + models + settings)
          ─► query_analysis.analyze  (intent → modality weights, symbols, follow-ups)
          ─► embed query ONCE
          ─► retriever.HybridRetriever   dense (FAISS) ‖ BM25 ‖ fuzzy (short keys)
                                         → weighted RRF + bounded v3 feature bonuses
          ─► reranker (cross-encoder top-20 → top-k)
          ─► vision (lazy, cached, local only) for figure candidates
          ─► evidence gate ──fail──► "NOT FOUND IN DOCUMENT"
          ─► _select_evidence (dedupe, modality quota, per-document balance)
          ─► generator.build_messages → llm (streamed)   | extractive_answer (no LLM)
          ─► generator.validate_answer (invalid refs removed, uncited sentences and
                                        unsupported values flagged)
          ─► evaluator (retrieval + answer metrics, Response Quality Score)
          ─► ui.components (escaped HTML, [REF-n] popups, sources, metrics)
```

## Files — purpose and connections

| File | Purpose | Used by / uses |
|---|---|---|
| `app.py` | Streamlit UI: sidebar (mode, admin, model, retrieval knobs, status), tabs Ask / Documents / Browse / Diagnostics, streaming, feedback | uses `config`, `rag.pipeline`, `ui.*`, `rag.security.RateLimiter` |
| `config.py` | Single source of settings: env > Streamlit secrets > `.env` > profile (`local`/`cloud`/`test`); masks secrets for display | read by everything; nothing else reads `os.environ` |
| `rag/schema.py` | Dataclasses `Chunk`, `DocumentRecord`, `Candidate`, `Source`, `AnswerResult` with (de)serialisation | shared vocabulary of all modules |
| `rag/text_utils.py` | token estimate, normalisation, technical tokenizer (keeps `V_OUT`, `±5%`, part numbers), metadata/value extraction | chunker, processor, BM25, generator, evaluator |
| `rag/chunker.py` | Recursive splitting with packing, tiny-chunk merge, one word-aligned overlap, parent-context prefix | `document_processor` |
| `rag/document_processor.py` | PDF → chunks (rejects encrypted / over-long PDFs): headings, text, tables, rows/pins, equations, figures + captions, OCR option, scanned-page warnings | `pipeline.ingest`; uses PyMuPDF, `chunker`, `text_utils` |
| `rag/embeddings.py` | Provider-independent embedding engines: Ollama (nomic prefixes, batch), FastEmbed (ONNX), SentenceTransformers, Hash (tests) | `pipeline`; namespace from `config.embedding_namespace` |
| `rag/vector_store.py` | FAISS `IndexFlatIP` per (namespace, doc): add / update / delete / search with document filter, atomic save | `pipeline`, `retriever` |
| `rag/keyword_index.py` | BM25 and fuzzy indexes over chunks (rapidfuzz) | `retriever`, `cache.FeedbackStore` |
| `rag/query_analysis.py` | Detects intent (pin/spec/equation/figure/compare), symbols, follow-up expansion → modality weights | `pipeline`, `retriever`, `generator` |
| `rag/retriever.py` | Hybrid retrieval: parallel channels, weighted RRF, section/context/metadata/symbol/modality features | `pipeline` |
| `rag/reranker.py` | Cross-encoder (fastembed) → lexical → none fallback chain | `pipeline` |
| `rag/llm.py` | LLM engines: Ollama (local/remote), OpenAI-compatible SSE (llama.cpp, vLLM, Ollama `/v1`), think-tag filter, concurrency semaphore, health | `pipeline` |
| `rag/vision.py` | Figure description via Ollama llava or OpenAI-compatible vision; result stored on the figure chunk (persisted, re-embedded) | `pipeline` (lazy) |
| `rag/generator.py` | System prompt, evidence blocks `[REF-n]`, extractive answers, answer validation | `pipeline` |
| `rag/evaluator.py` | Retrieval metrics, answer metrics, Response Quality Score (not "confidence") | `pipeline`, shown by `ui.components` |
| `rag/security.py` | PDF validation (header, size), safe filenames, query sanitising, rate limiter | `pipeline`, `app` |
| `rag/pipeline.py` | Orchestrator: `build_engines`, `RAGPipeline.ingest / reindex / delete / answer / health` | `app`, `benchmark`, tests |
| `storage/backends.py` | `LocalStorage`, `HFHubStorage` (pull on start, push after writes) | `pipeline` |
| `storage/documents.py` | Document registry, chunks, records, figures, original PDFs; path-safe resolution | `pipeline`, `ui` (figures) |
| `storage/cache.py` | `QueryCache` (memory LRU + disk) and `FeedbackStore` (JSONL, document-scoped overrides) | `pipeline` |
| `ui/styles.py` | Dark theme CSS, citation popups (hover on desktop, tap on mobile) | `app` |
| `ui/components.py` | Escaped answer HTML, NOT FOUND box, source cards, metrics, document stats, browse panel | `app` |
| `benchmark.py` | Measured v3 vs v4 comparison (timings, chunk stats, hit rates) | standalone |
| `tests/*` | Offline test-suite with synthetic datasheets; AppTest UI smoke test | `pytest` |
| `Dockerfile`, `docker-compose.yml`, `Caddyfile`, `scripts/*` | Self-hosting on an always-on VM (Ollama + app + optional HTTPS) | deployment |
| `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `.env.example`, `.gitignore` | Runtime/UI config, secret templates, keep secrets and data out of git | deployment |

## Storage layout

```
data/
  metadata/registry.json              list of documents
  metadata/<doc_id>/record.json       name, pages, stats, warnings, chunker signature
  metadata/<doc_id>/chunks.jsonl      all chunks (text/table/row/pin/equation/figure, incl. cached vision text)
  indexes/<namespace>/<doc_id>.faiss  vectors (+ <doc_id>.ids.json)
  documents/<doc_id>.pdf              original (optional)
  figures/<doc_id>/*.png              rendered figures
  feedback/feedback.jsonl             ratings and human corrections
cache/
  queries/*.json                      answer cache
```

`doc_id` = first 16 hex chars of the SHA-256 of the PDF bytes, so the same file is
never indexed twice and figures of different PDFs never overwrite each other.

## Key design decisions

* **Embed the query once** (v3 embedded it four times) and search all channels in parallel.
* **Fusion by rank (RRF)** instead of adding cosine, fuzzy and bonus points on
  different scales; v3's useful bonuses are kept as small bounded features.
* **Evidence gate before generation**: if the best evidence is too weak, the LLM is not
  called and NOT FOUND is returned — the main hallucination guard.
* **Validation after generation**: citations to non-existent refs are removed,
  numeric values not present in the cited evidence are flagged in the UI.
* **Namespaces per embedding model**: switching models never mixes vector dimensions.
* **Graceful degradation**: missing reranker / LLM / vision downgrades with a visible
  warning; the app keeps answering (extractive) instead of crashing.
