"""
rag/pipeline.py — the only object the UI talks to.

INGESTION (once per PDF):
  validate → hash → already indexed? → parse (document_processor) → store chunks
  → batch-embed → FAISS index → registry → optional cloud mirror

QUERY (every question):
  feedback override? → query cache? → analyze query → embed query ONCE
  → hybrid retrieval (dense + BM25 + fuzzy, RRF) → rerank → lazy vision
  → EVIDENCE GATE (low evidence ⇒ "NOT FOUND IN DOCUMENT", LLM never called)
  → evidence selection (dedupe, modality quota, per-document balance)
  → generation (stream) or extractive → citation validation → metrics → cache

The pipeline never imports Streamlit; it is fully testable and reusable from a
CLI or another UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from storage.backends import StorageBackend, LocalStorage
from storage.cache import FeedbackStore, QueryCache
from storage.documents import DocumentRepository, compute_doc_id

from .document_processor import ParseOptions, PDFProcessingError, parse_pdf
from .embeddings import EmbeddingEngine, EmbeddingError, create_embedding_engine
from .evaluator import answer_metrics, retrieval_metrics
from .generator import NOT_FOUND, build_evidence, build_messages, extractive_answer, validate_answer
from .keyword_index import BM25Index, FuzzyIndex
from .llm import LLMEngine, LLMError, create_llm_engine
from .query_analysis import QueryAnalysis, analyze_query
from .reranker import Reranker, create_reranker, key_term_coverage
from .retriever import HybridRetriever
from .schema import AnswerResult, Candidate, Chunk, DocumentRecord, Source
from .security import ValidationError, safe_filename, validate_pdf
from .text_utils import tokenize
from .vector_store import FaissVectorStore, VectorIndexError
from .vision import VisionEngine, create_vision_engine

ProgressFn = Callable[[float, str], None]
TokenFn = Callable[[str], None]


class IngestionError(Exception):
    """Human-readable ingestion failure."""


_DEFAULT_LLM = object()   # sentinel: "use the pipeline's own LLM"


@dataclass
class Engines:
    embedder: EmbeddingEngine
    llm: Optional[LLMEngine]
    reranker: Reranker
    vision: Optional[VisionEngine]
    warnings: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, str]:
        return {
            "embedding": self.embedder.name,
            "llm": self.llm.name if self.llm else "extractive (no LLM)",
            "reranker": getattr(self.reranker, "model", self.reranker.name),
            "vision": self.vision.name if self.vision else "disabled",
        }


def build_llm(settings) -> Tuple[Optional[LLMEngine], str]:
    """The answer model alone (cheap to create), so switching models never reloads embeddings."""
    try:
        return create_llm_engine(settings), ""
    except LLMError as e:
        return None, f"{e} Falling back to extractive answers."


def build_engines(settings, include_llm: bool = True) -> Engines:
    warnings: List[str] = []
    embedder = create_embedding_engine(settings)          # failures here are fatal → shown in UI
    llm = None
    if include_llm:
        llm, w = build_llm(settings)
        if w:
            warnings.append(w)
    reranker, w = create_reranker(settings)
    if w:
        warnings.append(w)
    try:
        vision = create_vision_engine(settings)
    except Exception as e:  # noqa: BLE001
        vision = None
        warnings.append(f"Vision disabled: {e}")
    return Engines(embedder, llm, reranker, vision, warnings)


class RAGPipeline:
    def __init__(self, settings, engines: Engines, storage: Optional[StorageBackend] = None):
        self.s = settings
        self.e = engines
        self.data_dir = Path(settings.data_dir)
        self.storage = storage or LocalStorage(self.data_dir)
        self.repo = DocumentRepository(self.data_dir)
        self.namespace = settings.embedding_namespace
        self.store = FaissVectorStore(self.data_dir / "indexes", self.namespace)
        self.cache = QueryCache(Path(settings.cache_dir))
        self.feedback = FeedbackStore(self.data_dir)
        self._lexical: Dict[Tuple[str, ...], Tuple[Dict[str, Chunk], BM25Index, FuzzyIndex]] = {}

    # ================================================================ docs
    def documents(self) -> List[DocumentRecord]:
        return self.repo.list()

    def is_stale(self, rec: DocumentRecord) -> bool:
        return rec.chunker_signature != self.s.chunker_signature

    def _parse_opts(self) -> ParseOptions:
        s = self.s
        return ParseOptions(s.max_chunk_tokens, s.overlap_tokens, s.min_chunk_tokens, s.exclude_table_text,
                            s.enable_ocr, s.max_figures_per_page, s.max_pages)

    def ingest(self, pdf_bytes: bytes, filename: str, progress: Optional[ProgressFn] = None,
               force: bool = False) -> Tuple[DocumentRecord, str]:
        """Returns (record, status) where status ∈ {'new', 'cached', 'vectors-rebuilt', 'reindexed'}."""
        progress = progress or (lambda f, m: None)
        try:
            validate_pdf(pdf_bytes, self.s.max_upload_mb)
        except ValidationError as e:
            raise IngestionError(str(e)) from e
        doc_id = compute_doc_id(pdf_bytes)
        existing = self.repo.get(doc_id)
        if existing and not force and not self.is_stale(existing) and self.repo.load_chunks(doc_id):
            status = "cached"
            if not self.store.has_document(doc_id):
                self._embed_and_index(doc_id, self.repo.load_chunks(doc_id), progress)
                status = "vectors-rebuilt"
                self._register_namespace(existing)
            return existing, status

        name = safe_filename(filename)
        t0 = time.time()
        try:
            parsed = parse_pdf(
                pdf_bytes, doc_id, name, self.repo.figures_dir(doc_id), f"figures/{doc_id}", self._parse_opts(),
                progress=lambda f, m: progress(0.6 * f, m),
            )
        except PDFProcessingError as e:
            raise IngestionError(str(e)) from e
        t_parse = time.time() - t0
        self.repo.save_chunks(doc_id, parsed.chunks)
        if self.s.keep_original_pdfs:
            self.repo.save_pdf(doc_id, pdf_bytes)
        t1 = time.time()
        self._embed_and_index(doc_id, parsed.chunks, lambda f, m: progress(0.6 + 0.38 * f, m))
        t_embed = time.time() - t1
        rec = DocumentRecord(
            doc_id=doc_id, name=name, pages=parsed.page_count, size_bytes=len(pdf_bytes),
            ingested_at=time.time(), chunker_signature=self.s.chunker_signature,
            stats={**parsed.stats, "parse_ms": int(t_parse * 1000), "embed_ms": int(t_embed * 1000)},
            warnings=parsed.warnings, namespaces=[self.namespace], has_pdf=self.s.keep_original_pdfs,
        )
        self.repo.save_record(rec)
        self._invalidate(doc_id)
        progress(0.99, "Backing up")
        w = self.storage.push(self.repo.rel_paths_for(doc_id, self.namespace))
        if w:
            rec.warnings.append(w)
        progress(1.0, "Done")
        return rec, ("reindexed" if existing else "new")

    def _register_namespace(self, rec: DocumentRecord) -> None:
        if self.namespace not in rec.namespaces:
            rec.namespaces.append(self.namespace)
            self.repo.save_record(rec)

    def _embed_and_index(self, doc_id: str, chunks: List[Chunk], progress: ProgressFn) -> None:
        texts = [c.search_text for c in chunks]
        try:
            vecs = self.e.embedder.embed_documents(texts, progress=lambda f, m: progress(f, f"Embedding {len(texts)} chunks"))
        except EmbeddingError as e:
            raise IngestionError(str(e)) from e
        self.store.add_document(doc_id, [c.chunk_id for c in chunks], vecs)

    def ensure_vectors(self, doc_ids: Sequence[str], progress: Optional[ProgressFn] = None) -> List[str]:
        """Make sure every selected document has vectors for the CURRENT embedding model.
        Re-embeds from stored chunks (no re-parsing) when vectors are missing or corrupted."""
        rebuilt = []
        for d in doc_ids:
            ok = self.store.has_document(d)
            if ok:
                try:
                    self.store.dim(d)
                except VectorIndexError:
                    ok = False
            if not ok:
                chunks = self.repo.load_chunks(d)
                if chunks:
                    self._embed_and_index(d, chunks, progress or (lambda f, m: None))
                    rec = self.repo.get(d)
                    if rec:
                        self._register_namespace(rec)
                    self.storage.push(self.repo.rel_paths_for(d, self.namespace)[-2:])
                    rebuilt.append(d)
        return rebuilt

    def reindex(self, doc_id: str, progress: Optional[ProgressFn] = None) -> DocumentRecord:
        pdf = self.repo.load_pdf(doc_id)
        rec = self.repo.get(doc_id)
        if pdf is None or rec is None:
            raise IngestionError("The original PDF is not stored (KEEP_ORIGINAL_PDFS=false). Please upload it again.")
        self.store.delete_document(doc_id)
        new_rec, _ = self.ingest(pdf, rec.name, progress, force=True)
        return new_rec

    def delete(self, doc_id: str) -> str:
        rels = self.repo.rel_paths_for(doc_id, self.namespace)
        self.store.delete_document(doc_id)
        self.repo.delete(doc_id)
        self._invalidate(doc_id)
        self.cache.clear()
        return self.storage.remove([r for r in rels if r != "metadata/registry.json"]) or self.storage.push(["metadata/registry.json"])

    def _invalidate(self, doc_id: str) -> None:
        for k in [k for k in self._lexical if doc_id in k]:
            self._lexical.pop(k, None)

    def _lexical_for(self, doc_ids: Sequence[str]):
        key = tuple(sorted(doc_ids))
        if key not in self._lexical:
            chunks: Dict[str, Chunk] = {}
            for d in key:
                for c in self.repo.load_chunks(d):
                    chunks[c.chunk_id] = c
            lst = list(chunks.values())
            self._lexical[key] = (chunks, BM25Index(lst), FuzzyIndex(lst))
        return self._lexical[key]

    # ================================================================ query
    def fingerprint(self, llm: Optional[LLMEngine] = None) -> str:
        s = self.s
        return "|".join(map(str, [
            s.chunker_signature, self.namespace, llm.name if llm else "extractive",
            self.e.reranker.name, s.top_k_retrieval, s.top_k_fusion, s.top_k_final, s.rerank_min_score,
            s.dense_min_score, s.lexical_min_coverage, self.e.vision.name if self.e.vision else "novision",
        ]))

    def answer(self, question: str, doc_ids: Sequence[str], history: Optional[List[str]] = None,
               on_token: Optional[TokenFn] = None, on_status: Optional[Callable[[str], None]] = None,
               use_cache: bool = True, llm: Any = _DEFAULT_LLM) -> AnswerResult:
        """llm: the answer model for this question (None = extractive). Omitted → the
        pipeline's own engine. Passing it per call lets every user pick a model without
        duplicating the loaded indexes."""
        llm = self.e.llm if llm is _DEFAULT_LLM else llm
        status = on_status or (lambda m: None)
        history = history or []
        T: Dict[str, float] = {}
        t_start = time.perf_counter()
        doc_ids = [d for d in doc_ids if self.repo.get(d)]
        if not doc_ids:
            return AnswerResult("error", "Upload or select at least one document first.")

        # 1. human override (feedback) ------------------------------------------------
        fb = self.feedback.find_override(question, doc_ids)
        if fb:
            return AnswerResult("feedback_override", fb["correction"],
                                warnings=["Showing a human-corrected answer saved earlier for this question."],
                                timings={"total": round(time.perf_counter() - t_start, 3)})

        # 2. query cache --------------------------------------------------------------
        key = QueryCache.make_key(question, doc_ids, self.fingerprint(llm), history)
        if use_cache and self.s.enable_query_cache:
            hit = self.cache.get(key)
            if hit:
                res = AnswerResult.from_dict(hit)
                res.from_cache = True
                res.timings = {**res.timings, "total_cached": round(time.perf_counter() - t_start, 4)}
                if on_token:
                    on_token(res.answer)
                return res

        # 3. analyse + embed query ----------------------------------------------------
        warnings: List[str] = []
        qa = analyze_query(question, history)
        status("Searching documents…")
        t = time.perf_counter()
        try:
            self.ensure_vectors(doc_ids)
        except IngestionError as e:
            warnings.append(f"Vector index unavailable ({e}); using keyword search only.")
        T["index_check"] = time.perf_counter() - t

        t = time.perf_counter()
        qvec = None
        try:
            qvec = self.e.embedder.embed_query(qa.retrieval_query)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Embedding the question failed ({e}); using keyword search only.")
        T["query_embedding"] = time.perf_counter() - t

        # 4. hybrid retrieval ---------------------------------------------------------
        t = time.perf_counter()
        chunks, bm25, fuzzy = self._lexical_for(doc_ids)
        retriever = HybridRetriever(chunks, bm25, fuzzy, self.store,
                                    [d for d in doc_ids if self.store.has_document(d)],
                                    self.s.top_k_retrieval, self.s.rrf_k)
        try:
            cands = retriever.retrieve(qa, qvec, self.s.top_k_fusion)
        except VectorIndexError as e:
            warnings.append(f"{e} Using keyword search only.")
            cands = retriever.retrieve(qa, None, self.s.top_k_fusion)
        T["retrieval"] = time.perf_counter() - t

        # 5. rerank -------------------------------------------------------------------
        t = time.perf_counter()
        try:
            ranked = self.e.reranker.rerank(qa, cands)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Reranker failed ({e}); using fusion order.")
            from .reranker import NoReranker
            ranked = NoReranker().rerank(qa, cands)
        T["rerank"] = time.perf_counter() - t

        # 6. lazy vision --------------------------------------------------------------
        t = time.perf_counter()
        if self.e.vision and ranked:
            if self._run_vision(qa, ranked, status, warnings):
                ranked = self.e.reranker.rerank(qa, ranked)
        T["vision"] = time.perf_counter() - t

        diag = {
            "query_analysis": {
                "retrieval_query": qa.retrieval_query, "intents": qa.intents,
                "modality_weights": qa.modality_weights, "key_terms": qa.key_terms, "symbols": qa.symbols,
            },
            "candidates": [c.diag() for c in ranked],
        }

        # 7. evidence gate ------------------------------------------------------------
        ok, gate_reason = self._evidence_gate(qa, ranked)
        diag["gate"] = gate_reason
        if not ok:
            T["total"] = time.perf_counter() - t_start
            res = AnswerResult(
                "not_found", NOT_FOUND, timings={k: round(v, 4) for k, v in T.items()}, diagnostics=diag,
                warnings=warnings, retrieval_metrics=retrieval_metrics(qa, [], ranked, []),
            )
            if on_token:
                on_token(NOT_FOUND)
            if self.s.enable_query_cache and not warnings:
                self.cache.put(key, res.to_dict())
            return res

        # 8. evidence selection -------------------------------------------------------
        used = self._select_evidence(qa, ranked)
        blocks = build_evidence([c.chunk for c in used])

        # 9. generation ---------------------------------------------------------------
        t = time.perf_counter()
        answer = ""
        first_token_at: Optional[float] = None
        if llm:
            status("Generating…")
            msgs = build_messages(question, blocks, history)
            try:
                parts: List[str] = []
                for delta in llm.stream(msgs):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(delta)
                    if on_token:
                        on_token(delta)
                answer = "".join(parts).strip()
            except LLMError as e:
                warnings.append(f"{e} Showing extractive evidence instead.")
                answer = ""
        if not answer:
            answer = extractive_answer(qa, blocks)
            if on_token:
                on_token(answer)
        T["generation"] = time.perf_counter() - t
        if first_token_at:
            T["time_to_first_token"] = first_token_at - t_start

        # 10. validation + metrics ----------------------------------------------------
        v = validate_answer(answer, blocks)
        if v.invalid_refs:
            warnings.append(f"Removed citation(s) to non-existent evidence: {', '.join(f'REF-{n}' for n in v.invalid_refs)}.")
        if not v.is_not_found and not v.cited_refs:
            warnings.append("The model gave no citations. Treat this answer with caution and check the sources below.")
        if v.unsupported_values:
            warnings.append("Values not found verbatim in the cited evidence: " + ", ".join(v.unsupported_values[:8]))

        sources = [
            Source(ref=b.ref, chunk_id=b.chunk.chunk_id, doc_id=b.chunk.doc_id, doc_name=b.chunk.doc_name,
                   page=b.chunk.page, section=b.chunk.section, parent_ctx=b.chunk.parent_ctx,
                   modality=b.chunk.modality,
                   snippet=(b.chunk.raw_content + (f"\n\n[vision] {b.chunk.vision_text}" if b.chunk.vision_text else ""))[:900],
                   score=round(c.rerank or 0.0, 4), chunk_idx=b.chunk.chunk_idx, depth=b.chunk.depth,
                   meta={k: v2 for k, v2 in (b.chunk.meta or {}).items() if k not in ("bbox",)},
                   figure_path=b.chunk.figure_path)
            for b, c in zip(blocks, used)
        ]
        T["total"] = time.perf_counter() - t_start
        res = AnswerResult(
            kind="not_found" if v.is_not_found else "answer", answer=v.answer, sources=sources,
            retrieval_metrics=retrieval_metrics(qa, used, ranked, v.cited_refs),
            answer_metrics=answer_metrics(qa, v, blocks),
            timings={k: round(val, 4) for k, val in T.items()}, diagnostics=diag, warnings=warnings,
        )
        if self.s.enable_query_cache and not any("fail" in w.lower() or "unavailable" in w.lower() for w in warnings):
            self.cache.put(key, res.to_dict())
        return res

    # ---------------------------------------------------------------------------------
    def _evidence_gate(self, qa: QueryAnalysis, ranked: List[Candidate]) -> Tuple[bool, str]:
        if not ranked:
            return False, "no candidates retrieved"
        top = ranked[0]
        cov = key_term_coverage(qa, top.chunk.search_text)
        if any(c.features.get("symbol", 0) >= 0.12 for c in ranked[:5]) and cov >= 0.3:
            return True, "exact symbol / part-number match"
        if self.e.reranker.is_model:
            best = max(c.rerank or 0 for c in ranked)
            if best >= self.s.rerank_min_score:
                return True, f"reranker score {best:.3f} ≥ {self.s.rerank_min_score}"
            return False, f"best reranker score {best:.3f} < {self.s.rerank_min_score}"
        best_dense = max((c.dense or 0) for c in ranked)
        best_cov = max(key_term_coverage(qa, c.chunk.search_text) for c in ranked[:10])
        if best_dense >= self.s.dense_min_score:
            return True, f"dense similarity {best_dense:.3f} ≥ {self.s.dense_min_score}"
        if best_cov >= self.s.lexical_min_coverage:
            return True, f"key-term coverage {best_cov:.2f} ≥ {self.s.lexical_min_coverage}"
        return False, f"dense {best_dense:.3f} and key-term coverage {best_cov:.2f} below thresholds"

    def _select_evidence(self, qa: QueryAnalysis, ranked: List[Candidate]) -> List[Candidate]:
        k = self.s.top_k_final
        top = ranked[0].rerank or 0.0
        floor = (self.s.rerank_min_score * 0.5) if self.e.reranker.is_model else 0.0
        pool = [c for c in ranked if (c.rerank or 0) >= max(floor, top * 0.15)] or ranked[:1]

        chosen: List[Candidate] = []
        seen_sig = set()

        def sig(c: Candidate):
            toks = tokenize(c.chunk.raw_content)[:40]
            return (c.chunk.doc_id, c.chunk.page, " ".join(toks[:25]))

        def take(c: Candidate) -> bool:
            s = sig(c)
            if s in seen_sig or any(x is c for x in chosen):
                return False
            # skip a row whose whole table part is already chosen (and vice-versa it's fine)
            tid = (c.chunk.meta or {}).get("table_id")
            if c.chunk.modality in ("row", "pin") and any(
                x.chunk.modality == "table" and (x.chunk.meta or {}).get("table_id") == tid
                and x.chunk.doc_id == c.chunk.doc_id and c.chunk.raw_content.split("\n")[-1] in x.chunk.raw_content
                for x in chosen
            ):
                return False
            seen_sig.add(s)
            chosen.append(c)
            return True

        # per-document balance for comparison questions
        if qa.is_comparison:
            docs = list(dict.fromkeys(c.chunk.doc_id for c in pool))
            for d in docs:
                for c in pool:
                    if c.chunk.doc_id == d and take(c):
                        break
        # modality quota: at least one block of each preferred modality, if decent
        for m in qa.preferred_modalities():
            for c in pool[:15]:
                if c.chunk.modality == m and (c.rerank or 0) >= top * 0.3:
                    take(c)
                    break
        for c in pool:
            if len(chosen) >= k:
                break
            take(c)
        chosen = chosen[:k]
        chosen.sort(key=lambda c: c.final, reverse=True)
        return chosen

    def _run_vision(self, qa: QueryAnalysis, ranked: List[Candidate], status, warnings) -> bool:
        """Describe the top relevant figure(s) once; cache + re-embed. Returns True if anything changed."""
        visual = "figure" in qa.intents
        figs = [c for c in ranked[:8] if c.chunk.modality == "figure"]
        if not figs or not (visual or figs[0] is ranked[0]):
            return False
        changed = False
        model = self.e.vision.name
        for c in figs[: self.s.max_vision_figures]:
            ch = c.chunk
            if ch.vision_text and ch.vision_model == model:
                continue
            path = self.repo.resolve(ch.figure_path)
            if not path:
                continue
            status(f"Analysing figure on page {ch.page}…")
            try:
                ch.vision_text = self.e.vision.describe(path, ch.caption or "")[:2000]
                ch.vision_model = model
                self.repo.update_chunk(ch)
                vec = self.e.embedder.embed_documents([ch.search_text])[0]
                self.store.update_vector(ch.doc_id, ch.chunk_id, vec)
                self._invalidate(ch.doc_id)
                self.storage.push([f"metadata/{ch.doc_id}"])
                changed = True
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Vision analysis failed for the figure on page {ch.page}: {e}")
        return changed

    # ================================================================ health
    def health(self, llm: Any = _DEFAULT_LLM) -> Dict[str, str]:
        llm = self.e.llm if llm is _DEFAULT_LLM else llm
        out = {"embedding": self.e.embedder.health()}
        out["llm"] = llm.health() if llm else ""
        out["storage"] = self.storage.last_error
        return out
