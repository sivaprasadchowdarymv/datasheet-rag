"""
rag/retriever.py — first-stage retrieval: wide, cheap, high-recall.

    query ─► QueryAnalysis
            ├─► dense   (FAISS cosine, query embedded ONCE)      ┐
            ├─► BM25    (technical tokens)                       ├─► weighted RRF fusion
            └─► fuzzy   (sections / breadcrumbs / part numbers)  ┘        │
                                                                            ▼
                   v3 feature bonuses (bounded): section match, parent-context
                   match, metadata match, modality intent, depth preference
                                                                            │
                                                                            ▼
                                                              top-K_FUSION candidates

Why RRF instead of v3's `max(semantic, fuzzy) + bonuses`?  Cosine (0..1),
BM25 (unbounded) and fuzzy (0..100) live on different scales; adding them
lets one channel dominate arbitrarily. Reciprocal-rank fusion only uses ranks,
so each channel contributes fairly and no per-channel threshold tuning is
needed. v3's useful heuristics survive as small multiplicative features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

from .keyword_index import BM25Index, FuzzyIndex, partial_ratio
from .query_analysis import QueryAnalysis
from .schema import Candidate, Chunk
from .vector_store import FaissVectorStore

CHANNEL_WEIGHTS = {"dense": 1.0, "bm25": 0.9, "fuzzy": 0.45}


class HybridRetriever:
    def __init__(
        self,
        chunks: Dict[str, Chunk],
        bm25: BM25Index,
        fuzzy: FuzzyIndex,
        store: FaissVectorStore,
        doc_ids: Sequence[str],
        top_k: int = 20,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.fuzzy = fuzzy
        self.store = store
        self.doc_ids = list(doc_ids)
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        qa: QueryAnalysis,
        query_vec: Optional[np.ndarray],
        k_out: int,
        modalities: Optional[Sequence[str]] = None,
    ) -> List[Candidate]:
        allowed = set(modalities) if modalities else None
        keep = (lambda cid: self.chunks[cid].modality in allowed) if allowed else None
        q = qa.retrieval_query
        k = self.top_k

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_dense = ex.submit(self.store.search, query_vec, k, self.doc_ids, keep) if query_vec is not None else None
            f_bm25 = ex.submit(self.bm25.search, q, k, keep)
            f_fuzzy = ex.submit(self.fuzzy.search, q, k, keep)
            channels = {
                "dense": f_dense.result() if f_dense else [],
                "bm25": f_bm25.result(),
                "fuzzy": f_fuzzy.result(),
            }

        # Modality-aware recall: guarantee a few candidates of each preferred
        # modality even when plain text dominates the global ranking.
        if allowed is None and query_vec is not None:
            for m in qa.preferred_modalities():
                extra = self.store.search(query_vec, 4, self.doc_ids, lambda cid, m=m: self.chunks[cid].modality == m)
                channels.setdefault(f"dense_{m}", extra)

        cands: Dict[str, Candidate] = {}
        bm25_max = self.bm25.max_possible(q)
        for ch, hits in channels.items():
            w = CHANNEL_WEIGHTS.get(ch, 0.6)
            for rank, (cid, score) in enumerate(hits):
                if cid not in self.chunks:
                    continue
                c = cands.setdefault(cid, Candidate(chunk=self.chunks[cid]))
                c.ranks[ch] = rank + 1
                c.fused += w / (self.rrf_k + rank + 1)
                if ch.startswith("dense"):
                    c.dense = max(c.dense or -1.0, score)
                elif ch == "bm25":
                    c.bm25 = min(1.0, score / bm25_max)
                elif ch == "fuzzy":
                    c.fuzzy = score

        # Dense score for lexical-only hits (needed by the evidence gate / diagnostics)
        if query_vec is not None:
            missing = [c for c in cands.values() if c.dense is None]
            if missing:
                by_doc: Dict[str, List[Candidate]] = {}
                for c in missing:
                    by_doc.setdefault(c.chunk.doc_id, []).append(c)
                for doc_id, cs in by_doc.items():
                    try:
                        index, ids = self.store._load(doc_id)
                        pos = {cid: i for i, cid in enumerate(ids)}
                        for c in cs:
                            if c.chunk.chunk_id in pos:
                                v = index.reconstruct(pos[c.chunk.chunk_id])
                                c.dense = float(np.dot(v, query_vec))
                    except Exception:
                        pass

        for c in cands.values():
            self._apply_features(c, qa)
        ranked = sorted(cands.values(), key=lambda c: c.fused, reverse=True)
        return ranked[:k_out]

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_features(c: Candidate, qa: QueryAnalysis) -> None:
        ch = c.chunk
        ql = qa.retrieval_query.lower()
        words = [w for w in qa.key_terms if len(w) > 3]
        f: Dict[str, float] = {}

        # v3 "section heading match bonus"
        if ch.section and (ch.section.lower() in ql or any(w in ch.section.lower() for w in words)):
            f["section"] = 0.10
        # v3 "parent context match bonus"
        if ch.parent_ctx and any(w in ch.parent_ctx.lower() for w in words):
            f["parent_ctx"] = 0.06
        # v3 "metadata keyword bonus" — now bounded, and exact for symbols/part numbers
        meta_vals = " ".join(
            " ".join(map(str, v)) if isinstance(v, list) else str(v) for v in (ch.meta or {}).values()
        ).lower()
        if meta_vals and any(w in meta_vals for w in words):
            f["meta"] = 0.06
        if qa.symbols:
            hay = (ch.search_text + " " + meta_vals).lower()
            exact = sum(1 for s in qa.symbols if s.lower() in hay)
            if exact:
                f["symbol"] = min(0.25, 0.12 * exact)
            elif any(partial_ratio(s.lower(), hay) >= 90 for s in qa.symbols if len(s) >= 5):
                f["symbol"] = 0.06
        # modality intent
        mw = qa.weight(ch.modality)
        if mw != 1.0:
            f["modality"] = mw - 1.0
        # v3 "depth penalty" — prefer structurally complete chunks, gently
        if ch.depth >= 5:
            f["depth"] = -0.04

        c.features = f
        c.fused *= (1.0 + sum(f.values()))
