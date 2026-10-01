"""
rag/reranker.py — second stage: score (query, chunk) pairs precisely.

    Reranker
    ├── CrossEncoderReranker  ms-marco-MiniLM-L-6-v2 via FastEmbed/ONNX (22M params,
    │                         ~80 MB, ~20-60 ms for 20 pairs on 2 CPU cores)
    ├── LexicalReranker       no model: dense cosine + key-term coverage + exact symbols
    └── NoReranker            keep fusion order

All rerankers write `candidate.rerank` in 0..1 and `candidate.final`.
The cross-encoder logit is squashed with a sigmoid; it is NOT a calibrated
probability, it is only used for ordering and for the evidence gate.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

from .query_analysis import QueryAnalysis
from .schema import Candidate
from .text_utils import tokenize


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def key_term_coverage(qa: QueryAnalysis, text: str) -> float:
    terms = [t for t in qa.key_terms if len(t) > 1]
    if not terms:
        return 0.0
    toks = set(tokenize(text))
    return sum(1 for t in terms if t in toks) / len(terms)


class Reranker(ABC):
    name = "base"
    is_model = False

    @abstractmethod
    def rerank(self, qa: QueryAnalysis, cands: List[Candidate]) -> List[Candidate]: ...

    def _finish(self, qa: QueryAnalysis, cands: List[Candidate]) -> List[Candidate]:
        for c in cands:
            # keep a small share of modality intent and exact-symbol evidence
            bonus = 0.08 * max(0.0, qa.weight(c.chunk.modality) - 1.0) + 0.5 * c.features.get("symbol", 0.0)
            c.final = (c.rerank or 0.0) + bonus
        return sorted(cands, key=lambda c: c.final, reverse=True)

    def health(self) -> str:
        return ""


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"
    is_model = True

    def __init__(self, model: str, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = model
        self._m = TextCrossEncoder(model_name=model, cache_dir=cache_dir)

    def rerank(self, qa: QueryAnalysis, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        docs = [c.chunk.search_text[:1500] for c in cands]
        scores = list(self._m.rerank(qa.retrieval_query, docs, batch_size=32))
        for c, s in zip(cands, scores):
            c.rerank = _sigmoid(float(s))
        return self._finish(qa, cands)


class LexicalReranker(Reranker):
    name = "lexical"

    def rerank(self, qa: QueryAnalysis, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        top_fused = max(c.fused for c in cands) or 1.0
        for c in cands:
            cov = key_term_coverage(qa, c.chunk.search_text)
            dense = max(0.0, c.dense or 0.0)
            c.rerank = 0.45 * dense + 0.35 * cov + 0.20 * (c.fused / top_fused)
        return self._finish(qa, cands)


class NoReranker(Reranker):
    name = "none"

    def rerank(self, qa: QueryAnalysis, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        top = max(c.fused for c in cands) or 1.0
        for c in cands:
            c.rerank = c.fused / top
        return self._finish(qa, cands)


def create_reranker(settings) -> tuple[Reranker, str]:
    """Returns (reranker, warning). Falls back to lexical if the model cannot load."""
    if not settings.enable_reranker or settings.reranker_provider == "none":
        return NoReranker(), ""
    if settings.reranker_provider == "lexical":
        return LexicalReranker(), ""
    if settings.reranker_provider in ("fastembed", "cross_encoder", "cross-encoder"):
        try:
            return CrossEncoderReranker(settings.reranker_model, cache_dir=f"{settings.cache_dir}/models"), ""
        except Exception as e:  # noqa: BLE001
            return LexicalReranker(), f"Cross-encoder reranker unavailable ({e}); using lexical reranking."
    return LexicalReranker(), f"Unknown RERANKER_PROVIDER '{settings.reranker_provider}'; using lexical."
