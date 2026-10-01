"""
rag/evaluator.py — metrics, split into RETRIEVAL and ANSWER groups.

None of these are calibrated probabilities. The headline number is therefore
called "Response Quality Score (heuristic)", not "confidence" as in v3.

Retrieval metrics
  top_score            best reranker score (0..1)
  mean_topk_score      mean reranker score of the evidence given to the LLM
  score_margin         top − 2nd score (low margin = ambiguous evidence)
  source_diversity     distinct modalities / documents / pages in the evidence
  key_term_recall      fraction of question key terms present in the evidence
  precision_at_k       fraction of evidence blocks the answer actually cited
                       (a usage-based proxy; true precision needs labelled data —
                       see benchmark.py for labelled evaluation)

Answer metrics
  citation_coverage    factual sentences carrying ≥1 valid citation
  citation_validity    valid / all [REF-n] the model wrote (fabricated refs lower it)
  numeric_grounding    values-with-units in the answer that appear in evidence
  faithfulness_lexical content words of the answer found in cited evidence
  completeness         question key terms addressed by the answer
  groundedness         combination of the above three grounding signals
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .generator import EvidenceBlock, Validation
from .query_analysis import QueryAnalysis
from .schema import Candidate
from .text_utils import content_words, numbers_with_units, tokenize


def retrieval_metrics(qa: QueryAnalysis, used: Sequence[Candidate], all_cands: Sequence[Candidate],
                      cited: Sequence[int]) -> Dict[str, Any]:
    scores = [c.rerank or 0.0 for c in used]
    ordered = sorted((c.rerank or 0.0 for c in all_cands), reverse=True)
    ev_text = " ".join(c.chunk.search_text for c in used)
    ev_tokens = set(tokenize(ev_text))
    terms = [t for t in qa.key_terms if len(t) > 1]
    return {
        "top_score": round(ordered[0], 4) if ordered else 0.0,
        "mean_topk_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "score_margin": round(ordered[0] - ordered[1], 4) if len(ordered) > 1 else (round(ordered[0], 4) if ordered else 0.0),
        "modalities": sorted({c.chunk.modality for c in used}),
        "documents": len({c.chunk.doc_id for c in used}),
        "pages": len({(c.chunk.doc_id, c.chunk.page) for c in used}),
        "source_diversity": round(len({c.chunk.modality for c in used}) / 6.0, 3),
        "key_term_recall": round(sum(1 for t in terms if t in ev_tokens) / len(terms), 3) if terms else 0.0,
        "precision_at_k": round(len(set(cited)) / len(used), 3) if used else 0.0,
        "candidates_considered": len(all_cands),
    }


def answer_metrics(qa: QueryAnalysis, v: Validation, blocks: Sequence[EvidenceBlock]) -> Dict[str, Any]:
    if v.is_not_found:
        return {"not_found": True, "response_quality_score": None}
    answer_body = v.answer
    total_refs = len(v.cited_refs) + len(v.invalid_refs)
    citation_validity = (len(v.cited_refs) / total_refs) if total_refs else 0.0
    citation_coverage = (1 - v.uncited_sentences / v.total_sentences) if v.total_sentences else (1.0 if v.cited_refs else 0.0)

    vals = numbers_with_units(answer_body)
    numeric_grounding = 1.0 if not vals else max(0.0, 1 - len(v.unsupported_values) / len(set(vals)))

    cited_text = " ".join(b.text for b in blocks if b.ref in v.cited_refs) or " ".join(b.text for b in blocks)
    ctx_words = content_words(cited_text)
    ans_words = content_words(answer_body) - {"ref", "interpretation", "not", "found", "document"}
    faithfulness = len(ans_words & ctx_words) / len(ans_words) if ans_words else 0.0

    terms = [t for t in qa.key_terms if len(t) > 1]
    ans_tokens = set(tokenize(answer_body))
    completeness = sum(1 for t in terms if t in ans_tokens) / len(terms) if terms else 1.0

    groundedness = 0.4 * citation_coverage + 0.35 * numeric_grounding + 0.25 * faithfulness
    score = 100 * (0.45 * groundedness + 0.25 * citation_validity + 0.15 * completeness + 0.15 * faithfulness)
    return {
        "citation_coverage": round(citation_coverage, 3),
        "citation_validity": round(citation_validity, 3),
        "numeric_grounding": round(numeric_grounding, 3),
        "faithfulness_lexical": round(faithfulness, 3),
        "completeness": round(completeness, 3),
        "groundedness": round(groundedness, 3),
        "unsupported_values": v.unsupported_values,
        "invalid_refs": v.invalid_refs,
        "response_quality_score": round(score, 1),
    }
