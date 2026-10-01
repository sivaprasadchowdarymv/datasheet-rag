"""
rag/keyword_index.py — lexical retrieval channels.

BM25Index   Okapi BM25 over technical tokens. Catches exact symbols the dense
            model blurs ("VCC", "I2C", "tPD", "LM317", "3.3V").
FuzzyIndex  RapidFuzz over SHORT keys (section, breadcrumb, part numbers,
            first line). This is v3's fuzzy matching + section matching,
            but scoped to fields where token_set_ratio is meaningful — on whole
            chunks v3's token_set_ratio returned ~100 whenever the query words
            appeared anywhere, which inflated scores past the thresholds.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import Chunk
from .text_utils import tokenize

try:
    from rapidfuzz import fuzz, process

    RAPIDFUZZ_AVAILABLE = True
except ImportError:  # graceful degradation, as in v3
    RAPIDFUZZ_AVAILABLE = False


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.4, b: float = 0.72):
        self.k1, self.b = k1, b
        self.ids: List[str] = [c.chunk_id for c in chunks]
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: List[int] = []
        for i, c in enumerate(chunks):
            toks = tokenize(c.search_text)
            self.doc_len.append(len(toks))
            for t, f in Counter(toks).items():
                self.postings[t].append((i, f))
        n = max(1, len(self.ids))
        self.avgdl = (sum(self.doc_len) / n) if self.doc_len else 1.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int, keep=None) -> List[Tuple[str, float]]:
        q = list(dict.fromkeys(tokenize(query)))
        if not q or not self.ids:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                dl = self.doc_len[i] or 1
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Tuple[str, float]] = []
        for i, s in ranked:
            cid = self.ids[i]
            if keep and not keep(cid):
                continue
            out.append((cid, s))
            if len(out) >= k:
                break
        return out

    def max_possible(self, query: str) -> float:
        """Upper bound used to normalise BM25 into 0..1 for diagnostics/gating."""
        q = list(dict.fromkeys(tokenize(query)))
        return sum(self.idf.get(t, 0.0) * (self.k1 + 1) for t in q) or 1.0


class FuzzyIndex:
    def __init__(self, chunks: Sequence[Chunk]):
        self.ids = [c.chunk_id for c in chunks]
        self.keys: List[str] = []
        for c in chunks:
            first = (c.raw_content or "").split("\n", 1)[0][:120]
            pns = " ".join((c.meta or {}).get("part_numbers", [])[:5])
            cap = c.caption or ""
            self.keys.append(f"{c.section} | {c.parent_ctx} | {cap} | {pns} | {first}".lower())

    def search(self, query: str, k: int, keep=None, min_score: float = 60.0) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        ql = query.lower()
        if RAPIDFUZZ_AVAILABLE:
            res = process.extract(ql, self.keys, scorer=fuzz.token_set_ratio, limit=max(k * 4, k))
            pairs = [(self.ids[i], float(s)) for _, s, i in res]
        else:
            words = set(ql.split())
            pairs = [
                (cid, 100.0 * sum(w in key for w in words) / max(1, len(words)))
                for cid, key in zip(self.ids, self.keys)
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
        out = [(cid, s) for cid, s in pairs if s >= min_score and (not keep or keep(cid))]
        return out[:k]


def fuzzy_ratio(a: str, b: str) -> float:
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.ratio(a, b))
    sa, sb = set(a.split()), set(b.split())
    return 100.0 * len(sa & sb) / max(1, len(sa | sb))


def partial_ratio(a: str, b: str) -> float:
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.partial_ratio(a, b))
    return 100.0 if a in b else 0.0
