"""
storage/cache.py — query-level cache (temporary) and feedback store (persistent).

QueryCache
  Key = hash(normalised question, selected doc_ids, chunker signature,
  embedding namespace, LLM name, retrieval settings). Any change to documents
  or models changes the key, so a cached answer can never be stale.
  Lives in cache/ (temporary by design; losing it only costs speed).

FeedbackStore (v3 feature, fixed)
  v3 matched feedback with substring tests across ALL documents, so a correction
  saved for datasheet A could override answers for datasheet B, and "current"
  matched every question containing that word. v4 scopes feedback to the
  document set and requires a near-exact question match (RapidFuzz ratio ≥ 92).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.keyword_index import fuzzy_ratio
from rag.text_utils import content_words


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.µ°%±-]", " ", (q or "").lower())).strip()


class QueryCache:
    def __init__(self, cache_dir: Path, max_items: int = 256):
        self.dir = Path(cache_dir) / "queries"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.max_items = max_items
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(question: str, doc_ids: Sequence[str], fingerprint: str, history: Sequence[str] = ()) -> str:
        raw = json.dumps([_norm_q(question), sorted(doc_ids), fingerprint, [_norm_q(h) for h in history[-2:]]])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self.mem:
                self.mem.move_to_end(key)
                self.hits += 1
                return self.mem[key]
        p = self.dir / f"{key}.json"
        if p.exists():
            try:
                val = json.loads(p.read_text(encoding="utf-8"))
                with self._lock:
                    self.mem[key] = val
                    self.hits += 1
                return val
            except Exception:
                p.unlink(missing_ok=True)      # corrupted cache entry → drop it
        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self.mem[key] = value
            self.mem.move_to_end(key)
            while len(self.mem) > self.max_items:
                self.mem.popitem(last=False)
        try:
            tmp = self.dir / f"{key}.json.tmp"
            tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
            os.replace(tmp, self.dir / f"{key}.json")
        except Exception:
            pass

    def clear(self) -> int:
        n = 0
        with self._lock:
            self.mem.clear()
        for f in self.dir.glob("*.json"):
            f.unlink(missing_ok=True)
            n += 1
        return n

    def size(self) -> int:
        return len(list(self.dir.glob("*.json")))


class FeedbackStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "feedback" / "feedback.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, question: str, answer: str, doc_ids: Sequence[str], rating: int,
            correction: str = "", note: str = "") -> None:
        rec = {
            "ts": time.time(), "question": question, "q_norm": _norm_q(question), "answer": answer[:4000],
            "doc_ids": sorted(doc_ids), "rating": int(rating), "correction": correction[:4000], "note": note[:500],
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def find_override(self, question: str, doc_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Latest human correction for (almost) the same question on overlapping documents."""
        qn = _norm_q(question)
        docs = set(doc_ids)
        for rec in reversed(self.all()):
            if not rec.get("correction"):
                continue
            if not docs & set(rec.get("doc_ids", [])):
                continue
            prev = rec.get("q_norm", "")
            # Near-identical wording AND the same content terms: "minimum" vs
            # "maximum" differs by a few characters but is a different question.
            if fuzzy_ratio(qn, prev) >= 92 and content_words(qn) == content_words(prev):
                return rec
        return None
