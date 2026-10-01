"""
rag/vector_store.py — persistent FAISS store replacing v3's brute-force cosine loop.

Layout on disk:
    data/indexes/<embedding-namespace>/<doc_id>.faiss
    data/indexes/<embedding-namespace>/<doc_id>.ids.json

Why one index per document?
  • delete a document  = delete two files (no index rebuild, no ID remapping)
  • metadata filtering by document = only search the selected indexes
  • switching embedding model = a new namespace; old vectors are never mixed
    with new ones (v3 crashed or silently mis-scored when the embed model
    changed, because cached 768-d vectors met a 1024-d query vector)

IndexFlatIP on normalised vectors gives EXACT cosine search. For datasheet-scale
corpora (≤ ~200k chunks) this is both the most accurate and fast enough
(single-digit milliseconds per 10k vectors on CPU).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np


class VectorIndexError(Exception):
    pass


class FaissVectorStore:
    def __init__(self, root: Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._indexes: Dict[str, faiss.Index] = {}
        self._ids: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- paths
    def _paths(self, doc_id: str) -> Tuple[Path, Path]:
        return self.dir / f"{doc_id}.faiss", self.dir / f"{doc_id}.ids.json"

    def has_document(self, doc_id: str) -> bool:
        fi, fj = self._paths(doc_id)
        return doc_id in self._indexes or (fi.exists() and fj.exists())

    def documents(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.faiss"))

    # ---------------------------------------------------------------- write
    def add_document(self, doc_id: str, chunk_ids: List[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(chunk_ids) != vectors.shape[0]:
            raise VectorIndexError("chunk_ids and vectors length mismatch")
        index = faiss.IndexFlatIP(vectors.shape[1])
        if len(chunk_ids):
            index.add(vectors)
        fi, fj = self._paths(doc_id)
        with self._lock:
            tmp_i, tmp_j = fi.with_suffix(".faiss.tmp"), fj.with_suffix(".json.tmp")
            faiss.write_index(index, str(tmp_i))
            tmp_j.write_text(json.dumps(chunk_ids), encoding="utf-8")
            os.replace(tmp_i, fi)      # atomic: a crash never leaves a half-written index
            os.replace(tmp_j, fj)
            self._indexes[doc_id] = index
            self._ids[doc_id] = list(chunk_ids)

    def update_vector(self, doc_id: str, chunk_id: str, vector: np.ndarray) -> None:
        """Replace one vector (used when a lazily computed vision description is added)."""
        with self._lock:
            index, ids = self._load(doc_id)
            if chunk_id not in ids:
                return
            mat = index.reconstruct_n(0, index.ntotal)
            mat[ids.index(chunk_id)] = np.asarray(vector, dtype=np.float32)
            self.add_document(doc_id, ids, mat)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._indexes.pop(doc_id, None)
            self._ids.pop(doc_id, None)
            for p in self._paths(doc_id):
                p.unlink(missing_ok=True)

    # ---------------------------------------------------------------- read
    def _load(self, doc_id: str) -> Tuple[faiss.Index, List[str]]:
        with self._lock:
            if doc_id in self._indexes:
                return self._indexes[doc_id], self._ids[doc_id]
            fi, fj = self._paths(doc_id)
            if not (fi.exists() and fj.exists()):
                raise VectorIndexError(f"No vector index for document {doc_id} in {self.namespace}")
            try:
                index = faiss.read_index(str(fi))
                ids = json.loads(fj.read_text(encoding="utf-8"))
            except Exception as e:  # corrupted file → caller re-embeds from stored chunks
                raise VectorIndexError(f"Vector index for {doc_id} is corrupted: {e}") from e
            if index.ntotal != len(ids):
                raise VectorIndexError(f"Vector index for {doc_id} is inconsistent; it will be rebuilt.")
            self._indexes[doc_id], self._ids[doc_id] = index, ids
            return index, ids

    def dim(self, doc_id: str) -> int:
        return self._load(doc_id)[0].d

    def search(
        self,
        query: np.ndarray,
        k: int,
        doc_ids: Iterable[str],
        keep: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Exact cosine search across the selected documents.
        `keep(chunk_id)` is an optional metadata filter (modality etc.).
        """
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        hits: List[Tuple[str, float]] = []
        for doc_id in doc_ids:
            index, ids = self._load(doc_id)
            if index.ntotal == 0:
                continue
            if index.d != q.shape[1]:
                raise VectorIndexError(
                    f"Embedding dimension mismatch for {doc_id} ({index.d} vs {q.shape[1]}); re-index needed."
                )
            want = index.ntotal if keep else min(index.ntotal, k)
            scores, idx = index.search(q, want)
            got = 0
            for s, i in zip(scores[0], idx[0]):
                if i < 0:
                    continue
                cid = ids[i]
                if keep and not keep(cid):
                    continue
                hits.append((cid, float(s)))
                got += 1
                if got >= k:
                    break
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def stats(self) -> Dict[str, int]:
        out = {}
        for d in self.documents():
            try:
                out[d] = self._load(d)[0].ntotal
            except VectorIndexError:
                out[d] = -1
        return out
