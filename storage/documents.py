"""
storage/documents.py — persistent document library.

    data/
    ├── documents/<doc_id>.pdf            original upload (optional, for re-index)
    ├── metadata/registry.json            DocumentRecord per document
    ├── metadata/<doc_id>/chunks.jsonl    every Chunk (text, tables, rows, eqs, figures)
    ├── figures/<doc_id>/p3_0.png         rendered figures (namespaced per doc)
    ├── indexes/<embed-namespace>/…       FAISS (see rag/vector_store.py)
    └── feedback/feedback.jsonl           user feedback

doc_id = first 16 hex chars of SHA-256(pdf bytes): uploading the same PDF twice
(even under another name) never re-processes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

from rag.schema import Chunk, DocumentRecord


def compute_doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class DocumentRepository:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        for sub in ("documents", "metadata", "figures", "indexes", "feedback"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "metadata" / "registry.json"
        self._lock = threading.RLock()
        self._chunk_cache: Dict[str, List[Chunk]] = {}

    # ------------------------------------------------------------ registry
    def _read_registry(self) -> Dict[str, DocumentRecord]:
        if not self.registry_path.exists():
            return {}
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return {k: DocumentRecord.from_dict(v) for k, v in raw.items()}
        except Exception:
            # corrupted registry → rebuild from what is on disk
            return self._rebuild_registry()

    def _rebuild_registry(self) -> Dict[str, DocumentRecord]:
        recs: Dict[str, DocumentRecord] = {}
        for d in (self.root / "metadata").iterdir():
            f = d / "record.json"
            if d.is_dir() and f.exists():
                try:
                    recs[d.name] = DocumentRecord.from_dict(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
        self._write_registry(recs)
        return recs

    def _write_registry(self, recs: Dict[str, DocumentRecord]) -> None:
        _atomic_write(self.registry_path, json.dumps({k: v.to_dict() for k, v in recs.items()}, indent=1))

    def list(self) -> List[DocumentRecord]:
        with self._lock:
            return sorted(self._read_registry().values(), key=lambda r: r.ingested_at)

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            return self._read_registry().get(doc_id)

    def save_record(self, rec: DocumentRecord) -> None:
        with self._lock:
            recs = self._read_registry()
            recs[rec.doc_id] = rec
            self._write_registry(recs)
            _atomic_write(self.root / "metadata" / rec.doc_id / "record.json", json.dumps(rec.to_dict(), indent=1))

    # ------------------------------------------------------------ chunks
    def chunks_path(self, doc_id: str) -> Path:
        return self.root / "metadata" / doc_id / "chunks.jsonl"

    def save_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        with self._lock:
            _atomic_write(self.chunks_path(doc_id), "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks))
            self._chunk_cache[doc_id] = chunks

    def load_chunks(self, doc_id: str) -> List[Chunk]:
        with self._lock:
            if doc_id in self._chunk_cache:
                return self._chunk_cache[doc_id]
            p = self.chunks_path(doc_id)
            if not p.exists():
                return []
            chunks = [Chunk.from_dict(json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self._chunk_cache[doc_id] = chunks
            return chunks

    def update_chunk(self, chunk: Chunk) -> None:
        chunks = self.load_chunks(chunk.doc_id)
        for i, c in enumerate(chunks):
            if c.chunk_id == chunk.chunk_id:
                chunks[i] = chunk
                break
        self.save_chunks(chunk.doc_id, chunks)

    # ------------------------------------------------------------ files
    def pdf_path(self, doc_id: str) -> Path:
        return self.root / "documents" / f"{doc_id}.pdf"

    def save_pdf(self, doc_id: str, data: bytes) -> None:
        p = self.pdf_path(doc_id)
        p.write_bytes(data)

    def load_pdf(self, doc_id: str) -> Optional[bytes]:
        p = self.pdf_path(doc_id)
        return p.read_bytes() if p.exists() else None

    def figures_dir(self, doc_id: str) -> Path:
        return self.root / "figures" / doc_id

    def resolve(self, rel_path: Optional[str]) -> Optional[Path]:
        """Turn a stored relative path into an absolute one INSIDE data/ (never outside)."""
        if not rel_path:
            return None
        p = (self.root / rel_path).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            return None
        return p if p.exists() else None

    def rel_paths_for(self, doc_id: str, namespace: Optional[str] = None) -> List[str]:
        rels = [f"documents/{doc_id}.pdf", f"metadata/{doc_id}", f"figures/{doc_id}", "metadata/registry.json"]
        if namespace:
            rels += [f"indexes/{namespace}/{doc_id}.faiss", f"indexes/{namespace}/{doc_id}.ids.json"]
        return rels

    def delete(self, doc_id: str) -> None:
        with self._lock:
            recs = self._read_registry()
            recs.pop(doc_id, None)
            self._write_registry(recs)
            self._chunk_cache.pop(doc_id, None)
            self.pdf_path(doc_id).unlink(missing_ok=True)
            shutil.rmtree(self.root / "metadata" / doc_id, ignore_errors=True)
            shutil.rmtree(self.figures_dir(doc_id), ignore_errors=True)
            for ns in (self.root / "indexes").glob("*"):
                for f in ns.glob(f"{doc_id}.*"):
                    f.unlink(missing_ok=True)
