"""
rag/schema.py — plain data objects shared by every layer.

`Chunk` is the v4 successor of the v3 `Node` dataclass. All v3 fields are kept
(type → modality, section, content, raw_content, page, chunk_idx, depth,
parent_ctx, meta, figure_path, caption, vision_text) and three are added:
chunk_id, doc_id and doc_name, so citations stay unambiguous across documents.
Embeddings are NOT stored on the chunk any more — they live in the FAISS index.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

MODALITIES = ("text", "table", "row", "pin", "equation", "figure")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    modality: str                 # text | table | row | pin | equation | figure
    section: str
    content: str                  # what is embedded and shown to the LLM (with context prefix)
    raw_content: str              # original text shown to the user as evidence
    page: int
    chunk_idx: int = 0
    depth: int = 0
    parent_ctx: str = ""          # heading breadcrumb ("Electrical Characteristics > Absolute Max")
    meta: Dict[str, Any] = field(default_factory=dict)
    figure_path: Optional[str] = None   # relative to DATA_DIR
    caption: Optional[str] = None
    vision_text: Optional[str] = None
    vision_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    @property
    def search_text(self) -> str:
        """Text used by keyword search: content + vision description when available."""
        if self.vision_text:
            return f"{self.content}\n{self.vision_text}"
        return self.content


@dataclass
class DocumentRecord:
    doc_id: str
    name: str
    pages: int
    size_bytes: int
    ingested_at: float
    chunker_signature: str
    stats: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    namespaces: List[str] = field(default_factory=list)   # embedding namespaces indexed
    has_pdf: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentRecord":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Candidate:
    """A chunk travelling through retrieval → fusion → reranking, with diagnostics."""
    chunk: Chunk
    dense: Optional[float] = None       # cosine similarity
    bm25: Optional[float] = None        # normalised 0..1 within the query
    fuzzy: Optional[float] = None       # 0..100
    ranks: Dict[str, int] = field(default_factory=dict)
    fused: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    rerank: Optional[float] = None      # 0..1
    final: float = 0.0

    def diag(self) -> Dict[str, Any]:
        c = self.chunk
        return {
            "chunk_id": c.chunk_id, "doc": c.doc_name, "page": c.page, "type": c.modality,
            "section": c.section[:60], "dense": _r(self.dense), "bm25": _r(self.bm25),
            "fuzzy": _r(self.fuzzy), "fused": _r(self.fused), "rerank": _r(self.rerank),
            "final": _r(self.final), **{f"f_{k}": _r(v) for k, v in self.features.items()},
        }


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


@dataclass
class Source:
    """One [REF-N] citation as shown in the UI."""
    ref: int
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    section: str
    parent_ctx: str
    modality: str
    snippet: str
    score: Optional[float]
    chunk_idx: int = 0
    depth: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    figure_path: Optional[str] = None

    @property
    def label(self) -> str:
        return f"[REF-{self.ref}]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Source":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class AnswerResult:
    kind: str                       # answer | not_found | feedback_override | error
    answer: str
    sources: List[Source] = field(default_factory=list)
    retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    answer_metrics: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnswerResult":
        d = dict(d)
        d["sources"] = [Source.from_dict(s) for s in d.get("sources", [])]
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)
