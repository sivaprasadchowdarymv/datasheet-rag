"""
rag/embeddings.py — EmbeddingEngine interface + implementations.

    EmbeddingEngine
    ├── OllamaEmbeddingEngine            (LOCAL: nomic-embed-text, mxbai-embed-large …)
    ├── FastEmbedEngine                  (CLOUD: ONNX transformer, CPU, ~100–200 MB RAM)
    ├── SentenceTransformerEngine        (optional, needs torch — not recommended on free cloud)
    └── HashEmbeddingEngine              (tests / emergency fallback, zero downloads)

Every engine returns L2-normalised float32 vectors, so inner product == cosine
and FAISS IndexFlatIP gives exact cosine search.

v3 issues fixed here:
  • v3 embedded the query 4× per question (once per engine). v4 embeds once.
  • v3 sent one HTTP request per chunk. v4 batches (Ollama /api/embed accepts lists).
  • nomic-embed-text is trained with "search_query:" / "search_document:"
    prefixes; v3 did not use them, which measurably lowers retrieval quality.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingError(Exception):
    """Human-readable embedding failure (model missing, server down, timeout …)."""


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class EmbeddingEngine(ABC):
    provider: str = "base"

    def __init__(self, model: str, max_chars: int = 2000, batch_size: int = 32):
        self.model = model
        self.max_chars = max_chars
        self.batch_size = max(1, batch_size)
        self._dim: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_query("dimension probe").shape[-1])
        return self._dim

    def _clip(self, texts: List[str]) -> List[str]:
        return [(t or " ")[: self.max_chars] for t in texts]

    @abstractmethod
    def _embed_docs(self, texts: List[str]) -> np.ndarray: ...

    @abstractmethod
    def _embed_query(self, text: str) -> np.ndarray: ...

    def embed_documents(self, texts: List[str], progress=None) -> np.ndarray:
        texts = self._clip(texts)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            out.append(_normalise(self._embed_docs(texts[i:i + self.batch_size])))
            if progress:
                progress(min(1.0, (i + self.batch_size) / max(1, len(texts))), "Embedding")
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.vstack(out)
        self._dim = mat.shape[1]
        return mat

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._embed_query(self._clip([text])[0]))[0]

    def health(self) -> str:
        """Returns '' when healthy, otherwise a human-readable problem."""
        try:
            _ = self.dim
            return ""
        except Exception as e:  # noqa: BLE001
            return str(e)


# ---------------------------------------------------------------------------
class OllamaEmbeddingEngine(EmbeddingEngine):
    provider = "ollama"

    def __init__(self, model: str, host: str, auth_header: str = "", timeout: int = 120, **kw):
        super().__init__(model, **kw)
        try:
            import ollama
        except ImportError as e:
            raise EmbeddingError("The 'ollama' package is not installed (pip install ollama).") from e
        kwargs = {"timeout": timeout}
        if auth_header:
            kwargs["headers"] = {"Authorization": auth_header}
        self._client = ollama.Client(host=host, **kwargs)
        self.host = host
        is_nomic = "nomic" in model.lower()
        self._doc_prefix = "search_document: " if is_nomic else ""
        self._query_prefix = "search_query: " if is_nomic else ""

    def _call(self, inputs: List[str]) -> np.ndarray:
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._client.embed(model=self.model, input=inputs, truncate=True)
                embs = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
                return np.asarray(embs, dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e).lower()
                if "not found" in msg or "pull" in msg:
                    raise EmbeddingError(
                        f"Ollama embedding model '{self.model}' is not installed. Run: ollama pull {self.model}"
                    ) from e
                time.sleep(1.5 * (attempt + 1))
        raise EmbeddingError(
            f"Could not reach Ollama at {self.host} for embeddings. Is `ollama serve` running? ({last})"
        )

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        return self._call([self._doc_prefix + t for t in texts])

    def _embed_query(self, text: str) -> np.ndarray:
        return self._call([self._query_prefix + text])


# ---------------------------------------------------------------------------
class FastEmbedEngine(EmbeddingEngine):
    """ONNX-runtime transformer embeddings (no torch). Downloads the model once."""
    provider = "fastembed"

    def __init__(self, model: str, cache_dir: Optional[str] = None, threads: Optional[int] = None, **kw):
        super().__init__(model, **kw)
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise EmbeddingError("The 'fastembed' package is not installed (pip install fastembed).") from e
        try:
            self._m = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=threads)
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(f"Could not load embedding model '{model}': {e}") from e

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        return np.asarray(list(self._m.passage_embed(texts, batch_size=self.batch_size)), dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self._m.query_embed(text)), dtype=np.float32)


# ---------------------------------------------------------------------------
class SentenceTransformerEngine(EmbeddingEngine):
    provider = "sentence_transformers"

    def __init__(self, model: str, **kw):
        super().__init__(model, **kw)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers is not installed (pip install sentence-transformers). "
                "On free cloud tiers prefer EMBEDDING_PROVIDER=fastembed (no torch)."
            ) from e
        self._m = SentenceTransformer(model, device="cpu", trust_remote_code="nomic" in model.lower())
        self._nomic = "nomic" in model.lower()

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        if self._nomic:
            texts = ["search_document: " + t for t in texts]
        return np.asarray(self._m.encode(texts, batch_size=self.batch_size), dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        if self._nomic:
            text = "search_query: " + text
        return np.asarray(self._m.encode([text]), dtype=np.float32)


# ---------------------------------------------------------------------------
class HashEmbeddingEngine(EmbeddingEngine):
    """
    Deterministic hashed bag of word + character-trigram features.
    Not semantic, but surprisingly decent for spec look-ups and perfect for tests.
    """
    provider = "hash"

    def __init__(self, model: str = "hash-512", **kw):
        super().__init__(model, **kw)
        m = re.search(r"(\d+)", model)
        self._dim = int(m.group(1)) if m else 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9.]+", text.lower())
        feats = words + [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
            v[h % self._dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v / (math.sqrt(float((v * v).sum())) or 1.0)

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self._dim), np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)[None, :]


def create_embedding_engine(settings) -> EmbeddingEngine:
    p = settings.embedding_provider
    common = dict(max_chars=settings.embed_max_chars, batch_size=settings.embed_batch_size)
    if p == "ollama":
        return OllamaEmbeddingEngine(
            settings.embedding_model, host=settings.ollama_host,
            auth_header=settings.ollama_auth_header, **common,
        )
    if p == "fastembed":
        return FastEmbedEngine(settings.embedding_model, cache_dir=f"{settings.cache_dir}/models", **common)
    if p in ("sentence_transformers", "sentence-transformers", "st"):
        return SentenceTransformerEngine(settings.embedding_model, **common)
    if p == "hash":
        return HashEmbeddingEngine(settings.embedding_model or "hash-512", **common)
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{p}'. Use ollama | fastembed | sentence_transformers | hash.")
