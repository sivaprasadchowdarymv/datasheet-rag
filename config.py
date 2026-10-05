"""
config.py — the ONE place where configuration lives.

Resolution order for every setting (first wins):
  1. Real environment variables (shell, Docker, Oracle VM, CI)
  2. Streamlit secrets (.streamlit/secrets.toml locally, "Secrets" box on Streamlit Cloud)
  3. .env file in the project root (local development)
  4. APP_MODE profile defaults below (local / cloud)

Nothing else in the code base reads os.environ directly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Optional loaders (never fatal)
# ---------------------------------------------------------------------------
def _bridge_streamlit_secrets() -> None:
    """Copy root-level Streamlit secrets into os.environ (env vars win)."""
    try:
        import streamlit as st

        secrets = st.secrets  # raises if no secrets file exists
        for key in list(secrets.keys()):
            val = secrets[key]
            if isinstance(val, (str, int, float, bool)) and key not in os.environ:
                os.environ[key] = str(val)
    except Exception:
        pass


_bridge_streamlit_secrets()   # secrets first: they must win over .env
try:  # .env for local development
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:  # python-dotenv missing → fine
    pass



# ---------------------------------------------------------------------------
# Profiles: sensible defaults per mode. Any key can be overridden by env.
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict[str, str]] = {
    # Your PC: everything through Ollama, exactly like v3.
    "local": {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "mistral",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "VISION_PROVIDER": "ollama",
        "VISION_MODEL": "llava:7b",
        "RERANKER_PROVIDER": "fastembed",
        "ENABLE_VISION": "true",
        "ALLOW_MODE_SWITCH": "true",
    },
    # Public deployment: no Ollama on the host. Small ONNX models run in-process;
    # generation is extractive unless you point LLM_BASE_URL at an open-model server.
    "cloud": {
        "LLM_PROVIDER": "auto",  # openai_compat if LLM_BASE_URL set, else extractive
        "LLM_MODEL": "qwen2.5:3b",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "VISION_PROVIDER": "none",
        "VISION_MODEL": "",
        "RERANKER_PROVIDER": "fastembed",
        "ENABLE_VISION": "false",
        "ALLOW_MODE_SWITCH": "false",
    },
    # Offline unit tests: deterministic, no downloads, no network.
    "test": {
        "LLM_PROVIDER": "extractive",
        "LLM_MODEL": "",
        "EMBEDDING_PROVIDER": "hash",
        "EMBEDDING_MODEL": "hash-512",
        "VISION_PROVIDER": "none",
        "VISION_MODEL": "",
        "RERANKER_PROVIDER": "lexical",
        "ENABLE_VISION": "false",
        "ALLOW_MODE_SWITCH": "false",
    },
}


def _raw(key: str, mode: str, default: str = "") -> str:
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    return PROFILES.get(mode, {}).get(key, default)


def _bool(key: str, mode: str, default: bool) -> bool:
    return _raw(key, mode, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(key: str, mode: str, default: int) -> int:
    try:
        return int(_raw(key, mode, str(default)))
    except ValueError:
        return default


def _float(key: str, mode: str, default: float) -> float:
    try:
        return float(_raw(key, mode, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- mode -------------------------------------------------------------
    app_mode: str = "local"
    allow_mode_switch: bool = True

    # --- models -----------------------------------------------------------
    llm_provider: str = "ollama"          # ollama | openai_compat | extractive | auto
    llm_model: str = "mistral"
    llm_base_url: str = ""                # for openai_compat (…/v1)
    llm_api_key: str = ""                 # only if your endpoint needs one (never hardcode)
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700
    llm_num_ctx: int = 4096               # Ollama context window
    llm_timeout_s: int = 180
    llm_label: str = ""                   # display name of the primary LLM in the "Answer model" menu
    # Extra answer models offered in the sidebar menu: (label, provider, base_url, model, api_key).
    # Filled from LLM2_* … LLM5_* settings, e.g. Kimi via OpenRouter or Ollama Cloud.
    llm_choices: Tuple[Tuple[str, str, str, str, str], ...] = ()

    embedding_provider: str = "ollama"    # ollama | fastembed | sentence_transformers | hash
    embedding_model: str = "nomic-embed-text"
    embed_batch_size: int = 32
    embed_max_chars: int = 2000

    vision_provider: str = "ollama"       # ollama | openai_compat | none
    vision_model: str = "llava:7b"
    enable_vision: bool = True
    max_vision_figures: int = 1

    reranker_provider: str = "fastembed"  # fastembed | lexical | none
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True

    ollama_host: str = "http://localhost:11434"
    ollama_auth_header: str = ""          # e.g. "Bearer xyz" if Ollama sits behind a proxy

    # --- chunking (changing these marks documents as stale) ---------------
    max_chunk_tokens: int = 400
    overlap_tokens: int = 60
    min_chunk_tokens: int = 40
    exclude_table_text: bool = True
    enable_ocr: bool = False
    max_figures_per_page: int = 6

    # --- retrieval --------------------------------------------------------
    top_k_retrieval: int = 20             # per channel candidate pool
    top_k_fusion: int = 20                # after fusion, sent to reranker
    top_k_final: int = 6                  # evidence blocks given to the LLM
    rrf_k: int = 60
    rerank_min_score: float = 0.02        # sigmoid(cross-encoder logit); gate for NOT FOUND
    dense_min_score: float = 0.45         # cosine gate used when reranker is off
    lexical_min_coverage: float = 0.5     # fraction of key query terms found
    enable_query_cache: bool = True

    # --- storage ----------------------------------------------------------
    data_dir: str = str(PROJECT_ROOT / "data")
    cache_dir: str = str(PROJECT_ROOT / "cache")
    storage_backend: str = "local"        # local | hf
    hf_token: str = ""
    hf_dataset_repo: str = ""             # e.g. "yourname/rag-agent-store" (private!)
    keep_original_pdfs: bool = True

    # --- security / limits -----------------------------------------------
    max_upload_mb: int = 50
    max_pages: int = 400
    max_query_chars: int = 1000
    queries_per_minute: int = 10
    max_concurrent_generations: int = 2
    admin_password: str = ""              # if set, delete/clear/reindex need it
    public_uploads: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def chunker_signature(self) -> str:
        """Identifies chunking behaviour; stored with each document."""
        return (
            f"v4|max={self.max_chunk_tokens}|ov={self.overlap_tokens}|min={self.min_chunk_tokens}"
            f"|xtab={int(self.exclude_table_text)}|ocr={int(self.enable_ocr)}"
        )

    @property
    def embedding_namespace(self) -> str:
        """Vectors are stored per embedding model so switching models never mixes dimensions."""
        raw = f"{self.embedding_provider}__{self.embedding_model}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openai_compat" if self.llm_base_url else "extractive"

    def with_overrides(self, **kw: Any) -> "Settings":
        return replace(self, **kw)

    def public_dict(self) -> Dict[str, Any]:
        """Safe-to-display view (no secrets)."""
        d = asdict(self)
        for secret in ("llm_api_key", "hf_token", "admin_password", "ollama_auth_header"):
            d[secret] = "•••" if d.get(secret) else ""
        d["llm_choices"] = [[c[0], c[1], c[2], c[3], "•••" if c[4] else ""] for c in self.llm_choices]
        return d


def _llm_choices(m: str, primary_base: str, primary_key: str) -> Tuple[Tuple[str, str, str, str, str], ...]:
    """LLM2_* … LLM5_*: extra answer models selectable in the sidebar.

    LLMn_MODEL      required, e.g. moonshotai/kimi-k2.6:free  or  kimi-k2.6:cloud
    LLMn_LABEL      shown in the menu (default: the model name)
    LLMn_BASE_URL   OpenAI-compatible URL ending in /v1; empty → same server as the main LLM,
                    or local Ollama when the main LLM has no URL either
    LLMn_API_KEY    empty → reuse the main key when the URL is the same server
    """
    out = []
    for i in range(2, 6):
        model = _raw(f"LLM{i}_MODEL", m).strip()
        if not model:
            continue
        base = _raw(f"LLM{i}_BASE_URL", m).strip().rstrip("/") or primary_base
        key = _raw(f"LLM{i}_API_KEY", m).strip() or (primary_key if base == primary_base else "")
        provider = "openai_compat" if base else "ollama"
        label = _raw(f"LLM{i}_LABEL", m).strip() or model
        out.append((label, provider, base, model, key))
    return tuple(out)


def load_settings(mode: str | None = None) -> Settings:
    mode = (mode or os.environ.get("APP_MODE", "local")).strip().lower()
    if mode not in PROFILES:
        mode = "local"
    m = mode
    hf_token = _raw("HF_TOKEN", m)
    hf_repo = _raw("HF_DATASET_REPO", m)
    default_backend = "hf" if (hf_token and hf_repo) else "local"
    return Settings(
        app_mode=m,
        allow_mode_switch=_bool("ALLOW_MODE_SWITCH", m, True),
        llm_provider=_raw("LLM_PROVIDER", m, "ollama").lower(),
        llm_model=_raw("LLM_MODEL", m, "mistral"),
        llm_base_url=_raw("LLM_BASE_URL", m).rstrip("/"),
        llm_api_key=_raw("LLM_API_KEY", m),
        llm_temperature=_float("LLM_TEMPERATURE", m, 0.1),
        llm_max_tokens=_int("LLM_MAX_TOKENS", m, 700),
        llm_num_ctx=_int("LLM_NUM_CTX", m, 4096),
        llm_timeout_s=_int("LLM_TIMEOUT_S", m, 180),
        llm_label=_raw("LLM_LABEL", m),
        llm_choices=_llm_choices(m, _raw("LLM_BASE_URL", m).rstrip("/"), _raw("LLM_API_KEY", m)),
        embedding_provider=_raw("EMBEDDING_PROVIDER", m, "ollama").lower(),
        embedding_model=_raw("EMBEDDING_MODEL", m, "nomic-embed-text"),
        embed_batch_size=_int("EMBED_BATCH_SIZE", m, 32),
        embed_max_chars=_int("EMBED_MAX_CHARS", m, 2000),
        vision_provider=_raw("VISION_PROVIDER", m, "none").lower(),
        vision_model=_raw("VISION_MODEL", m),
        enable_vision=_bool("ENABLE_VISION", m, False),
        max_vision_figures=_int("MAX_VISION_FIGURES", m, 1),
        reranker_provider=_raw("RERANKER_PROVIDER", m, "fastembed").lower(),
        reranker_model=_raw("RERANKER_MODEL", m, "Xenova/ms-marco-MiniLM-L-6-v2"),
        enable_reranker=_bool("ENABLE_RERANKER", m, True),
        ollama_host=_raw("OLLAMA_HOST", m, "http://localhost:11434"),
        ollama_auth_header=_raw("OLLAMA_AUTH_HEADER", m),
        max_chunk_tokens=_int("MAX_CHUNK_TOKENS", m, 400),
        overlap_tokens=_int("OVERLAP_TOKENS", m, 60),
        min_chunk_tokens=_int("MIN_CHUNK_TOKENS", m, 40),
        exclude_table_text=_bool("EXCLUDE_TABLE_TEXT", m, True),
        enable_ocr=_bool("ENABLE_OCR", m, False),
        max_figures_per_page=_int("MAX_FIGURES_PER_PAGE", m, 6),
        top_k_retrieval=_int("TOP_K_RETRIEVAL", m, 20),
        top_k_fusion=_int("TOP_K_FUSION", m, 20),
        top_k_final=_int("TOP_K_FINAL", m, 6),
        rrf_k=_int("RRF_K", m, 60),
        rerank_min_score=_float("RERANK_MIN_SCORE", m, 0.02),
        dense_min_score=_float("DENSE_MIN_SCORE", m, 0.45),
        lexical_min_coverage=_float("LEXICAL_MIN_COVERAGE", m, 0.5),
        enable_query_cache=_bool("ENABLE_QUERY_CACHE", m, True),
        data_dir=_raw("DATA_DIR", m, str(PROJECT_ROOT / "data")),
        cache_dir=_raw("CACHE_DIR", m, str(PROJECT_ROOT / "cache")),
        storage_backend=_raw("STORAGE_BACKEND", m, default_backend).lower(),
        hf_token=hf_token,
        hf_dataset_repo=hf_repo,
        keep_original_pdfs=_bool("KEEP_ORIGINAL_PDFS", m, True),
        max_upload_mb=_int("MAX_UPLOAD_MB", m, 50),
        max_pages=_int("MAX_PAGES", m, 400),
        max_query_chars=_int("MAX_QUERY_CHARS", m, 1000),
        queries_per_minute=_int("QUERIES_PER_MINUTE", m, 10),
        max_concurrent_generations=_int("MAX_CONCURRENT_GENERATIONS", m, 2),
        admin_password=_raw("ADMIN_PASSWORD", m),
        public_uploads=_bool("PUBLIC_UPLOADS", m, True),
    )
