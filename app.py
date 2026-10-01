"""
app.py — Streamlit frontend for RAG Agent v4.

    streamlit run app.py                      (local, Ollama)
    APP_MODE=cloud streamlit run app.py       (cloud profile)

The UI only talks to rag.pipeline.RAGPipeline. Expensive objects (models,
storage restore) are created once per process with st.cache_resource; each
browser session keeps only its chat history and preferences.
"""
from __future__ import annotations

import hmac
import os
import time
import traceback
import uuid
from dataclasses import replace
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Agent", page_icon="📡", layout="wide")

from config import PROFILES, Settings, load_settings  # noqa: E402
from rag.embeddings import EmbeddingError  # noqa: E402
from rag.pipeline import Engines, IngestionError, RAGPipeline, build_engines  # noqa: E402
from rag.schema import AnswerResult  # noqa: E402
from rag.security import RateLimiter, sanitize_query  # noqa: E402
from storage.backends import create_storage  # noqa: E402
from ui.components import (  # noqa: E402
    render_answer, render_browse, render_doc_stats, render_metrics, render_sources,
)
from ui.styles import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)
BASE = load_settings()

# ============================================================== cached resources
ENGINE_FIELDS = ("app_mode", "llm_provider", "llm_model", "llm_base_url", "llm_api_key", "embedding_provider",
                 "embedding_model", "vision_provider", "vision_model", "enable_vision", "reranker_provider",
                 "reranker_model", "enable_reranker", "ollama_host", "ollama_auth_header", "llm_timeout_s",
                 "llm_num_ctx", "llm_max_tokens", "llm_temperature", "max_concurrent_generations")


def engine_key(s: Settings) -> Tuple:
    return tuple(getattr(s, f) for f in ENGINE_FIELDS)


@st.cache_resource(show_spinner="Loading models (first start can take a minute)…", max_entries=3)
def get_engines(key: Tuple, _settings: Settings) -> Engines:
    return build_engines(_settings)


@st.cache_resource(show_spinner="Restoring document library…")
def get_storage(data_dir: str, backend: str, repo: str):
    s = BASE.with_overrides(data_dir=data_dir, storage_backend=backend, hf_dataset_repo=repo)
    storage = create_storage(s)
    storage.pull_all()                    # no-op locally; restores from HF on a fresh cloud container
    return storage


@st.cache_resource(max_entries=6)
def get_pipeline(settings_key: str, _settings: Settings, _engines: Engines) -> RAGPipeline:
    storage = get_storage(_settings.data_dir, _settings.storage_backend, _settings.hf_dataset_repo)
    return RAGPipeline(_settings, _engines, storage)


@st.cache_data(ttl=30, show_spinner=False)
def cached_health(key: str, _pipe: RAGPipeline) -> Dict[str, str]:
    return _pipe.health()


@st.cache_resource
def get_rate_limiter(per_minute: int) -> RateLimiter:
    return RateLimiter(per_minute)


# ============================================================== session state
ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex)
ss.setdefault("messages", [])           # [{"role": "user"/"assistant", "content": str, "result": dict}]
ss.setdefault("admin", not bool(BASE.admin_password))
ss.setdefault("rated", {})
ss.setdefault("mode", BASE.app_mode)
ss.setdefault("last_result", None)

# ============================================================== sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    modes = ["local", "cloud"] if BASE.allow_mode_switch else [BASE.app_mode]
    mode = st.radio("Mode", modes, index=modes.index(ss.mode) if ss.mode in modes else 0, horizontal=True,
                    help="LOCAL = Ollama on this machine. CLOUD = in-process ONNX models + optional remote LLM.")
    ss.mode = mode
    S = load_settings(mode) if mode != BASE.app_mode else BASE

    if S.admin_password and not ss.admin:
        with st.expander("🔐 Admin unlock"):
            pw = st.text_input("Admin password", type="password")
            if st.button("Unlock") and pw:
                ss.admin = hmac.compare_digest(pw.encode(), S.admin_password.encode())
                st.toast("Admin unlocked" if ss.admin else "Wrong password")
                st.rerun()

    st.caption(f"LLM: **{S.resolved_llm_provider}** {S.llm_model if S.resolved_llm_provider != 'extractive' else ''}")
    if ss.admin and S.resolved_llm_provider == "ollama":
        S = S.with_overrides(llm_model=st.text_input("Ollama LLM model", S.llm_model,
                                                     help="Any model you have pulled: mistral, llama3.1, qwen2.5:7b, phi3 …"))
    st.caption(f"Embeddings: **{S.embedding_provider}** {S.embedding_model}")

    st.subheader("🔎 Retrieval")
    top_k_final = st.slider("Evidence blocks (top-K)", 2, 12, S.top_k_final)
    top_k_ret = st.slider("Candidate pool per channel", 5, 60, S.top_k_retrieval, step=5)
    use_rerank = st.toggle("Reranker", value=S.enable_reranker)
    rerank_min = st.slider("Evidence threshold (reranker)", 0.0, 0.5, float(S.rerank_min_score), 0.005,
                           help="Below this score the app answers NOT FOUND instead of letting the LLM guess.")
    dense_min = st.slider("Evidence threshold (no reranker, cosine)", 0.0, 0.9, float(S.dense_min_score), 0.01)
    S = S.with_overrides(top_k_final=top_k_final, top_k_retrieval=top_k_ret, top_k_fusion=max(top_k_ret, top_k_final),
                         enable_reranker=use_rerank, rerank_min_score=rerank_min, dense_min_score=dense_min)

    with st.expander("🔪 Chunking (applies to new / re-indexed documents)", expanded=False):
        disabled = not ss.admin
        mct = st.slider("Max chunk tokens", 100, 800, S.max_chunk_tokens, 20, disabled=disabled)
        ovl = st.slider("Overlap tokens", 0, 150, S.overlap_tokens, 10, disabled=disabled)
        S = S.with_overrides(max_chunk_tokens=mct, overlap_tokens=ovl)
        if disabled:
            st.caption("Unlock admin to change chunking.")

# ============================================================== engines + pipeline
try:
    engines = get_engines(engine_key(S), S)
except EmbeddingError as e:
    st.error(f"**Embedding model could not be loaded.** {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"**Startup failed:** {type(e).__name__}: {e}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()

if not S.enable_reranker:
    from rag.reranker import NoReranker
    engines = replace(engines, reranker=NoReranker())
pipe_key = repr((engine_key(S), S.chunker_signature, S.top_k_final, S.top_k_retrieval, S.enable_reranker,
                 S.rerank_min_score, S.dense_min_score))
pipe = get_pipeline(pipe_key, S, engines)
limiter = get_rate_limiter(S.queries_per_minute)

with st.sidebar:
    st.subheader("🩺 Status")
    health = cached_health(pipe_key, pipe)
    desc = engines.describe()
    for comp, label in (("embedding", "Embeddings"), ("llm", "LLM")):
        problem = health.get(comp, "")
        (st.error if problem else st.success)(f"{label}: {desc[comp]}" + (f" — {problem}" if problem else ""), icon="⚠️" if problem else "✅")
    st.caption(f"Reranker: {desc['reranker']} · Vision: {desc['vision']}")
    st.caption(f"Storage: {pipe.storage.describe()}")
    for w in engines.warnings:
        st.warning(w)
    if health.get("storage"):
        st.warning(health["storage"])

# ============================================================== header
st.markdown("## 📡 RAG Agent")
st.caption("Ask questions about your datasheets. Every answer is grounded in the uploaded PDFs and cited "
           "with [REF-n] badges you can hover or tap. Missing information is reported as NOT FOUND, never guessed.")

docs = pipe.documents()
tab_ask, tab_docs, tab_browse, tab_diag = st.tabs(["💬 Ask", "📁 Documents", "🗂 Browse", "🔬 Diagnostics"])


# ============================================================== documents tab
def ingest_uploads(files) -> None:
    for f in files:
        data = f.getvalue()
        bar = st.progress(0.0, text=f"{f.name}: starting…")
        try:
            rec, status = pipe.ingest(data, f.name, progress=lambda frac, msg: bar.progress(min(1.0, frac), text=f"{f.name}: {msg}"))
            bar.empty()
            msg = {"new": "indexed", "cached": "already indexed — loaded instantly",
                   "vectors-rebuilt": "vectors rebuilt for the current embedding model",
                   "reindexed": "re-indexed"}[status]
            st.success(f"**{rec.name}** {msg} · {rec.stats.get('total', 0)} chunks")
            for w in rec.warnings:
                st.warning(f"{rec.name}: {w}")
        except IngestionError as e:
            bar.empty()
            st.error(f"**{f.name}:** {e}")
        except MemoryError:
            bar.empty()
            st.error(f"**{f.name}:** the server ran out of memory. Try a smaller PDF.")
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(f"**{f.name}:** unexpected error ({type(e).__name__}).")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


with tab_docs:
    can_upload = S.public_uploads or ss.admin
    if can_upload:
        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader("Upload datasheet PDFs", type=["pdf"], accept_multiple_files=True,
                                     help=f"Max {S.max_upload_mb} MB and {S.max_pages} pages per file. "
                                          "Each PDF is processed once; re-uploading the same file is instant.")
            submitted = st.form_submit_button("📥 Index documents", type="primary")
        if submitted and files:
            ingest_uploads(files)
            docs = pipe.documents()
    else:
        st.info("Uploads are restricted to the administrator on this deployment.")

    if not docs:
        st.info("No documents yet. Upload one or more datasheet PDFs above.")
    for rec in docs:
        stale = pipe.is_stale(rec)
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**📄 {rec.name}** · {rec.pages} pages · {rec.size_bytes / 1e6:.2f} MB · id `{rec.doc_id}`"
                            + (" · ⚠️ *stale (chunking settings changed)*" if stale else ""))
            with c2:
                if ss.admin:
                    b1, b2 = st.columns(2)
                    if b1.button("🔄", key=f"re_{rec.doc_id}", help="Re-index (re-parse with current settings)"):
                        with st.spinner("Re-indexing…"):
                            try:
                                pipe.reindex(rec.doc_id)
                                st.toast("Re-indexed")
                            except IngestionError as e:
                                st.error(str(e))
                        st.rerun()
                    if b2.button("🗑️", key=f"del_{rec.doc_id}", help="Delete document, chunks, figures and vectors"):
                        w = pipe.delete(rec.doc_id)
                        if w:
                            st.warning(w)
                        st.rerun()
            render_doc_stats(rec)
            for w in rec.warnings:
                st.caption(f"⚠️ {w}")
            st.caption(f"Parse {rec.stats.get('parse_ms', 0)} ms · embed {rec.stats.get('embed_ms', 0)} ms · "
                       f"indexed for: {', '.join(rec.namespaces)}")
    if ss.admin and docs:
        if st.button("🧹 Clear answer cache"):
            st.toast(f"Removed {pipe.cache.clear()} cached answers")

# ============================================================== ask tab
with tab_ask:
    if not docs:
        st.info("Upload a datasheet in the **📁 Documents** tab to begin.")
    else:
        names = {d.doc_id: d.name for d in docs}
        scope = st.multiselect("Search in", options=list(names), default=list(names), format_func=lambda d: names[d],
                               help="Select several datasheets to compare them.")
        for i, m in enumerate(ss.messages):
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(m["content"])
                    continue
                res = AnswerResult.from_dict(m["result"])
                render_answer(res)
                render_sources(res.sources, pipe.repo.resolve, key_prefix=str(i))
                render_metrics(res)
                if res.kind in ("answer", "not_found"):
                    fb = st.feedback("thumbs", key=f"fb_{i}")
                    if fb is not None and ss.rated.get(i) != fb:
                        ss.rated[i] = fb
                        pipe.feedback.add(m["question"], res.answer, m["doc_ids"], 1 if fb == 1 else -1)
                        st.toast("Thanks for the feedback")
                    with st.expander("✏️ Submit the correct answer"):
                        corr = st.text_area("Correct answer (shown next time this exact question is asked)", key=f"corr_{i}")
                        if st.button("Save correction", key=f"save_{i}") and corr.strip():
                            pipe.feedback.add(m["question"], res.answer, m["doc_ids"], -1, correction=corr.strip())
                            st.toast("Correction saved")

        question = st.chat_input("Ask about the selected datasheets… e.g. What is the maximum output current?")
        if question:
            q = sanitize_query(question, S.max_query_chars)
            allowed, wait = limiter.check(ss.sid)
            if not q:
                st.warning("Please type a question.")
            elif not scope:
                st.warning("Select at least one document.")
            elif not allowed:
                st.warning(f"Too many questions in a short time. Please wait {wait} s.")
            else:
                history = [m["content"] for m in ss.messages if m["role"] == "user"][-3:]
                ss.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    status_box = st.empty()
                    stream_box = st.empty()
                    buf: List[str] = []
                    last = [0.0]

                    def on_token(t: str) -> None:
                        buf.append(t)
                        if time.time() - last[0] > 0.08:
                            stream_box.markdown("".join(buf) + " ▌")
                            last[0] = time.time()

                    status_box.caption("Searching documents…")
                    try:
                        res = pipe.answer(q, scope, history, on_token=on_token,
                                          on_status=lambda msg: status_box.caption(msg))
                    except Exception as e:  # noqa: BLE001
                        res = AnswerResult("error", f"Something went wrong while answering ({type(e).__name__}). "
                                                    "Please try again; if it persists, check the Diagnostics tab.",
                                           diagnostics={"traceback": traceback.format_exc()})
                    status_box.empty()
                    stream_box.empty()
                    render_answer(res)
                    render_sources(res.sources, pipe.repo.resolve, key_prefix="new")
                    render_metrics(res)
                ss.messages.append({"role": "assistant", "content": res.answer, "result": res.to_dict(),
                                    "question": q, "doc_ids": list(scope)})
                ss.last_result = res.to_dict()
                st.rerun()
        if ss.messages and st.button("🧽 New conversation"):
            ss.messages, ss.rated = [], {}
            st.rerun()

# ============================================================== browse tab
with tab_browse:
    if not docs:
        st.info("Nothing to browse yet.")
    else:
        names = {d.doc_id: d.name for d in docs}
        c1, c2 = st.columns([2, 1])
        doc_id = c1.selectbox("Document", list(names), format_func=lambda d: names[d])
        mode_b = c2.selectbox("Show", ["Text chunks", "Tables", "Rows / pins", "Equations", "Figures"])
        render_browse(pipe.repo.load_chunks(doc_id), mode_b, pipe.repo.resolve)

# ============================================================== diagnostics tab
with tab_diag:
    st.markdown("#### Last query")
    lr = ss.last_result
    if not lr:
        st.caption("Ask a question to see retrieval diagnostics.")
    else:
        res = AnswerResult.from_dict(lr)
        d = res.diagnostics or {}
        st.write(f"**Result:** {res.kind} · **evidence gate:** {d.get('gate', '—')} · cached: {res.from_cache}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Latency (s)**")
            st.dataframe(pd.DataFrame([{"stage": k, "seconds": v} for k, v in res.timings.items()]), width="stretch")
        with c2:
            st.markdown("**Query analysis**")
            st.json(d.get("query_analysis", {}))
        if d.get("candidates"):
            st.markdown("**Candidates after fusion + reranking** (dense = cosine, bm25 = normalised, fuzzy = 0-100, "
                        "rerank = 0-1, f_* = v3 feature bonuses)")
            st.dataframe(pd.DataFrame(d["candidates"]), width="stretch", height=320)
        if d.get("traceback") and ss.admin:
            with st.expander("Error details"):
                st.code(d["traceback"])
    st.markdown("#### System")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Cached answers", pipe.cache.size())
    c3.metric("Cache hits (this process)", pipe.cache.hits)
    st.markdown("**Models**")
    st.json(engines.describe())
    st.markdown("**Vector store** (chunks per document, namespace `" + pipe.namespace + "`)")
    st.json(pipe.store.stats())
    if ss.admin:
        with st.expander("Effective configuration (secrets hidden)"):
            st.json(S.public_dict())
