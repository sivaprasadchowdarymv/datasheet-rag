"""
ui/components.py — rendering only. Takes plain data (AnswerResult, Chunk,
DocumentRecord) and draws it. No retrieval, no model calls.

Everything that comes from a document or a model is HTML-escaped before it is
put into unsafe_allow_html markup (v3 inserted raw LLM output into HTML).
"""
from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from rag.generator import NOT_FOUND
from rag.schema import AnswerResult, Chunk, DocumentRecord, Source

TYPE_ICONS = {"text": "📝", "table": "📊", "row": "🔢", "pin": "📌", "equation": "➗", "figure": "🖼️"}
_REF_GROUP = re.compile(r"\[(REF-\d+(?:\]\[REF-\d+)*)\]")


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def popup_html(src: Source) -> str:
    icon = TYPE_ICONS.get(src.modality, "📄")
    score = f" · score {src.score:.3f}" if src.score is not None else ""
    depth = f" · depth {src.depth}, chunk {src.chunk_idx}" if src.depth else ""
    crumb = f'<div class="pop-breadcrumb">📂 {_esc(src.parent_ctx)}</div>' if src.parent_ctx else ""
    chips = ""
    meta = {k: v for k, v in (src.meta or {}).items() if k not in ("headers", "table_id", "part", "parts", "rows")}
    if meta:
        bits = []
        for k, v in list(meta.items())[:4]:
            val = ", ".join(map(str, v[:3])) if isinstance(v, list) else str(v)[:40]
            if val:
                bits.append(f'<span class="pop-meta-chip">{_esc(k)}: {_esc(val)}</span>')
        chips = f'<div class="pop-meta-row">{"".join(bits)}</div>' if bits else ""
    return (
        f'<span class="cite-wrap" tabindex="0"><span class="cite-badge">REF-{src.ref}</span>'
        f'<span class="cite-popup">'
        f'<div class="pop-header">{icon} {_esc(src.modality.upper())} · page {src.page}{score}{depth}</div>'
        f'<div class="pop-doc">📄 {_esc(src.doc_name)}</div>'
        f'<div class="pop-section">Section: {_esc(src.section)}</div>{crumb}{chips}'
        f'<div class="pop-body">{_esc(src.snippet[:700])}</div></span></span>'
    )


def answer_html(answer: str, sources: List[Source]) -> str:
    ref_map = {s.ref: s for s in sources}
    out_lines = []
    in_list = False
    for raw_line in (answer or "").split("\n"):
        line = _esc(raw_line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)

        def sub(m: re.Match) -> str:
            n = int(m.group(1))
            return popup_html(ref_map[n]) if n in ref_map else f'<span class="cite-badge bad">REF-{n}?</span>'

        line = re.sub(r"\[REF-(\d+)\]", sub, line)
        bullet = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if bullet:
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            out_lines.append(f"<li>{bullet.group(1)}</li>")
            continue
        if in_list:
            out_lines.append("</ul>")
            in_list = False
        out_lines.append(line + "<br>" if line.strip() else "<br>")
    if in_list:
        out_lines.append("</ul>")
    return f'<div class="answer-box">{"".join(out_lines)}</div>'


def render_answer(res: AnswerResult, resolve_path=None) -> None:
    if res.kind == "feedback_override":
        st.info("⚡ **Human-corrected answer** saved earlier for this question.")
    if res.kind == "not_found" and res.answer.strip().upper().startswith(NOT_FOUND):
        st.markdown(
            f'<div class="nf-box"><b>{NOT_FOUND}</b><br><span style="opacity:.75;font-size:.85rem">'
            "The selected documents do not contain enough evidence to answer this. "
            "Nothing was guessed. Try rephrasing, selecting other documents, or check Diagnostics.</span></div>",
            unsafe_allow_html=True,
        )
    elif res.kind == "error":
        st.error(res.answer)
    else:
        st.markdown(answer_html(res.answer, res.sources), unsafe_allow_html=True)
    for w in res.warnings:
        st.warning(w, icon="⚠️")
    bits = []
    if res.timings.get("total") is not None:
        bits.append(f"⏱ {res.timings['total']:.2f}s")
    if res.timings.get("time_to_first_token"):
        bits.append(f"first token {res.timings['time_to_first_token']:.2f}s")
    if res.from_cache:
        bits.append("⚡ cached")
    if res.sources:
        bits.append(f"{len(res.sources)} sources · hover or tap a REF badge")
    if bits:
        st.caption(" · ".join(bits))


def render_sources(sources: List[Source], resolve_path=None, key_prefix: str = "") -> None:
    if not sources:
        return
    with st.expander(f"📚 Evidence ({len(sources)} sources)", expanded=False):
        for s in sources:
            icon = TYPE_ICONS.get(s.modality, "📄")
            crumb = f" · {s.parent_ctx}" if s.parent_ctx else ""
            st.markdown(
                f"**{icon} REF-{s.ref}** — `{s.doc_name}` · page **{s.page}** · {s.modality.upper()} · "
                f"score {s.score if s.score is not None else '—'}  \n"
                f"<span style='opacity:.7;font-size:.8rem'>Section: {_esc(s.section)}{_esc(crumb)} · "
                f"chunk {s.chunk_idx}, depth {s.depth} · id {s.chunk_id}</span>",
                unsafe_allow_html=True,
            )
            if s.modality == "figure" and resolve_path:
                p = resolve_path(s.figure_path)
                if p:
                    st.image(str(p), width=420)
            st.markdown(f'<div class="src-body">{_esc(s.snippet)}</div>', unsafe_allow_html=True)
            meta = {k: v for k, v in (s.meta or {}).items() if v}
            if meta:
                st.caption("Metadata: " + "; ".join(
                    f"{k}={', '.join(map(str, v[:4])) if isinstance(v, list) else v}" for k, v in list(meta.items())[:6]))
            st.divider()


def _metric_cells(defs) -> None:
    cols = st.columns(len(defs))
    for col, (name, val, tip) in zip(cols, defs):
        pct = int(round(100 * max(0.0, min(1.0, float(val or 0)))))
        bc = "#3fb950" if pct >= 70 else "#d29922" if pct >= 40 else "#f85149"
        with col:
            st.markdown(
                f'<div class="metric-cell"><div class="metric-name">{name}</div>'
                f'<div class="metric-val" style="color:{bc};">{pct}%</div>'
                f'<div class="metric-bar"><div class="metric-fill" style="width:{pct}%;background:{bc};"></div></div>'
                f'<div class="metric-tip">{tip}</div></div>', unsafe_allow_html=True)


def render_metrics(res: AnswerResult) -> None:
    am, rm = res.answer_metrics or {}, res.retrieval_metrics or {}
    if not am and not rm:
        return
    with st.expander("📊 Response quality", expanded=False):
        rqs = am.get("response_quality_score")
        if rqs is not None:
            color = "#3fb950" if rqs >= 75 else "#d29922" if rqs >= 50 else "#f85149"
            st.markdown(
                f'<div class="rqs-badge" style="border-color:{color};background:{color}18;">'
                f'<span style="color:{color};font-size:1.2rem;font-weight:700;">Response Quality Score {rqs}/100</span>'
                f'<span style="opacity:.7;font-size:.8rem"> &nbsp;heuristic, not a calibrated confidence</span></div>',
                unsafe_allow_html=True)
            st.markdown("**Answer**")
            _metric_cells([
                ("Groundedness", am.get("groundedness"), "citations + values + wording"),
                ("Citation coverage", am.get("citation_coverage"), "factual sentences cited"),
                ("Citation validity", am.get("citation_validity"), "no invented REF numbers"),
                ("Numeric grounding", am.get("numeric_grounding"), "values found in evidence"),
                ("Completeness", am.get("completeness"), "question terms addressed"),
            ])
        if rm:
            st.markdown("**Retrieval**")
            _metric_cells([
                ("Top evidence score", rm.get("top_score"), "best reranker score"),
                ("Mean evidence score", rm.get("mean_topk_score"), "of blocks sent to LLM"),
                ("Key-term recall", rm.get("key_term_recall"), "question terms in evidence"),
                ("Evidence used", rm.get("precision_at_k"), "blocks actually cited"),
                ("Source diversity", rm.get("source_diversity"), "modalities / 6"),
            ])
            st.caption(f"Modalities: {', '.join(rm.get('modalities', [])) or '—'} · documents: {rm.get('documents', 0)} · "
                       f"pages: {rm.get('pages', 0)} · candidates considered: {rm.get('candidates_considered', 0)}")
        with st.expander("Raw values"):
            st.json({"answer": am, "retrieval": rm, "timings": res.timings})


def render_doc_stats(rec: DocumentRecord) -> None:
    s = rec.stats
    cols = st.columns(7)
    vals = [
        ("📄 pages", rec.pages), ("📝 text", s.get("text", 0)), ("📊 tables", s.get("table", 0)),
        ("🔢 rows/pins", s.get("row", 0) + s.get("pin", 0)), ("➗ equations", s.get("equation", 0)),
        ("🖼️ figures", s.get("figure", 0)), ("📦 chunks", s.get("total", 0)),
    ]
    for col, (lbl, v) in zip(cols, vals):
        col.metric(lbl, v)


def render_browse(chunks: List[Chunk], mode: str, resolve_path) -> None:
    if mode == "Text chunks":
        text = [c for c in chunks if c.modality == "text"]
        depths = pd.Series([c.depth for c in text]).value_counts().sort_index() if text else pd.Series(dtype=int)
        if len(depths):
            st.caption("Chunk depth distribution (structural split level)")
            st.bar_chart(depths)
        for c in text[:400]:
            label = f"[p{c.page}] {c.section}" + (f" › depth {c.depth}, chunk {c.chunk_idx}" if c.depth else "")
            with st.expander(label + (f" | {c.parent_ctx}" if c.parent_ctx else "")):
                st.markdown(f'<div class="src-body">{_esc(c.raw_content[:1500])}</div>', unsafe_allow_html=True)
                if c.meta:
                    st.json(c.meta)
    elif mode == "Tables":
        tables = [c for c in chunks if c.modality == "table"]
        if not tables:
            st.info("No tables detected.")
        for c in tables:
            with st.expander(f"[p{c.page}] {c.section} · part {c.meta.get('part', 1)}/{c.meta.get('parts', 1)}"):
                lines = c.raw_content.split("\n")
                try:
                    rows = [l.split(" | ") for l in lines]
                    width = len(rows[0])
                    df = pd.DataFrame([r + [""] * (width - len(r)) for r in rows[1:]], columns=rows[0])
                    st.dataframe(df, width="stretch")
                except Exception:
                    st.text(c.raw_content[:2000])
    elif mode == "Rows / pins":
        rows = [c for c in chunks if c.modality in ("row", "pin")]
        if not rows:
            st.info("No table rows detected.")
        else:
            st.dataframe(pd.DataFrame([{"Page": c.page, "Type": c.modality, "Section": c.section,
                                        "Row": c.raw_content.split("\n")[-1]} for c in rows]), width="stretch")
    elif mode == "Equations":
        eqs = [c for c in chunks if c.modality == "equation"]
        if eqs:
            st.dataframe(pd.DataFrame([{"Page": c.page, "Section": c.section, "Expression": c.raw_content}
                                       for c in eqs]), width="stretch")
        else:
            st.info("No equations detected.")
    elif mode == "Figures":
        figs = [c for c in chunks if c.modality == "figure"]
        if not figs:
            st.info("No figures found.")
        cols = st.columns(3)
        for i, c in enumerate(figs):
            with cols[i % 3]:
                p = resolve_path(c.figure_path)
                if p:
                    st.image(str(p), caption=c.caption or f"p{c.page}", width="stretch")
                if c.vision_text:
                    with st.expander("Vision description"):
                        st.write(c.vision_text)
