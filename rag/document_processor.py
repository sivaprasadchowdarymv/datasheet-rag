"""
rag/document_processor.py — PDF → list[Chunk]  (runs ONCE per document).

Lanes (same modalities as v3):
  TEXT     font-aware heading detection → heading hierarchy breadcrumb →
           recursive_chunk() with parent-context prefix + overlap
  TABLE    page.find_tables() → table parts (header repeated) + ROW / PIN nodes
           rendered as "Header: value; …" so every row is self-contained
  EQUATION tightened detector + neighbouring lines as context
  FIGURE   raster images AND vector drawings (most datasheet plots are vector,
           which v3 never saw), rendered to PNG, caption found by geometry

Changes vs v3 (all deliberate, see docs/FEATURE_CHECKLIST.md):
  • headings carry across pages (v3 restarted at "General" on every page)
  • table text is not duplicated into the text lane (configurable)
  • figure files are namespaced per document (v3 overwrote fig_p1_0.png
    across different PDFs — a real cross-document bug)
  • captions are matched to the nearest "Figure N" block, not the first
    block on the page that contains the word "fig"
  • scanned pages are detected and reported; optional OCR via Tesseract
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pymupdf

from .chunker import recursive_chunk, with_context_prefix
from .schema import Chunk
from .text_utils import extract_metadata, normalize, tok

ProgressFn = Callable[[float, str], None]


class PDFProcessingError(Exception):
    """Raised with a human-readable message for malformed/encrypted/empty PDFs."""


@dataclass
class ParseOptions:
    max_chunk_tokens: int = 400
    overlap_tokens: int = 60
    min_chunk_tokens: int = 40
    exclude_table_text: bool = True
    enable_ocr: bool = False
    max_figures_per_page: int = 6
    max_pages: int = 400


@dataclass
class ParsedDocument:
    chunks: List[Chunk]
    page_count: int
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


# v3 regex heading detector — used as a fallback when a PDF has no font signal.
_HEADING_RE = re.compile(
    r"^([A-Z][A-Z0-9 \-()/.]{4,}|\d+(?:\.\d+)*[.)]?\s+[A-Z].{3,60}|[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,7})$"
)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]")
_CAPTION_RE = re.compile(r"^\s*(fig\.?|figure)\s*[0-9]+", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\s*table\s*[0-9]+", re.IGNORECASE)
_PIN_HEADER_RE = re.compile(r"\b(pin|pins|pad|ball|signal|gpio|terminal|port)\b", re.IGNORECASE)

# Equations: an assignment whose right-hand side is an EXPRESSION, a relation
# with a math operator, or a numbered display equation. Plain "VCC = 3.3 V"
# spec lines are deliberately excluded (they live in text/table nodes and
# metadata) — v3's detector matched almost every spec line.
_EQ_RHS_OP = re.compile(r"[+*/^×÷√∑∫·]|\b(?:log|ln|exp|sqrt|sin|cos)\b|[A-Za-z]\w*\s*[(+\-*/]")
_EQ_SYMBOLS = re.compile(r"[√∑∫≈∝]")


def is_equation_line(line: str) -> bool:
    s = line.strip()
    if not (4 < len(s) < 220) or s.count(" ") > 40:
        return False
    if _EQ_SYMBOLS.search(s) and re.search(r"[A-Za-z]", s):
        return True
    if "=" in s:
        left, _, right = s.partition("=")
        if not re.search(r"[A-Za-z]", left) or len(left) > 40 or not right.strip():
            return False
        if _EQ_RHS_OP.search(right):
            return True
        if re.search(r"\(\d+\)\s*$", s):          # "... (3)" numbered display equation
            return True
    return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _inside(bbox: Tuple[float, float, float, float], rects: List[pymupdf.Rect]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r.contains(pymupdf.Point(cx, cy)) for r in rects)


def _overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return inter.get_area() / max(1e-6, a.get_area())


def _body_font_size(doc: pymupdf.Document, pages: int) -> float:
    sizes: Dict[float, int] = {}
    for pno in range(min(pages, 15)):
        try:
            d = doc[pno].get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        except Exception:
            continue
        for b in d.get("blocks", []):
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    t = sp.get("text", "").strip()
                    if t:
                        key = round(sp.get("size", 0) * 2) / 2
                        sizes[key] = sizes.get(key, 0) + len(t)
    return max(sizes, key=sizes.get) if sizes else 10.0


@dataclass
class _Line:
    text: str
    bbox: Tuple[float, float, float, float]
    size: float
    bold: bool
    block_no: int


def _page_lines(page: pymupdf.Page, textpage=None) -> List[_Line]:
    kw = {"flags": pymupdf.TEXTFLAGS_TEXT}
    if textpage is not None:
        kw["textpage"] = textpage
    d = page.get_text("dict", **kw)
    out: List[_Line] = []
    for bno, b in enumerate(d.get("blocks", [])):
        for ln in b.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = normalize("".join(s["text"] for s in ln["spans"])).strip()
            if not text:
                continue
            size = max(s.get("size", 0) for s in spans)
            bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
            out.append(_Line(text, tuple(ln["bbox"]), size, bold, bno))
    return out


def _is_heading(line: _Line, body: float, font_mode: bool) -> bool:
    t = line.text
    if not (3 < len(t) < 100) or t.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADING_RE.match(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3 or letters / max(1, len(t)) < 0.5:
        return False
    if re.match(r"^[\d\s.,\-+±%]+$", t):
        return False
    if font_mode:
        if line.size >= body * 1.18:
            return True
        if line.bold and line.size >= body * 0.95 and len(t) < 80:
            return True
        return bool(_NUMBERED_HEADING_RE.match(t)) and (line.bold or line.size > body * 1.05)
    return bool(_HEADING_RE.match(t))


def _clean_cell(c) -> str:
    return re.sub(r"\s+", " ", normalize(str(c))).strip() if c is not None else ""


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def parse_pdf(
    pdf_bytes: bytes,
    doc_id: str,
    doc_name: str,
    figures_dir: Path,
    figures_rel_prefix: str,
    opts: ParseOptions,
    progress: Optional[ProgressFn] = None,
) -> ParsedDocument:
    progress = progress or (lambda f, m: None)
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFProcessingError(f"This file could not be opened as a PDF ({type(e).__name__}).") from e
    if doc.needs_pass:
        raise PDFProcessingError("This PDF is password-protected. Please upload an unlocked copy.")
    n_pages = len(doc)
    if n_pages == 0:
        raise PDFProcessingError("This PDF has no pages.")
    if n_pages > opts.max_pages:
        raise PDFProcessingError(f"This PDF has {n_pages} pages; the limit is {opts.max_pages}.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    body = _body_font_size(doc, n_pages)
    warnings: List[str] = []
    chunks: List[Chunk] = []

    # Decide font-mode vs regex-mode once per document.
    font_mode = False
    for pno in range(min(n_pages, 5)):
        try:
            if any(_is_heading(l, body, True) for l in _page_lines(doc[pno])):
                font_mode = True
                break
        except Exception:
            pass

    heading_stack: List[Tuple[float, str]] = []   # (font size, title) — carried across pages
    scanned_pages: List[int] = []

    def breadcrumb() -> str:
        return " > ".join(t for _, t in heading_stack[-2:])

    def current_section() -> str:
        return heading_stack[-1][1] if heading_stack else "General"

    def push_heading(line: _Line) -> None:
        if font_mode:
            # real hierarchy: a heading closes every open heading of equal/smaller font
            while heading_stack and heading_stack[-1][0] <= line.size + 0.01:
                heading_stack.pop()
        elif len(heading_stack) >= 8:
            heading_stack.pop(0)          # regex mode: v3 behaviour (last two headings seen)
        heading_stack.append((line.size, line.text[:90]))

    def add(**kw) -> Chunk:
        c = Chunk(chunk_id=f"{doc_id}-{len(chunks):05d}", doc_id=doc_id, doc_name=doc_name, **kw)
        chunks.append(c)
        return c

    for pno in range(n_pages):
        page = doc[pno]
        page_num = pno + 1
        progress(pno / n_pages, f"Parsing page {page_num}/{n_pages}")
        page_rect = page.rect

        # ---------------- tables first (their bboxes mask the text lane) -----
        tables = []
        try:
            tables = list(page.find_tables().tables)
        except Exception:
            tables = []
        table_rects = [pymupdf.Rect(t.bbox) for t in tables]

        # ---------------- text lines (with optional OCR) ----------------------
        try:
            lines = _page_lines(page)
        except Exception:
            lines = []
        char_count = sum(len(l.text) for l in lines)
        if char_count < 25 and (page.get_images() or page.get_drawings()):
            if opts.enable_ocr:
                try:
                    tp = page.get_textpage_ocr(full=True, dpi=200)
                    lines = _page_lines(page, textpage=tp)
                except Exception:
                    scanned_pages.append(page_num)
            else:
                scanned_pages.append(page_num)

        text_lines = [l for l in lines if not (opts.exclude_table_text and _inside(l.bbox, table_rects))]
        heading_positions: List[Tuple[float, str, str]] = [(-1.0, current_section(), breadcrumb())]

        # ---------------- TEXT lane ------------------------------------------
        sections: List[Tuple[str, str, str]] = []   # (section, breadcrumb, text)
        buf: List[str] = []
        last_block = None

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                sections.append((current_section(), breadcrumb(), text))
            buf.clear()

        for ln in text_lines:
            if _is_heading(ln, body, font_mode):
                flush()
                push_heading(ln)
                heading_positions.append((ln.bbox[1], current_section(), breadcrumb()))
                last_block = ln.block_no
                continue
            if last_block is not None and ln.block_no != last_block:
                buf.append("")                    # paragraph boundary for the chunker
            buf.append(ln.text)
            last_block = ln.block_no
        flush()

        for section, crumb, text in sections:
            for piece in recursive_chunk(text, opts.max_chunk_tokens, opts.overlap_tokens, opts.min_chunk_tokens):
                add(
                    modality="text", section=section, content=with_context_prefix(piece.text, crumb),
                    raw_content=piece.text, page=page_num, chunk_idx=piece.chunk_idx,
                    depth=piece.depth, parent_ctx=crumb, meta=extract_metadata(piece.text),
                )

        def section_at(y: float) -> Tuple[str, str]:
            sec, crumb = heading_positions[0][1], heading_positions[0][2]
            for hy, hs, hc in heading_positions:
                if hy <= y:
                    sec, crumb = hs, hc
            return sec, crumb

        # ---------------- EQUATION lane --------------------------------------
        seen_eq = set()
        for i, ln in enumerate(text_lines):
            if not is_equation_line(ln.text) or ln.text in seen_eq:
                continue
            seen_eq.add(ln.text)
            sec, crumb = section_at(ln.bbox[1])
            prev_l = text_lines[i - 1].text if i > 0 else ""
            next_l = text_lines[i + 1].text if i + 1 < len(text_lines) else ""
            ctx = "\n".join(x for x in (prev_l, ln.text, next_l) if x)
            add(
                modality="equation", section=sec, content=with_context_prefix(ctx, crumb),
                raw_content=ln.text, page=page_num, parent_ctx=crumb, meta=extract_metadata(ctx),
            )

        # ---------------- TABLE lane ------------------------------------------
        all_blocks = []
        try:
            all_blocks = page.get_text("blocks") or []
        except Exception:
            pass
        for t_idx, t in enumerate(tables):
            try:
                rows_raw = t.extract() or []
                raw_names = [_clean_cell(n) for n in t.header.names]
                if not t.header.external and rows_raw:
                    rows_raw = rows_raw[1:]
                elif t.header.external and rows_raw and sum(1 for n in raw_names if n) <= max(1, len(raw_names) // 2):
                    # PyMuPDF sometimes takes the section title above the grid as an
                    # "external header"; the real column names are then the first row.
                    raw_names = [_clean_cell(n) for n in rows_raw[0]]
                    rows_raw = rows_raw[1:]
                names = [(n or f"Col{i + 1}") for i, n in enumerate(raw_names)]
            except Exception:
                continue
            rows = [[_clean_cell(c) for c in r] for r in rows_raw]
            rows = [r for r in rows if any(r)]
            if not rows and not any(names):
                continue
            trect = pymupdf.Rect(t.bbox)
            sec, crumb = section_at(trect.y0)
            caption = ""
            for b in all_blocks:
                btxt = _clean_cell(b[4])
                if _TABLE_CAPTION_RE.match(btxt) and abs(b[3] - trect.y0) < 60:
                    caption = btxt[:160]
                    break
            header_line = " | ".join(names)
            row_lines = [" | ".join(r) for r in rows]
            table_id = f"p{page_num}t{t_idx}"
            is_pin = bool(_PIN_HEADER_RE.search(header_line + " " + caption + " " + sec))
            title = f"TABLE {caption or sec} (page {page_num})"

            # table parts: ≤ 1.5× chunk size, header repeated in every part
            parts: List[List[str]] = []
            cur: List[str] = []
            for rl in row_lines:
                if cur and tok(header_line + "\n" + "\n".join(cur + [rl])) > opts.max_chunk_tokens * 1.5:
                    parts.append(cur)
                    cur = []
                cur.append(rl)
            if cur or not parts:
                parts.append(cur)
            for p_idx, part in enumerate(parts):
                raw = header_line + ("\n" + "\n".join(part) if part else "")
                add(
                    modality="table", section=sec, content=with_context_prefix(f"{title}\n{raw}", crumb),
                    raw_content=raw, page=page_num, chunk_idx=p_idx, parent_ctx=crumb,
                    meta={**extract_metadata(raw), "table_id": table_id, "headers": names[:12],
                          "part": p_idx + 1, "parts": len(parts), "rows": len(part), "caption": caption},
                )
            # ROW / PIN nodes — header always injected (v3), plus key:value rendering
            for r_idx, r in enumerate(rows):
                kv = "; ".join(f"{h}: {v}" for h, v in zip(names, r) if v)
                if not kv:
                    continue
                raw = f"{header_line}\n{' | '.join(r)}"
                add(
                    modality="pin" if is_pin else "row", section=sec,
                    content=with_context_prefix(f"{title} row: {kv}", crumb),
                    raw_content=raw, page=page_num, chunk_idx=r_idx, parent_ctx=crumb,
                    meta={**extract_metadata(kv), "table_id": table_id},
                )

        # ---------------- FIGURE lane -------------------------------------------
        fig_rects: List[pymupdf.Rect] = []
        page_area = page_rect.get_area()
        try:
            for img in page.get_images(full=True):
                for r in page.get_image_rects(img[0]):
                    if r.width >= 80 and r.height >= 60 and r.get_area() >= 0.02 * page_area:
                        fig_rects.append(pymupdf.Rect(r))
        except Exception:
            pass
        try:
            for r in page.cluster_drawings():
                r = pymupdf.Rect(r)
                area = r.get_area()
                if not (0.04 * page_area <= area <= 0.85 * page_area) or r.width < 100 or r.height < 60:
                    continue
                if any(_overlap_ratio(r, tr) > 0.5 for tr in table_rects):
                    continue
                fig_rects.append((r + (-14, -14, 14, 14)) & page_rect)   # include axis labels
        except Exception:
            pass
        uniq: List[pymupdf.Rect] = []
        for r in sorted(fig_rects, key=lambda x: -x.get_area()):
            if all(_overlap_ratio(r, u) < 0.5 and _overlap_ratio(u, r) < 0.5 for u in uniq):
                uniq.append(r)
        for f_idx, r in enumerate(sorted(uniq, key=lambda x: (x.y0, x.x0))[: opts.max_figures_per_page]):
            caption = ""
            best = 1e9
            for b in all_blocks:
                btxt = _clean_cell(b[4])
                if not _CAPTION_RE.match(btxt):
                    continue
                dist = (b[1] - r.y1) if b[1] >= r.y1 - 5 else (r.y0 - b[3]) * 1.5
                if -5 <= dist < best and dist < 120:
                    best, caption = dist, btxt[:220]
            nearby = " ".join(
                _clean_cell(b[4]) for b in all_blocks
                if pymupdf.Rect(b[:4]).intersects(r + (-40, -40, 40, 40)) and not _CAPTION_RE.match(_clean_cell(b[4]))
            )[:300]
            fname = f"p{page_num}_{f_idx}.png"
            try:
                pix = page.get_pixmap(clip=r, dpi=150)
                pix.save(str(figures_dir / fname))
            except Exception:
                continue
            sec, crumb = section_at(r.y0)
            label = caption or f"Figure {f_idx + 1} on page {page_num}"
            body_txt = f"FIGURE (page {page_num}): {label}" + (f"\nNearby text: {nearby}" if nearby else "")
            add(
                modality="figure", section=sec, content=with_context_prefix(body_txt, crumb),
                raw_content=label, page=page_num, chunk_idx=f_idx, parent_ctx=crumb,
                meta={"bbox": [round(v, 1) for v in r], **extract_metadata(caption)},
                figure_path=f"{figures_rel_prefix}/{fname}", caption=caption or None,
            )

    progress(1.0, "Parsing complete")
    doc.close()

    if scanned_pages:
        pages_txt = ", ".join(map(str, scanned_pages[:15])) + ("…" if len(scanned_pages) > 15 else "")
        if opts.enable_ocr:
            warnings.append(f"OCR failed on page(s) {pages_txt}; their content is not searchable.")
        else:
            warnings.append(
                f"Page(s) {pages_txt} have no text layer (scanned). Set ENABLE_OCR=true "
                "and install Tesseract to make them searchable."
            )
    if not chunks:
        raise PDFProcessingError(
            "No text, tables or figures could be extracted. The PDF may be scanned; enable OCR."
        )

    stats: Dict[str, int] = {}
    for c in chunks:
        stats[c.modality] = stats.get(c.modality, 0) + 1
    stats["total"] = len(chunks)
    depths = [c.depth for c in chunks if c.modality == "text"]
    stats["avg_depth_x100"] = int(100 * statistics.mean(depths)) if depths else 0
    stats["scanned_pages"] = len(scanned_pages)
    return ParsedDocument(chunks=chunks, page_count=n_pages, warnings=warnings, stats=stats)
