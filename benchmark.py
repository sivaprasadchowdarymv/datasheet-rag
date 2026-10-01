"""
benchmark.py — measured comparison of the ORIGINAL v3 app vs the NEW v4 pipeline.

Nothing in the output is estimated: every number is timed or counted on your
machine, on your PDFs. If a stage cannot run (e.g. Ollama not reachable for
the original app) it is reported as "skipped" with the reason.

Usage
-----
  python benchmark.py --pdf datasheets/LM317.pdf --questions questions.txt
  python benchmark.py --pdf a.pdf b.pdf --original ../amalbro_ds_1.py --mode local
  python benchmark.py --pdf a.pdf --mode test          # fully offline smoke run

questions.txt — one question per line, optionally "question | expected text":
  What is the maximum output current? | 1.5
  What is the dropout voltage? | 1.3
  What is the price of this part? | NOT FOUND

Metrics
-------
  ingest_s            wall time to parse + embed + index one PDF (cold, no cache)
  reload_s            wall time to ingest the same PDF again (cache hit)
  chunks_by_type      number of retrievable units per modality
  avg_chunk_tokens    mean size of text chunks (≈ chars/4, same estimator as v3)
  answer_s            per-question latency, cold (no query cache)
  answer_cached_s     per-question latency on repeat (v4 query cache)
  retrieval_hit       expected text appears in a retrieved source
  answer_hit          expected text appears in the answer
  not_found_correct   question expected NOT FOUND and the system said so
  embed_calls         (v4) embedding calls per question — v3 embeds the query 4×
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_questions(path: Optional[str]) -> List[Tuple[str, str]]:
    if not path:
        return [("What is the maximum output current?", ""), ("What is the pin configuration?", "")]
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, _, exp = line.partition("|")
        out.append((q.strip(), exp.strip()))
    return out


def _hit(expected: str, text: str) -> Optional[bool]:
    if not expected or expected.upper() == "NOT FOUND":
        return None
    return expected.lower().replace(" ", "") in (text or "").lower().replace(" ", "")


def _summary(xs: List[float]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "max": round(max(xs), 3)}


def _rate(xs: List[Optional[bool]]) -> Optional[str]:
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)}" if xs else None


# ---------------------------------------------------------------------------
# NEW (v4)
# ---------------------------------------------------------------------------
class _CountingEmbedder:
    """Wraps the embedding engine to count calls/texts (does not change results)."""

    def __init__(self, inner):
        self.inner, self.calls, self.texts = inner, 0, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed_query(self, text):
        self.calls += 1
        self.texts += 1
        return self.inner.embed_query(text)

    def embed_documents(self, texts, progress=None):
        self.calls += 1
        self.texts += len(texts)
        return self.inner.embed_documents(texts, progress=progress)


def bench_new(pdfs: List[Path], questions, mode: str, work: Path) -> Dict[str, Any]:
    os.environ["APP_MODE"] = mode
    from config import load_settings
    from rag.pipeline import RAGPipeline, build_engines
    from rag.text_utils import tok

    s = load_settings(mode).with_overrides(
        data_dir=str(work / "data"), cache_dir=str(work / "cache"), storage_backend="local")
    engines = build_engines(s)
    counter = _CountingEmbedder(engines.embedder)
    engines.embedder = counter
    pipe = RAGPipeline(s, engines)

    res: Dict[str, Any] = {"system": f"v4 ({mode})", "models": {
        "embedding": f"{s.embedding_provider}:{s.embedding_model}",
        "reranker": (f"{s.reranker_provider}:{s.reranker_model}" if s.reranker_provider == "fastembed"
                     else s.reranker_provider) if s.enable_reranker else "off",
        "llm": f"{s.resolved_llm_provider}:{s.llm_model}"}, "docs": []}
    doc_ids = []
    for p in pdfs:
        data = p.read_bytes()
        t0 = time.perf_counter()
        rec, _ = pipe.ingest(data, p.name)
        t_ingest = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipe.ingest(data, p.name)
        t_reload = time.perf_counter() - t0
        chunks = pipe.repo.load_chunks(rec.doc_id)
        by_type: Dict[str, int] = {}
        for c in chunks:
            by_type[c.modality] = by_type.get(c.modality, 0) + 1
        text_sizes = [tok(c.raw_content) for c in chunks if c.modality == "text"]
        res["docs"].append({"name": p.name, "pages": rec.pages, "ingest_s": round(t_ingest, 3),
                            "reload_s": round(t_reload, 4), "chunks_by_type": by_type,
                            "avg_chunk_tokens": round(statistics.mean(text_sizes), 1) if text_sizes else 0,
                            "warnings": rec.warnings})
        doc_ids.append(rec.doc_id)

    rows = []
    for q, exp in questions:
        before = counter.calls
        t0 = time.perf_counter()
        r = pipe.answer(q, doc_ids, use_cache=False)
        t_cold = time.perf_counter() - t0
        calls = counter.calls - before
        t0 = time.perf_counter()
        pipe.answer(q, doc_ids)          # populate cache
        r2 = pipe.answer(q, doc_ids)     # cached
        t_cached = time.perf_counter() - t0
        src_text = " ".join(f"{x.snippet} {x.meta}" for x in r.sources)
        rows.append({
            "question": q, "expected": exp, "kind": r.kind,
            "answer_s": round(t_cold, 3), "answer_cached_s": round(t_cached / 2, 4),
            "cached_flag": r2.from_cache, "embed_calls": calls,
            "retrieval_hit": _hit(exp, src_text), "answer_hit": _hit(exp, r.answer),
            "not_found_correct": (r.kind == "not_found") if exp.upper() == "NOT FOUND" else None,
            "rqs": (r.answer_metrics or {}).get("response_quality_score"),
            "sources": len(r.sources), "answer": r.answer[:300],
        })
    res["questions"] = rows
    return res


# ---------------------------------------------------------------------------
# ORIGINAL (v3) — imported as a module; its UI main() is never called
# ---------------------------------------------------------------------------
def bench_original(path: Path, pdfs: List[Path], questions, work: Path) -> Dict[str, Any]:
    res: Dict[str, Any] = {"system": "v3 (original)", "docs": [], "questions": []}
    cwd = os.getcwd()
    os.chdir(work)   # v3 writes .rag_cache/, feedback.json and figures relative to cwd
    try:
        spec = importlib.util.spec_from_file_location("rag_v3", path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(mod)
        ollama_ok = False
        if getattr(mod, "OLLAMA_AVAILABLE", False):
            try:
                mod.ollama.list()
                ollama_ok = True
            except Exception as e:  # noqa: BLE001
                res["note"] = f"Ollama not reachable ({type(e).__name__}); v3 embeddings/answers skipped"
        else:
            res["note"] = "ollama package not installed; v3 embeddings/answers skipped"
        res["models"] = {"embedding": f"ollama:{mod.EMBED_MODEL}", "llm": f"ollama:{mod.LLM_MODEL}",
                         "reranker": "none"}

        graphs = []
        for p in pdfs:
            data = p.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                t0 = time.perf_counter()
                g = mod.build_graph(data)
                t_ingest = time.perf_counter() - t0
                t0 = time.perf_counter()
                mod.build_graph(data)
                t_reload = time.perf_counter() - t0
            by_type: Dict[str, int] = {}
            for n in g["nodes"]:
                by_type[n.type] = by_type.get(n.type, 0) + 1
            by_type["figure"] = len(g["figures"])
            sizes = [mod.tok(n.content) for n in g["nodes"] if n.type == "text"]
            res["docs"].append({"name": p.name, "ingest_s": round(t_ingest, 3), "reload_s": round(t_reload, 4),
                                "chunks_by_type": by_type,
                                "avg_chunk_tokens": round(statistics.mean(sizes), 1) if sizes else 0,
                                "embedded": sum(1 for n in g["nodes"] if n.embedding)})
            graphs.append(g)

        if not ollama_ok:
            return res
        # v3 answers over ONE document at a time; benchmark against the first PDF.
        g = graphs[0]
        for q, exp in questions:
            with contextlib.redirect_stderr(io.StringIO()):
                t0 = time.perf_counter()
                r = mod.answer_query(q, g)
                t = time.perf_counter() - t0
            src_text = " ".join(f"{s.get('snippet', '')} {s.get('meta', '')}" for s in r.get("sources", []))
            ans = r.get("answer", "")
            res["questions"].append({
                "question": q, "expected": exp, "kind": r.get("type"), "answer_s": round(t, 3),
                "answer_cached_s": None, "retrieval_hit": _hit(exp, src_text), "answer_hit": _hit(exp, ans),
                "not_found_correct": ("NOT FOUND" in ans.upper() or "not found" in ans.lower())
                if exp.upper() == "NOT FOUND" else None,
                "confidence(v3)": (r.get("metrics") or {}).get("confidence"), "sources": len(r.get("sources", [])),
                "answer": ans[:300]})
    finally:
        os.chdir(cwd)
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def summarise(r: Dict[str, Any]) -> Dict[str, Any]:
    qs = r.get("questions", [])
    return {
        "system": r["system"],
        "ingest_s": _summary([d["ingest_s"] for d in r["docs"]]),
        "reload_s": _summary([d["reload_s"] for d in r["docs"]]),
        "answer_s": _summary([q["answer_s"] for q in qs]),
        "answer_cached_s": _summary([q.get("answer_cached_s") for q in qs]),
        "retrieval_hit": _rate([q.get("retrieval_hit") for q in qs]),
        "answer_hit": _rate([q.get("answer_hit") for q in qs]),
        "not_found_correct": _rate([q.get("not_found_correct") for q in qs]),
        "embed_calls_per_q": _summary([q.get("embed_calls") for q in qs]),
    }


def to_markdown(results: List[Dict[str, Any]]) -> str:
    sums = [summarise(r) for r in results]
    keys = ["ingest_s", "reload_s", "answer_s", "answer_cached_s", "retrieval_hit", "answer_hit",
            "not_found_correct", "embed_calls_per_q"]

    def cell(v):
        if isinstance(v, dict):
            return "—" if not v.get("n") else f"mean {v['mean']} / med {v['median']}"
        return "—" if v is None else str(v)

    lines = ["| metric | " + " | ".join(s["system"] for s in sums) + " |",
             "|---|" + "---|" * len(sums)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(cell(s[k]) for s in sums) + " |")
    lines.append("")
    for r in results:
        lines.append(f"### {r['system']}")
        if r.get("note"):
            lines.append(f"_Note: {r['note']}_")
        if r.get("models"):
            lines.append("Models: " + ", ".join(f"{k}={v}" for k, v in r["models"].items()))
        for d in r["docs"]:
            lines.append(f"- {d['name']}: chunks {d['chunks_by_type']}, avg text chunk "
                         f"{d['avg_chunk_tokens']} tok, ingest {d['ingest_s']} s, reload {d['reload_s']} s")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--questions")
    ap.add_argument("--mode", default=os.environ.get("APP_MODE", "local"), choices=["local", "cloud", "test"])
    ap.add_argument("--original", help="path to the original amalbro_ds_1.py (optional)")
    ap.add_argument("--out", default="benchmark_results")
    a = ap.parse_args()

    pdfs = [Path(p) for p in a.pdf]
    questions = load_questions(a.questions)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if a.original:
            (tmp / "v3").mkdir()
            results.append(bench_original(Path(a.original).resolve(), [p.resolve() for p in pdfs],
                                          questions, tmp / "v3"))
        (tmp / "v4").mkdir()
        results.append(bench_new(pdfs, questions, a.mode, tmp / "v4"))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"benchmark-{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = to_markdown(results)
    (out / f"benchmark-{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved: {out}/benchmark-{stamp}.json and .md")


if __name__ == "__main__":
    main()
