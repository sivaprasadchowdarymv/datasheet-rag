"""
End-to-end UI smoke test with Streamlit's AppTest (no browser, no network).

A document is ingested into a temporary DATA_DIR first, then the real app.py
is executed: the page must render without exceptions, the document must be
listed, and asking a question must produce an answer with a [REF-n] citation.
"""
from __future__ import annotations

import pytest

from tests.conftest import ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_renders_and_answers(monkeypatch, tmp_path, pdf_a):
    monkeypatch.setenv("APP_MODE", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from config import load_settings
    from rag.pipeline import RAGPipeline, build_engines

    s = load_settings("test")
    assert s.data_dir == str(tmp_path / "data")
    pipe = RAGPipeline(s, build_engines(s))
    rec, _ = pipe.ingest(pdf_a, "LM317.pdf")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    at.chat_input[0].set_value("What is the maximum output current?").run()
    assert not at.exception, at.exception
    rendered = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "1.5" in rendered and "REF-" in rendered, rendered[:500]
