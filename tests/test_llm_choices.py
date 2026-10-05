"""Extra answer models (LLM2_* … LLM5_*), e.g. Kimi via OpenRouter or Ollama Cloud."""
from __future__ import annotations

import pytest

from tests.conftest import ROOT, FakeLLM


def test_llm_choices_are_parsed_and_secrets_masked(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "main-key")
    monkeypatch.setenv("LLM2_LABEL", "Kimi K2.6 (OpenRouter free)")
    monkeypatch.setenv("LLM2_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("LLM2_MODEL", "moonshotai/kimi-k2.6:free")
    monkeypatch.setenv("LLM2_API_KEY", "sk-or-secret")
    monkeypatch.setenv("LLM3_MODEL", "kimi-k2.6:cloud")          # same server as the main LLM
    from config import load_settings

    s = load_settings("cloud")
    assert s.llm_choices == (
        ("Kimi K2.6 (OpenRouter free)", "openai_compat", "https://openrouter.ai/api/v1",
         "moonshotai/kimi-k2.6:free", "sk-or-secret"),
        ("kimi-k2.6:cloud", "openai_compat", "https://llm.example.com/v1", "kimi-k2.6:cloud", "main-key"),
    )
    shown = str(s.public_dict())
    assert "sk-or-secret" not in shown and "main-key" not in shown


def test_answer_uses_the_model_passed_per_call(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    kimi = FakeLLM()
    kimi.name = "fake:kimi"
    res = pipeline.answer("What is the maximum output current?", [rec.doc_id], use_cache=False, llm=kimi)
    assert kimi.calls == 1 and "[REF-1]" in res.answer
    # extractive chosen explicitly → no LLM call, still cited
    res2 = pipeline.answer("What is the maximum output current?", [rec.doc_id], use_cache=False, llm=None)
    assert kimi.calls == 1 and "REF-" in res2.answer


def test_cache_is_separate_per_answer_model(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    a, b = FakeLLM(), FakeLLM()
    a.name, b.name = "fake:a", "fake:b"
    q = "What is the maximum output current?"
    pipeline.answer(q, [rec.doc_id], llm=a)
    pipeline.answer(q, [rec.doc_id], llm=b)          # different model → must not reuse a's cached answer
    assert a.calls == 1 and b.calls == 1


def test_app_shows_answer_model_menu(monkeypatch, tmp_path):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("APP_MODE", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM2_LABEL", "Kimi K2.6")
    monkeypatch.setenv("LLM2_BASE_URL", "http://127.0.0.1:9/v1")      # unreachable on purpose
    monkeypatch.setenv("LLM2_MODEL", "kimi-k2.6:cloud")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    menu = at.sidebar.selectbox[0]
    assert menu.options == ["Kimi K2.6", "Extractive (no LLM, verbatim excerpts)"]
    at.sidebar.selectbox[0].set_value("Extractive (no LLM, verbatim excerpts)").run()
    assert not at.exception, at.exception
