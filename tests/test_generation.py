"""Grounded generation, NOT FOUND behaviour, LLM failure fallback, feedback override."""
from dataclasses import replace

from conftest import FakeLLM
from rag.generator import NOT_FOUND
from rag.llm import LLMError, _ThinkFilter


def test_grounded_answer_with_fake_llm(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    llm = FakeLLM()
    pipeline.e = replace(pipeline.e, llm=llm)
    streamed = []
    res = pipeline.answer("What is the maximum output current?", [rec.doc_id], on_token=streamed.append)
    assert res.kind == "answer" and llm.calls == 1
    assert "[REF-1]" in res.answer and "".join(streamed).strip()
    assert res.answer_metrics["citation_validity"] == 1.0


def test_not_found_skips_llm(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    llm = FakeLLM()
    pipeline.e = replace(pipeline.e, llm=llm)
    res = pipeline.answer("What is the price in euros and the shipping date?", [rec.doc_id])
    assert res.kind == "not_found" and res.answer == NOT_FOUND
    assert llm.calls == 0                       # the LLM was never asked to guess


def test_llm_saying_not_found_is_respected(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    pipeline.e = replace(pipeline.e, llm=FakeLLM(reply=NOT_FOUND))
    res = pipeline.answer("What is the maximum output current at 200 C?", [rec.doc_id], use_cache=False)
    assert res.kind == "not_found"


def test_llm_failure_falls_back_to_extractive(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")

    class Broken(FakeLLM):
        def stream(self, messages):
            raise LLMError("Cannot reach Ollama")
            yield  # pragma: no cover

    pipeline.e = replace(pipeline.e, llm=Broken())
    res = pipeline.answer("What is the maximum output current?", [rec.doc_id])
    assert res.kind == "answer" and "[REF-" in res.answer
    assert any("Cannot reach Ollama" in w for w in res.warnings)


def test_think_tags_are_hidden():
    f = _ThinkFilter()
    out = "".join(f.feed(x) for x in ["Hel", "lo <thi", "nk>secret</th", "ink> world"]) + f.flush()
    assert out == "Hello  world"


def test_feedback_override_is_scoped_to_documents(pipeline, pdf_a, pdf_b):
    a, _ = pipeline.ingest(pdf_a, "a.pdf")
    b, _ = pipeline.ingest(pdf_b, "b.pdf")
    pipeline.feedback.add("What is the maximum output current?", "x", [a.doc_id], -1, correction="1.5 A (human)")
    assert pipeline.answer("what is the maximum output current", [a.doc_id]).kind == "feedback_override"
    assert pipeline.answer("What is the maximum output current?", [b.doc_id]).kind != "feedback_override"
    assert pipeline.answer("What is the minimum output current?", [a.doc_id]).kind != "feedback_override"
