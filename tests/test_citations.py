"""Citation generation, validation, popups and multi-document citation correctness."""
from rag.generator import build_evidence, validate_answer
from ui.components import answer_html


def test_invalid_refs_are_removed_and_reported(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    blocks = build_evidence(pipeline.repo.load_chunks(rec.doc_id)[:3])
    v = validate_answer("The output is regulated [REF-1]. Pin 2 is VOUT [REF-9].", blocks)
    assert v.invalid_refs == [9] and v.cited_refs == [1] and "REF-9" not in v.answer


def test_grouped_refs_are_normalised(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    blocks = build_evidence(pipeline.repo.load_chunks(rec.doc_id)[:3])
    v = validate_answer("Value given in two places [REF-1, REF-2].", blocks)
    assert "[REF-1][REF-2]" in v.answer and v.cited_refs == [1, 2]


def test_unsupported_values_are_flagged(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    row = [c for c in pipeline.repo.load_chunks(rec.doc_id) if "Output current" in c.raw_content and c.modality == "row"]
    blocks = build_evidence(row)
    assert validate_answer("The maximum output current is 1.5 A [REF-1].", blocks).unsupported_values == []
    assert "7a" in validate_answer("The maximum output current is 7 A [REF-1].", blocks).unsupported_values


def test_popup_html_is_escaped(pipeline, pdf_a):
    res = pipeline.answer("What is the maximum output current?", [pipeline.ingest(pdf_a, "a.pdf")[0].doc_id])
    html = answer_html("<script>alert(1)</script> value [REF-1]", res.sources)
    assert "<script>" not in html and "cite-popup" in html and "a.pdf" in html


def test_multi_document_citations_point_to_right_doc_and_page(pipeline, pdf_a, pdf_b):
    a, _ = pipeline.ingest(pdf_a, "LM317.pdf")
    b, _ = pipeline.ingest(pdf_b, "LT1085.pdf")
    res = pipeline.answer("Compare the maximum output current of LM317 and LT1085", [a.doc_id, b.doc_id])
    assert res.kind == "answer"
    docs = {s.doc_name for s in res.sources}
    assert {"LM317.pdf", "LT1085.pdf"} <= docs                      # both documents represented
    for s in res.sources:                                           # each citation is self-consistent
        chunk = next(c for c in pipeline.repo.load_chunks(s.doc_id) if c.chunk_id == s.chunk_id)
        assert chunk.doc_name == s.doc_name and chunk.page == s.page
    cur = [s for s in res.sources if s.modality == "row" and "Output current" in s.snippet]
    assert any("1.5" in s.snippet and s.doc_name == "LM317.pdf" and s.page == 1 for s in cur)
    assert any("3.0" in s.snippet and s.doc_name == "LT1085.pdf" and s.page == 1 for s in cur)
