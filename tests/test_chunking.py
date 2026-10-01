"""PDF extraction + chunking: text lane, recursive chunker, tables, equations, figures."""
from rag.chunker import recursive_chunk
from rag.document_processor import is_equation_line
from rag.text_utils import extract_metadata, tok, tokenize


def test_recursive_chunk_respects_size_and_packs_small_lines():
    text = "\n".join(f"Line {i} describes parameter number {i} of the regulator." for i in range(200))
    pieces = recursive_chunk(text, max_tokens=120, overlap_tokens=20, min_tokens=20)
    assert len(pieces) > 1
    # packing: far fewer chunks than lines (v3 produced ~one chunk per line here)
    assert len(pieces) < 60
    # overlap only adds a bounded tail, never compounds
    assert all(tok(p.text) <= 120 * 1.15 + 25 for p in pieces)
    assert pieces[1].text.startswith("…")


def test_recursive_chunk_short_text_single_piece():
    pieces = recursive_chunk("Short text.", 400, 60, 40)
    assert len(pieces) == 1 and pieces[0].text == "Short text." and pieces[0].depth == 0


def test_equation_detector_is_precise():
    assert is_equation_line("VOUT = VREF * (1 + R2 / R1) + IADJ * R2      (1)")
    assert is_equation_line("P = I^2 * R")
    assert not is_equation_line("VCC = 3.3 V")            # spec value, not an equation
    assert not is_equation_line("The supply is 5 V nominal.")


def test_metadata_and_tokenizer_keep_technical_tokens():
    m = extract_metadata("VIN = 12 V, IOUT = 1.5 A at TA = 25 °C, LM317T")
    assert "voltage" in m and "current" in m and "LM317T" in m["part_numbers"]
    toks = tokenize("Supply 3.3V on GPIO12 via I2C")
    assert "3.3v" in toks and "3.3" in toks and "gpio12" in toks and "i2c" in toks


def test_pdf_extracts_all_modalities(pipeline, pdf_a):
    rec, status = pipeline.ingest(pdf_a, "lm317.pdf")
    assert status == "new"
    s = rec.stats
    assert s["text"] >= 2 and s["table"] >= 2 and s["row"] >= 4 and s["pin"] == 3
    assert s["equation"] >= 1 and s["figure"] >= 1
    chunks = pipeline.repo.load_chunks(rec.doc_id)
    row = next(c for c in chunks if c.modality == "row" and "Output current" in c.raw_content)
    assert "Max: 1.5" in row.content and "Parameter | Symbol" in row.raw_content     # header injected
    assert row.parent_ctx.endswith("Absolute Maximum Ratings")                        # parent context
    fig = next(c for c in chunks if c.modality == "figure")
    assert fig.caption.startswith("Figure 1") and pipeline.repo.resolve(fig.figure_path) is not None
    assert fig.figure_path.startswith(f"figures/{rec.doc_id}/")                       # namespaced per doc


def test_malformed_and_empty_pdfs_are_rejected(pipeline):
    import pytest
    from rag.pipeline import IngestionError

    with pytest.raises(IngestionError, match="not a PDF"):
        pipeline.ingest(b"hello world", "x.pdf")
    with pytest.raises(IngestionError):
        pipeline.ingest(b"%PDF-1.7\n garbage", "broken.pdf")
