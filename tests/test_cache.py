"""Index save / reload, no re-processing, corruption recovery, query cache."""
from rag.pipeline import RAGPipeline


def test_same_pdf_is_not_reprocessed(pipeline, pdf_a):
    rec1, s1 = pipeline.ingest(pdf_a, "first-name.pdf")
    rec2, s2 = pipeline.ingest(pdf_a, "renamed.pdf")
    assert s1 == "new" and s2 == "cached" and rec1.doc_id == rec2.doc_id


def test_index_survives_restart(settings, pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    r1 = pipeline.answer("maximum output current", [rec.doc_id], use_cache=False)
    fresh = RAGPipeline(settings, pipeline.e)          # simulates an app / container restart
    assert [d.doc_id for d in fresh.documents()] == [rec.doc_id]
    r2 = fresh.answer("maximum output current", [rec.doc_id], use_cache=False)
    assert [s.chunk_id for s in r1.sources] == [s.chunk_id for s in r2.sources]


def test_corrupted_index_is_rebuilt_from_chunks(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    fi = pipeline.store.dir / f"{rec.doc_id}.faiss"
    pipeline.store._indexes.clear()
    fi.write_bytes(b"corrupted")
    res = pipeline.answer("maximum output current", [rec.doc_id], use_cache=False)
    assert res.kind == "answer" and fi.stat().st_size > 100


def test_new_embedding_model_gets_its_own_namespace(settings, pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    other = settings.with_overrides(embedding_model="hash-256")
    from rag.pipeline import build_engines

    p2 = RAGPipeline(other, build_engines(other))
    assert p2.namespace != pipeline.namespace
    res = p2.answer("maximum output current", [rec.doc_id])   # re-embeds stored chunks, no re-parse
    assert res.kind == "answer" and p2.store.dim(rec.doc_id) == 256


def test_query_cache_hit(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    r1 = pipeline.answer("maximum output current", [rec.doc_id])
    r2 = pipeline.answer("Maximum output current", [rec.doc_id])
    assert not r1.from_cache and r2.from_cache and r1.answer == r2.answer


def test_delete_removes_everything(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    pipeline.delete(rec.doc_id)
    assert pipeline.documents() == [] and not pipeline.store.has_document(rec.doc_id)
    assert not pipeline.repo.figures_dir(rec.doc_id).exists()
