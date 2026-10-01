"""Semantic, keyword, hybrid retrieval and reranking."""
from rag.query_analysis import analyze_query
from rag.reranker import LexicalReranker, NoReranker
from rag.retriever import HybridRetriever


def _retriever(pipeline, doc_ids):
    chunks, bm25, fuzzy = pipeline._lexical_for(doc_ids)
    return HybridRetriever(chunks, bm25, fuzzy, pipeline.store, doc_ids, top_k=20), bm25, chunks


def test_keyword_retrieval_finds_exact_symbol(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    _, bm25, chunks = _retriever(pipeline, [rec.doc_id])
    hits = bm25.search("TSTG storage temperature", 5)
    assert hits and "TSTG" in chunks[hits[0][0]].raw_content


def test_semantic_retrieval_uses_faiss(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    q = pipeline.e.embedder.embed_query("pin configuration adjustment input")
    hits = pipeline.store.search(q, 5, [rec.doc_id])
    assert len(hits) == 5 and hits[0][1] >= hits[-1][1]


def test_hybrid_retrieval_is_modality_aware(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    r, _, _ = _retriever(pipeline, [rec.doc_id])
    qa = analyze_query("What is the pin configuration?")
    assert "pin" in qa.intents
    q = pipeline.e.embedder.embed_query(qa.retrieval_query)
    ranked = LexicalReranker().rerank(qa, r.retrieve(qa, q, 20))
    assert ranked[0].chunk.modality in ("pin", "table")
    qa2 = analyze_query("Show the equation used to set the output voltage")
    ranked2 = LexicalReranker().rerank(qa2, r.retrieve(qa2, pipeline.e.embedder.embed_query(qa2.retrieval_query), 20))
    assert any(c.chunk.modality == "equation" for c in ranked2[:3])


def test_reranking_produces_scores_and_order(pipeline, pdf_a):
    rec, _ = pipeline.ingest(pdf_a, "a.pdf")
    r, _, _ = _retriever(pipeline, [rec.doc_id])
    qa = analyze_query("maximum output current")
    cands = r.retrieve(qa, pipeline.e.embedder.embed_query(qa.retrieval_query), 20)
    for rr in (LexicalReranker(), NoReranker()):
        ranked = rr.rerank(qa, list(cands))
        assert all(c.rerank is not None and 0 <= c.rerank <= 1.0001 for c in ranked)
        assert [c.final for c in ranked] == sorted([c.final for c in ranked], reverse=True)
    assert "Output current" in LexicalReranker().rerank(qa, list(cands))[0].chunk.raw_content


def test_follow_up_question_is_expanded():
    qa = analyze_query("and its max?", history=["What is the output current of LM317?"])
    assert "LM317" in qa.retrieval_query
