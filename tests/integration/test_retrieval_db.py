"""PostgreSQL → embedding → hybrid retrieval → reranker → RAG, on real fixture records."""

import pytest

from app.ingestion.chunker import Chunker
from app.ingestion.client import FileSource
from app.ingestion.embedder import embed_chunks
from app.ingestion.pipeline import SyncPipeline, chunking_config
from app.llm.base import LLMProvider, LLMResult
from app.rag.pipeline import RAGPipeline
from app.rag.structured import answer_structured
from app.retrieval.embeddings import EmbeddingError
from app.retrieval.hybrid import HybridSearcher
from app.retrieval.query import SearchFilters, analyze_query
from app.retrieval.reranker import RerankerError
from tests.integration.conftest import word_tokens

pytestmark = pytest.mark.integration
TYPES = ["ebook", "blog", "forum", "qna_type1", "qna_type2"]


@pytest.fixture
async def indexed(sm, snapshot_dir, isettings, embedder):
    pipe = SyncPipeline(sm, FileSource(snapshot_dir), isettings, Chunker(word_tokens, chunking_config(isettings)))
    await pipe.run(TYPES)
    await embed_chunks(sm, embedder, isettings)
    return sm


class OverlapReranker:
    """Fake cross-encoder: score = fraction of query words present in the text."""

    def __init__(self, fail=False):
        self.fail = fail

    def score(self, query, texts):
        if self.fail:
            raise RerankerError("boom")
        q = set(query.split())
        return [len(q & set(t.split())) / max(1, len(q)) for t in texts]


def _searcher(isettings, embedder, sm, reranker=False):
    return HybridSearcher(isettings, embedder=embedder, reranker=reranker, sessionmaker=sm)


async def test_hybrid_finds_exact_qna(indexed, isettings, embedder):
    docs, trace = await _searcher(isettings, embedder, indexed).search("ডিসিআর ফি কি অনলাইনে দেয়া যাবে")
    assert trace.vector_hits and trace.lexical_hits
    top = docs[0].metadata
    assert (top["doc_source_type"], top["doc_source_id"]) == ("qna_type2", "24776")
    assert top["vector_score"] is not None and top["lexical_rank"] is not None


async def test_lexical_only_when_embedding_fails(indexed, isettings, embedder, monkeypatch):
    s = _searcher(isettings, embedder, indexed)

    async def broken(_):
        raise EmbeddingError("down")
    monkeypatch.setattr(s.embedder, "aembed_query", broken)
    docs, trace = await s.search("ডিসিআর ফি অনলাইনে")
    assert trace.embedding_failed and trace.vector_hits == 0 and docs
    assert docs[0].metadata["doc_source_id"] == "24776"


async def test_metadata_filters(indexed, isettings, embedder):
    s = _searcher(isettings, embedder, indexed)
    docs, _ = await s.search("ভূমি", SearchFilters(source_types=["ebook"]), final_k=10)
    assert docs and all(d.metadata["doc_source_type"] == "ebook" for d in docs)
    docs, _ = await s.search("ভূমি", SearchFilters(document_ids=["241"]), final_k=10)
    assert docs and {d.metadata["doc_source_id"] for d in docs} == {"241"}
    docs, _ = await s.search("ভূমি", SearchFilters(categories=["namjari"]), final_k=10)
    assert all(d.metadata["category"] == "namjari" for d in docs)


async def test_section_number_query_boosts_that_section(indexed, isettings, embedder):
    q = analyze_query("পার্বত্য চট্টগ্রাম ভূমি-বিরোধ নিষ্পত্তি কমিশন আইন ধারা ২")
    docs, _ = await _searcher(isettings, embedder, indexed).search(q, SearchFilters(document_ids=["241"]))
    assert docs[0].metadata["section_number"] == "2"


async def test_reranker_and_fallback(indexed, isettings, embedder):
    docs, trace = await _searcher(isettings, embedder, indexed, OverlapReranker()).search("খতিয়ান কী")
    assert trace.reranked and docs[0].metadata["rerank_score"] is not None
    scores = [d.metadata["rerank_score"] for d in docs]
    assert scores[0] == max(scores)
    docs, trace = await _searcher(isettings, embedder, indexed, OverlapReranker(fail=True)).search("খতিয়ান কী")
    assert trace.rerank_failed and not trace.reranked and docs


async def test_structured_count_matches_database(indexed):
    async with indexed() as s:
        res = await answer_structured(s, {"kind": "count_ebooks", "doc_type": "পরিপত্র", "year": None})
    assert res["count"] == 2  # fixture circulars 838 and 291
    assert "২টি" in res["answer"]


class EchoLLM(LLMProvider):
    name = "echo"

    @property
    def model(self):
        return "echo"

    async def generate(self, messages, max_tokens=None):
        return LLMResult(text="অনলাইনে নামজারির আবেদন করলে ডিসিআর ফি অনলাইনে পরিশোধ করা যায় [1]।",
                         model="echo", latency_ms=1)


async def test_end_to_end_rag_with_db(indexed, isettings, embedder):
    p = RAGPipeline(isettings, searcher=_searcher(isettings, embedder, indexed), llm=EchoLLM(), sessionmaker=indexed)
    r = await p.answer("ডিসিআর ফি কি অনলাইনে দেয়া যাবে?")
    assert r.grounded and r.sources[0]["source_id"] == "24776" and r.sources[0]["source_type"] == "qna_type2"
    r = await p.answer("কতটি পরিপত্র আছে?")
    assert r.route == "structured" and r.results_count == 2


async def test_detail_expansion_returns_whole_provisions_in_order(indexed, isettings, embedder):
    from sqlalchemy import text as sql

    from app.rag.context import expand_document, expand_sections

    async with indexed() as s:
        # a provision that the (small test) chunker split into several chunks
        row = (await s.execute(sql(
            "SELECT document_id, section_number, array_agg(id ORDER BY chunk_index) ids FROM document_chunks "
            "WHERE section_number IS NOT NULL GROUP BY 1, 2 HAVING count(*) > 1 LIMIT 1"))).one()
    q = analyze_query("ধারা বিস্তারিত বলুন")
    docs, _ = await _searcher(isettings, embedder, indexed).search("ভূমি", final_k=10)
    async with indexed() as s:
        from app.rag.context import _docs_for
        one = list((await _docs_for(s, [row.ids[-1]], q)).values())
        expanded = await expand_sections(s, one + docs, q, max_sections=1)
        assert [d.metadata["chunk_id"] for d in expanded[:len(row.ids)]] == list(row.ids)
        ordered, cov = await expand_document(s, one, q)
    assert ordered[0].metadata["chunk_metadata"].get("role") == "overview"
    assert [d.metadata["chunk_index"] for d in ordered] == sorted(d.metadata["chunk_index"] for d in ordered)
    assert cov.total >= 1 and cov.label in ("ধারা", "বিধি", "অনুচ্ছেদ", "প্রবিধান")
