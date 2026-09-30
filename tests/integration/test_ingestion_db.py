"""API snapshot → validation → normalisation → chunking → PostgreSQL → embeddings (pgvector)."""

import json

import pytest
from sqlalchemy import func, select, text

from app.db.models import Document, DocumentChunk, Source
from app.ingestion.chunker import Chunker
from app.ingestion.client import FileSource
from app.ingestion.embedder import count_pending, embed_chunks
from app.ingestion.pipeline import SyncPipeline, chunking_config
from tests.integration.conftest import word_tokens

pytestmark = pytest.mark.integration
TYPES = ["ebook", "blog", "forum", "qna_type1", "qna_type2"]


def _pipeline(sm, directory, settings):
    return SyncPipeline(sm, FileSource(directory), settings, Chunker(word_tokens, chunking_config(settings)))


async def _count(sm, model, *where):
    async with sm() as s:
        return (await s.execute(select(func.count()).select_from(model).where(*where))).scalar_one()


async def test_migrations_created_indexes(sm):
    async with sm() as s:
        idx = set((await s.execute(text("SELECT indexname FROM pg_indexes WHERE tablename='document_chunks'")))
                  .scalars())
        method = (await s.execute(text(
            "SELECT am.amname FROM pg_class c JOIN pg_am am ON am.oid=c.relam "
            "WHERE c.relname='ix_chunks_embedding_hnsw'"))).scalar()
    assert {"ix_chunks_embedding_hnsw", "ix_chunks_search_vector"} <= idx and method == "hnsw"


async def test_full_then_incremental_sync(sm, snapshot_dir, isettings):
    res = await _pipeline(sm, snapshot_dir, isettings).run(TYPES)
    assert res.status == "succeeded", res.errors
    assert res.total("created") == sum(len(json.loads((snapshot_dir / f"{t}.json").read_text())) for t in TYPES)
    # source identity preserved
    async with sm() as s:
        d = (await s.execute(select(Document).where(Document.source_type == "qna_type2",
                                                    Document.source_id == "24776"))).scalar_one()
        assert d.is_active and d.category == "namjari"
        sec = (await s.execute(select(DocumentChunk).where(DocumentChunk.source_type == "section",
                                                           DocumentChunk.doc_source_type == "ebook"))).scalars().first()
        assert sec.parent_source_id and sec.section_number and sec.title in sec.context
    # placeholder records stored for audit but not searchable
    assert await _count(sm, Source, Source.source_type == "ebook", Source.source_id == "633") == 1
    assert await _count(sm, Document, Document.source_id == "633", Document.is_active.is_(True)) == 0

    res2 = await _pipeline(sm, snapshot_dir, isettings).run(TYPES)
    assert res2.total("skipped") == res.total("created") and not res2.changed


async def test_changed_record_reprocessed_and_unchanged_embeddings_reused(sm, snapshot_dir, isettings, embedder):
    await _pipeline(sm, snapshot_dir, isettings).run(TYPES)
    await embed_chunks(sm, embedder, isettings)
    async with sm() as s:
        assert await count_pending(s, isettings.embedding_model) == 0

    ebooks = json.loads((snapshot_dir / "ebook.json").read_text())
    act = next(e for e in ebooks if e["id"] == 164)
    act["sections"][0]["heading"] = (act["sections"][0].get("heading") or "") + " (সংশোধিত)"
    (snapshot_dir / "ebook.json").write_text(json.dumps(ebooks, ensure_ascii=False))

    res = await _pipeline(sm, snapshot_dir, isettings).run(["ebook"])
    st = res.per_type["ebook"]
    assert st.updated == 1 and st.skipped == len(ebooks) - 1
    assert st.embeddings_reused > 0  # untouched sections kept their vectors
    async with sm() as s:
        pending = await count_pending(s, isettings.embedding_model)
    assert 0 < pending < st.chunks_written


async def test_deleted_upstream_is_soft_deleted(sm, snapshot_dir, isettings):
    await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2"])
    rows = json.loads((snapshot_dir / "qna_type2.json").read_text())
    removed = rows.pop()
    (snapshot_dir / "qna_type2.json").write_text(json.dumps(rows, ensure_ascii=False))
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2"])
    assert res.per_type["qna_type2"].deleted == 1
    async with sm() as s:
        src = (await s.execute(select(Source).where(Source.source_id == str(removed["id"])))).scalar_one()
        assert src.is_deleted and src.raw["id"] == removed["id"]  # raw kept for audit
        doc = (await s.execute(select(Document).where(Document.source_id == str(removed["id"])))).scalar_one()
        assert not doc.is_active
    assert await _count(sm, DocumentChunk, DocumentChunk.document_id == doc.id) == 0
    # Reappearing upstream revives it.
    rows.append(removed)
    (snapshot_dir / "qna_type2.json").write_text(json.dumps(rows, ensure_ascii=False))
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2"])
    assert res.per_type["qna_type2"].updated == 1


async def test_mass_delete_guard(sm, snapshot_dir, isettings):
    # Synthetic test-only records (not real data): enough rows for the guard to apply.
    rows = [{"id": 900000 + i, "question": f"প্রশ্ন {i}", "answer": f"উত্তর {i}", "category": "", "keyword": ""}
            for i in range(12)]
    (snapshot_dir / "qna_type1.json").write_text(json.dumps(rows, ensure_ascii=False))
    await _pipeline(sm, snapshot_dir, isettings).run(["qna_type1"])
    (snapshot_dir / "qna_type1.json").write_text("[]")  # e.g. a truncated upstream response
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type1"])
    assert res.per_type["qna_type1"].deleted == 0 and "refusing to delete" in res.errors[0]["error"]
    assert await _count(sm, Document, Document.source_type == "qna_type1", Document.is_active.is_(True)) == 12
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type1"], allow_mass_delete=True)
    assert res.per_type["qna_type1"].deleted == 12


async def test_fetch_failure_applies_no_deletions(sm, snapshot_dir, isettings):
    await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2", "blog"])
    (snapshot_dir / "blog.json").unlink()  # simulates the API being down for this type
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2", "blog"])
    assert res.status == "partial"
    assert res.per_type["blog"].fetch_failed and res.per_type["blog"].deleted == 0
    assert res.per_type["qna_type2"].skipped == 6
    assert await _count(sm, Source, Source.is_deleted.is_(True)) == 0


async def test_invalid_record_reported_not_deleted(sm, snapshot_dir, isettings):
    await _pipeline(sm, snapshot_dir, isettings).run(["qna_type1"])
    rows = json.loads((snapshot_dir / "qna_type1.json").read_text())
    rows[0]["answer"] = "   "  # now fails validation
    (snapshot_dir / "qna_type1.json").write_text(json.dumps(rows, ensure_ascii=False))
    res = await _pipeline(sm, snapshot_dir, isettings).run(["qna_type1"])
    assert res.per_type["qna_type1"].errors == 1 and res.per_type["qna_type1"].deleted == 0
    assert await _count(sm, Document, Document.source_id == str(rows[0]["id"]), Document.is_active.is_(True)) == 1


async def test_model_change_marks_chunks_pending(sm, snapshot_dir, isettings, embedder):
    await _pipeline(sm, snapshot_dir, isettings).run(["qna_type2"])
    job = await embed_chunks(sm, embedder, isettings)
    assert job.status == "succeeded" and job.chunks_embedded == job.chunks_total > 0
    other = isettings.model_copy(update={"embedding_model": "some/other-model"})
    async with sm() as s:
        assert await count_pending(s, other.embedding_model) == job.chunks_total
    # full reindex with the current model re-embeds everything without touching sources
    job2 = await embed_chunks(sm, embedder, isettings, mode="full")
    assert job2.chunks_embedded == job.chunks_total

