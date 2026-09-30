"""Prebuilt index export → import on a real PostgreSQL + pgvector database."""

import io
import tarfile

import pytest
from sqlalchemy import text

from app.ingestion.chunker import Chunker
from app.ingestion.client import FileSource
from app.ingestion.embedder import embed_chunks
from app.ingestion.pipeline import SyncPipeline, chunking_config
from app.ingestion.seed import SeedError, export_index, import_index, read_manifest
from tests.integration.conftest import word_tokens

pytestmark = pytest.mark.integration


@pytest.fixture
async def built(sm, snapshot_dir, isettings, embedder):
    pipe = SyncPipeline(sm, FileSource(snapshot_dir), isettings, Chunker(word_tokens, chunking_config(isettings)))
    await pipe.run(["ebook", "qna_type2"])
    await embed_chunks(sm, embedder, isettings)
    return sm


async def _fingerprint(sm):
    async with sm() as s:
        return (await s.execute(text(
            "SELECT count(*), md5(string_agg(content_hash || embedding::text, ',' ORDER BY id)) FROM document_chunks"
        ))).one()


async def _wipe(sm):
    async with sm() as s, s.begin():
        await s.execute(text("TRUNCATE document_chunks, documents, sources, ingestion_runs RESTART IDENTITY CASCADE"))


async def test_roundtrip_restores_identical_index(built, isettings, tmp_path):
    before = await _fingerprint(built)
    seed = tmp_path / "seed.tar.gz"
    rep = export_index(seed, isettings)
    assert rep.rows["document_chunks"] == before[0] > 0
    await _wipe(built)
    rep2 = import_index(seed, isettings)
    assert rep2.rows == rep.rows and rep2.warnings == []
    assert await _fingerprint(built) == before  # same chunks, same vectors
    async with built() as s:
        # generated FTS column rebuilt; sequences advanced past imported ids; audit row written
        assert (await s.execute(text("SELECT count(*) FROM document_chunks WHERE search_vector IS NULL "
                                     "AND lexical_body <> ''"))).scalar() == 0
        await s.execute(text("INSERT INTO ingestion_runs (status) VALUES ('queued')"))
        assert (await s.execute(text("SELECT trigger FROM ingestion_runs ORDER BY id LIMIT 1"))).scalar() == "seed"


async def test_import_refuses_non_empty_database_unless_forced(built, isettings, tmp_path):
    seed = tmp_path / "seed.tar.gz"
    export_index(seed, isettings)
    with pytest.raises(SeedError, match="already contains"):
        import_index(seed, isettings)
    assert import_index(seed, isettings, force=True).rows["sources"] > 0


async def test_corrupted_seed_is_rejected(built, isettings, tmp_path):
    seed = tmp_path / "seed.tar.gz"
    export_index(seed, isettings)
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(seed) as src, tarfile.open(bad, "w") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name == "sources.tsv.gz":
                data = data[:-10] + b"0123456789"  # flip the tail
            m.size = len(data)
            dst.addfile(m, io.BytesIO(data))
    await _wipe(built)
    with pytest.raises(SeedError, match="checksum"):
        import_index(bad, isettings)


async def test_incompatible_embeddings_are_refused(built, isettings, tmp_path):
    seed = tmp_path / "seed.tar.gz"
    export_index(seed, isettings)
    await _wipe(built)
    with pytest.raises(SeedError, match="dimension"):
        import_index(seed, isettings.model_copy(update={"embedding_dim": 768}))
    with pytest.raises(SeedError, match="re-embedded"):
        import_index(seed, isettings.model_copy(update={"embedding_model": "other/model"}))
    assert read_manifest(seed)["embedding_model"] == isettings.embedding_model


async def test_export_refuses_unembedded_index(sm, snapshot_dir, isettings, tmp_path):
    pipe = SyncPipeline(sm, FileSource(snapshot_dir), isettings, Chunker(word_tokens, chunking_config(isettings)))
    await pipe.run(["qna_type2"])  # no embedding pass
    with pytest.raises(SeedError, match="not embedded"):
        export_index(tmp_path / "x.tar.gz", isettings)
