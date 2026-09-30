"""Integration fixtures: a real PostgreSQL + pgvector database (TEST_DATABASE_URL).

Uses the real migrations and SQL; replaces only the BGE models with deterministic fakes so
the suite runs in seconds on any machine. Skips cleanly if the database is unreachable.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://bhumipedia:bhumipedia@localhost:5433/bhumipedia_test")
if not TEST_DB_URL.rsplit("/", 1)[-1].endswith("_test"):
    raise RuntimeError("refusing to run integration tests against a database not named *_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.core.text import tokenize  # noqa: E402
from app.retrieval.embeddings import EmbeddingService  # noqa: E402
from tests.conftest import SNAPSHOT, make_settings  # noqa: E402


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder (1024-d). Texts sharing words are similar."""

    def _vec(self, t: str) -> list[float]:
        v = np.zeros(1024, dtype=np.float32)
        for tok in tokenize(t):
            for piece in {tok, tok[:4]}:
                v[int(hashlib.md5(piece.encode()).hexdigest(), 16) % 1024] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v + 1e-3).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _ensure_database():
    base, name = TEST_DB_URL.rsplit("/", 1)
    admin = create_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            if not c.execute(text("SELECT 1 FROM pg_database WHERE datname=:n"), {"n": name}).scalar():
                c.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def migrated_db():
    try:
        _ensure_database()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"test database unavailable: {e}")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(cfg, "head")
    command.upgrade(cfg, "head")  # idempotent
    return TEST_DB_URL


@pytest.fixture
async def sm(migrated_db):
    engine = create_async_engine(migrated_db)
    async with engine.begin() as c:
        await c.execute(text("TRUNCATE document_chunks, documents, sources, embedding_jobs, ingestion_runs, "
                             "conversation_messages, conversations RESTART IDENTITY CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def isettings():
    return make_settings(database_url=TEST_DB_URL, chunk_target_tokens=120, chunk_max_tokens=180,
                         chunk_overlap_tokens=20, chunk_min_tokens=10, min_vector_similarity=0.2)


@pytest.fixture
def embedder(isettings):
    return EmbeddingService(isettings, backend=HashingEmbeddings())


@pytest.fixture
def snapshot_dir(tmp_path):
    d = tmp_path / "snap"
    shutil.copytree(SNAPSHOT, d)
    return d


def word_tokens(t: str) -> int:
    return len(re.findall(r"\S+", t or ""))
