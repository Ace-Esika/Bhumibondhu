"""Embedding pass: fills `document_chunks.embedding` for pending chunks.

Pending = no vector yet, or a vector produced by a different EMBEDDING_MODEL (a model
change therefore triggers re-embedding automatically without touching source data).
`mode="full"` re-embeds everything with the current model (e.g. after changing
EMBEDDING_MAX_SEQ_LENGTH). Existing vectors keep serving queries until they are replaced.

Keyset pagination by id keeps memory flat (the corpus is never loaded into RAM), and each
page commits independently so progress survives interruption.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.cache import bump_corpus_version
from app.core.config import Settings, get_settings
from app.db.models import DocumentChunk, EmbeddingJob
from app.retrieval.embeddings import EmbeddingError, EmbeddingService

log = logging.getLogger(__name__)


def _pending_filter(model: str):
    return or_(DocumentChunk.embedding.is_(None), DocumentChunk.embedding_model != model)


async def count_pending(session: AsyncSession, model: str) -> int:
    return (await session.execute(
        select(func.count()).select_from(DocumentChunk).where(_pending_filter(model))
    )).scalar_one()


async def embed_chunks(sessionmaker: async_sessionmaker[AsyncSession], embedder: EmbeddingService,
                       settings: Settings | None = None, run_id: int | None = None, mode: str = "pending",
                       limit: int | None = None, stop_event: asyncio.Event | None = None) -> EmbeddingJob:
    settings = settings or get_settings()
    model = settings.embedding_model
    page_size = max(settings.embedding_batch_size * 8, 16)
    started = datetime.now(UTC)

    async with sessionmaker() as session, session.begin():
        where = [] if mode == "full" else [_pending_filter(model)]
        total = (await session.execute(select(func.count()).select_from(DocumentChunk).where(*where))).scalar_one()
        if limit:
            total = min(total, limit)
        job = EmbeddingJob(run_id=run_id, model=model, mode=mode, status="running", chunks_total=total)
        session.add(job)
        await session.flush()
        job_id = job.id

    done = failed = 0
    last_id = 0
    t0 = time.perf_counter()
    status, error = "succeeded", None
    try:
        while done + failed < total:
            if stop_event is not None and stop_event.is_set():
                status, error = "interrupted", "stopped by shutdown signal"
                break
            async with sessionmaker() as session:
                q = (select(DocumentChunk.id, DocumentChunk.context, DocumentChunk.content)
                     .where(DocumentChunk.id > last_id).order_by(DocumentChunk.id)
                     .limit(min(page_size, total - done - failed)))
                if mode != "full":
                    q = q.where(_pending_filter(model))
                if mode == "full":
                    q = q.where(or_(DocumentChunk.embedding_created_at.is_(None),
                                    DocumentChunk.embedding_created_at < started,
                                    DocumentChunk.embedding_model != model))
                rows = (await session.execute(q)).all()
            if not rows:
                break
            last_id = rows[-1].id
            texts = [f"{r.context}\n\n{r.content}" if r.context else r.content for r in rows]
            try:
                vectors: list[list[float] | None] = await asyncio.to_thread(embedder.embed_documents, texts)
            except EmbeddingError:
                # Isolate the bad chunk(s) instead of failing the whole page.
                vectors = []
                for t in texts:
                    try:
                        vectors.append((await asyncio.to_thread(embedder.embed_documents, [t]))[0])
                    except EmbeddingError as e:
                        vectors.append(None)
                        log.error("chunk embedding failed", extra={"error": str(e)})
            now = datetime.now(UTC)
            async with sessionmaker() as session, session.begin():
                for r, v in zip(rows, vectors, strict=True):
                    if v is None:
                        failed += 1
                        continue
                    await session.execute(
                        update(DocumentChunk).where(DocumentChunk.id == r.id)
                        .values(embedding=v, embedding_model=model, embedding_created_at=now)
                    )
                    done += 1
                await session.execute(update(EmbeddingJob).where(EmbeddingJob.id == job_id)
                                      .values(chunks_embedded=done, chunks_failed=failed))
            rate = done / max(1e-6, time.perf_counter() - t0)
            log.info("embedding progress", extra={"done": done, "failed": failed, "total": total,
                                                  "chunks_per_s": round(rate, 2),
                                                  "eta_min": round((total - done - failed) / max(rate, 1e-6) / 60, 1)})
        if failed:
            status = "partial" if status == "succeeded" else status
    except Exception as e:
        status, error = "failed", f"{type(e).__name__}: {e}"
        log.exception("embedding pass failed")
    finally:
        async with sessionmaker() as session, session.begin():
            await session.execute(update(EmbeddingJob).where(EmbeddingJob.id == job_id).values(
                status=status, chunks_embedded=done, chunks_failed=failed, completed_at=datetime.now(UTC),
                error=error))
        if done:
            await bump_corpus_version()
    async with sessionmaker() as session:
        return await session.get(EmbeddingJob, job_id)
