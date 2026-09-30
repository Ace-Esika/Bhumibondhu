"""Run orchestration shared by the CLI, the scheduler and admin-triggered jobs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.db.database import get_engine, get_sessionmaker
from app.db.models import IngestionRun
from app.ingestion.client import BhumipediaClient, FileSource
from app.ingestion.embedder import embed_chunks
from app.ingestion.pipeline import SyncPipeline, create_run, finish_run
from app.retrieval.embeddings import get_embedding_service

log = logging.getLogger(__name__)

SYNC_LOCK_KEY = 727274002


class SyncAlreadyRunning(RuntimeError):
    pass


@asynccontextmanager
async def sync_lock():
    """Session-level advisory lock held on a dedicated connection for the whole run."""
    async with get_engine().connect() as conn:
        got = (await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": SYNC_LOCK_KEY})).scalar()
        await conn.commit()  # session-level lock survives; don't sit "idle in transaction"
        if not got:
            raise SyncAlreadyRunning("another sync/reindex is in progress")
        try:
            yield
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": SYNC_LOCK_KEY})
            await conn.commit()


async def execute_run(run_id: int, settings: Settings | None = None,
                      stop_event: asyncio.Event | None = None) -> IngestionRun:
    """Execute a queued ingestion run (sync and/or embedding) under the global lock."""
    settings = settings or get_settings()
    sm = get_sessionmaker()
    async with sm() as session, session.begin():
        run = await session.get(IngestionRun, run_id)
        run.status = "running"
        run.started_at = datetime.now(UTC)
        types = list(run.source_types or settings.sync_source_types)
        opts = dict(run.options or {})

    result = None
    fatal = None
    extra: dict = {}
    try:
        async with sync_lock():
            if not opts.get("embed_only"):
                source = FileSource(opts["from_dir"]) if opts.get("from_dir") else BhumipediaClient(settings)
                async with source:
                    pipeline = SyncPipeline(sm, source, settings)
                    result = await pipeline.run(types, run_id=run_id, force=bool(opts.get("force")),
                                                allow_mass_delete=bool(opts.get("allow_mass_delete")))
            if not opts.get("no_embed"):
                job = await embed_chunks(sm, get_embedding_service(), settings, run_id=run_id,
                                         mode="full" if opts.get("reindex_full") else "pending",
                                         stop_event=stop_event)
                extra["embedding"] = {"job_id": job.id, "status": job.status, "embedded": job.chunks_embedded,
                                      "failed": job.chunks_failed, "total": job.chunks_total}
    except SyncAlreadyRunning as e:
        fatal = str(e)
    except Exception as e:
        log.exception("ingestion run failed", extra={"run_id": run_id})
        fatal = f"{type(e).__name__}: {e}"

    async with sm() as session, session.begin():
        if result is None and opts.get("embed_only") and not fatal:
            run = await session.get(IngestionRun, run_id)
            run.status = "succeeded" if extra.get("embedding", {}).get("status") == "succeeded" else "partial"
            run.completed_at = datetime.now(UTC)
            run.stats = extra
        else:
            await finish_run(session, run_id, result, fatal, extra)
    async with sm() as session:
        run = await session.get(IngestionRun, run_id)
    log.info("ingestion run finished", extra={"run_id": run_id, "status": run.status, "counts": {
        "created": run.records_created, "updated": run.records_updated, "deleted": run.records_deleted,
        "skipped": run.records_skipped, "errors": run.errors}})
    return run


async def enqueue_run(source_types: list[str] | None, trigger: str, options: dict | None = None) -> int:
    settings = get_settings()
    async with get_sessionmaker()() as session, session.begin():
        run = await create_run(session, source_types or list(settings.sync_source_types), trigger, options)
        return run.id


async def claim_next_queued_run() -> int | None:
    """Atomically claim the oldest queued run (safe with multiple workers)."""
    async with get_sessionmaker()() as session, session.begin():
        rid = (await session.execute(text(
            "SELECT id FROM ingestion_runs WHERE status = 'queued' ORDER BY requested_at "
            "FOR UPDATE SKIP LOCKED LIMIT 1"))).scalar()
        if rid is not None:
            await session.execute(text("UPDATE ingestion_runs SET status='running', started_at=now() WHERE id=:i"),
                                  {"i": rid})
        return rid


async def recover_stale_runs() -> int:
    """Mark runs/jobs left 'running' by a crashed process as failed.

    Only safe while holding the sync lock: if we hold it, nothing else can be running."""
    async with sync_lock(), get_sessionmaker()() as session, session.begin():
        n = (await session.execute(text(
            "UPDATE ingestion_runs SET status='failed', completed_at=now(), "
            "error_details = error_details || '[{\"error\": \"interrupted (process exited)\"}]'::jsonb "
            "WHERE status='running'"))).rowcount
        await session.execute(text(
            "UPDATE embedding_jobs SET status='interrupted', completed_at=now() WHERE status='running'"))
        return n
