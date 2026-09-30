"""Background ingestion worker: `python -m app.workers.sync`.

A single, dependency-free loop (no Celery/beat):
- executes runs queued by `POST /api/admin/sync` (claimed with FOR UPDATE SKIP LOCKED);
- enqueues a scheduled sync when the last scheduled one is older than SYNC_INTERVAL_HOURS
  (state lives in `ingestion_runs`, so restarts don't reset the schedule);
- the global advisory lock guarantees at most one sync/embedding pass across all workers.
SIGTERM/SIGINT stop gracefully: the embedding pass checkpoints after the current batch.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import signal
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.cache import bump_corpus_version
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import dispose_engine, get_sessionmaker
from app.db.models import IngestionRun
from app.ingestion.runner import (
    SyncAlreadyRunning,
    claim_next_queued_run,
    enqueue_run,
    execute_run,
    recover_stale_runs,
    sync_lock,
)
from app.rag.memory import SQLConversationStore

log = logging.getLogger("app.worker")


async def schedule_due(interval_hours: float) -> bool:
    async with get_sessionmaker()() as session:
        last = (await session.execute(
            select(IngestionRun.requested_at).where(IngestionRun.trigger == "schedule")
            .order_by(IngestionRun.requested_at.desc()).limit(1)
        )).scalar()
    return last is None or datetime.now(UTC) - last >= timedelta(hours=interval_hours)


async def bootstrap_from_seed() -> None:
    """Empty database + seed file present → import the prebuilt index (once, under the lock)."""
    from pathlib import Path

    from app.ingestion.seed import SeedError, database_is_empty, import_index

    s = get_settings()
    seed = Path(s.index_seed_path)
    if not s.index_seed_auto_import or not seed.is_file():
        return
    try:
        async with sync_lock():
            if not await asyncio.to_thread(database_is_empty, s):
                return
            log.info("empty database: importing prebuilt index", extra={"seed": str(seed)})
            report = await asyncio.to_thread(import_index, seed, s)
            await bump_corpus_version()
            log.info("prebuilt index imported", extra={"rows": report.rows, "seconds": round(report.seconds, 1),
                                                       "warnings": report.warnings})
    except SyncAlreadyRunning:
        log.info("another worker holds the sync lock; skipping seed import")
    except SeedError as e:
        log.error("seed import refused; falling back to a full sync", extra={"error": str(e)})


async def worker_loop(stop: asyncio.Event) -> None:
    s = get_settings()
    await bootstrap_from_seed()
    try:
        if n := await recover_stale_runs():
            log.warning("recovered stale runs", extra={"count": n})
    except SyncAlreadyRunning:
        log.info("another worker holds the sync lock")
    first = True
    last_purge = 0.0
    while not stop.is_set():
        if time.monotonic() - last_purge > 3600:
            last_purge = time.monotonic()
            try:
                n = await SQLConversationStore(get_sessionmaker()).purge_older_than(s.conversation_retention_days)
                if n:
                    log.info("purged expired conversations", extra={"count": n})
            except Exception:
                log.exception("conversation purge failed")
        try:
            run_id = await claim_next_queued_run()
            if run_id is None and s.sync_enabled and (
                    (first and s.sync_on_startup and await schedule_due(0)) or await schedule_due(s.sync_interval_hours)):
                run_id = await enqueue_run(None, "schedule")
                async with get_sessionmaker()() as session, session.begin():
                    run = await session.get(IngestionRun, run_id)
                    run.status = "running"
            first = False
            if run_id is not None:
                log.info("starting ingestion run", extra={"run_id": run_id})
                await execute_run(run_id, stop_event=stop)
                continue
        except Exception:
            log.exception("worker iteration failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=s.worker_poll_seconds)
        except TimeoutError:
            pass


async def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, json_logs=s.log_json)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    log.info("worker started", extra={"sync_enabled": s.sync_enabled, "interval_h": s.sync_interval_hours,
                                      "types": s.sync_source_types})
    try:
        await worker_loop(stop)
    finally:
        await dispose_engine()
        log.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
