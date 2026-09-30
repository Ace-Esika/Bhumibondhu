from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from app.api.schemas import IngestionStatus, SyncQueued, SyncRequest
from app.core.security import rate_limit, require_admin
from app.db.database import get_sessionmaker
from app.db.models import EmbeddingJob, IngestionRun
from app.db.stats import document_counts
from app.ingestion.runner import enqueue_run

router = APIRouter(prefix="/api/admin", tags=["admin"],
                   dependencies=[Depends(require_admin), Depends(rate_limit("admin", "admin_rate_limit_per_minute"))])


@router.post("/sync", response_model=SyncQueued, status_code=status.HTTP_202_ACCEPTED)
async def trigger_sync(body: SyncRequest) -> SyncQueued:
    """Queue a sync; the worker process executes it (the API never runs ingestion itself)."""
    opts = {"force": body.force, "no_embed": not body.embed, "reindex_full": body.reindex_full,
            "allow_mass_delete": body.allow_mass_delete}
    run_id = await enqueue_run(body.source_types, "admin", opts)
    return SyncQueued(run_id=run_id, status="queued")


@router.get("/ingestion-status", response_model=IngestionStatus)
async def ingestion_status(limit: int = 10) -> IngestionStatus:
    limit = max(1, min(limit, 50))
    async with get_sessionmaker()() as s:
        runs = (await s.execute(select(IngestionRun).order_by(IngestionRun.id.desc()).limit(limit))).scalars().all()
        jobs = (await s.execute(select(EmbeddingJob).order_by(EmbeddingJob.id.desc()).limit(limit))).scalars().all()
    return IngestionStatus(
        runs=[{"id": r.id, "status": r.status, "trigger": r.trigger, "source_types": r.source_types,
               "requested_at": r.requested_at, "started_at": r.started_at, "completed_at": r.completed_at,
               "records_fetched": r.records_fetched, "records_created": r.records_created,
               "records_updated": r.records_updated, "records_deleted": r.records_deleted,
               "records_skipped": r.records_skipped, "errors": r.errors,
               "error_details": (r.error_details or [])[:20]} for r in runs],
        embedding_jobs=[{"id": j.id, "run_id": j.run_id, "model": j.model, "mode": j.mode, "status": j.status,
                         "chunks_total": j.chunks_total, "chunks_embedded": j.chunks_embedded,
                         "chunks_failed": j.chunks_failed, "started_at": j.started_at,
                         "completed_at": j.completed_at} for j in jobs],
        counts=await document_counts(),
    )
