from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import get_sessionmaker

router = APIRouter(tags=["health"])
log = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict:
    """Liveness: the process is up and serving."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: database + pgvector reachable, embedding model loaded, index non-empty."""
    s = get_settings()
    checks: dict[str, object] = {}
    ok = True
    try:
        async with get_sessionmaker()() as session:
            checks["pgvector"] = (await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'"))).scalar()
            checks["embedded_chunks"] = (await session.execute(text(
                "SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL AND embedding_model = :m"),
                {"m": s.embedding_model})).scalar()
        checks["database"] = "ok"
        ok = ok and bool(checks["pgvector"]) and checks["embedded_chunks"] > 0
    except Exception:
        log.exception("readiness: database check failed")
        checks["database"] = "unavailable"
        ok = False
    pipeline = getattr(request.app.state, "pipeline", None)
    checks["embedding_model_loaded"] = bool(pipeline and pipeline.searcher.embedder.loaded)
    checks["reranker"] = "enabled" if s.reranker_enabled else "disabled"
    checks["llm_configured"] = s.groq_configured
    ok = ok and checks["embedding_model_loaded"] and s.groq_configured
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)
