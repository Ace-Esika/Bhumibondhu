"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.api.routes import admin, chat, health, search
from app.core.cache import close_redis
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import dispose_engine
from app.llm.base import LLMError
from app.rag.guardrails import InvalidInput
from app.rag.pipeline import RAGPipeline

log = logging.getLogger(__name__)


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def create_app(pipeline: RAGPipeline | None = None, warmup: bool = True) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline(settings)
        if warmup:
            # Load the embedding model (and reranker if enabled) before taking traffic.
            try:
                p = app.state.pipeline
                await asyncio.to_thread(p.searcher.embedder.embed_query, "উষ্ণকরণ")
                if p.searcher.reranker is not None:
                    await asyncio.to_thread(p.searcher.reranker.score, "উষ্ণকরণ", ["উষ্ণকরণ"])
            except Exception:
                log.exception("model warm-up failed; /ready will report not_ready")
        if not settings.groq_configured:
            log.warning("GROQ_API_KEY/GROQ_MODEL not set: /api/chat will return 503; /api/search works")
        yield
        await close_redis()
        await dispose_engine()

    app = FastAPI(
        title="Bhumipedia RAG Chatbot",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"], allow_credentials=False)
    app.add_middleware(RequestContextMiddleware, timeout_seconds=settings.request_timeout_seconds)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=64 * 1024)

    @app.exception_handler(InvalidInput)
    async def _invalid(request: Request, exc: InvalidInput):
        return JSONResponse({"detail": str(exc), "request_id": _rid(request)}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # Don't echo user input back; report locations and messages only.
        errors = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse({"detail": errors, "request_id": _rid(request)}, status_code=422)

    @app.exception_handler(LLMError)
    async def _llm(request: Request, exc: LLMError):
        log.error("llm error", extra={"error_type": type(exc).__name__})
        return JSONResponse({"detail": exc.user_message, "request_id": _rid(request)}, status_code=exc.http_status)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"detail": "Internal server error", "request_id": _rid(request)}, status_code=500)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(search.router)
    app.include_router(admin.router)
    return app


app = create_app()
