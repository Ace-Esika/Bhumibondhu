"""Corpus statistics shared by the CLI and the admin API."""

from __future__ import annotations

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.database import get_sessionmaker
from app.db.models import Document, DocumentChunk


async def document_counts() -> dict:
    model = get_settings().embedding_model
    async with get_sessionmaker()() as s:
        docs = (await s.execute(select(Document.source_type, Document.is_active, func.count())
                                .group_by(Document.source_type, Document.is_active))).all()
        chunks = (await s.execute(select(
            DocumentChunk.doc_source_type, func.count(),
            func.count().filter(DocumentChunk.embedding.is_not(None) & (DocumentChunk.embedding_model == model)),
        ).group_by(DocumentChunk.doc_source_type))).all()
    return {
        "documents": {f"{t}{'' if a else ' (inactive)'}": n for t, a, n in docs},
        "chunks": {t: {"total": n, "embedded": e, "pending": n - e} for t, n, e in chunks},
        "embedding_model": model,
    }
