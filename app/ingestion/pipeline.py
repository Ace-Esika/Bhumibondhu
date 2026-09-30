"""Incremental synchronisation: API → validation → normalisation → chunking → PostgreSQL.

Change detection per upstream record uses (source_type, source_id) + a content hash:

    unchanged hash            → skip (only last_seen_at is touched)
    new / changed hash        → re-normalise, re-chunk; chunks whose text hash is unchanged
                                keep their existing embedding (no re-embedding)
    missing from the payload  → soft-delete the source, deactivate its documents and drop
                                their chunks (guarded by SYNC_MAX_DELETE_FRACTION)

Each record is processed in its own transaction, so one bad record never aborts a run.
Records that fail validation are reported and left untouched (never treated as deleted).
Embedding is a separate pass (`app.ingestion.embedder`) over chunks lacking a vector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.cache import bump_corpus_version
from app.core.config import Settings, get_settings
from app.core.text import lexical_text
from app.db.models import Document, DocumentChunk, IngestionRun, Source
from app.ingestion.chunker import Chunker, ChunkingConfig
from app.ingestion.hasher import PIPELINE_VERSION, canonical_json, source_hash, text_hash
from app.ingestion.normalizer import (
    NormalizedDocument,
    RecordValidationError,
    normalize,
    parse_datetime,
    record_id,
)
from app.ingestion.tokens import get_token_counter

log = logging.getLogger(__name__)

MAX_ERROR_DETAILS = 200


class RecordSource(Protocol):
    async def fetch(self, source_type: str) -> list[dict[str, Any]]: ...


@dataclass
class TypeStats:
    fetched: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    duplicates: int = 0
    excluded: int = 0
    errors: int = 0
    chunks_written: int = 0
    embeddings_reused: int = 0
    fetch_failed: bool = False
    seconds: float = 0.0


@dataclass
class SyncResult:
    per_type: dict[str, TypeStats] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def total(self, attr: str) -> int:
        return sum(getattr(s, attr) for s in self.per_type.values())

    @property
    def changed(self) -> bool:
        return any(self.total(a) for a in ("created", "updated", "deleted"))

    @property
    def status(self) -> str:
        if not self.per_type:
            return "failed"
        if all(s.fetch_failed for s in self.per_type.values()):
            return "failed"
        return "partial" if (self.errors or any(s.fetch_failed for s in self.per_type.values())) else "succeeded"


def chunking_config(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        target_tokens=settings.chunk_target_tokens, max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens, min_tokens=settings.chunk_min_tokens,
    )


def _jsonable(v: Any) -> Any:
    """Round-trip through canonical JSON so dates etc. are stored as strings in JSONB."""
    import json

    return json.loads(canonical_json(v))


class SyncPipeline:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], source: RecordSource,
                 settings: Settings | None = None, chunker: Chunker | None = None):
        self.sessionmaker = sessionmaker
        self.source = source
        self.settings = settings or get_settings()
        cfg = chunking_config(self.settings)
        self.chunker = chunker or Chunker(get_token_counter(self.settings.embedding_model), cfg)
        # Anything that changes processing output must be part of the hash signature.
        self.signature = f"{self.chunker.cfg.signature}|x={self.settings.exclude_title_regex}"

    # ------------------------------------------------------------------ entry point
    async def run(self, source_types: list[str], run_id: int | None = None, force: bool = False,
                  allow_mass_delete: bool = False) -> SyncResult:
        result = SyncResult()
        for st in source_types:
            t0 = time.perf_counter()
            stats = TypeStats()
            result.per_type[st] = stats
            try:
                records = await self.source.fetch(st)
            except Exception as e:
                stats.fetch_failed = True
                self._error(result, st, None, f"fetch failed: {e}")
                log.error("source fetch failed; skipping type (no deletions applied)",
                          extra={"source_type": st, "error": str(e)})
                continue
            await self._sync_type(st, records, stats, result, run_id, force, allow_mass_delete)
            stats.seconds = round(time.perf_counter() - t0, 2)
            log.info("synced source type", extra={"source_type": st, "stats": stats.__dict__})
        if result.changed:
            await bump_corpus_version()
        return result

    def _error(self, result: SyncResult, st: str, rid: str | None, msg: str) -> None:
        if st in result.per_type:
            result.per_type[st].errors += 1
        if len(result.errors) < MAX_ERROR_DETAILS:
            result.errors.append({"source_type": st, "source_id": rid, "error": msg[:500]})

    # ------------------------------------------------------------------ per type
    async def _sync_type(self, st: str, records: list[dict], stats: TypeStats, result: SyncResult,
                         run_id: int | None, force: bool, allow_mass_delete: bool) -> None:
        stats.fetched = len(records)
        by_id: dict[str, dict] = {}
        for raw in records:
            rid = record_id(st, raw)
            if rid is None:
                self._error(result, st, None, "record without id")
                continue
            if rid in by_id:
                stats.duplicates += 1  # upstream duplicate: last occurrence wins
            by_id[rid] = raw

        async with self.sessionmaker() as session:
            rows = (await session.execute(
                select(Source.id, Source.source_id, Source.content_hash, Source.is_deleted)
                .where(Source.source_type == st)
            )).all()
        existing = {r.source_id: r for r in rows}

        unchanged_pks: list[int] = []
        for rid, raw in by_id.items():
            h = source_hash(raw, self.signature)
            prev = existing.get(rid)
            if prev is not None and prev.content_hash == h and not prev.is_deleted and not force:
                stats.skipped += 1
                unchanged_pks.append(prev.id)
                continue
            try:
                docs = normalize(st, raw, self.settings)
            except RecordValidationError as e:
                self._error(result, st, rid, f"validation: {e.detail}")
                continue
            except Exception as e:  # normaliser bug on unexpected data: isolate the record
                self._error(result, st, rid, f"normalise: {type(e).__name__}: {e}")
                continue
            if not docs:
                stats.excluded += 1
            try:
                async with self.sessionmaker() as session, session.begin():
                    written, reused = await self._upsert_record(session, st, rid, raw, h, docs, run_id, force)
                stats.chunks_written += written
                stats.embeddings_reused += reused
                if prev is None:
                    stats.created += 1
                else:
                    stats.updated += 1
            except Exception as e:
                log.exception("record upsert failed", extra={"source_type": st, "source_id": rid})
                self._error(result, st, rid, f"db: {type(e).__name__}: {e}")

        if unchanged_pks:
            async with self.sessionmaker() as session, session.begin():
                for i in range(0, len(unchanged_pks), 5000):
                    await session.execute(
                        update(Source).where(Source.id.in_(unchanged_pks[i:i + 5000]))
                        .values(last_seen_at=func.now(), last_run_id=run_id)
                    )

        # ---- deletions: present in DB, absent upstream
        live_existing = {sid for sid, r in existing.items() if not r.is_deleted}
        missing = sorted(live_existing - set(by_id))
        if missing:
            frac = len(missing) / max(1, len(live_existing))
            if frac > self.settings.sync_max_delete_fraction and len(live_existing) >= 10 and not allow_mass_delete:
                self._error(result, st, None,
                            f"refusing to delete {len(missing)}/{len(live_existing)} records ({frac:.0%}) — "
                            f"exceeds SYNC_MAX_DELETE_FRACTION; rerun with --allow-mass-delete if intended")
            else:
                async with self.sessionmaker() as session, session.begin():
                    stats.deleted = await self._soft_delete(session, st, missing, run_id)

    # ------------------------------------------------------------------ writes
    async def _upsert_record(self, session: AsyncSession, st: str, rid: str, raw: dict, h: str,
                             docs: list[NormalizedDocument], run_id: int | None, force: bool) -> tuple[int, int]:
        stmt = pg_insert(Source).values(
            source_type=st, source_id=rid, raw=raw, content_hash=h,
            source_created_at=parse_datetime(raw.get("created_date")), last_run_id=run_id,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_sources_type_id",
            set_={"raw": stmt.excluded.raw, "content_hash": stmt.excluded.content_hash,
                  "source_created_at": stmt.excluded.source_created_at, "last_seen_at": func.now(),
                  "last_run_id": run_id, "is_deleted": False, "deleted_at": None},
        ).returning(Source.id)
        source_pk = (await session.execute(stmt)).scalar_one()

        existing_docs = {
            (d.source_type, d.source_id): d
            for d in (await session.execute(select(Document).where(Document.source_pk == source_pk))).scalars()
        }
        written = reused = 0
        produced: set[tuple[str, str]] = set()
        for doc in docs:
            produced.add((doc.source_type, doc.source_id))
            w, r = await self._upsert_document(session, source_pk, doc, existing_docs.get(
                (doc.source_type, doc.source_id)), force)
            written += w
            reused += r
        # Documents this record no longer produces (e.g. a forum topic removed).
        for key, d in existing_docs.items():
            if key not in produced and d.is_active:
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == d.id))
                d.is_active = False
        return written, reused

    async def _upsert_document(self, session: AsyncSession, source_pk: int, doc: NormalizedDocument,
                               existing: Document | None, force: bool) -> tuple[int, int]:
        chunks = self.chunker.chunk(doc)
        # PIPELINE_VERSION is included so derived fields (e.g. lexical text) are rebuilt when
        # processing logic changes even if the chunk text itself did not.
        doc_hash = text_hash(PIPELINE_VERSION, canonical_json(_jsonable({
            "title": doc.title, "url": doc.url, "file_url": doc.file_url, "meta": doc.metadata,
            "authority": doc.authority, "category": doc.category, "year": doc.year,
        })), *[c.content_hash for c in chunks])

        values = dict(
            source_pk=source_pk, source_type=doc.source_type, source_id=doc.source_id,
            parent_source_id=doc.parent_source_id, title=doc.title, url=doc.url, file_url=doc.file_url,
            doc_type=doc.doc_type, category=doc.category, year=doc.year, act_number=doc.act_number,
            publication_date=doc.publication_date, author=doc.author, authority=doc.authority,
            metadata_=_jsonable(doc.metadata), content_hash=doc_hash, is_active=True,
            source_updated_at=doc.source_updated_at, ingested_at=datetime.now(UTC),
        )
        if existing is None:
            existing = Document(**values)
            session.add(existing)
            await session.flush()
        else:
            if existing.content_hash == doc_hash and existing.is_active and not force:
                return 0, 0
            for k, v in values.items():
                setattr(existing, k, v)
            await session.flush()

        # Reuse vectors of chunks whose exact text is unchanged (same model only).
        old = {
            r.content_hash: r
            for r in (await session.execute(
                select(DocumentChunk.content_hash, DocumentChunk.embedding, DocumentChunk.embedding_model,
                       DocumentChunk.embedding_created_at)
                .where(DocumentChunk.document_id == existing.id, DocumentChunk.embedding.is_not(None))
            )).all()
        }
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == existing.id))

        rows, reused = [], 0
        for c in chunks:
            prev = old.get(c.content_hash)
            keep = prev is not None and prev.embedding_model == self.settings.embedding_model
            reused += keep
            rows.append(dict(
                document_id=existing.id, chunk_index=c.chunk_index, source_type=c.source_type,
                source_id=c.source_id, parent_source_id=c.parent_source_id, doc_source_type=doc.source_type,
                doc_type=doc.doc_type, category=doc.category, year=doc.year, authority=doc.authority,
                title=doc.title, section_number=c.section_number, section_heading=c.section_heading,
                context=c.context, content=c.content,
                lexical_title=lexical_text(f"{doc.title} {c.section_heading or ''} "
                                           f"{doc.metadata.get('keyword', '')} {doc.category or ''}"),
                lexical_body=lexical_text(c.embed_text), token_count=c.token_count,
                metadata_=_jsonable(c.metadata), content_hash=c.content_hash,
                embedding=prev.embedding if keep else None,
                embedding_model=prev.embedding_model if keep else None,
                embedding_created_at=prev.embedding_created_at if keep else None,
            ))
        if rows:
            await session.execute(pg_insert(DocumentChunk), rows)
        return len(rows), reused

    async def _soft_delete(self, session: AsyncSession, st: str, source_ids: list[str], run_id: int | None) -> int:
        pks = (await session.execute(
            update(Source).where(Source.source_type == st, Source.source_id.in_(source_ids))
            .values(is_deleted=True, deleted_at=func.now(), last_run_id=run_id).returning(Source.id)
        )).scalars().all()
        if pks:
            doc_ids = (await session.execute(
                update(Document).where(Document.source_pk.in_(pks)).values(is_active=False).returning(Document.id)
            )).scalars().all()
            if doc_ids:
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids)))
        log.info("soft-deleted records removed upstream", extra={"source_type": st, "count": len(pks)})
        return len(pks)


# ---------------------------------------------------------------------- run bookkeeping

async def create_run(session: AsyncSession, source_types: list[str], trigger: str, options: dict | None = None,
                     status: str = "queued") -> IngestionRun:
    run = IngestionRun(status=status, trigger=trigger, source_types=source_types, options=options or {},
                       error_details=[], stats={})
    session.add(run)
    await session.flush()
    return run


async def finish_run(session: AsyncSession, run_id: int, result: SyncResult | None,
                     fatal_error: str | None = None, extra_stats: dict | None = None) -> None:
    run = await session.get(IngestionRun, run_id)
    if run is None:
        return
    run.completed_at = datetime.now(UTC)
    if result is not None:
        run.records_fetched = result.total("fetched")
        run.records_created = result.total("created")
        run.records_updated = result.total("updated")
        run.records_deleted = result.total("deleted")
        run.records_skipped = result.total("skipped")
        run.errors = len(result.errors) + (1 if fatal_error else 0)
        run.error_details = result.errors[:MAX_ERROR_DETAILS]
        run.stats = {**{k: v.__dict__ for k, v in result.per_type.items()}, **(extra_stats or {})}
        run.status = "failed" if fatal_error else result.status
    else:
        run.status = "failed"
        run.errors = 1
    if fatal_error:
        run.error_details = [*(run.error_details or []), {"error": fatal_error[:1000]}]
