"""Prebuilt index ("seed") export/import.

A seed is one `.tar.gz` holding the indexed corpus — `sources`, `documents` and
`document_chunks` including their embeddings — plus a `manifest.json` describing how it was
built (pipeline version, chunking signature, embedding model/dimension, row counts, sha256
of every member). It lets a fresh clone or a new server start with a complete index instead
of re-fetching and re-embedding everything (hours on CPU).

    python -m app.cli export-index                 # → seed/bhumipedia-index.tar.gz
    python -m app.cli import-index <file>          # manual restore into an empty database

The worker imports INDEX_SEED_PATH automatically when the database is empty; the regular
incremental sync afterwards only processes records that changed upstream since the seed.

Data moves with PostgreSQL COPY (text format), so no pg_dump/pg_restore binaries are
needed and the file is independent of the PostgreSQL major version. Generated columns
(`search_vector`) are rebuilt by PostgreSQL on import. Conversations and ingestion-run
history are never exported.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from app.core.config import Settings, get_settings
from app.ingestion.hasher import PIPELINE_VERSION
from app.ingestion.pipeline import chunking_config

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
TABLES = ("sources", "documents", "document_chunks")  # FK order
# Columns rewritten on export: ingestion-run history is not part of the seed.
NULLED_COLUMNS = {("sources", "last_run_id")}


class SeedError(RuntimeError):
    pass


@dataclass
class SeedReport:
    path: str
    rows: dict[str, int]
    manifest: dict
    seconds: float
    warnings: list[str]


def processing_signature(settings: Settings) -> str:
    """Must match SyncPipeline.signature: anything that changes processing output."""
    return f"{chunking_config(settings).signature}|x={settings.exclude_title_regex}"


def _dsn(settings: Settings) -> str:
    url = settings.database_url
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _columns(cur: psycopg.Cursor, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = %s AND is_generated = 'NEVER' ORDER BY ordinal_position", (table,))
    return [r[0] for r in cur.fetchall()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------- export

def export_index(output: str | Path, settings: Settings | None = None) -> SeedReport:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    files: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp, psycopg.connect(_dsn(settings)) as conn:
        # One consistent, read-only snapshot across all tables for the whole export.
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks WHERE embedding IS NULL "
                        "OR embedding_model <> %s", (settings.embedding_model,))
            pending = cur.fetchone()[0]
            if pending:
                raise SeedError(f"{pending} chunks are not embedded with {settings.embedding_model}; "
                                "run `python -m app.cli embed` before exporting")
            columns = {t: _columns(cur, t) for t in TABLES}
            for table in TABLES:
                cols = columns[table]
                select = ", ".join(f"NULL AS {c}" if (table, c) in NULLED_COLUMNS else c for c in cols)
                member = Path(tmp) / f"{table}.tsv.gz"
                n = 0
                with gzip.open(member, "wb", compresslevel=9) as gz, \
                        cur.copy(f"COPY (SELECT {select} FROM {table} ORDER BY id) TO STDOUT") as copy:
                    for block in copy:
                        gz.write(block)
                        n += bytes(block).count(b"\n")
                rows[table] = n
                files[member.name] = {"table": table, "columns": cols, "rows": n, "sha256": _sha256(member)}
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            pgvector = cur.fetchone()[0]
            conn.rollback()

        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "processing_signature": processing_signature(settings),
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "source_api_base_url": settings.source_api_base_url,
            "pgvector_version": pgvector,
            "rows": rows,
            "files": files,
        }
        tmp_out = output.with_suffix(output.suffix + ".partial")
        with tarfile.open(tmp_out, "w") as tar:  # members are already gzip-compressed
            data = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
            for name in files:
                tar.add(Path(tmp) / name, arcname=name)
        tmp_out.replace(output)  # atomic: never leave a half-written seed behind
    log.info("index exported", extra={"path": str(output), "rows": rows})
    return SeedReport(str(output), rows, manifest, time.perf_counter() - t0, [])


# ---------------------------------------------------------------------------------- import

def read_manifest(path: str | Path) -> dict:
    with tarfile.open(path, "r") as tar:
        member = tar.extractfile("manifest.json")
        if member is None:
            raise SeedError("seed has no manifest.json")
        return json.loads(member.read())


def check_compatibility(manifest: dict, settings: Settings, allow_model_mismatch: bool = False) -> list[str]:
    """Raise on hard incompatibilities; return warnings for soft ones."""
    if manifest.get("format_version") != FORMAT_VERSION:
        raise SeedError(f"unsupported seed format {manifest.get('format_version')} (expected {FORMAT_VERSION})")
    if manifest.get("embedding_dim") != settings.embedding_dim:
        raise SeedError(f"seed vectors have dimension {manifest.get('embedding_dim')}, "
                        f"this deployment expects {settings.embedding_dim}")
    warnings = []
    if manifest.get("embedding_model") != settings.embedding_model:
        msg = (f"seed was embedded with {manifest.get('embedding_model')}, this deployment uses "
               f"{settings.embedding_model}: every chunk would be re-embedded")
        if not allow_model_mismatch:
            raise SeedError(msg + " (pass --allow-model-mismatch to import the text anyway)")
        warnings.append(msg)
    if manifest.get("pipeline_version") != PIPELINE_VERSION or \
            manifest.get("processing_signature") != processing_signature(settings):
        warnings.append("seed was built with different processing code/settings "
                        f"({manifest.get('pipeline_version')} vs {PIPELINE_VERSION}): the next sync reprocesses "
                        "records, but chunks whose text is unchanged keep their embeddings")
    return warnings


def import_index(path: str | Path, settings: Settings | None = None, force: bool = False,
                 allow_model_mismatch: bool = False) -> SeedReport:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    path = Path(path)
    if not path.exists():
        raise SeedError(f"seed file not found: {path}")
    manifest = read_manifest(path)
    warnings = check_compatibility(manifest, settings, allow_model_mismatch)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path, "r") as tar:
            for name, meta in manifest["files"].items():
                member = tar.getmember(name)
                if not member.isfile() or "/" in name or name.startswith("."):
                    raise SeedError(f"unexpected member in seed: {name}")
                tar.extract(member, tmp, filter="data")
                if _sha256(Path(tmp) / name) != meta["sha256"]:
                    raise SeedError(f"checksum mismatch for {name}: the seed file is corrupted")

        with psycopg.connect(_dsn(settings)) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sources")
            existing = cur.fetchone()[0]
            if existing and not force:
                raise SeedError(f"database already contains {existing} sources; "
                                "import only into an empty database (or pass --force to replace it)")
            if existing:
                cur.execute("TRUNCATE document_chunks, documents, sources RESTART IDENTITY CASCADE")
            rows: dict[str, int] = {}
            for name, meta in manifest["files"].items():
                table, cols = meta["table"], meta["columns"]
                target = _columns(cur, table)
                missing = set(cols) - set(target)
                if missing:
                    raise SeedError(f"{table}: seed has columns {sorted(missing)} unknown to this schema "
                                    "(run migrations / use a matching code version)")
                with gzip.open(Path(tmp) / name, "rb") as gz, \
                        cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as copy:
                    for block in iter(lambda: gz.read(1 << 20), b""):
                        copy.write(block)
                cur.execute(f"SELECT count(*) FROM {table}")
                rows[table] = cur.fetchone()[0]
                if rows[table] != meta["rows"]:
                    raise SeedError(f"{table}: imported {rows[table]} rows, manifest says {meta['rows']}")
                cur.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                            f"COALESCE((SELECT max(id) FROM {table}), 1))")
            cur.execute(
                "INSERT INTO ingestion_runs (status, trigger, source_types, options, started_at, completed_at, "
                "records_fetched, records_created, stats) VALUES ('succeeded', 'seed', '{}', %s, now(), now(), "
                "%s, %s, %s)",
                (json.dumps({"seed": path.name}), rows["sources"], rows["sources"],
                 json.dumps({"seed_manifest": {k: manifest[k] for k in manifest if k != "files"}, "warnings": warnings},
                            ensure_ascii=False)))
            conn.commit()
    for w in warnings:
        log.warning("seed import: %s", w)
    log.info("index imported from seed", extra={"path": str(path), "rows": rows})
    return SeedReport(str(path), rows, manifest, time.perf_counter() - t0, warnings)


def database_is_empty(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    with psycopg.connect(_dsn(settings)) as conn, conn.cursor() as cur:
        cur.execute("SELECT NOT EXISTS (SELECT 1 FROM sources)")
        return bool(cur.fetchone()[0])
