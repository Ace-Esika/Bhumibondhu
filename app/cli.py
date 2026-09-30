"""Operations CLI.

    python -m app.cli sync [--source ebook --source qna_type2] [--force] [--no-embed]
                           [--from-dir DIR] [--allow-mass-delete]
    python -m app.cli embed                 # embed pending/changed chunks
    python -m app.cli reindex [--full]      # re-embed changed (default) or everything
    python -m app.cli status                # recent ingestion runs + embedding jobs
    python -m app.cli counts                # document / chunk counts
    python -m app.cli snapshot DIR          # save raw API payloads for offline replay
    python -m app.cli evaluate DATASET.jsonl [--k 5 --k 10] [--no-rerank] [--output report.json]
    python -m app.cli build-eval-set OUT.jsonl [--per-type N]
    python -m app.cli groq-models           # list models available to GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import signal
import sys
from pathlib import Path

from sqlalchemy import select, text

from app.core.config import SOURCE_TYPES, get_settings
from app.core.logging import configure_logging
from app.db.database import dispose_engine, get_sessionmaker
from app.db.models import EmbeddingJob, IngestionRun
from app.db.stats import document_counts


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


async def cmd_sync(args) -> int:
    from app.ingestion.runner import enqueue_run, execute_run

    opts = {"force": args.force, "no_embed": args.no_embed, "allow_mass_delete": args.allow_mass_delete}
    if args.from_dir:
        opts["from_dir"] = str(Path(args.from_dir).resolve())
    run_id = await enqueue_run(args.source or None, "cli", opts)
    run = await execute_run(run_id)
    _print({"run_id": run.id, "status": run.status, "fetched": run.records_fetched,
            "created": run.records_created, "updated": run.records_updated, "deleted": run.records_deleted,
            "skipped": run.records_skipped, "errors": run.errors, "error_details": run.error_details[:20],
            "stats": run.stats})
    return 0 if run.status in ("succeeded", "partial") else 1


async def cmd_embed(args, full: bool = False) -> int:
    from app.ingestion.runner import enqueue_run, execute_run

    run_id = await enqueue_run([], "cli", {"embed_only": True, "reindex_full": full})
    run = await execute_run(run_id)
    _print({"run_id": run.id, "status": run.status, "stats": run.stats, "errors": run.error_details})
    return 0 if run.status == "succeeded" else 1


async def cmd_status(args) -> int:
    async with get_sessionmaker()() as s:
        runs = (await s.execute(select(IngestionRun).order_by(IngestionRun.id.desc()).limit(args.limit))).scalars()
        jobs = (await s.execute(select(EmbeddingJob).order_by(EmbeddingJob.id.desc()).limit(args.limit))).scalars()
        _print({
            "runs": [{"id": r.id, "status": r.status, "trigger": r.trigger, "types": r.source_types,
                      "started": r.started_at, "completed": r.completed_at, "fetched": r.records_fetched,
                      "created": r.records_created, "updated": r.records_updated, "deleted": r.records_deleted,
                      "skipped": r.records_skipped, "errors": r.errors} for r in runs],
            "embedding_jobs": [{"id": j.id, "run_id": j.run_id, "model": j.model, "mode": j.mode,
                                "status": j.status, "embedded": j.chunks_embedded, "failed": j.chunks_failed,
                                "total": j.chunks_total, "started": j.started_at, "completed": j.completed_at}
                               for j in jobs],
        })
    return 0


async def cmd_counts(args) -> int:
    _print(await document_counts())
    return 0


async def cmd_snapshot(args) -> int:
    from app.ingestion.client import BhumipediaClient

    out = Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    async with BhumipediaClient() as client:
        for st in args.source or SOURCE_TYPES:
            records = await client.fetch(st)
            (out / f"{st}.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            print(f"{st}: {len(records)} records → {out / (st + '.json')}")
    return 0


async def cmd_evaluate(args) -> int:
    from app.evaluation.benchmark import run_benchmark

    report = await run_benchmark(args.dataset, ks=args.k or [5, 10], rerank=not args.no_rerank,
                                 generate=args.generate, delay_s=args.delay)
    _print(report["summary"])
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"full report → {args.output}")
    return 0


async def cmd_build_eval(args) -> int:
    from app.evaluation.dataset import build_self_retrieval_set

    n = await build_self_retrieval_set(args.output, per_type=args.per_type, seed=args.seed)
    print(f"wrote {n} examples → {args.output}")
    return 0


async def cmd_groq_models(args) -> int:
    import httpx

    s = get_settings()
    if not s.groq_api_key:
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 1
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {s.groq_api_key.get_secret_value()}"})
        r.raise_for_status()
        for m in sorted(r.json().get("data", []), key=lambda m: m["id"]):
            print(m["id"], f"(ctx={m.get('context_window')})")
    return 0


async def cmd_db_check(args) -> int:
    async with get_sessionmaker()() as s:
        v = (await s.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))).scalar()
        print(f"database ok; pgvector {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps stacks
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    p = argparse.ArgumentParser(prog="python -m app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sync", help="fetch sources and index changes")
    sp.add_argument("--source", action="append", choices=SOURCE_TYPES)
    sp.add_argument("--force", action="store_true", help="reprocess even unchanged records")
    sp.add_argument("--no-embed", action="store_true", help="skip the embedding pass")
    sp.add_argument("--from-dir", help="read <type>.json snapshots instead of the live API")
    sp.add_argument("--allow-mass-delete", action="store_true")

    sub.add_parser("embed", help="embed pending chunks")
    rp = sub.add_parser("reindex", help="re-embed chunks")
    rp.add_argument("--full", action="store_true", help="re-embed every chunk, not only pending ones")
    stp = sub.add_parser("status")
    stp.add_argument("--limit", type=int, default=10)
    sub.add_parser("counts")
    snp = sub.add_parser("snapshot")
    snp.add_argument("directory")
    snp.add_argument("--source", action="append", choices=SOURCE_TYPES)
    ep = sub.add_parser("evaluate")
    ep.add_argument("dataset")
    ep.add_argument("--k", action="append", type=int)
    ep.add_argument("--no-rerank", action="store_true")
    ep.add_argument("--generate", action="store_true", help="also call the LLM and check citations")
    ep.add_argument("--delay", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    ep.add_argument("--output")
    bp = sub.add_parser("build-eval-set")
    bp.add_argument("output")
    bp.add_argument("--per-type", type=int, default=40)
    bp.add_argument("--seed", type=int, default=13)
    sub.add_parser("groq-models")
    sub.add_parser("db-check")

    args = p.parse_args(argv)
    handlers = {
        "sync": cmd_sync, "embed": cmd_embed, "reindex": lambda a: cmd_embed(a, full=a.full),
        "status": cmd_status, "counts": cmd_counts, "snapshot": cmd_snapshot, "evaluate": cmd_evaluate,
        "build-eval-set": cmd_build_eval, "groq-models": cmd_groq_models, "db-check": cmd_db_check,
    }

    async def runner() -> int:
        if args.cmd in ("sync", "embed", "reindex"):
            from app.ingestion.runner import SyncAlreadyRunning, recover_stale_runs

            try:
                if n := await recover_stale_runs():
                    print(f"marked {n} stale run(s) as failed", file=sys.stderr)
            except SyncAlreadyRunning:
                print("another sync is running; this run will report it and exit", file=sys.stderr)
        try:
            return await handlers[args.cmd](args)
        finally:
            await dispose_engine()

    return asyncio.run(runner())


if __name__ == "__main__":
    sys.exit(main())
