"""Retrieval / RAG evaluation.

Dataset: JSONL (or a JSON array), one example per line:

    {"id": "q1", "question": "...", "expected_source_ids": ["qna_type2:24776", "section:1675"],
     "expected_answer": "... (optional, for human review)", "tags": ["namjari"]}

`expected_source_ids` entries are either document-level `<doc_source_type>:<doc_source_id>`
(e.g. `ebook:241`, `qna_type2:24776`) or element-level `<element_type>:<element_id>` (e.g.
`section:1675`, `subsection:4317`). A retrieved chunk is relevant if either key matches.
An empty list marks an *unanswerable* question: the system should refuse.

Metrics (retrieval): Recall@k (hit rate: ≥1 relevant in top k), full Recall@k (fraction of
expected ids found), MRR@maxk, Precision@k. With `--generate` (needs Groq): citation
precision (cited sources that are relevant), citation hit rate (answer cites ≥1 relevant
source), refusal accuracy on unanswerable questions, grounded rate.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.llm.base import LLMError
from app.rag.guardrails import evidence_sufficient


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip() and not line.startswith("//")]
    for i, r in enumerate(rows):
        if "question" not in r or "expected_source_ids" not in r:
            raise ValueError(f"example {i} missing question/expected_source_ids")
        r.setdefault("id", f"ex{i}")
    return rows


def doc_keys(meta: dict[str, Any]) -> set[str]:
    return {f"{meta.get('doc_source_type')}:{meta.get('doc_source_id')}",
            f"{meta.get('source_type')}:{meta.get('source_id')}"}


def source_keys(src: dict[str, Any]) -> set[str]:
    return {f"{src.get('source_type')}:{src.get('source_id')}", f"{src.get('element_type')}:{src.get('element_id')}"}


def retrieval_metrics(docs: list[Document], expected: set[str], ks: list[int]) -> dict[str, float]:
    rel = [bool(doc_keys(d.metadata) & expected) for d in docs]
    out: dict[str, float] = {}
    for k in ks:
        top = docs[:k]
        found = set().union(*(doc_keys(d.metadata) for d in top)) & expected if top else set()
        out[f"hit@{k}"] = float(any(rel[:k]))
        out[f"recall@{k}"] = len(found) / len(expected)
        out[f"precision@{k}"] = sum(rel[:k]) / k
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    return out


async def run_benchmark(dataset_path: str, ks: list[int] | None = None, rerank: bool = True,
                        generate: bool = False, delay_s: float = 0.0) -> dict[str, Any]:
    from app.rag.pipeline import RAGPipeline
    from app.retrieval.hybrid import HybridSearcher

    settings = get_settings()
    ks = sorted(set(ks or [5, 10]))
    examples = load_dataset(dataset_path)
    searcher = HybridSearcher(settings)
    pipeline = RAGPipeline(settings, searcher=searcher) if generate else None

    per_example, by_tag = [], defaultdict(list)
    answerable = [e for e in examples if e["expected_source_ids"]]
    unanswerable = [e for e in examples if not e["expected_source_ids"]]
    latencies = []

    for ex in examples:
        expected = set(ex["expected_source_ids"])
        t0 = time.perf_counter()
        docs, trace = await searcher.search(ex["question"], final_k=max(ks), rerank=rerank)
        latencies.append((time.perf_counter() - t0) * 1000)
        row: dict[str, Any] = {"id": ex["id"], "question": ex["question"], "tags": ex.get("tags", []),
                               "retrieved": [sorted(doc_keys(d.metadata))[0] for d in docs[:max(ks)]]}
        gate_docs = docs[: settings.final_context_k]
        passes_gate, why = evidence_sufficient(gate_docs, settings, trace.reranked)
        row["evidence_gate"] = {"pass": passes_gate, "why": why}
        if expected:
            row.update(retrieval_metrics(docs, expected, ks))
            for t in ex.get("tags", []) or ["untagged"]:
                by_tag[t].append(row)
        if generate and pipeline is not None:
            if delay_s and per_example:
                await asyncio.sleep(delay_s)  # stay under provider rate limits (e.g. Groq free tier TPM)
            try:
                res = await pipeline.answer(ex["question"])
            except LLMError as e:
                row["llm_error"] = type(e).__name__
                per_example.append(row)
                continue
            cited = [s for s in res.sources if s.get("cited")]
            relevant = [s for s in cited if source_keys(s) & expected]
            row["answer"] = res.answer
            row["grounded"] = res.grounded
            row["refused"] = res.reason is not None and res.reason.startswith(("insufficient", "llm_reported"))
            row["citation_precision"] = len(relevant) / len(cited) if cited else 0.0
            row["citation_hit"] = float(bool(relevant))
        per_example.append(row)

    def mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return round(statistics.mean(vals), 4) if vals else None

    answered_rows = [r for r in per_example if r["id"] in {e["id"] for e in answerable}]
    unans_rows = [r for r in per_example if r["id"] in {e["id"] for e in unanswerable}]
    summary: dict[str, Any] = {
        "dataset": str(dataset_path), "examples": len(examples), "answerable": len(answerable),
        "unanswerable": len(unanswerable), "reranker": bool(rerank and searcher.reranker is not None),
        "fusion": settings.fusion_method, "weights": [settings.vector_weight, settings.lexical_weight],
        "embedding_model": settings.embedding_model,
        "retrieval_latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
    }
    for k in ks:
        summary[f"hit@{k}"] = mean(answered_rows, f"hit@{k}")
        summary[f"recall@{k}"] = mean(answered_rows, f"recall@{k}")
        summary[f"precision@{k}"] = mean(answered_rows, f"precision@{k}")
    summary["mrr"] = mean(answered_rows, "mrr")
    summary["evidence_gate_pass_rate_answerable"] = mean(
        [{"v": float(r["evidence_gate"]["pass"])} for r in answered_rows], "v")
    summary["evidence_gate_reject_rate_unanswerable"] = mean(
        [{"v": float(not r["evidence_gate"]["pass"])} for r in unans_rows], "v")
    if generate:
        summary["citation_precision"] = mean(answered_rows, "citation_precision")
        summary["citation_hit_rate"] = mean(answered_rows, "citation_hit")
        answered_rows_ok = [r for r in answered_rows if "grounded" in r]
        unans_ok = [r for r in unans_rows if "refused" in r]
        summary["grounded_rate"] = mean([{"v": float(r["grounded"])} for r in answered_rows_ok], "v")
        summary["refusal_accuracy_unanswerable"] = mean([{"v": float(r["refused"])} for r in unans_ok], "v")
        summary["llm_errors"] = sum(1 for r in per_example if "llm_error" in r)
        summary["false_refusal_rate_answerable"] = mean([{"v": float(r["refused"])} for r in answered_rows_ok], "v")
    summary["by_tag"] = {t: {"n": len(rows), "mrr": mean(rows, "mrr"),
                             **{f"hit@{k}": mean(rows, f"hit@{k}") for k in ks}} for t, rows in sorted(by_tag.items())}
    return {"summary": summary, "examples": per_example}
