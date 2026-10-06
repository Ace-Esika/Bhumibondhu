"""Hybrid retrieval: dense + lexical candidates → score fusion → boosts → (optional) rerank.

Exposed as a LangChain `BaseRetriever` (`BhumipediaRetriever`) returning `Document`s whose
metadata carries everything citations need (built from DB rows, never from the LLM).

Fusion methods (FUSION_METHOD):
- `rrf` (default): weighted Reciprocal Rank Fusion, w_v/(k+rank_v) + w_l/(k+rank_l).
  Robust to the very different score scales of cosine similarity and ts_rank.
- `linear`: min-max normalise each list, then w_v*v + w_l*l.
Both use VECTOR_WEIGHT / LEXICAL_WEIGHT; benchmark with `python -m app.cli evaluate`.

Post-fusion multiplicative boosts:
- authority: 1 + AUTHORITY_BOOST * authority/100 (official law > Q&A > blog > forum)
- section match: 1 + SECTION_MATCH_BOOST when the query names "ধারা N" and the chunk is N
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.core.cache import cache_get, cache_set, make_key
from app.core.config import Settings, get_settings
from app.core.text import lexical_coverage, normalize_query
from app.db.database import get_sessionmaker
from app.retrieval.embeddings import EmbeddingError, EmbeddingService, get_embedding_service
from app.retrieval.query import DEFINITION_TERMS, AnalyzedQuery, SearchFilters, analyze_query
from app.retrieval.reranker import BGEReranker, RerankerError, get_reranker
from app.retrieval.search import Hit, fetch_chunks, lexical_search, section_lookup, vector_search

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    chunk_id: int
    vector_score: float | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    fused: float = 0.0
    rerank_score: float | None = None
    exact: bool = False  # found by the structural "<act> ধারা N" lookup
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    embedding_ms: float = 0.0
    vector_ms: float = 0.0
    lexical_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    vector_hits: int = 0
    lexical_hits: int = 0
    candidates: int = 0
    reranked: bool = False
    rerank_failed: bool = False
    embedding_failed: bool = False


def fuse(vector_hits: list[Hit], lexical_hits: list[Hit], method: str, w_vec: float, w_lex: float,
         rrf_k: int = 60) -> dict[int, Candidate]:
    cands: dict[int, Candidate] = {}
    for h in vector_hits:
        c = cands.setdefault(h.chunk_id, Candidate(h.chunk_id))
        c.vector_score, c.vector_rank = h.score, h.rank
    for h in lexical_hits:
        c = cands.setdefault(h.chunk_id, Candidate(h.chunk_id))
        c.lexical_score, c.lexical_rank = h.score, h.rank

    if method == "rrf":
        for c in cands.values():
            c.fused = (w_vec / (rrf_k + c.vector_rank) if c.vector_rank else 0.0) + \
                      (w_lex / (rrf_k + c.lexical_rank) if c.lexical_rank else 0.0)
    elif method == "linear":
        def norm(hits: list[Hit]) -> dict[int, float]:
            if not hits:
                return {}
            lo, hi = min(h.score for h in hits), max(h.score for h in hits)
            span = hi - lo
            return {h.chunk_id: (h.score - lo) / span if span > 0 else 1.0 for h in hits}

        nv, nl = norm(vector_hits), norm(lexical_hits)
        for c in cands.values():
            c.fused = w_vec * nv.get(c.chunk_id, 0.0) + w_lex * nl.get(c.chunk_id, 0.0)
    else:
        raise ValueError(f"unknown fusion method {method}")
    return cands


_OVERVIEW_ROLES = ("overview", "toc", "preamble")


def apply_boosts(cands: list[Candidate], query: AnalyzedQuery, settings: Settings) -> None:
    """Multiplicative post-fusion boosts.

    The year in a query is deliberately NOT used here: it is usually copied from the act's
    title, while the stored `year` comes from a separate upstream field that disagrees for
    some records, so matching on it would punish the right act.
    """
    for c in cands:
        authority = int(c.row.get("authority") or 0)
        boost = 1.0 + settings.authority_boost * authority / 100.0
        if query.section_number and c.row.get("section_number") == query.section_number:
            boost *= 1.0 + settings.section_match_boost
        role = (c.row.get("chunk_metadata") or {}).get("role")
        if role in _OVERVIEW_ROLES and not query.wants_act_overview:
            # Every act has an overview chunk full of its title; it would otherwise win any
            # query that merely names the act while asking about one of its provisions.
            boost *= settings.overview_demotion
        c.fused *= boost


def _title_terms(q: AnalyzedQuery) -> list[str]:
    """Query terms that can identify an act: drop the section number itself, provision words
    and definition helpers."""
    drop = {q.section_number, q.subsection_number, q.clause, "ধারা:*", "ধারা", "বিধি", "অনুচ্ছেদ:*", "প্রবিধান:*",
            "section", "rule", "article", "উপ", "উপধারা", "উপধারা:*", "দফা", "নম্বর", "বলা", "বলে", "বলা:*",
            "subsection", "clause", *DEFINITION_TERMS}
    return [t for t in q.terms if t not in drop and not t.rstrip(":*").isdigit()]


def promote_exact_matches(cands: list[Candidate], exact_ids: list[int], settings: Settings) -> None:
    """An explicit, structurally exact match ("ধারা N" of the named act) outranks fuzzy
    matches: place it just above the current best, preserving the lookup order."""
    if not exact_ids or not cands:
        return
    top = max(c.fused for c in cands) or 1e-6
    by_id = {c.chunk_id: c for c in cands}
    for rank, cid in enumerate(exact_ids):
        if cid in by_id:
            by_id[cid].exact = True
            by_id[cid].fused = max(by_id[cid].fused, top * (1 + settings.section_match_boost) - rank * 1e-9)


def candidate_to_document(c: Candidate, query: AnalyzedQuery) -> Document:
    r = c.row
    meta = {
        "chunk_id": c.chunk_id, "document_id": r["document_id"], "chunk_index": r["chunk_index"],
        "source_type": r["source_type"], "source_id": r["source_id"], "parent_source_id": r["parent_source_id"],
        "doc_source_type": r["doc_source_type"], "doc_source_id": r["doc_source_id"], "doc_type": r["doc_type"],
        "title": r["title"], "section_number": r["section_number"], "section_heading": r["section_heading"],
        "category": r["category"], "year": r["year"], "authority": r["authority"], "url": r["url"],
        "file_url": r["file_url"], "act_number": r["act_number"], "author": r["author"],
        "publication_date": r["publication_date"].isoformat() if r["publication_date"] else None,
        "chunk_metadata": r["chunk_metadata"] or {}, "doc_metadata": r["doc_metadata"] or {},
        "context": r["context"], "content": r["content"],
        "vector_score": c.vector_score, "vector_rank": c.vector_rank, "lexical_score": c.lexical_score,
        "lexical_rank": c.lexical_rank, "fused_score": c.fused, "rerank_score": c.rerank_score,
        "lexical_coverage": lexical_coverage(query.terms, r.get("lexical_body") or ""), "exact_match": c.exact,
    }
    page = f"{r['context']}\n\n{r['content']}" if r["context"] else r["content"]
    return Document(page_content=page, metadata=meta, id=str(c.chunk_id))


class HybridSearcher:
    def __init__(self, settings: Settings | None = None, embedder: EmbeddingService | None = None,
                 reranker: BGEReranker | None | bool = True, sessionmaker=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_service()
        self.reranker = get_reranker() if reranker is True else (reranker or None)
        self.sessionmaker = sessionmaker or get_sessionmaker()

    async def search(self, query: AnalyzedQuery | str, filters: SearchFilters | None = None,
                     final_k: int | None = None, rerank: bool = True,
                     query_vector: list[float] | None = None) -> tuple[list[Document], RetrievalTrace]:
        s = self.settings
        q = analyze_query(query) if isinstance(query, str) else query
        trace = RetrievalTrace()
        t_start = time.perf_counter()
        final_k = final_k or s.final_context_k

        # 1) query embedding (degrades to lexical-only retrieval on failure)
        emb_key = make_key("qemb", s.embedding_model, q.text) if s.cache_enabled else None
        if query_vector is None and emb_key:
            query_vector = await cache_get(emb_key)
        if query_vector is None:
            try:
                query_vector, trace.embedding_ms = await self.embedder.aembed_query(q.text)
                if emb_key:
                    await cache_set(emb_key, query_vector, s.query_embedding_cache_ttl_seconds)
            except EmbeddingError:
                log.exception("query embedding failed; continuing lexical-only")
                trace.embedding_failed = True

        # 2) candidates from both retrievers (one transaction; SET LOCAL stays scoped)
        async with self.sessionmaker() as session, session.begin():
            vec_hits: list[Hit] = []
            if query_vector is not None:
                vec_hits, trace.vector_ms = await vector_search(session, query_vector, s.embedding_model,
                                                                s.vector_top_k, filters, s.hnsw_ef_search)
            lex_hits, trace.lexical_ms = await lexical_search(session, q.terms, s.lexical_top_k, filters)
            trace.vector_hits, trace.lexical_hits = len(vec_hits), len(lex_hits)
            exact: list[Hit] = []
            if q.section_number:
                exact = await section_lookup(session, q.section_number, _title_terms(q), 6, filters, q.year,
                                             q.subsection_number, q.clause)

            cands = fuse(vec_hits, lex_hits, s.fusion_method, s.vector_weight, s.lexical_weight, s.rrf_k)
            for h in exact:
                cands.setdefault(h.chunk_id, Candidate(h.chunk_id))
            rows = await fetch_chunks(session, list(cands))
        for cid, row in rows.items():
            cands[cid].row = row
        ranked = [c for c in cands.values() if c.row]
        apply_boosts(ranked, q, s)
        promote_exact_matches(ranked, [h.chunk_id for h in exact], s)
        ranked.sort(key=lambda c: c.fused, reverse=True)
        pool = ranked[: s.reranker_top_k]
        trace.candidates = len(pool)

        # 3) optional rerank of the top pool
        if rerank and self.reranker is not None and pool:
            t0 = time.perf_counter()
            try:
                docs = [candidate_to_document(c, q) for c in pool]
                scores = await asyncio.to_thread(self.reranker.score, q.text, [d.page_content for d in docs])
                for c, sc in zip(pool, scores, strict=True):
                    c.rerank_score = sc
                pool.sort(key=lambda c: (c.rerank_score or 0.0) * (1 + 0.05 * (c.row.get("authority") or 0) / 100),
                          reverse=True)
                trace.reranked = True
            except RerankerError:
                trace.rerank_failed = True
                log.exception("reranker failed")
                if not s.reranker_fallback_on_error:
                    raise
            trace.rerank_ms = (time.perf_counter() - t0) * 1000

        final = [candidate_to_document(c, q) for c in pool[:final_k]]
        trace.total_ms = (time.perf_counter() - t_start) * 1000
        return final, trace


class BhumipediaRetriever(BaseRetriever):
    """LangChain retriever over the hybrid pgvector + PostgreSQL FTS index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    searcher: Any
    filters: SearchFilters | None = None
    k: int | None = None
    rerank: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError("BhumipediaRetriever is async-only; use `ainvoke`.")

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]:
        docs, _ = await self.searcher.search(query, self.filters, self.k, self.rerank)
        return docs


def cache_key_parts(query: AnalyzedQuery, filters: SearchFilters | None) -> tuple:
    return (normalize_query(query.text), filters.model_dump(exclude_none=True) if filters else {})

