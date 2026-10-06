"""Dense (pgvector) and lexical (PostgreSQL FTS) candidate retrieval.

Every value reaches SQL as a bind parameter. Filter *columns* come from a fixed whitelist;
nothing user-supplied is ever interpolated into SQL text.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.text import bn_to_ascii_digits, lexical_coverage, lexical_text, normalize_section_number
from app.retrieval.query import SearchFilters


@dataclass
class Hit:
    chunk_id: int
    score: float
    rank: int


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def filter_clause(filters: SearchFilters | None, params: dict) -> str:
    if filters is None or filters.is_empty():
        return ""
    parts = []
    if filters.source_types:
        parts.append("c.doc_source_type = ANY(:f_source_types)")
        params["f_source_types"] = list(filters.source_types)
    if filters.doc_types:
        parts.append("c.doc_type = ANY(:f_doc_types)")
        params["f_doc_types"] = list(filters.doc_types)
    if filters.categories:
        parts.append("c.category = ANY(:f_categories)")
        params["f_categories"] = [c.strip().lower() for c in filters.categories]
    if filters.year:
        parts.append("c.year = :f_year")
        params["f_year"] = filters.year
    if filters.document_ids:
        parts.append("c.document_id IN (SELECT id FROM documents WHERE source_id = ANY(:f_doc_ids) "
                     "AND source_type = 'ebook')")
        params["f_doc_ids"] = list(filters.document_ids)
    return " AND " + " AND ".join(parts)


async def vector_search(session: AsyncSession, query_vec: list[float], model: str, k: int,
                        filters: SearchFilters | None = None, ef_search: int = 80) -> tuple[list[Hit], float]:
    t0 = time.perf_counter()
    params: dict = {"qvec": _vector_literal(query_vec), "model": model, "k": k}
    where = filter_clause(filters, params)
    # SET LOCAL scopes the setting to this transaction. Values are validated ints.
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    if where:
        # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the filter.
        await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = (await session.execute(text(f"""
        SELECT c.id, 1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
        FROM document_chunks c
        WHERE c.embedding IS NOT NULL AND c.embedding_model = :model {where}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :k
    """), params)).all()
    hits = [Hit(r.id, float(r.similarity), i + 1) for i, r in enumerate(rows)]
    hits.sort(key=lambda h: h.score, reverse=True)  # relaxed_order may be slightly out of order
    for i, h in enumerate(hits):
        h.rank = i + 1
    return hits, (time.perf_counter() - t0) * 1000


async def lexical_search(session: AsyncSession, terms: list[str], k: int,
                         filters: SearchFilters | None = None) -> tuple[list[Hit], float]:
    """OR-combined tsquery over the weighted (title=A, body=B) tsvector.

    `terms` come from `app.core.text.query_terms`, which only emits [letters/digits](:*)?
    so the tsquery string is always well-formed; it is still passed as a bind parameter.
    ts_rank_cd normalisation 1|32: divide by log(length) (so long manual chunks don't win on
    sheer size) and map into [0, 1).
    """
    if not terms:
        return [], 0.0
    t0 = time.perf_counter()
    params: dict = {"tsq": " | ".join(terms), "k": k}
    where = filter_clause(filters, params)
    rows = (await session.execute(text(f"""
        SELECT c.id, ts_rank_cd(c.search_vector, q, 33) AS rank
        FROM document_chunks c, to_tsquery('simple', :tsq) q
        WHERE c.search_vector @@ q {where}
        ORDER BY rank DESC, c.authority DESC, c.id
        LIMIT :k
    """), params)).all()
    hits = [Hit(r.id, float(r.rank), i + 1) for i, r in enumerate(rows)]
    return hits, (time.perf_counter() - t0) * 1000


async def section_lookup(session: AsyncSession, section_number: str, title_terms: list[str], k: int,
                         filters: SearchFilters | None = None, year: int | None = None,
                         subsection: str | None = None, clause: str | None = None) -> list[Hit]:
    """Exact structural match for queries that name a provision ("<act> এর ধারা ৫ উপ-ধারা (২)").

    1. Acts are ranked by how well their *document title* (not the section heading) covers the
       query's remaining terms; the year named in the query is only a tie-breaker, because an
       act's title year and its `act_year` field disagree for some records.
    2. Every chunk of that provision is returned in document order, with the requested
       subsection / clause first, so a long provision split over several chunks is still found
       by its subsection number.

    Title-less requests ("ধারা ৫ কী?") are ambiguous across acts and return nothing.
    """
    if not title_terms:
        return []
    params: dict = {"n": section_number}
    where = filter_clause(filters, params)
    rows = (await session.execute(text(f"""
        SELECT c.id, c.document_id, c.chunk_index, c.metadata, c.content, d.title
        FROM document_chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.section_number = :n AND d.is_active {where}
    """), params)).mappings().all()
    return rank_section_rows([dict(r) for r in rows], title_terms, k, year, subsection, clause)


def rank_section_rows(rows: list[dict], title_terms: list[str], k: int, year: int | None = None,
                      subsection: str | None = None, clause: str | None = None) -> list[Hit]:
    """Pure ranking step of `section_lookup` (unit-testable without a database)."""
    if not rows:
        return []
    year_token = str(year) if year else None
    scores: dict[int, tuple[float, int, int]] = {}
    for r in rows:
        if r["document_id"] in scores:
            continue
        body = lexical_text(r["title"])
        tokens = body.split()
        scores[r["document_id"]] = (lexical_coverage(title_terms, body),
                                    1 if year_token and year_token in tokens else 0,
                                    -len(tokens))  # prefer the tighter title ("The X Act" over "The X Act (Repealed)")
    best = max(scores.values())
    if best[0] < 0.5:
        return []
    docs = {d for d, sc in scores.items() if sc == best}
    if len(docs) > 2:
        return []  # "ভূমি আইনের ধারা ৫": the title names no single act, so claiming an exact match would be a guess
    picked = [r for r in rows if r["document_id"] in docs]
    picked.sort(key=lambda r: (r["document_id"], _subsection_rank(r, subsection, clause), r["chunk_index"]))
    return [Hit(r["id"], 1.0, i + 1) for i, r in enumerate(picked[: max(k, 6)])]


def _marker(label: str) -> re.Pattern:
    """A list marker at the start of a line: "(২)", "২)", "দফা (ক)"."""
    return re.compile(rf"(?:^|\n)(?:উপ-)?(?:দফা\s*|তফসিল\s*)?\(?{re.escape(label)}\)", re.IGNORECASE)


def _subsection_rank(row: dict, subsection: str | None, clause: str | None) -> int:
    """0 = the chunk holds exactly what was asked; larger = further away. Without a subsection or
    clause in the query every chunk of the provision ties and document order decides."""
    if not subsection and not clause:
        return 0
    meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
    content = bn_to_ascii_digits(row["content"])
    # Upstream stores both উপ-ধারা "(২)" and দফা "(ক)" as subsections, so either kind of
    # reference is checked against the chunk's recorded subsection numbers first.
    nums = {normalize_section_number(n) for n in (meta.get("subsection_numbers") or [meta.get("subsection_number")])
            if n}
    rank = 0
    if subsection and subsection not in nums:
        rank += 1 if _marker(subsection).search(content) else 4
    if clause and clause not in nums and not _marker(clause).search(content):
        rank += 2
    return rank


async def section_chunk_ids(session: AsyncSession, document_id: int, section_number: str) -> list[int]:
    """All chunks of one provision, in document order (a long section may span several)."""
    rows = await session.execute(text(
        "SELECT id FROM document_chunks WHERE document_id = :d AND section_number = :n ORDER BY chunk_index"),
        {"d": document_id, "n": section_number})
    return [r.id for r in rows]


async def document_chunk_ids(session: AsyncSession, document_id: int, limit: int = 400) -> list[int]:
    """A document's chunks in reading order (overview/TOC first)."""
    rows = await session.execute(text(
        "SELECT id FROM document_chunks WHERE document_id = :d ORDER BY chunk_index LIMIT :k"),
        {"d": document_id, "k": limit})
    return [r.id for r in rows]


async def document_section_count(session: AsyncSession, document_id: int) -> int:
    return (await session.execute(text(
        "SELECT count(DISTINCT section_number) FROM document_chunks WHERE document_id = :d "
        "AND section_number IS NOT NULL"), {"d": document_id})).scalar_one()


async def fetch_chunks(session: AsyncSession, chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    rows = (await session.execute(text("""
        SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.source_type, c.source_id, c.parent_source_id,
               c.doc_source_type, c.doc_type, c.category, c.year, c.authority, c.title, c.section_number,
               c.section_heading, c.context, c.content, c.metadata AS chunk_metadata, c.lexical_body,
               d.source_id AS doc_source_id, d.url, d.file_url, d.act_number, d.publication_date,
               d.metadata AS doc_metadata, d.author
        FROM document_chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(:ids) AND d.is_active
    """), {"ids": chunk_ids})).mappings().all()
    return {r["chunk_id"]: dict(r) for r in rows}
