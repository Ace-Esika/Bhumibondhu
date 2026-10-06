"""Context expansion for detailed questions.

Retrieval returns the best *chunks*, but a detailed legal answer needs whole *provisions*:

- `expand_sections`: when a retrieved chunk is part of a provision that was split across
  several chunks (long sections split at subsection boundaries), pull in its sibling chunks
  in document order, so "ধারা ৪ বিস্তারিত" sees the complete section.
- `expand_document`: for "all sections of <act>" requests, return the act's chunks in
  reading order (overview/table of contents first). Whatever does not fit the context
  budget is reported as a coverage note so the model says the answer is partial instead
  of guessing about sections it was not shown.

Both only ever add real chunks from the database; nothing is synthesised.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.text import ascii_to_bn_digits
from app.ingestion.normalizer import SECTION_LABELS
from app.retrieval.hybrid import Candidate, candidate_to_document
from app.retrieval.query import AnalyzedQuery
from app.retrieval.search import document_chunk_ids, document_section_count, fetch_chunks, section_chunk_ids


@dataclass
class Coverage:
    title: str
    label: str
    total: int


def _key(d: Document) -> tuple[int, str] | None:
    m = d.metadata
    if m.get("doc_source_type") == "ebook" and m.get("section_number"):
        return m["document_id"], m["section_number"]
    return None


async def _docs_for(session, ids: list[int], q: AnalyzedQuery) -> dict[int, Document]:
    rows = await fetch_chunks(session, ids)
    return {cid: candidate_to_document(Candidate(cid, row=row), q) for cid, row in rows.items()}


async def expand_sections(session, docs: list[Document], q: AnalyzedQuery, max_sections: int) -> list[Document]:
    """Complete up to `max_sections` provisions among the top results with their sibling chunks."""
    keys: list[tuple[int, str]] = []
    for d in docs:
        k = _key(d)
        if k and k not in keys:
            keys.append(k)
        if len(keys) >= max_sections:
            break
    if not keys:
        return docs
    siblings: dict[tuple[int, str], list[int]] = {}
    for k in keys:
        siblings[k] = await section_chunk_ids(session, *k)
    have = {d.metadata["chunk_id"] for d in docs}
    missing = [cid for ids in siblings.values() for cid in ids if cid not in have]
    if not missing:
        return docs
    extra = await _docs_for(session, missing, q)
    by_id = {d.metadata["chunk_id"]: d for d in docs} | extra
    out: list[Document] = []
    placed: set[int] = set()
    for d in docs:
        k = _key(d)
        group = siblings.get(k, [d.metadata["chunk_id"]]) if k else [d.metadata["chunk_id"]]
        for cid in group:  # the whole provision, in order, where its best chunk ranked
            if cid not in placed and cid in by_id:
                out.append(by_id[cid])
                placed.add(cid)
    return out


async def expand_document(session, docs: list[Document], q: AnalyzedQuery) -> tuple[list[Document], Coverage | None]:
    """For "all sections" requests: the top-ranked act's chunks in reading order."""
    target = next((d for d in docs if d.metadata.get("doc_source_type") == "ebook"), None)
    if target is None:
        return docs, None
    doc_id = target.metadata["document_id"]
    ids = await document_chunk_ids(session, doc_id)
    by_id = await _docs_for(session, ids, q)
    # TOC first, then provisions; the preamble would only spend budget ahead of the provisions.
    ordered = [by_id[i] for i in ids if i in by_id
               and (by_id[i].metadata.get("chunk_metadata") or {}).get("role") != "preamble"]
    total = await document_section_count(session, doc_id)
    label = SECTION_LABELS.get(target.metadata.get("doc_type") or "", "অনুচ্ছেদ")
    return ordered, Coverage(title=target.metadata.get("title") or "", label=label, total=total)


def coverage_note(cov: Coverage, used: list[Document]) -> str | None:
    """Describe which provisions actually made it into the context (after budgeting)."""
    from app.llm.prompts import COVERAGE_NOTE

    nums: list[str] = []
    for d in used:
        n = d.metadata.get("section_number")
        if n and n not in nums:
            nums.append(n)
    if cov.total == 0 or len(nums) >= cov.total:
        return None
    covered = ", ".join(ascii_to_bn_digits(n) for n in nums) if nums else "শুধু পরিচিতি ও সূচি"
    # Bengali genitive: ধারা→ধারার, বিধি→বিধির, অনুচ্ছেদ→অনুচ্ছেদের
    label_gen = cov.label + ("র" if cov.label[-1] in "ািীুূেৈোৌ" else "ের")
    return COVERAGE_NOTE.format(title=cov.title, total=ascii_to_bn_digits(str(cov.total)),
                                included=ascii_to_bn_digits(str(len(nums))), covered=f"{cov.label} {covered}",
                                label=cov.label, label_gen=label_gen)


def focus_note(q: AnalyzedQuery) -> str | None:
    """Prompt note for questions that name a provision, so the model answers from that exact
    provision and admits it when the requested subsection / clause was not retrieved."""
    from app.llm.prompts import FOCUS_NOTE

    if not q.section_number:
        return None
    parts = [f"ধারা/বিধি {ascii_to_bn_digits(q.section_number)}"]
    missing = "provision"
    if q.subsection_number:
        parts.append(f"উপ-ধারা {ascii_to_bn_digits(q.subsection_number)}")
        missing = "subsection"
    if q.clause:
        parts.append(f"দফা ({q.clause})")
        missing = "clause"
    return FOCUS_NOTE.format(target=", ".join(parts), missing=missing)
