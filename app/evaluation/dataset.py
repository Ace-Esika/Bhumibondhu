"""Auto-generated *self-retrieval* evaluation set built from indexed records.

These examples are derived mechanically from real records (the question of a Q&A pair, or a
"what does section N of <act> say" query). They measure whether the index can find a known
record, catch regressions, and let fusion weights / chunk sizes be compared. They are NOT a
substitute for a curated set of real user questions (see evaluation/curated.jsonl).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sqlalchemy import select

from app.core.text import ascii_to_bn_digits
from app.db.database import get_sessionmaker
from app.db.models import Document, DocumentChunk
from app.ingestion.normalizer import SECTION_LABELS


async def build_self_retrieval_set(output: str, per_type: int = 40, seed: int = 13) -> int:
    rng = random.Random(seed)
    examples: list[dict] = []
    async with get_sessionmaker()() as s:
        for st in ("qna_type1", "qna_type2"):
            docs = (await s.execute(select(Document.source_id, Document.title)
                                    .where(Document.source_type == st, Document.is_active))).all()
            for d in rng.sample(docs, min(per_type, len(docs))):
                examples.append({"id": f"self-{st}-{d.source_id}", "question": d.title,
                                 "expected_source_ids": [f"{st}:{d.source_id}"], "tags": [f"self_{st}"]})
        rows = (await s.execute(
            select(DocumentChunk.source_id, DocumentChunk.section_number, DocumentChunk.section_heading,
                   DocumentChunk.doc_type, Document.title, Document.source_id.label("doc_id"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.source_type == "section", DocumentChunk.section_number.is_not(None),
                   DocumentChunk.section_heading.is_not(None), Document.is_active)
        )).all()
        seen: set[str] = set()
        rows = [r for r in rows if not (r.source_id in seen or seen.add(r.source_id))]
        for r in rng.sample(rows, min(per_type, len(rows))):
            label = SECTION_LABELS.get(r.doc_type or "", "অনুচ্ছেদ")
            examples.append({
                "id": f"self-section-{r.source_id}",
                "question": f"{r.title} এর {label} {ascii_to_bn_digits(r.section_number)} এ কী বলা হয়েছে?",
                "expected_source_ids": [f"section:{r.source_id}"], "tags": ["self_section_by_number"]})
            examples.append({
                "id": f"self-heading-{r.source_id}",
                "question": r.section_heading.rstrip("।").strip(),
                "expected_source_ids": [f"ebook:{r.doc_id}"], "tags": ["self_section_heading_to_act"]})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(examples)


async def build_hierarchy_set(output: str, per_type: int = 40, seed: int = 13) -> int:
    """Hierarchy-aware examples for the legal acts: every question names a position in the
    act → section → subsection tree, and the expected id is that exact element.

    - `h_section_by_number`   "<act> এর ধারা N"
    - `h_subsection_by_number` "<act> এর ধারা N এর উপ-ধারা (k)"
    - `h_section_by_heading`  "<act> - <heading> সম্পর্কে কী বলা আছে"
    - `h_passage`             the opening words of a provision's own text (paraphrase-free)
    """
    import re

    rng = random.Random(seed)
    examples: list[dict] = []
    async with get_sessionmaker()() as s:
        rows = (await s.execute(
            select(DocumentChunk.source_type, DocumentChunk.source_id, DocumentChunk.section_number,
                   DocumentChunk.section_heading, DocumentChunk.doc_type, DocumentChunk.content,
                   DocumentChunk.metadata_, Document.title)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.source_type == "ebook", Document.is_active,
                   DocumentChunk.source_type.in_(("section", "subsection")),
                   DocumentChunk.section_number.is_not(None))
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )).all()

    def label(r) -> str:
        return SECTION_LABELS.get(r.doc_type or "", "অনুচ্ছেদ")

    def section_id(r) -> str | None:
        return next((u["id"] for u in (r.metadata_ or {}).get("path", []) if u["type"] == "section"), None)

    sections = {}
    for r in rows:
        sid = section_id(r)
        if sid and sid not in sections and re.fullmatch(r"\d+", r.section_number or ""):
            sections[sid] = r
    pool = list(sections.items())
    rng.shuffle(pool)
    for sid, r in pool[:per_type]:
        examples.append({"id": f"h-sec-{sid}", "tags": ["h_section_by_number"], "expected_source_ids": [f"section:{sid}"],
                         "question": f"{r.title} এর {label(r)} {ascii_to_bn_digits(r.section_number)} এ কী বলা হয়েছে?"})
    for sid, r in [p for p in pool[per_type:] if p[1].section_heading][:per_type]:
        examples.append({"id": f"h-head-{sid}", "tags": ["h_section_by_heading"], "expected_source_ids": [f"section:{sid}"],
                         "question": f"{r.title} - {r.section_heading.rstrip('।').strip()} সম্পর্কে কী বলা আছে"})
    subs = []
    for r in rows:
        m = r.metadata_ or {}
        nums, covers = m.get("subsection_numbers") or [], m.get("covers") or []
        sid = section_id(r)
        if r.source_type != "subsection" or not sid or not re.fullmatch(r"\d+", r.section_number or ""):
            continue
        if len(nums) == 1:
            subs.append((r, nums[0], r.source_id, sid))
        elif nums and len(nums) == len(covers):
            subs.extend((r, n, cid, sid) for n, cid in zip(nums, covers))
    rng.shuffle(subs)
    for r, num, eid, sid in subs[:per_type]:
        examples.append({"id": f"h-sub-{eid}", "tags": ["h_subsection_by_number"],
                         "expected_source_ids": [f"subsection:{eid}"],
                         "question": f"{r.title} এর {label(r)} {ascii_to_bn_digits(r.section_number)} এর "
                                     f"উপ-{label(r)} {num.strip('()')}"})
    passages = [r for r in rows if r.source_type == "section" and len(r.content.split()) >= 30]
    rng.shuffle(passages)
    for r in passages[:per_type]:
        words = r.content.split()
        examples.append({"id": f"h-pass-{r.source_id}", "tags": ["h_passage"],
                         "expected_source_ids": [f"section:{r.source_id}"], "question": " ".join(words[:22])})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(examples)
