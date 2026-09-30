"""Auto-generated *self-retrieval* evaluation set built from indexed records.

These examples are derived mechanically from real records (the question of a Q&A pair, or a
"what does section N of <act> say" query). They measure whether the index can find a known
record, catch regressions, and let fusion weights / chunk sizes be compared. They are NOT a
substitute for a curated set of real user questions (see evaluation/curated.jsonl).
"""

from __future__ import annotations

import json
import random
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
