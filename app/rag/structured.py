"""Exact/structured queries answered from the database instead of generation.

Only question shapes backed by reliable structured fields are handled (currently: counts of
approved ebooks by `ebooks_type`, optionally per year). Anything else goes to RAG.
The source API only returns approved records, so every active ebook is an approved one.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.text import ascii_to_bn_digits
from app.db.models import Document

MAX_LISTED = 15


async def answer_structured(session: AsyncSession, spec: dict[str, Any]) -> dict[str, Any] | None:
    if spec.get("kind") != "count_ebooks":
        return None
    doc_type, year = spec["doc_type"], spec.get("year")
    where = [Document.source_type == "ebook", Document.is_active.is_(True), Document.doc_type == doc_type]
    if year:
        where.append(Document.year == year)
    total = (await session.execute(select(func.count()).select_from(Document).where(*where))).scalar_one()
    rows = (await session.execute(
        select(Document.source_id, Document.title, Document.year, Document.url, Document.file_url)
        .where(*where).order_by(Document.year.desc().nulls_last(), Document.title).limit(MAX_LISTED)
    )).all()

    scope = f"{ascii_to_bn_digits(str(year))} সালের " if year else ""
    if total == 0:
        answer = f"ভূমিপিডিয়ার বর্তমান তথ্যভান্ডারে {scope}'{doc_type}' ধরনের কোনো অনুমোদিত দলিল পাওয়া যায়নি।"
    else:
        answer = (f"ভূমিপিডিয়ার বর্তমান তথ্যভান্ডারে {scope}'{doc_type}' ধরনের মোট "
                  f"{ascii_to_bn_digits(str(total))}টি অনুমোদিত দলিল রয়েছে।")
        if rows:
            shown = "\n".join(f"- {r.title}" for r in rows)
            more = f"\n(প্রথম {ascii_to_bn_digits(str(len(rows)))}টি দেখানো হলো)" if total > len(rows) else ""
            answer += f"\n\n{shown}{more}"
    base = get_settings().source_api_base_url.rstrip("/")
    sources = [{
        "index": 1, "title": "ভূমিপিডিয়া ই-বুক তালিকা (অনুমোদিত রেকর্ড)", "source_type": "ebook",
        "element_type": "database_count", "source_id": None, "element_id": None, "doc_type": doc_type,
        "section": None, "year": year, "authority": "সরকারি দলিল", "url": f"{base}/api/ebooks/full/",
        "snippet": f"doc_type={doc_type}" + (f", year={year}" if year else ""), "cited": True,
    }]
    for i, r in enumerate(rows, start=2):
        sources.append({
            "index": i, "title": r.title, "source_type": "ebook", "element_type": "ebook", "source_id": r.source_id,
            "element_id": r.source_id, "doc_type": doc_type, "section": None, "year": r.year,
            "authority": "সরকারি দলিল", "url": r.url or r.file_url, "snippet": "", "cited": False,
        })
    return {"answer": answer, "sources": sources, "count": total}
