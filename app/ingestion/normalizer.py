"""Normalise validated API records into documents with a preserved structural hierarchy.

Output is a list of `NormalizedDocument`s, each carrying a tree of `Unit`s:

    ebook (act)                       source_type="ebook"
     ├── section                      source_type="section"
     │    ├── subsection              source_type="subsection"
     │    │    └── schedule           source_type="schedule"
     │    │         └── subschedule   source_type="subschedule"
     │    └── schedule (section-level)
     └── act-level free text (`schedules` field) → "ebook_text"

Nothing is invented: fields missing upstream stay None.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.text import (
    clean_unicode,
    html_to_text,
    normalize_section_number,
    normalize_whitespace,
    parse_year,
)
from app.ingestion.schemas import SCHEMAS, BlogIn, EbookIn, ForumIn, QnaIn, ScheduleIn, SectionIn

log = logging.getLogger(__name__)

# Source authority (0-100). Higher = more authoritative. Used for ranking boosts and for
# telling the LLM which evidence to prefer.
AUTHORITY_GAZETTE_LAW = 100      # গেজেট-published legal instruments
AUTHORITY_OFFICIAL_DOC = 90      # other approved official ebooks (circulars, manuals, ...)
AUTHORITY_OFFICIAL_QNA = 70
AUTHORITY_OFFICIAL_BLOG = 50
AUTHORITY_BLOG = 40
AUTHORITY_FORUM_OFFICIAL = 30
AUTHORITY_FORUM = 10

AUTHORITY_LABELS = {
    AUTHORITY_GAZETTE_LAW: "সরকারি আইনি দলিল (গেজেট)",
    AUTHORITY_OFFICIAL_DOC: "সরকারি দলিল",
    AUTHORITY_OFFICIAL_QNA: "সরকারি প্রশ্নোত্তর",
    AUTHORITY_OFFICIAL_BLOG: "সরকারি ব্লগ",
    AUTHORITY_BLOG: "ব্লগ",
    AUTHORITY_FORUM_OFFICIAL: "ফোরাম (অফিসিয়াল)",
    AUTHORITY_FORUM: "ফোরাম আলোচনা (অযাচাইকৃত)",
}

# Ebook types whose provisions are called ধারা / বিধি / প্রবিধান.
SECTION_LABELS = {
    "আইন": "ধারা", "অধ্যাদেশ": "ধারা", "রাষ্ট্রপতির আদেশ": "অনুচ্ছেদ",
    "বিধিমালা": "বিধি", "প্রবিধান": "প্রবিধান",
}
OFFICIAL_AUTHORS = ("ভূমি মন্ত্রণালয়",)


@dataclass
class Unit:
    source_type: str
    source_id: str
    parent_source_id: str | None
    text: str
    number: str | None = None  # display form, e.g. "৫" or "(১)"
    heading: str | None = None
    children: list[Unit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    source_type: str
    source_id: str
    title: str
    parent_source_id: str | None = None
    url: str | None = None
    file_url: str | None = None
    doc_type: str | None = None
    category: str | None = None
    year: int | None = None
    act_number: str | None = None
    publication_date: date | None = None
    author: str | None = None
    authority: int = 0
    source_updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    units: list[Unit] = field(default_factory=list)
    section_label: str = "ধারা"


class RecordValidationError(ValueError):
    def __init__(self, source_type: str, record_id: Any, detail: str):
        super().__init__(f"{source_type}[{record_id}]: {detail}")
        self.source_type = source_type
        self.record_id = record_id
        self.detail = detail


def record_id(source_type: str, raw: dict[str, Any]) -> str | None:
    rid = raw.get("id")
    return None if rid is None else str(rid)


def validate(source_type: str, raw: dict[str, Any]):
    try:
        return SCHEMAS[source_type].model_validate(raw)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors()[:5])
        raise RecordValidationError(source_type, raw.get("id"), errs) from None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_date(value: str | None) -> date | None:
    """Accepts 'YYYY-MM-DD' and 'DD-MM-YYYY' in ASCII or Bengali digits."""
    if not value:
        return None
    from app.core.text import bn_to_ascii_digits

    v = bn_to_ascii_digits(value).strip()
    for pat, order in ((r"^(\d{4})-(\d{1,2})-(\d{1,2})", "ymd"), (r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", "dmy")):
        m = re.match(pat, v)
        if m:
            a, b, c = map(int, m.groups())
            y, mth, d = (a, b, c) if order == "ymd" else (c, b, a)
            try:
                return date(y, mth, d)
            except ValueError:
                return None
    return None


def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = normalize_whitespace(clean_unicode(s))
    return s or None


def _urls(values: list[str]) -> list[str]:
    return [v.strip() for v in values if isinstance(v, str) and re.match(r"^https?://", v.strip())]


def _template(tpl: str, **kw) -> str | None:
    return tpl.format(**kw) if tpl else None


# --------------------------------------------------------------------------------------
# Ebooks
# --------------------------------------------------------------------------------------

def _schedule_unit(sc: ScheduleIn, parent_id: str) -> Unit | None:
    children = []
    for ss in sc.subschedules:
        text = html_to_text(ss.content)
        if text or ss.heading:
            children.append(Unit("subschedule", str(ss.id), str(sc.id), text,
                                 number=_clean(ss.number), heading=_clean(ss.heading),
                                 metadata={"note": _clean(ss.note)} if ss.note else {}))
    text = html_to_text(sc.content)
    if not (text or children or sc.heading):
        return None
    return Unit("schedule", str(sc.id), parent_id, text, number=_clean(sc.number), heading=_clean(sc.heading),
                children=children, metadata={"note": _clean(sc.note)} if sc.note else {})


def _section_unit(sec: SectionIn, act_id: str) -> Unit | None:
    children: list[Unit] = []
    for ss in sec.subsections:
        sub_children = [u for u in (_schedule_unit(sc, str(ss.id)) for sc in ss.schedules) if u]
        text = html_to_text(ss.content)
        if not (text or sub_children):
            continue  # upstream contains empty placeholder subsections numbered "0"
        number = _clean(ss.number)
        if normalize_section_number(number) is None:
            number = None
        children.append(Unit("subsection", str(ss.id), str(sec.id), text, number=number,
                             heading=_clean(ss.heading), children=sub_children,
                             metadata={"note": _clean(ss.note)} if getattr(ss, "note", None) else {}))
    for sc in sec.schedules:
        u = _schedule_unit(sc, str(sec.id))
        if u:
            children.append(u)
    text = html_to_text(sec.content)
    if not (text or children):
        return None
    meta = {"note": _clean(sec.note)} if sec.note else {}
    return Unit("section", str(sec.id), act_id, text, number=_clean(sec.number), heading=_clean(sec.heading),
                children=children, metadata=meta)


def normalize_ebook(e: EbookIn, settings: Settings) -> NormalizedDocument:
    act_id = str(e.id)
    doc_type = _clean(e.ebooks_type)
    gazette = (e.publication_by or "").strip() == "গেজেট"
    units = [u for u in (_section_unit(s, act_id) for s in e.sections) if u]
    free_text = html_to_text(e.schedules)
    if free_text:
        # For structured acts this field holds the appendix/schedule text; for circulars,
        # manuals, guidelines etc. (no sections) it is the entire document body.
        units.append(Unit("ebook_text", act_id, None, free_text,
                          metadata={"role": "appendix" if e.sections else "body"}))
    title = _clean(e.title_of_act)
    title_derived = False
    if not title:
        # Some approved circulars have an empty title upstream. Derive one from real fields
        # (type + memo number, else the first line of the body) and flag it.
        title_derived = True
        first_line = next((ln for ln in free_text.split("\n") if ln.strip()), "")
        if doc_type and _clean(e.number):
            title = f"{doc_type} (নং {_clean(e.number)})"
        else:
            title = (first_line[:120] + ("…" if len(first_line) > 120 else "")) or f"{doc_type or 'দলিল'} #{act_id}"
    signature = ", ".join(x for x in (_clean(e.signature_by), _clean(e.signature_position)) if x)
    metadata = {
        "title_derived": title_derived or None,
        "issued_date": _clean(e.created_at_en) or _clean(e.created_at_bn),
        "applicable_date": _clean(e.applicable_date_en) or _clean(e.applicable_date_bn),
        "branch": ", ".join(x for x in (_clean(e.branch), _clean(e.sub_branch)) if x) or None,
        "signature": signature or None,
        "ebooks_type": doc_type,
        "act_year_raw": e.act_year or None,
        "publication_date_raw": e.publication_date or None,
        "publication_by": _clean(e.publication_by),
        "proposal": html_to_text(e.proposal) or None,
        "objective": html_to_text(e.objective) or None,
        "motto": html_to_text(e.motto) or None,
        "heading": html_to_text(e.heading) or None,
        "meta_keywords": [k for k in (_clean(x) for x in e.meta_keywords) if k and k != "[]"],
        "reference_links": _urls(e.multiple_reference_link),
        "files": {k: v for k, v in (("file", e.file), ("merged_file", e.merged_file),
                                    ("system_generated_pdf", e.system_generated_pdf)) if v},
        "section_count": len(e.sections),
    }
    return NormalizedDocument(
        source_type="ebook",
        source_id=act_id,
        title=title,
        url=_template(settings.ebook_url_template, id=e.id),
        file_url=e.file or e.merged_file or e.system_generated_pdf or None,
        doc_type=doc_type,
        year=parse_year(e.act_year),
        act_number=_clean(e.number),
        publication_date=parse_date(e.publication_date),
        authority=AUTHORITY_GAZETTE_LAW if gazette else AUTHORITY_OFFICIAL_DOC,
        source_updated_at=parse_datetime(e.created_date),
        metadata={k: v for k, v in metadata.items() if v not in (None, [], {})},
        units=units,
        section_label=SECTION_LABELS.get(doc_type or "", "অনুচ্ছেদ"),
    )


# --------------------------------------------------------------------------------------
# Q&A, blogs, forums
# --------------------------------------------------------------------------------------

def normalize_category(value: str | None) -> str | None:
    """Upstream categories have stray whitespace and casing variants ('dag ', 'Khotian')."""
    v = (value or "").strip().lower()
    return v or None


def normalize_qna(q: QnaIn, source_type: str) -> NormalizedDocument:
    question = _clean(q.question) or ""
    answer = html_to_text(q.answer)
    category = normalize_category(q.category)
    keyword = _clean(q.keyword)
    text = f"প্রশ্ন: {question}\nউত্তর: {answer}"
    return NormalizedDocument(
        source_type=source_type,
        source_id=str(q.id),
        title=question,
        category=category,
        authority=AUTHORITY_OFFICIAL_QNA,
        metadata={k: v for k, v in {"keyword": keyword, "category_raw": q.category or None}.items() if v},
        units=[Unit(source_type, str(q.id), None, text,
                    metadata={"question": question, "keyword": keyword})],
    )


def normalize_blog(b: BlogIn, settings: Settings) -> NormalizedDocument:
    author = _clean(b.author)
    official = author in OFFICIAL_AUTHORS
    created = parse_datetime(b.created_date)
    return NormalizedDocument(
        source_type="blog",
        source_id=str(b.id),
        title=_clean(b.title_name) or str(b.id),
        url=_template(settings.blog_url_template, id=b.id),
        author=author,
        authority=AUTHORITY_OFFICIAL_BLOG if official else AUTHORITY_BLOG,
        source_updated_at=created,
        publication_date=created.date() if created else None,
        metadata={k: v for k, v in {"cover": b.cover, "featured": b.featured}.items() if v is not None},
        units=[Unit("blog", str(b.id), None, html_to_text(b.content))],
    )


def normalize_forum(f: ForumIn, settings: Settings) -> list[NormalizedDocument]:
    official = (f.badge or "").lower() in ("official", "verified")
    authority = AUTHORITY_FORUM_OFFICIAL if official else AUTHORITY_FORUM
    docs: list[NormalizedDocument] = []
    group_meta = {"forum_name": _clean(f.name), "badge": f.badge, "group_type": f.group_type}
    desc = html_to_text(f.description)
    if desc:
        docs.append(NormalizedDocument(
            source_type="forum", source_id=str(f.id), title=_clean(f.name) or str(f.id), authority=authority,
            source_updated_at=parse_datetime(f.created_date),
            metadata={k: v for k, v in group_meta.items() if v},
            units=[Unit("forum", str(f.id), None, desc)],
        ))
    for t in f.topics:
        text = html_to_text(t.description)
        title = _clean(t.title) or str(t.id)
        if not (text or title):
            continue
        docs.append(NormalizedDocument(
            source_type="topic", source_id=str(t.id), parent_source_id=str(f.id), title=title,
            url=_template(settings.forum_topic_url_template, id=t.id, group=f.id),
            authority=authority, source_updated_at=parse_datetime(t.created_date),
            metadata={k: v for k, v in {**group_meta, "status": t.status, "is_archived": t.is_archived}.items()
                      if v not in (None, "")},
            units=[Unit("topic", str(t.id), str(f.id), text or title)],
        ))
    return docs


def _excluded(title: str | None, settings: Settings) -> bool:
    return bool(settings.exclude_title_regex and title and re.search(settings.exclude_title_regex, title))


def normalize(source_type: str, raw: dict[str, Any], settings: Settings | None = None) -> list[NormalizedDocument]:
    """Validate + normalise one raw API record. Raises RecordValidationError.

    Returns [] for records removed by the EXCLUDE_TITLE_REGEX data-quality filter."""
    settings = settings or get_settings()
    rec = validate(source_type, raw)
    if source_type == "ebook":
        docs = [normalize_ebook(rec, settings)]
    elif source_type in ("qna_type1", "qna_type2"):
        docs = [normalize_qna(rec, source_type)]
    elif source_type == "blog":
        docs = [normalize_blog(rec, settings)]
    elif source_type == "forum":
        if _excluded(rec.name, settings):
            return []
        docs = normalize_forum(rec, settings)
    else:
        raise ValueError(f"unknown source type {source_type}")
    return [d for d in docs if not _excluded(d.title, settings)]
