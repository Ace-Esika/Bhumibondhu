"""Query normalisation, analysis and lightweight routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.core.config import SOURCE_TYPES
from app.core.text import (bn_to_ascii_digits, clean_unicode, normalize_query, normalize_section_number,
                           query_terms)

# Logical source-type filter values accepted from clients (document-level types).
FILTERABLE_SOURCE_TYPES = (*SOURCE_TYPES, "topic")

_PROVISION = r"(?:ধারা|বিধি|অনুচ্ছেদ|প্রবিধান|section|rule|article|regulation)"
# "উপ-ধারা 2" must never be read as "ধারা 2": the lookbehind skips the sub- prefix.
_SECTION_RE = re.compile(
    rf"(?<!উপ)(?<!উপ-)(?<!sub-)(?<!sub){_PROVISION}\s*[-–:]?\s*(\d+[ক-হA-Za-z]?)(?:\s*\(\s*([০-৯0-9]+|[ক-হ])\s*\))?"
    r"|(\d+[ক-হ]?)\s*(?:নং|নম্বর|নাম্বার)?\s*(?:ধারা|বিধি|অনুচ্ছেদ)",
    re.IGNORECASE,
)
_SUB = r"(?:উপ-?\s*(?:ধারা|বিধি|অনুচ্ছেদ|প্রবিধান)|sub-?\s*(?:section|rule|regulation))"
_SUBSECTION_RES = (
    re.compile(rf"{_SUB}\s*[-–:]?\s*\(?\s*(\d+[ক-হ]?|[ক-হ]|[a-z])\s*\)?(?![ক-হ\w])", re.IGNORECASE),
    re.compile(rf"\(?\s*(\d+[ক-হ]?|[ক-হ])\s*\)?\s*(?:নং|নম্বর)?\s*{_SUB}", re.IGNORECASE),
)
_CLAUSE_RE = re.compile(r"(?:দফা|clause)\s*[-–:]?\s*\(?\s*([ক-হ]|[a-z]|\d+)\s*\)?"
                        r"|\(\s*([ক-হ])\s*\)\s*দফা", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)")

_SALAM = (r"(আস+ালামু?\s*আলাইকুম|আসসালামুআলাইকুম|assalamu?\s*(o\s*)?alaikum|assalamualaikum|salam|সালাম"
          r"|ওয়া\s*আলাইকুম\s*আস+ালাম|ওয়ালাইকুম\s*আস+ালাম)")
_THANKS = r"(ধন্যবাদ|শুক্রিয়া|thanks|thank you)"
_HELLO = (r"(হ্যালো|হাই|হেলো|নমস্কার|শুভ\s*(সকাল|সন্ধ্যা|দুপুর|অপরাহ্ন)|সুপ্রভাত|hi|hello|hey"
          r"|good\s*(morning|afternoon|evening))")
_GREETING_RE = re.compile(rf"^\s*({_SALAM}|{_THANKS}|{_HELLO})[\s!.,।?]*$", re.IGNORECASE)
_SALAM_RE = re.compile(rf"^\s*{_SALAM}", re.IGNORECASE)
_THANKS_RE = re.compile(rf"^\s*{_THANKS}", re.IGNORECASE)
_COUNT_RE = re.compile(r"(কতটি|কয়টি|কতগুলো|কতগুলি|কত সংখ্যক|মোট কত|সংখ্যা কত|how many|number of|count)",
                       re.IGNORECASE)
_LIST_TYPES = {
    # User-facing words → ebooks_type values that exist in the source data.
    "আইন": "আইন", "act": "আইন", "acts": "আইন",
    "অধ্যাদেশ": "অধ্যাদেশ", "ordinance": "অধ্যাদেশ",
    "বিধিমালা": "বিধিমালা", "rules": "বিধিমালা",
    "প্রবিধান": "প্রবিধান", "regulation": "প্রবিধান",
    "নীতিমালা": "নীতিমালা", "policy": "নীতিমালা",
    "নির্দেশিকা": "নির্দেশিকা", "guideline": "নির্দেশিকা",
    "পরিপত্র": "পরিপত্র", "circular": "পরিপত্র",
    "প্রজ্ঞাপন": "প্রজ্ঞাপন", "notification": "প্রজ্ঞাপন",
    "ম্যানুয়াল": "ম্যানুয়াল", "manual": "ম্যানুয়াল",
    "রাষ্ট্রপতির আদেশ": "রাষ্ট্রপতির আদেশ",
}


# Definition-seeking questions. Bengali statutes define terms as `"X" অর্থ ...` inside a
# section headed "সংজ্ঞা", so such queries also search for that heading (A-weighted).
_DEFINITION_RE = re.compile(
    r"(বলতে\s*(কী|কি)|কাকে\s*বলে|মানে\s*(কী|কি)|অর্থ\s*(কী|কি)|সংজ্ঞা|কী\s*জিনিস|কি\s*জিনিস|"
    r"what\s+is|meaning\s+of|definition|define)", re.IGNORECASE)
DEFINITION_TERMS = ("সংজ্ঞা:*", "অর্থ")

# The user explicitly wants a detailed / complete answer.
_DETAIL_RE = re.compile(
    r"(বিস্তারিত|বিশদ|সম্পূর্ণ|সম্পুর্ণ|পুরো|পূর্ণাঙ্গ|পুরোপুরি|ধাপে\s*ধাপে|ধাপসমূহ|ব্যাখ্যা|খুঁটিনাটি|সবকিছু|"
    r"in\s+detail|detailed|details|full\s+text|step\s*by\s*step|explain|elaborate|complete)", re.IGNORECASE)
# ...about every provision of a document ("সব ধারা", "ধারাসমূহ", "পুরো আইন").
_ALL_SECTIONS_RE = re.compile(
    r"((সব|সকল|সমস্ত|প্রতিটি)\s*(ধারা|বিধি|অনুচ্ছেদ)|(ধারা|বিধি|অনুচ্ছেদ)\s*(সমূহ|গুলো|গুলি)|"
    r"(পুরো|সম্পূর্ণ|পূর্ণাঙ্গ)\s*(আইন|বিধিমালা|অধ্যাদেশ|নীতিমালা)|all\s+(sections|rules|provisions))", re.IGNORECASE)


# The question is about the act as a whole (its contents, preamble, enactment date ...), so the
# overview / table-of-contents chunk is a legitimate answer rather than noise.
_ACT_LEVEL_RE = re.compile(
    r"(সূচি|সূচীপত্র|প্রস্তাবনা|উদ্দেশ্য|কোন\s*(আইন|বিধিমালা|অধ্যাদেশ|নীতিমালা)|কবে\s*(প্রণীত|জারি|প্রকাশ|কার্যকর)|"
    r"কত\s*সালে|কত\s*নম্বর|কতটি\s*ধারা|table of contents|overview|which act|how many sections)", re.IGNORECASE)


class QueryRoute(StrEnum):
    RAG = "rag"  # knowledge / legal question → retrieval + LLM
    STRUCTURED = "structured"  # exact factual query answerable from the database
    SMALLTALK = "smalltalk"  # greeting/thanks: canned reply, no retrieval


class SearchFilters(BaseModel):
    source_types: list[str] | None = Field(default=None, description="document-level source types")
    doc_types: list[str] | None = Field(default=None, description="ebooks_type values, e.g. আইন, পরিপত্র")
    categories: list[str] | None = None
    year: int | None = Field(default=None, ge=1700, le=2200)
    document_ids: list[str] | None = Field(default=None, description="restrict to specific ebook/act ids")

    @field_validator("source_types")
    @classmethod
    def _check_types(cls, v):
        if v:
            bad = set(v) - set(FILTERABLE_SOURCE_TYPES)
            if bad:
                raise ValueError(f"unknown source_types {sorted(bad)}")
        return v

    @field_validator("doc_types", "categories", "document_ids")
    @classmethod
    def _bound(cls, v):
        if v is not None and (len(v) > 50 or any(len(x) > 200 for x in v)):
            raise ValueError("too many / too long filter values")
        return v

    def is_empty(self) -> bool:
        return not any((self.source_types, self.doc_types, self.categories, self.year, self.document_ids))


@dataclass
class AnalyzedQuery:
    raw: str
    text: str  # cleaned query given to the embedder
    normalized: str  # canonical form for cache keys
    terms: list[str]  # tsquery terms
    section_number: str | None = None
    subsection_number: str | None = None  # "উপ-ধারা (২)" → "2"
    clause: str | None = None  # "দফা (ক)" → "ক"
    year: int | None = None
    is_definition: bool = False
    wants_detail: bool = False
    wants_all_sections: bool = False
    wants_act_overview: bool = False
    route: QueryRoute = QueryRoute.RAG
    structured: dict = field(default_factory=dict)


def _parse_provision(ascii_text: str) -> tuple[str | None, str | None]:
    """(section, subsection) named by the query, both normalised like the stored values."""
    m = _SECTION_RE.search(ascii_text)
    section = normalize_section_number(m.group(1) or m.group(3)) if m else None
    subsection = normalize_section_number(m.group(2)) if m and m.group(2) else None
    if subsection is None:
        for rx in _SUBSECTION_RES:
            sm = rx.search(ascii_text)
            if sm:
                subsection = normalize_section_number(sm.group(1))
                break
    return section, subsection


def _parse_clause(ascii_text: str) -> str | None:
    m = _CLAUSE_RE.search(ascii_text)
    return normalize_section_number(m.group(1) or m.group(2)) if m else None


def analyze_query(raw: str) -> AnalyzedQuery:
    text = re.sub(r"\s+", " ", clean_unicode(raw)).strip()
    ascii_digits = bn_to_ascii_digits(text)
    section, subsection = _parse_provision(ascii_digits)
    ym = _YEAR_RE.search(ascii_digits)
    terms = query_terms(text)
    is_definition = bool(_DEFINITION_RE.search(text))
    if is_definition:
        terms += [t for t in DEFINITION_TERMS if t not in terms]
    q = AnalyzedQuery(raw=raw, text=text, normalized=normalize_query(text), terms=terms,
                      section_number=section, subsection_number=subsection, clause=_parse_clause(ascii_digits),
                      year=int(ym.group(1)) if ym else None, is_definition=is_definition,
                      wants_all_sections=bool(_ALL_SECTIONS_RE.search(text)))
    q.wants_detail = bool(_DETAIL_RE.search(text)) or q.wants_all_sections
    q.wants_act_overview = q.wants_all_sections or bool(_ACT_LEVEL_RE.search(text))
    if _GREETING_RE.match(text):
        q.route = QueryRoute.SMALLTALK
        q.structured = {"greeting": "salam" if _SALAM_RE.match(text) else "thanks" if _THANKS_RE.match(text) else "hello"}
    elif _COUNT_RE.search(text):
        lowered = text.lower()
        # Longest match first so "রাষ্ট্রপতির আদেশ" wins over shorter words.
        for word in sorted(_LIST_TYPES, key=len, reverse=True):
            if word in lowered:
                q.route = QueryRoute.STRUCTURED
                q.structured = {"kind": "count_ebooks", "doc_type": _LIST_TYPES[word], "year": q.year}
                break
    return q
