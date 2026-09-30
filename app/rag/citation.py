"""Deterministic citation construction from retrieved-record metadata.

The LLM only emits bracketed indices ([1], [2]); titles, section labels and URLs come
exclusively from database rows. An index the LLM invents (out of range) is removed.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.core.text import ascii_to_bn_digits
from app.ingestion.normalizer import AUTHORITY_LABELS, SECTION_LABELS

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_BN_CITE_RE = re.compile(r"\[([০-৯]{1,2})\]")
# Some models (e.g. gpt-oss) cite with fullwidth/lenticular brackets: 【1】 ［1］ 〔1〕.
# gpt-oss may also append line ranges: 【2†L1-L5】.
_ALT_CITE_RE = re.compile(r"[【［〔]\s*([0-9০-৯]{1,2})\s*(?:†[^】］〕\n]{0,40})?[】］〕]")
_REPEATED_CITE_RE = re.compile(r"(\[\d{1,2}\])(?:\s*\1)+")


def authority_label(authority: int) -> str:
    best = max((a for a in AUTHORITY_LABELS if a <= authority), default=min(AUTHORITY_LABELS))
    return AUTHORITY_LABELS[best]


def section_label(meta: dict[str, Any]) -> str | None:
    """Human-readable location inside a document, e.g. 'ধারা ৫ — সংজ্ঞা, উপ-ধারা (২)'."""
    if meta.get("doc_source_type") != "ebook":
        return None
    label = SECTION_LABELS.get(meta.get("doc_type") or "", "অনুচ্ছেদ")
    parts = []
    if meta.get("section_number"):
        head = f"{label} {ascii_to_bn_digits(meta['section_number'])}"
        if meta.get("section_heading"):
            head += f" — {meta['section_heading'].rstrip('।').strip()}"
        parts.append(head)
    cm = meta.get("chunk_metadata") or {}
    if cm.get("subsection_number"):
        parts.append(f"উপ-{label} {cm['subsection_number']}")
    if cm.get("schedule_number"):
        parts.append(f"তফসিল {cm['schedule_number']}")
    elif cm.get("clause_number"):
        parts.append(f"দফা {cm['clause_number']}")
    elif cm.get("role") == "appendix":
        parts.append("তফসিল/সংযুক্তি")
    if not parts and cm.get("role") in ("overview", "toc"):
        parts.append("পরিচিতি ও সূচি")
    elif not parts and cm.get("role") == "preamble":
        parts.append("প্রস্তাবনা")
    return ", ".join(parts) or None


def build_source(index: int, doc: Document, cited: bool = True) -> dict[str, Any]:
    m = doc.metadata
    snippet = (m.get("content") or doc.page_content).strip().replace("\n", " ")
    return {
        "index": index,
        "title": m.get("title"),
        "source_type": m.get("doc_source_type"),
        "element_type": m.get("source_type"),
        "source_id": m.get("doc_source_id"),
        "element_id": m.get("source_id"),
        "doc_type": m.get("doc_type"),
        "section": section_label(m),
        "year": m.get("year"),
        "authority": authority_label(int(m.get("authority") or 0)),
        "url": m.get("url") or m.get("file_url"),
        "snippet": snippet[:240] + ("…" if len(snippet) > 240 else ""),
        "cited": cited,
    }


def normalise_citation_digits(answer: str) -> str:
    """Models sometimes write [১] or 【1】; convert to ASCII [n] so parsing is uniform."""
    from app.core.text import bn_to_ascii_digits

    answer = _ALT_CITE_RE.sub(lambda m: f"[{bn_to_ascii_digits(m.group(1))}]", answer)
    return _BN_CITE_RE.sub(lambda m: f"[{bn_to_ascii_digits(m.group(1))}]", answer)


def extract_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Return (answer with invalid indices removed, ordered list of valid cited indices)."""
    answer = normalise_citation_digits(answer)
    cited: list[int] = []

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        if 1 <= i <= n_sources:
            if i not in cited:
                cited.append(i)
            return m.group(0)
        return ""

    cleaned = _REPEATED_CITE_RE.sub(r"\1", _CITE_RE.sub(repl, answer))
    return cleaned, cited


def format_references(sources: list[dict[str, Any]]) -> str:
    """Plain-text reference block ('সূত্র:') for clients that render text only."""
    lines = ["সূত্র:"]
    for s in sources:
        label = s["title"] or ""
        if s.get("section"):
            label += f", {s['section']}"
        lines.append(f"[{ascii_to_bn_digits(str(s['index']))}] {label} ({s['authority']})")
        if s.get("url"):
            lines.append(f"    {s['url']}")
    return "\n".join(lines)
