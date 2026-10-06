"""Input validation, prompt-injection hardening, evidence sufficiency and output checks."""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document

from app.core.config import Settings

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAGLIKE = re.compile(r"</?\s*(sources?|system|instructions?|user|assistant)\b[^>]*>", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
# Only used for logging/monitoring: the model is instructed to ignore such text anyway.
_INJECTION_HINTS = re.compile(
    r"(ignore (all|any|the)? ?(previous|above|prior) (instructions|rules)|disregard (the )?(system|previous)|"
    r"you are now|system prompt|reveal (your|the) (prompt|instructions)|পূর্ববর্তী নির্দেশ(না)? উপেক্ষা)",
    re.IGNORECASE,
)


class InvalidInput(ValueError):
    pass


def clean_user_message(message: str, max_chars: int) -> str:
    if message is None:
        raise InvalidInput("message is required")
    msg = unicodedata.normalize("NFC", _CONTROL.sub(" ", message)).strip()
    msg = re.sub(r"\s+", " ", msg)
    if not msg:
        raise InvalidInput("message is empty")
    if len(msg) > max_chars:
        raise InvalidInput(f"message exceeds {max_chars} characters")
    if not re.search(r"[\wঀ-৿]", msg):
        raise InvalidInput("message contains no words")
    return msg


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_HINTS.search(text))


def sanitize_source_text(text: str) -> str:
    """Neutralise delimiter look-alikes so source text can't close/open prompt sections."""
    return _TAGLIKE.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text)


def evidence_sufficient(docs: list[Document], settings: Settings, reranked: bool) -> tuple[bool, str]:
    """Decide whether retrieved evidence is strong enough to call the LLM at all.

    - With the reranker: the best cross-encoder score must clear MIN_RERANK_SCORE.
    - Without it: the best dense similarity must clear MIN_VECTOR_SIMILARITY, or (when
      vectors are unavailable) lexical matches must cover most of the query terms.
    The LLM is additionally instructed to refuse when the evidence does not answer.
    """
    if not docs:
        return False, "no_results"
    # The user named a provision of a named act and the database has exactly that provision:
    # nothing about the dense similarity can make it less relevant.
    if any(d.metadata.get("exact_match") for d in docs):
        return True, "exact_provision"
    if reranked:
        best = max((d.metadata.get("rerank_score") or 0.0) for d in docs)
        return (best >= settings.min_rerank_score, f"rerank={best:.3f}")
    best_vec = max((d.metadata.get("vector_score") or 0.0) for d in docs)
    best_cov = max((d.metadata.get("lexical_coverage") or 0.0) for d in docs)
    if best_vec >= settings.min_vector_similarity:
        return True, f"vector={best_vec:.3f}"
    if best_vec == 0.0 and best_cov >= 0.6:
        return True, f"lexical_coverage={best_cov:.2f}"
    return False, f"vector={best_vec:.3f},coverage={best_cov:.2f}"


def strip_unknown_urls(answer: str, allowed: set[str]) -> str:
    """Remove URLs the model wrote that are not among the retrieved sources' URLs."""
    return URL_RE.sub(lambda m: m.group(0) if m.group(0).rstrip(".,।") in allowed else "", answer)
