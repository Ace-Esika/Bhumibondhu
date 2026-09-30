"""Follow-up question handling for multi-turn conversations.

A follow-up such as "এর ফি কত?" cannot be retrieved on its own. When history exists and
the question looks context-dependent, it is rewritten into a standalone question by the
LLM (CONDENSE_PROMPT); retrieval, routing and caching all use the standalone form. If the
rewrite fails, a deterministic fallback prefixes the previous user question, which still
gives retrieval the missing context. Self-contained questions skip the extra LLM call.
"""

from __future__ import annotations

import logging
import re

from app.core.text import query_terms
from app.llm.base import LLMError, LLMProvider
from app.llm.prompts import CONDENSE_PROMPT
from app.rag.memory import Turn

log = logging.getLogger(__name__)

# References to something said earlier (Bengali + English).
_ANAPHORA_RE = re.compile(
    r"(^|\s)(এটা|এটি|এটার|এটির|এর|এতে|এগুলো|এগুলোর|সেটা|সেটি|সেটার|সেটির|সেগুলো|তার|তাদের|উক্ত|ঐ|ওই|ওটা|"
    r"(এই|ওই|ঐ|সেই)\s*(আইন|ধারা|বিধি|বিধিমালা|পরিপত্র|দলিল|বিষয়|ফি|প্রক্রিয়া|জমি|আবেদন|মামলা)\S*|সেই|আগের|পূর্বের|উপরের|তাহলে|আর\s|আরও|"
    r"এছাড়া|এক্ষেত্রে|সেক্ষেত্রে|এখানে|it|its|this|that|those|these|them|same|above|previous)(?=\s|[?।,.!]|$)",
    re.IGNORECASE,
)
MAX_TURN_CHARS = 600


def looks_dependent(question: str) -> bool:
    """Heuristic: does the question need the conversation to be understood?"""
    return bool(_ANAPHORA_RE.search(question)) or len(query_terms(question)) <= 1


def render_turns(turns: list[Turn], max_chars: int) -> str:
    """Newest-last transcript within a character budget (older turns dropped first)."""
    lines: list[str] = []
    total = 0
    for t in reversed(turns):
        who = "ব্যবহারকারী" if t.role == "user" else "সহায়ক"
        text = re.sub(r"\s+", " ", t.content).strip()
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "…"
        line = f"{who}: {text}"
        if lines and total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(reversed(lines))


def fallback_standalone(question: str, turns: list[Turn]) -> str:
    prev = next((t.content for t in reversed(turns) if t.role == "user"), "")
    return f"{prev} — {question}" if prev else question


async def condense(llm: LLMProvider, turns: list[Turn], question: str, max_chars: int,
                   model: str | None = None) -> tuple[str, str]:
    """Return (standalone question, method) where method is llm|fallback."""
    messages = CONDENSE_PROMPT.format_messages(turns=render_turns(turns, max_chars), question=question)
    try:
        res = await llm.generate(messages, max_tokens=600, model=model or None)
        out = res.text.strip().strip('"“”').strip()
        out = out.splitlines()[0].strip() if out else ""
        # Reject degenerate rewrites (empty, an answer instead of a question, runaway text).
        if out and 3 <= len(out) <= max(400, 3 * len(question)) and "INSUFFICIENT" not in out:
            return out, "llm"
    except LLMError as e:
        log.warning("follow-up condensation failed; using fallback", extra={"error_type": type(e).__name__})
    return fallback_standalone(question, turns), "fallback"
