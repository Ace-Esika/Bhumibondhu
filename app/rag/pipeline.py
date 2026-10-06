"""RAG orchestration.

    message → validation → analysis/routing
        smalltalk  → canned reply
        structured → SQL answer (no generation)
        rag        → cache? → hybrid retrieval (→ rerank) → evidence gate
                   → context builder (sanitised, authority-labelled) → Groq
                   → citation validation → backend-built sources → cache
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.core.cache import cache_get, cache_set, get_corpus_version, make_key
from app.core.config import Settings, get_settings
from app.db.database import get_sessionmaker
from app.llm.base import LLMProvider, LLMRequestTooLarge, LLMResult
from app.llm.groq import get_llm_provider
from app.llm.prompts import (
    ANSWER_PROMPT,
    CONCISE_INSTRUCTION,
    DETAIL_INSTRUCTION,
    HISTORY_BLOCK,
    INSUFFICIENT_SENTINEL,
    REFUSAL_BN,
    REFUSAL_EN,
    SMALLTALK_BN,
    SMALLTALK_REPLIES_BN,
    TRUNCATION_NOTE_BN,
)
from app.rag.citation import authority_label, build_source, extract_citations, section_label
from app.rag.condense import condense, looks_dependent, render_turns
from app.rag.context import Coverage, coverage_note, expand_document, expand_sections
from app.rag.guardrails import (
    URL_RE,
    clean_user_message,
    evidence_sufficient,
    looks_like_injection,
    sanitize_source_text,
    strip_unknown_urls,
)
from app.rag.memory import ConversationStore, SQLConversationStore, Turn, new_conversation_id
from app.rag.structured import answer_structured
from app.retrieval.hybrid import HybridSearcher, RetrievalTrace
from app.retrieval.query import AnalyzedQuery, QueryRoute, SearchFilters, analyze_query

log = logging.getLogger(__name__)


@dataclass
class ChatResult:
    answer: str
    sources: list[dict[str, Any]]
    route: str
    grounded: bool
    query: str
    results_count: int
    conversation_id: str | None = None
    cached: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    reason: str | None = None
    standalone_query: str | None = None  # follow-up rewritten using the conversation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_bengali(text: str) -> bool:
    bn = sum(1 for ch in text if "ঀ" <= ch <= "৿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return bn >= latin


def build_context(docs: list[Document], max_chars: int) -> tuple[str, list[Document]]:
    """Render sources as delimited, authority-labelled blocks within a character budget."""
    blocks, used, total = [], [], 0
    for doc in docs:
        m = doc.metadata
        i = len(used) + 1
        header = [f"উৎসের ধরন: {authority_label(int(m.get('authority') or 0))} ({m.get('doc_source_type')})",
                  f"শিরোনাম: {m.get('title')}"]
        if sec := section_label(m):
            header.append(f"অবস্থান: {sec}")
        body = sanitize_source_text(doc.page_content)
        block = f'<source id="{i}">\n' + "\n".join(header) + f"\n---\n{body}\n</source>"
        if used and total + len(block) > max_chars:
            break
        if not used and len(block) > max_chars:
            # A single source larger than the whole budget: keep its head, say so explicitly.
            keep = max(200, len(body) - (len(block) - max_chars))
            block = (f'<source id="{i}">\n' + "\n".join(header) +
                     f"\n---\n{body[:keep]}\n…[সংক্ষেপিত: উৎসের বাকি অংশ এখানে দেখানো হয়নি]\n</source>")
        blocks.append(block)
        used.append(doc)
        total += len(block)
    return "\n\n".join(blocks), used


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, searcher: HybridSearcher | None = None,
                 llm: LLMProvider | None = None, sessionmaker=None, store: ConversationStore | None = None):
        self.settings = settings or get_settings()
        self.searcher = searcher or HybridSearcher(self.settings)
        self.llm = llm or get_llm_provider()
        self.sessionmaker = sessionmaker or get_sessionmaker()
        self.store = store or SQLConversationStore(self.sessionmaker)

    # ------------------------------------------------------------------ conversation layer
    async def answer(self, message: str, conversation_id: str | None = None,
                     filters: SearchFilters | None = None) -> ChatResult:
        """Answer one user message within a conversation (history is loaded and saved)."""
        s = self.settings
        t0 = time.perf_counter()
        text = clean_user_message(message, s.max_message_chars)
        if not s.conversation_enabled:
            return await self._answer(text, conversation_id, filters, [], t0)

        conv_id = conversation_id or new_conversation_id()
        history: list[Turn] = []
        if conversation_id:
            try:
                history = await self.store.recent(conv_id, s.conversation_history_turns)
            except Exception:
                log.exception("could not load conversation history; answering without it")

        question, method = text, None
        if history and s.condense_followups and looks_dependent(text):
            question, method = await condense(self.llm, history, text, s.conversation_history_chars,
                                              model=s.groq_condense_model or None)
            log.info("follow-up rewritten", extra={"method": method, "changed": question != text})

        result = await self._answer(question, conv_id, filters, history, t0)
        result.conversation_id = conv_id
        if question != text:
            result.standalone_query = question
        try:
            await self.store.append(conv_id, [
                Turn("user", text, {"standalone": question} if question != text else {}),
                Turn("assistant", result.answer, {
                    "route": result.route, "grounded": result.grounded, "reason": result.reason,
                    "sources": [f"{x.get('source_type')}:{x.get('source_id')}" for x in result.sources][:10]}),
            ])
        except Exception:
            log.exception("could not save conversation turn")
        return result

    async def _generate(self, q: AnalyzedQuery, docs: list[Document], budget: int, max_tokens: int,
                        coverage: Coverage | None, history: list[Turn] | None = None
                        ) -> tuple[LLMResult, list[Document]]:
        messages, used, max_tokens = self._fit_request(q, docs, budget, max_tokens, coverage, history or [])
        return await self.llm.generate(messages, max_tokens=max_tokens), used

    def _fit_request(self, q: AnalyzedQuery, docs: list[Document], budget: int, max_tokens: int,
                     coverage: Coverage | None, history: list[Turn]):
        """Build the prompt so it fits the provider's per-request token limits (if configured):
        trim the context until the estimated input fits, then cap the output budget."""
        s = self.settings
        cpt = s.llm_chars_per_token
        instructions = DETAIL_INSTRUCTION if q.wants_detail else CONCISE_INSTRUCTION
        history_block = (HISTORY_BLOCK.format(turns=sanitize_source_text(
            render_turns(history, s.conversation_history_chars))) if history else "")
        for _ in range(6):
            context, used = build_context(docs, budget)
            instr = instructions
            if coverage is not None and (note := coverage_note(coverage, used)):
                instr += note
            messages = ANSWER_PROMPT.format_messages(sources=context, question=q.text, instructions=instr,
                                                     history=history_block)
            est_input = int(sum(len(m.content) for m in messages) / cpt)
            if not s.llm_max_input_tokens or est_input <= s.llm_max_input_tokens or budget <= 1500:
                break
            over = est_input - s.llm_max_input_tokens
            budget = max(1500, int(len(context) - over * cpt * 1.1))
        if s.llm_max_total_tokens:
            max_tokens = max(512, min(max_tokens, s.llm_max_total_tokens - est_input))
        return messages, used, max_tokens

    def _refusal(self, q: AnalyzedQuery, reason: str, trace: RetrievalTrace | None, n: int,
                 conversation_id: str | None) -> ChatResult:
        return ChatResult(answer=REFUSAL_BN if is_bengali(q.text) else REFUSAL_EN, sources=[], route=q.route.value,
                          grounded=False, query=q.text, results_count=n, conversation_id=conversation_id,
                          timings_ms=_timings(trace), reason=reason)

    # ------------------------------------------------------------------ single question
    async def _answer(self, text: str, conversation_id: str | None, filters: SearchFilters | None,
                      history: list[Turn], t0: float) -> ChatResult:
        s = self.settings
        q = analyze_query(text)
        if looks_like_injection(text):
            log.info("possible prompt-injection attempt in user message")

        if q.route == QueryRoute.SMALLTALK:
            return ChatResult(answer=SMALLTALK_REPLIES_BN.get(q.structured.get("greeting"), SMALLTALK_BN), sources=[], route=q.route.value, grounded=True, query=q.text,
                              results_count=0, conversation_id=conversation_id)

        if q.route == QueryRoute.STRUCTURED and (filters is None or filters.is_empty()):
            async with self.sessionmaker() as session:
                res = await answer_structured(session, q.structured)
            if res is not None:
                return ChatResult(answer=res["answer"], sources=res["sources"], route=q.route.value, grounded=True,
                                  query=q.text, results_count=res["count"], conversation_id=conversation_id,
                                  timings_ms={"total": _ms(t0)})
            q.route = QueryRoute.RAG

        # ---- response cache (keyed by corpus version → invalidated by any source change)
        cache_key = None
        if s.cache_enabled and not history:
            cache_key = make_key("chat", await get_corpus_version(), q.normalized,
                                 filters.model_dump(exclude_none=True) if filters else {},
                                 s.embedding_model, s.reranker_enabled, s.final_context_k, s.groq_model,
                                 s.fusion_method, s.vector_weight, s.lexical_weight, q.wants_detail,
                                 q.wants_all_sections, s.groq_max_tokens, s.groq_max_tokens_detailed)
            if (hit := await cache_get(cache_key)) is not None:
                hit.update(cached=True, conversation_id=conversation_id, timings_ms={"total": _ms(t0)})
                return ChatResult(**hit)

        # ---- retrieval (detailed questions get more evidence)
        detail = q.wants_detail
        docs, trace = await self.searcher.search(q, filters, final_k=s.detail_context_k if detail else None)
        ok, why = evidence_sufficient(docs, s, trace.reranked)
        log.info("retrieval", extra={
            "query": q.text if s.debug_log_queries else None,
            "query_hash": hashlib.sha256(q.normalized.encode()).hexdigest()[:16],
            "retrieval": {**_timings(trace), "vector_hits": trace.vector_hits, "lexical_hits": trace.lexical_hits,
                          "reranked": trace.reranked, "rerank_failed": trace.rerank_failed,
                          "evidence": why, "chunk_ids": [d.metadata["chunk_id"] for d in docs],
                          "source_types": [d.metadata["doc_source_type"] for d in docs]}})
        if not ok:
            result = self._refusal(q, f"insufficient_evidence:{why}", trace, len(docs), conversation_id)
            result.timings_ms["total"] = _ms(t0)
            return result

        # ---- context: complete provisions / whole document for detailed questions
        coverage = None
        # Only legal documents have provisions to complete; skip the DB round-trip otherwise.
        has_ebook = any(d.metadata.get("doc_source_type") == "ebook" for d in docs)
        if has_ebook and (q.wants_all_sections or detail or q.section_number):
            async with self.sessionmaker() as session:
                if q.wants_all_sections:
                    docs, coverage = await expand_document(session, docs, q)
                else:
                    docs = await expand_sections(session, docs, q, s.expand_sections_max)
        budget = s.detail_max_context_chars if detail else s.max_context_chars
        max_tokens = s.groq_max_tokens_detailed if detail else s.groq_max_tokens

        # ---- generation (shrinks the context once if the provider rejects the request size)
        try:
            llm_res, used = await self._generate(q, docs, budget, max_tokens, coverage, history)
        except LLMRequestTooLarge:
            log.warning("llm request too large; retrying with half the context")
            llm_res, used = await self._generate(q, docs, budget // 2, max_tokens, coverage, history)
        if llm_res.truncated and max_tokens < s.groq_max_tokens_detailed:
            log.info("answer truncated; retrying with the detailed output budget")
            max_tokens = s.groq_max_tokens_detailed
            llm_res, used = await self._generate(q, docs, budget, max_tokens, coverage, history)
        raw_answer = llm_res.text.strip()

        timings = _timings(trace)
        timings["llm"] = round(llm_res.latency_ms, 1)
        if INSUFFICIENT_SENTINEL in raw_answer:
            result = self._refusal(q, "llm_reported_insufficient", trace, len(used), conversation_id)
            result.usage = llm_res.usage
            result.timings_ms = {**timings, "total": _ms(t0)}
            if cache_key:
                await cache_set(cache_key, _cacheable(result), s.cache_ttl_seconds)
            return result

        answer, cited = extract_citations(raw_answer, len(used))
        # URLs are allowed only if they come from the sources: record links or text quoted in them.
        allowed_urls = {u for d in used for u in (d.metadata.get("url"), d.metadata.get("file_url")) if u}
        allowed_urls |= {u.rstrip(".,।") for d in used for u in URL_RE.findall(d.page_content)}
        answer = strip_unknown_urls(answer, allowed_urls).strip()
        if cited:
            sources = [build_source(i, used[i - 1]) for i in cited]
        else:
            # The model answered without citing: attach what it was shown, flagged uncited.
            log.warning("llm answer without citations")
            sources = [build_source(i + 1, d, cited=False) for i, d in enumerate(used)]

        reason = None if cited else "uncited_answer"
        if llm_res.truncated:
            answer += TRUNCATION_NOTE_BN
            reason = reason or "truncated"
        result = ChatResult(answer=answer, sources=sources, route=q.route.value, grounded=bool(cited), query=q.text,
                            results_count=len(used), conversation_id=conversation_id, usage=llm_res.usage,
                            timings_ms={**timings, "total": _ms(t0)}, reason=reason)
        log.info("chat answered", extra={"llm": {"model": llm_res.model, "attempts": llm_res.attempts,
                                                 "usage": llm_res.usage}, "cited": cited})
        if cache_key:
            await cache_set(cache_key, _cacheable(result), s.cache_ttl_seconds)
        return result


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _timings(trace: RetrievalTrace | None) -> dict[str, float]:
    if trace is None:
        return {}
    return {"embedding": round(trace.embedding_ms, 1), "vector": round(trace.vector_ms, 1),
            "lexical": round(trace.lexical_ms, 1), "rerank": round(trace.rerank_ms, 1),
            "retrieval_total": round(trace.total_ms, 1)}


def _cacheable(result: ChatResult) -> dict[str, Any]:
    d = result.to_dict()
    d.pop("conversation_id", None)
    d.pop("cached", None)
    return d
