"""Groq provider built on LangChain's `ChatGroq`.

The SDK's own retries are disabled (max_retries=0) so the retry policy is explicit here:
exponential backoff with jitter on 429 / 5xx / timeouts / connection errors, honouring
Retry-After (capped), and an overall per-attempt timeout. Non-retryable errors (auth, bad
request) fail immediately. The API key is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.core.config import Settings, get_settings
from app.llm.base import (
    LLMBadRequest,
    LLMError,
    LLMNotConfigured,
    LLMProvider,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
)

log = logging.getLogger(__name__)

MAX_RETRY_AFTER_S = 20.0


def _retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(val) if val is not None else None
    except ValueError:
        return None


def classify(exc: Exception) -> tuple[type[LLMError], bool]:
    """Map a provider exception to (LLMError subclass, retryable)."""
    import groq
    import httpx

    if isinstance(exc, asyncio.TimeoutError | groq.APITimeoutError | httpx.TimeoutException):
        return LLMTimeout, True
    if isinstance(exc, groq.RateLimitError):
        # "Limit 8000, Requested 10400": this single request can never fit the per-minute
        # window, so waiting is useless — the caller must shrink it.
        m = re.search(r"Limit (\d+), Requested (\d+)", str(exc))
        if m and int(m.group(2)) > int(m.group(1)) and "per minute" in str(exc):
            return LLMRequestTooLarge, False
        return LLMRateLimited, True
    if isinstance(exc, groq.APIConnectionError | httpx.TransportError):
        return LLMUnavailable, True
    if isinstance(exc, groq.APIStatusError):
        status = exc.status_code
        if status == 429:
            return LLMRateLimited, True
        if status >= 500:
            return LLMUnavailable, True
        if status in (401, 403):
            return LLMNotConfigured, False
        if status == 413:
            return LLMRequestTooLarge, False
        return LLMBadRequest, False
    return LLMUnavailable, False


class GroqProvider(LLMProvider):
    """Groq chat models with explicit retries and an optional fallback model.

    Groq rate limits (RPM/TPM/TPD) are per model, so when the primary model is rate limited
    and GROQ_FALLBACK_MODEL is set, the request is answered by the fallback instead.
    A 429 whose Retry-After exceeds MAX_RETRY_AFTER_S (e.g. a daily token quota) fails fast
    rather than holding the request open.
    """

    name = "groq"

    def __init__(self, settings: Settings | None = None, chat_model: BaseChatModel | None = None,
                 fallback_chat_model: BaseChatModel | None = None):
        self.settings = settings or get_settings()
        self._chats: dict[str, BaseChatModel] = {}
        if chat_model is not None:
            self._chats[self.settings.groq_model] = chat_model
        if fallback_chat_model is not None and self.settings.groq_fallback_model:
            self._chats[self.settings.groq_fallback_model] = fallback_chat_model

    @property
    def model(self) -> str:
        return self.settings.groq_model

    def _chat_for(self, model: str) -> BaseChatModel:
        if model not in self._chats:
            if not self.settings.groq_configured:
                raise LLMNotConfigured("GROQ_API_KEY and GROQ_MODEL must be set")
            from langchain_groq import ChatGroq

            self._chats[model] = ChatGroq(
                model=model,
                api_key=self.settings.groq_api_key.get_secret_value(),
                temperature=self.settings.groq_temperature,
                max_tokens=self.settings.groq_max_tokens,
                timeout=self.settings.groq_timeout_seconds,
                max_retries=0,
            )
        return self._chats[model]

    @property
    def chat(self) -> BaseChatModel:
        return self._chat_for(self.model)

    async def generate(self, messages: list[BaseMessage], max_tokens: int | None = None,
                       model: str | None = None) -> LLMResult:
        primary = model or self.model
        fallback = self.settings.groq_fallback_model
        try:
            return await self._generate_with(primary, messages, max_tokens)
        except (LLMRateLimited, LLMUnavailable, LLMTimeout) as e:
            if not fallback or fallback == primary:
                raise
            log.warning("primary llm unavailable; using fallback model",
                        extra={"primary": primary, "fallback": fallback, "error_type": type(e).__name__})
            return await self._generate_with(fallback, messages, max_tokens)

    async def _generate_with(self, model: str, messages: list[BaseMessage], max_tokens: int | None) -> LLMResult:
        s = self.settings
        chat = self._chat_for(model)
        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        t0 = time.perf_counter()
        last: Exception | None = None
        for attempt in range(1, s.groq_max_retries + 2):
            try:
                msg = await asyncio.wait_for(chat.ainvoke(messages, **kwargs), timeout=s.groq_timeout_seconds + 5)
                usage = dict(getattr(msg, "usage_metadata", None) or {})
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                finish = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
                return LLMResult(text=text, model=model, latency_ms=(time.perf_counter() - t0) * 1000,
                                 attempts=attempt, finish_reason=finish,
                                 usage={k: int(v) for k, v in usage.items() if isinstance(v, int | float)})
            except Exception as e:  # noqa: BLE001 - classified below
                err_cls, retryable = classify(e)
                last = e
                ra = _retry_after(e)
                if ra is not None and ra > MAX_RETRY_AFTER_S:
                    retryable = False  # e.g. daily token quota: waiting minutes helps nobody
                log.warning("llm call failed", extra={"provider": self.name, "model": model, "attempt": attempt,
                                                      "error_type": type(e).__name__, "retryable": retryable,
                                                      "status": getattr(e, "status_code", None), "retry_after_s": ra})
                if not retryable or attempt > s.groq_max_retries:
                    raise err_cls(f"{type(e).__name__}") from e
                delay = min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.3)
                if ra is not None:
                    delay = max(delay, ra)
                await asyncio.sleep(delay)
        raise LLMUnavailable(type(last).__name__ if last else "unknown")  # pragma: no cover


@lru_cache
def get_llm_provider() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "groq":
        return GroqProvider(s)
    raise ValueError(f"unsupported LLM_PROVIDER {s.llm_provider}")
