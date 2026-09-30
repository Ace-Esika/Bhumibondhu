import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.base import LLMNotConfigured, LLMRateLimited, LLMUnavailable
from app.llm.groq import GroqProvider
from tests.conftest import make_settings


def _status_error(cls, status, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return cls("err", response=resp, body=None)


class ScriptedChat:
    """Minimal async chat model stub returning/raising items in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def fast(_):
        return None
    monkeypatch.setattr("app.llm.groq.asyncio.sleep", fast)


def _provider(script, **kw):
    s = make_settings(groq_api_key="gsk_test", groq_model="some-model", groq_max_retries=2, **kw)
    return GroqProvider(s, chat_model=ScriptedChat(script))


async def test_success_with_usage():
    msg = AIMessage(content="উত্তর [1]", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})
    p = _provider([msg])
    r = await p.generate([HumanMessage("q")])
    assert r.text == "উত্তর [1]" and r.usage["total_tokens"] == 13 and r.attempts == 1


async def test_retries_429_then_succeeds():
    p = _provider([_status_error(groq.RateLimitError, 429, {"retry-after": "2"}), AIMessage(content="ok")])
    r = await p.generate([HumanMessage("q")])
    assert r.text == "ok" and r.attempts == 2


async def test_persistent_429_raises_controlled_error():
    p = _provider([_status_error(groq.RateLimitError, 429)] * 3)
    with pytest.raises(LLMRateLimited):
        await p.generate([HumanMessage("q")])
    assert p._chats["some-model"].calls == 3


async def test_5xx_retried_then_unavailable():
    p = _provider([_status_error(groq.InternalServerError, 500)] * 3)
    with pytest.raises(LLMUnavailable):
        await p.generate([HumanMessage("q")])


async def test_auth_error_not_retried():
    p = _provider([_status_error(groq.AuthenticationError, 401)])
    with pytest.raises(LLMNotConfigured):
        await p.generate([HumanMessage("q")])
    assert p._chats["some-model"].calls == 1


def test_not_configured():
    with pytest.raises(LLMNotConfigured):
        _ = GroqProvider(make_settings()).chat


def test_error_message_never_contains_key():
    e = LLMRateLimited("RateLimitError")
    assert "gsk_" not in str(e) and "gsk_" not in e.user_message


async def test_long_retry_after_fails_fast_and_uses_fallback():
    quota = _status_error(groq.RateLimitError, 429, {"retry-after": "2744"})  # daily quota
    s = make_settings(groq_api_key="gsk_test", groq_model="primary", groq_fallback_model="backup", groq_max_retries=3)
    primary, backup = ScriptedChat([quota]), ScriptedChat([AIMessage(content="উত্তর [1]")])
    p = GroqProvider(s, chat_model=primary, fallback_chat_model=backup)
    r = await p.generate([HumanMessage("q")])
    assert primary.calls == 1  # no pointless retries against a long quota window
    assert r.text == "উত্তর [1]" and r.model == "backup"


async def test_no_fallback_configured_raises_controlled_error():
    p = _provider([_status_error(groq.RateLimitError, 429, {"retry-after": "3000"})])
    with pytest.raises(LLMRateLimited):
        await p.generate([HumanMessage("q")])
    assert p._chats["some-model"].calls == 1


async def test_non_availability_errors_do_not_trigger_fallback():
    s = make_settings(groq_api_key="gsk_test", groq_model="primary", groq_fallback_model="backup")
    backup = ScriptedChat([AIMessage(content="x")])
    p = GroqProvider(s, chat_model=ScriptedChat([_status_error(groq.AuthenticationError, 401)]),
                     fallback_chat_model=backup)
    with pytest.raises(LLMNotConfigured):
        await p.generate([HumanMessage("q")])
    assert backup.calls == 0


def test_per_minute_request_too_large_is_not_a_retryable_rate_limit():
    from app.llm.base import LLMRequestTooLarge
    from app.llm.groq import classify

    e = _status_error(groq.RateLimitError, 429)
    e.message = ""
    err = groq.RateLimitError("Rate limit reached ... on tokens per minute (TPM): Limit 8000, Requested 10400",
                              response=e.response, body=None)
    assert classify(err) == (LLMRequestTooLarge, False)
    ok = groq.RateLimitError("tokens per minute (TPM): Limit 8000, Used 7900, Requested 500",
                             response=e.response, body=None)
    assert classify(ok) == (LLMRateLimited, True)
