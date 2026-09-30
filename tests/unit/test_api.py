import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.llm.base import LLMRateLimited
from app.main import create_app
from app.rag.pipeline import ChatResult
from tests.conftest import make_settings


class FakePipeline:
    def __init__(self, exc=None):
        from app.rag.memory import InMemoryConversationStore

        self.exc = exc
        self.store = InMemoryConversationStore()
        self.searcher = type("S", (), {"embedder": type("E", (), {"loaded": True})()})()

    async def answer(self, message, conversation_id=None, filters=None):
        if self.exc:
            raise self.exc
        return ChatResult(answer="উত্তর [1]", sources=[{"index": 1, "title": "ভূমি আইন", "source_type": "ebook",
                                                         "section": "ধারা ৫", "url": "https://x/a.pdf"}],
                          route="rag", grounded=True, query=message, results_count=1,
                          conversation_id=conversation_id, timings_ms={"total": 1.0})


def _client(pipeline=None, **settings):
    app = create_app(pipeline=pipeline or FakePipeline(), warmup=False)
    s = make_settings(**settings)
    app.dependency_overrides[get_settings] = lambda: s
    return TestClient(app, raise_server_exceptions=False)


def test_health():
    with _client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert r.headers["X-Request-ID"] and r.headers["X-Content-Type-Options"] == "nosniff"


def test_chat_response_shape():
    with _client() as c:
        r = c.post("/api/chat", json={"message": "নামজারি করতে কী কী লাগে?", "conversation_id": "abc-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "উত্তর [1]" and body["conversation_id"] == "abc-1"
        assert body["sources"][0]["title"] == "ভূমি আইন"
        assert body["retrieval"]["results_count"] == 1 and body["request_id"]


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "x" * 5000}, {"message": "q", "evil": 1},
                                     {"message": "q", "conversation_id": "bad id with spaces"},
                                     {"message": "q", "filters": {"source_types": ["unknown"]}}])
def test_chat_validation(payload):
    with _client() as c:
        r = c.post("/api/chat", json=payload)
        assert r.status_code == 422
        assert "x" * 100 not in r.text  # user input is not echoed back


def test_body_size_limit():
    with _client() as c:
        r = c.post("/api/chat", content=b"{" + b" " * 70_000 + b"}", headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_llm_errors_are_controlled():
    with _client(FakePipeline(exc=LLMRateLimited("RateLimitError"))) as c:
        r = c.post("/api/chat", json={"message": "নামজারি"})
        assert r.status_code == 503 and "RateLimitError" not in r.text and r.json()["request_id"]


def test_unhandled_errors_hide_details():
    with _client(FakePipeline(exc=RuntimeError("secret internal detail"))) as c:
        r = c.post("/api/chat", json={"message": "নামজারি"})
        assert r.status_code == 500 and "secret internal detail" not in r.text and "Traceback" not in r.text


def test_rate_limit_429():
    with _client(rate_limit_enabled=True, rate_limit_per_minute=2) as c:
        codes = [c.post("/api/chat", json={"message": "নামজারি"}).status_code for _ in range(4)]
        assert codes[:2] == [200, 200] and codes[-1] == 429


def test_public_api_key_enforced_when_configured():
    with _client(public_api_keys="k1,k2") as c:
        assert c.post("/api/chat", json={"message": "নামজারি"}).status_code == 401
        assert c.post("/api/chat", json={"message": "নামজারি"}, headers={"X-API-Key": "k2"}).status_code == 200


def test_admin_requires_key():
    with _client() as c:
        assert c.post("/api/admin/sync", json={}).status_code == 401
        assert c.get("/api/admin/ingestion-status", headers={"X-Admin-API-Key": "wrong"}).status_code == 401


def test_admin_disabled_without_configured_key():
    with _client(admin_api_key=None) as c:
        assert c.post("/api/admin/sync", json={}, headers={"X-Admin-API-Key": ""}).status_code == 503


def test_admin_sync_validates_source_types():
    with _client() as c:
        r = c.post("/api/admin/sync", json={"source_types": ["nope"]}, headers={"X-Admin-API-Key": "test-admin-key"})
        assert r.status_code == 422


def test_conversation_endpoints():
    import asyncio

    from app.rag.memory import Turn

    fake = FakePipeline()
    asyncio.run(fake.store.append("conv-1", [Turn("user", "প্রশ্ন", {"standalone": "পূর্ণ প্রশ্ন"}),
                                              Turn("assistant", "উত্তর", {"sources": ["qna_type2:1"]})]))
    with _client(fake) as c:
        r = c.get("/api/conversations/conv-1")
        assert r.status_code == 200
        turns = r.json()["turns"]
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert turns[0]["standalone_query"] == "পূর্ণ প্রশ্ন" and turns[1]["sources"] == ["qna_type2:1"]
        assert c.get("/api/conversations/missing").status_code == 404
        assert c.get("/api/conversations/bad id!").status_code in (404, 422)
        assert c.delete("/api/conversations/conv-1").status_code == 204
        assert c.get("/api/conversations/conv-1").status_code == 404
