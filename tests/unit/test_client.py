import httpx
import pytest
import respx

from app.ingestion.client import BhumipediaClient, SourceAPIError
from tests.conftest import make_settings

BASE = "https://bhumipedia.test"


def _client(**kw):
    return BhumipediaClient(make_settings(source_api_base_url=BASE, source_api_max_retries=2, **kw))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def fast(_):
        return None
    monkeypatch.setattr("app.ingestion.client.asyncio.sleep", fast)


@respx.mock
async def test_plain_array():
    respx.get(f"{BASE}/api/v1/qna/type2/").respond(json=[{"id": 1, "question": "q", "answer": "a"}])
    async with _client() as c:
        assert await c.fetch("qna_type2") == [{"id": 1, "question": "q", "answer": "a"}]


@respx.mock
async def test_paginated_follows_next_on_same_host():
    respx.get(f"{BASE}/api/v1/qna/type1/", params={"page": "2"}).respond(
        json={"count": 2, "next": None, "results": [{"id": 2}]})
    respx.get(f"{BASE}/api/v1/qna/type1/").respond(
        json={"count": 2, "next": f"{BASE}/api/v1/qna/type1/?page=2", "results": [{"id": 1}]})
    async with _client() as c:
        assert [r["id"] for r in await c.fetch("qna_type1")] == [1, 2]


@respx.mock
async def test_refuses_foreign_pagination_host():
    respx.get(f"{BASE}/api/v1/qna/type1/").respond(
        json={"count": 2, "next": "https://evil.example/steal?page=2", "results": [{"id": 1}]})
    async with _client() as c:
        with pytest.raises(SourceAPIError, match="foreign host"):
            await c.fetch("qna_type1")


@respx.mock
async def test_count_mismatch_is_an_error():
    respx.get(f"{BASE}/api/blogs/full/").respond(json={"count": 5, "next": None, "results": [{"id": 1}]})
    async with _client() as c:
        with pytest.raises(SourceAPIError, match="expected 5"):
            await c.fetch("blog")


@respx.mock
async def test_retries_5xx_and_429_then_succeeds():
    route = respx.get(f"{BASE}/api/blogs/full/")
    route.side_effect = [httpx.Response(503), httpx.Response(429, headers={"Retry-After": "1"}),
                         httpx.Response(200, json=[])]
    async with _client() as c:
        assert await c.fetch("blog") == []
    assert route.call_count == 3


@respx.mock
async def test_gives_up_after_retries_and_on_timeout():
    respx.get(f"{BASE}/api/blogs/full/").mock(side_effect=httpx.ReadTimeout("slow"))
    async with _client() as c:
        with pytest.raises(SourceAPIError, match="3 attempt"):
            await c.fetch("blog")


@respx.mock
async def test_4xx_not_retried():
    route = respx.get(f"{BASE}/api/blogs/full/").respond(404)
    async with _client() as c:
        with pytest.raises(SourceAPIError):
            await c.fetch("blog")
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("body", [b"<html>oops</html>", b'{"detail": "x"}', b"[1, 2]"])
async def test_malformed_payloads(body):
    respx.get(f"{BASE}/api/blogs/full/").respond(content=body, headers={"content-type": "application/json"})
    async with _client() as c:
        with pytest.raises(SourceAPIError):
            await c.fetch("blog")
