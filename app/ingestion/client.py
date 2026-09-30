"""HTTP client for the Bhumipedia source APIs.

- Retries timeouts, connection errors, 429 and 5xx with exponential backoff (+ jitter),
  honouring Retry-After.
- Accepts both the plain-array response (default) and the paginated
  `{count, next, previous, results}` shape, following `next` links.
- Fails loudly on malformed top-level payloads: a sync must never interpret a broken
  response as "every record was deleted".
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.core.config import Settings, get_settings
from app.ingestion.schemas import ENDPOINTS

log = logging.getLogger(__name__)

MAX_PAGES = 10_000


class SourceAPIError(RuntimeError):
    """The upstream API could not deliver a trustworthy, complete payload."""


class BhumipediaClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=self.settings.source_api_base_url,
            timeout=httpx.Timeout(self.settings.source_api_timeout_seconds, connect=20.0),
            headers={"Accept": "application/json", "User-Agent": "bhumipedia-rag-sync/1.0"},
            verify=self.settings.source_api_verify_tls,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> BhumipediaClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, url: str) -> Any:
        retries = self.settings.source_api_max_retries
        for attempt in range(retries + 1):
            try:
                resp = await self._client.get(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"retryable status {resp.status_code}", request=resp.request,
                                                response=resp)
                resp.raise_for_status()
                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    raise SourceAPIError(f"malformed JSON from {url}") from e
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
                status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt >= retries:
                    raise SourceAPIError(f"GET {url} failed after {attempt + 1} attempt(s): "
                                         f"{type(e).__name__} status={status}") from e
                delay = min(60.0, 2 ** attempt + random.uniform(0, 1))
                if isinstance(e, httpx.HTTPStatusError):
                    ra = e.response.headers.get("retry-after")
                    if ra and ra.isdigit():
                        delay = min(120.0, float(ra))
                log.warning("source api retry", extra={"url": url, "attempt": attempt + 1, "status": status,
                                                       "delay_s": round(delay, 1)})
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def fetch(self, source_type: str) -> list[dict[str, Any]]:
        """Fetch every record for a source type."""
        path = ENDPOINTS[source_type]
        payload = await self._get_json(path)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            records = list(payload["results"])
            next_url, pages = payload.get("next"), 1
            base_host = urlparse(self.settings.source_api_base_url).netloc
            while next_url:
                pages += 1
                if pages > MAX_PAGES:
                    raise SourceAPIError(f"pagination for {source_type} exceeded {MAX_PAGES} pages")
                # Only follow pagination links on the configured host (no SSRF via `next`).
                parsed = urlparse(urljoin(self.settings.source_api_base_url, next_url))
                if parsed.netloc != base_host:
                    raise SourceAPIError(f"refusing to follow pagination link to foreign host {parsed.netloc}")
                page = await self._get_json(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
                if not isinstance(page, dict) or not isinstance(page.get("results"), list):
                    raise SourceAPIError(f"malformed page {pages} for {source_type}")
                records.extend(page["results"])
                next_url = page.get("next")
            if isinstance(payload.get("count"), int) and payload["count"] != len(records):
                raise SourceAPIError(f"{source_type}: expected {payload['count']} records, got {len(records)}")
        else:
            raise SourceAPIError(f"unexpected payload type for {source_type}: {type(payload).__name__}")
        non_dicts = sum(1 for r in records if not isinstance(r, dict))
        if non_dicts:
            raise SourceAPIError(f"{source_type}: {non_dicts} non-object records in payload")
        log.info("fetched source", extra={"source_type": source_type, "records": len(records)})
        return records


class FileSource:
    """Offline fallback: read `<source_type>.json` snapshots (same shape as the API) from a
    directory. Useful for air-gapped environments, reproducible tests and replaying an
    audited snapshot. Create one with `python -m app.cli snapshot <dir>`."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    async def __aenter__(self) -> FileSource:
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def fetch(self, source_type: str) -> list[dict[str, Any]]:
        path = self.directory / f"{source_type}.json"
        if not path.exists():
            raise SourceAPIError(f"snapshot file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        if not isinstance(data, list):
            raise SourceAPIError(f"snapshot {path} is not a JSON array")
        return data
