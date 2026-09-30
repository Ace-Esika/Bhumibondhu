"""Optional Redis cache.

Response-cache keys embed the *corpus version*: a counter bumped by every sync that
changes, adds or removes documents. Invalidation is therefore implicit and exact: stale
entries simply become unreachable and expire through their TTL. Nothing is cached forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import get_settings

log = logging.getLogger(__name__)

_redis = None
_redis_failed = False
CORPUS_VERSION_KEY = "bhumipedia:corpus_version"


async def get_redis():
    """Return a shared redis client, or None if Redis is not configured/unavailable."""
    global _redis, _redis_failed
    settings = get_settings()
    if not settings.redis_url or _redis_failed:
        return None
    if _redis is None:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2,
                                       decode_responses=True)
            await client.ping()
            _redis = client
        except Exception:
            log.warning("redis unavailable; caching disabled for this process")
            _redis_failed = True
            return None
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def make_key(namespace: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return f"bhumipedia:{namespace}:{hashlib.sha256(raw.encode()).hexdigest()[:40]}"


async def cache_get(key: str) -> Any | None:
    redis = await get_redis()
    if redis is None:
        return None
    try:
        val = await redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        log.warning("cache get failed", extra={"cache_key": key})
        return None


async def cache_set(key: str, value: Any, ttl: int) -> None:
    redis = await get_redis()
    if redis is None or ttl <= 0:
        return
    try:
        await redis.set(key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl)
    except Exception:
        log.warning("cache set failed", extra={"cache_key": key})


async def get_corpus_version() -> str:
    redis = await get_redis()
    if redis is None:
        return "0"
    try:
        return str(await redis.get(CORPUS_VERSION_KEY) or "0")
    except Exception:
        return "0"


async def bump_corpus_version() -> None:
    redis = await get_redis()
    if redis is None:
        return
    try:
        await redis.incr(CORPUS_VERSION_KEY)
    except Exception:
        log.warning("failed to bump corpus version; cached responses may be stale until TTL")
