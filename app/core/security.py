"""API-key authentication and rate limiting.

Rate limiting is a fixed-window counter per client and scope. It uses Redis when configured
(correct across multiple API replicas) and falls back to an in-process counter otherwise.
"""

from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.core.cache import get_redis
from app.core.config import Settings, get_settings

log = logging.getLogger(__name__)

admin_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)
public_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _matches(candidate: str | None, secret: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), secret.encode())


async def require_admin(
    key: str | None = Depends(admin_key_header), settings: Settings = Depends(get_settings)
) -> None:
    if not settings.admin_api_key or not settings.admin_api_key.get_secret_value():
        # Admin endpoints are disabled entirely when no key is configured.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin API is not configured")
    if not _matches(key, settings.admin_api_key.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin credentials")


async def require_public_key(
    key: str | None = Depends(public_key_header), settings: Settings = Depends(get_settings)
) -> None:
    if not settings.public_api_keys:
        return
    if not any(_matches(key, k.get_secret_value()) for k in settings.public_api_keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


def client_ip(request: Request, settings: Settings) -> str:
    if settings.trust_forwarded_for:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self) -> None:
        self._local: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    async def hit(self, key: str, limit: int, window: int = 60) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, seconds_until_reset)."""
        now = int(time.time())
        bucket = now // window
        reset = window - (now % window)
        redis = await get_redis()
        if redis is not None:
            rkey = f"rl:{key}:{bucket}"
            try:
                pipe = redis.pipeline()
                pipe.incr(rkey)
                pipe.expire(rkey, window + 1)
                count, _ = await pipe.execute()
                return int(count) <= limit, reset
            except Exception:  # Redis outage must not take the API down.
                log.warning("rate limiter redis failure; using local counter")
        b, count = self._local[key]
        count = count + 1 if b == bucket else 1
        self._local[key] = (bucket, count)
        if len(self._local) > 50_000:  # bound memory
            self._local.clear()
        return count <= limit, reset


_limiter = RateLimiter()


def rate_limit(scope: str, per_minute_attr: str = "rate_limit_per_minute"):
    async def dependency(request: Request, settings: Settings = Depends(get_settings)) -> None:
        if not settings.rate_limit_enabled:
            return
        limit = getattr(settings, per_minute_attr)
        allowed, reset = await _limiter.hit(f"{scope}:{client_ip(request, settings)}", limit)
        if not allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(reset)},
            )

    return dependency
