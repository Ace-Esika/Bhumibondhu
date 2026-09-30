"""ASGI middleware: request id, access log, body-size limit, request timeout, security headers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

log = logging.getLogger("app.access")

_RID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout_seconds: float):
        super().__init__(app)
        self.timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if 0 < len(incoming) <= 64 and set(incoming) <= _RID_OK else uuid.uuid4().hex
        token = request_id_var.set(rid)
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            try:
                response = await asyncio.wait_for(call_next(request), timeout=self.timeout)
            except TimeoutError:
                log.error("request timed out", extra={"path": request.url.path})
                response = JSONResponse({"detail": "Request timed out", "request_id": rid}, status_code=504)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            if request.url.path not in ("/health", "/ready"):
                log.info("request", extra={"method": request.method, "path": request.url.path, "status": status,
                                           "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Reject request bodies above `max_bytes` without buffering them (pure ASGI)."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)


class _BodyTooLarge(Exception):
    pass
