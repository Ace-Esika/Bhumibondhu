"""Structured JSON logging with request-id propagation and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

# (pattern, replacement) pairs for values that must never reach a log sink.
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "gsk_***"),  # Groq keys
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_***"),  # Hugging Face tokens
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]+"), r"\1***"),
    (re.compile(r"(?i)((?:x-admin-api-key|x-api-key|api[_-]?key|password)\"?\s*[:=]\s*\"?)[^\s\",}]+"),
     r"\1***"),
    (re.compile(r"((?:postgres(?:ql)?(?:\+\w+)?|redis)://[^:/@\s]*:)[^@\s]+@"), r"\1***@"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False, default=str))


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        rid = request_id_var.get()
        base = f"{self.formatTime(record)} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if rid:
            base += f" [rid={rid}]"
        if extras:
            base += " " + json.dumps(extras, ensure_ascii=False, default=str)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return redact(base)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stderr)  # stdout stays clean for CLI output
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Quiet noisy libraries; httpx at INFO logs full URLs of every request.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
