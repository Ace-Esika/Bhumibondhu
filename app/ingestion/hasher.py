"""Content hashing for change detection.

Engagement counters (views, likes, replies...) change constantly without any change in
content; they are excluded so a view-count bump never triggers re-chunking/re-embedding.

`PIPELINE_VERSION` is folded into the source hash: bump it whenever normalisation or
chunking logic changes so the next sync reprocesses every record (embeddings of chunks
whose text is unchanged are still reused).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PIPELINE_VERSION = "2026-09-30.3"

VOLATILE_FIELDS = frozenset(
    {
        "like_user_counter", "share_user_counter", "viewer_counter", "comment_counter",
        "view_count", "reply_count", "like_count", "member_count", "topic_count",
    }
)


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_FIELDS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_hash(raw: dict[str, Any], chunking_signature: str = "") -> str:
    """Hash of a raw API record's *content* plus the processing configuration."""
    return sha256(f"{PIPELINE_VERSION}|{chunking_signature}|{canonical_json(_strip_volatile(raw))}")


def text_hash(*parts: str) -> str:
    return sha256("\x1f".join(parts))
