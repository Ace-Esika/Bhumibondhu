"""Token counting with the embedding model's own tokenizer (XLM-R SentencePiece for BGE-M3).

Chunk sizes are enforced in *model tokens*, which matters for Bengali: a Bengali word is
often 2-4 tokens, so character-based limits are misleading. Only the small tokenizer files
are needed (no model weights). If the tokenizer can't be loaded (offline, no cache) we fall
back to a conservative character-based estimate and log a warning once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import lru_cache

log = logging.getLogger(__name__)

# Measured on this corpus with the BGE-M3 tokenizer: ~2.6 chars/token for Bengali legal text.
# The fallback deliberately over-estimates tokens (smaller chunks) to stay under limits.
FALLBACK_CHARS_PER_TOKEN = 2.3

TokenCounter = Callable[[str], int]


def approx_token_count(text: str) -> int:
    return max(1, int(len(text) / FALLBACK_CHARS_PER_TOKEN)) if text else 0


@lru_cache(maxsize=4)
def get_token_counter(model_name: str) -> TokenCounter:
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        tok.model_max_length = 10**9  # we only count; silence "sequence too long" warnings

        def count(text: str) -> int:
            if not text:
                return 0
            return len(tok(text, add_special_tokens=False)["input_ids"])

        return count
    except Exception as e:  # pragma: no cover - depends on environment
        log.warning("tokenizer unavailable; using approximate token counts",
                    extra={"model": model_name, "error": type(e).__name__})
        return approx_token_count
