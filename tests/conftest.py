"""Shared fixtures. Unit tests never touch the network, Redis, models or a database."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Isolate from any developer .env before app modules read settings.
os.environ.update({
    "APP_ENV": "test", "REDIS_URL": "", "CACHE_ENABLED": "false", "LOG_JSON": "false",
    "GROQ_API_KEY": "", "GROQ_MODEL": "", "ADMIN_API_KEY": "test-admin-key", "RERANKER_ENABLED": "false",
    "SYNC_ENABLED": "false", "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
})

from app.core.config import Settings, get_settings  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "snapshot"


def make_settings(**overrides) -> Settings:
    base = dict(app_env="test", redis_url="", cache_enabled=False, rate_limit_enabled=False,
                admin_api_key="test-admin-key", groq_api_key="", groq_model="")
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def load_snapshot(source_type: str) -> list[dict]:
    return json.loads((SNAPSHOT / f"{source_type}.json").read_text(encoding="utf-8"))


@pytest.fixture
def word_tokens():
    """Deterministic token counter (whitespace words) so chunker tests don't need a tokenizer."""
    return lambda text: len(text.split()) if text else 0
