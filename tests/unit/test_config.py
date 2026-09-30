import pytest

from app.core.config import Settings


def test_env_example_parses_cleanly(monkeypatch):
    for k in list(__import__("os").environ):
        if k.isupper() and k not in ("PATH", "HOME"):
            monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=".env.example")
    assert s.groq_model == "" and s.hf_home in (None, "")  # no inline comment leaked into a value
    assert s.sync_source_types == ["ebook", "blog", "forum", "qna_type1", "qna_type2"]
    assert s.exclude_title_regex == r"(?i)^\s*test\b"


def test_csv_lists_from_environment(monkeypatch):
    monkeypatch.setenv("SYNC_SOURCE_TYPES", "ebook, qna_type2")
    monkeypatch.setenv("PUBLIC_API_KEYS", "k1,k2")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example")
    s = Settings(_env_file=None)
    assert s.sync_source_types == ["ebook", "qna_type2"] and len(s.public_api_keys) == 2
    assert s.public_api_keys[0].get_secret_value() == "k1" and "k1" not in repr(s)


def test_invalid_config_rejected(monkeypatch):
    monkeypatch.setenv("SYNC_SOURCE_TYPES", "ebook,pinecone")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    monkeypatch.delenv("SYNC_SOURCE_TYPES")
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_env="production", admin_api_key=None)
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_overlap_tokens=600, chunk_target_tokens=500)


def test_plain_postgres_url_gets_psycopg_driver():
    assert Settings(_env_file=None, database_url="postgresql://u:p@h/db").database_url.startswith(
        "postgresql+psycopg://")
