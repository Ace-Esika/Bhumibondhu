"""Central configuration. Every tunable comes from environment variables (or `.env`).

Secrets (`GROQ_API_KEY`, `ADMIN_API_KEY`, `HF_API_TOKEN`) are `SecretStr` so they never
appear in reprs, logs or error messages.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

SOURCE_TYPES = ("ebook", "blog", "forum", "qna_type1", "qna_type2")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- application ---------------------------------------------------------------
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    debug_log_queries: bool = False  # log raw user queries (off in prod by default)

    # --- database ------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://bhumipedia:bhumipedia@localhost:5433/bhumipedia"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_statement_timeout_ms: int = 30_000

    # --- redis / cache -------------------------------------------------------------
    redis_url: str | None = None  # empty → caching and rate limiting fall back to in-process
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    query_embedding_cache_ttl_seconds: int = 7 * 24 * 3600

    # --- source API ----------------------------------------------------------------
    source_api_base_url: str = "https://bhumipedia.land.gov.bd"
    source_api_timeout_seconds: float = 180.0
    source_api_max_retries: int = 4
    source_api_verify_tls: bool = True
    # Source types synchronised by default (comma separated in env).
    sync_source_types: Annotated[list[str], NoDecode] = Field(default_factory=lambda: list(SOURCE_TYPES))
    # Page URL templates. The API does not expose public page URLs, so these are empty by
    # default and citations fall back to the record's real `file` link when present.
    # Example: EBOOK_URL_TEMPLATE=https://bhumipedia.land.gov.bd/acts/{id}
    ebook_url_template: str = ""
    blog_url_template: str = ""
    forum_topic_url_template: str = ""

    # Data-quality filter: records whose title (or forum group name) matches are kept in
    # `sources` for audit but produce no searchable documents. The live API currently
    # contains approved placeholder records such as "test-6-1" / "Test Open Forom".
    # Set to an empty string to disable.
    exclude_title_regex: str = r"(?i)^\s*test\b"

    # --- prebuilt index (seed) ---
    # When the database is empty on worker start, import this file instead of indexing from
    # scratch (see app/ingestion/seed.py). Relative paths resolve from the working directory.
    index_seed_path: str = "seed/bhumipedia-index.tar.gz"
    index_seed_auto_import: bool = True

    # --- sync scheduler --------------------------------------------------------------
    sync_enabled: bool = True
    sync_interval_hours: float = 6.0
    sync_on_startup: bool = True
    worker_poll_seconds: float = 10.0
    # Safety valve: refuse to soft-delete more than this fraction of a source type in
    # one run (protects against a truncated/empty upstream response).
    sync_max_delete_fraction: float = 0.3

    # --- chunking --------------------------------------------------------------------
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 700
    chunk_overlap_tokens: int = 80
    chunk_min_tokens: int = 30

    # --- embeddings ----------------------------------------------------------------
    # local       → langchain_huggingface.HuggingFaceEmbeddings (self-hosted, default)
    # hf_endpoint → langchain_huggingface.HuggingFaceEndpointEmbeddings (HF Inference API)
    embedding_provider: Literal["local", "hf_endpoint"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda", "auto"] = "cpu"
    embedding_batch_size: int = 4
    embedding_dim: int = 1024
    embedding_max_seq_length: int = 1024
    hf_api_token: SecretStr | None = None
    # Model weights cache + offline mode. These are exported to os.environ at startup (see
    # `export_runtime_env`) because huggingface_hub reads the process environment, not `.env`.
    hf_home: str | None = None  # a Docker volume in compose
    hf_hub_offline: bool = False  # set true once weights are cached: no Hub network calls

    # --- reranker ------------------------------------------------------------------
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["cpu", "cuda", "auto"] = "cpu"
    reranker_top_k: int = 20  # candidates sent to the reranker
    reranker_batch_size: int = 8
    reranker_max_length: int = 512  # ~2x faster than 1024 on CPU; chunk heads carry the context
    reranker_fallback_on_error: bool = True

    # --- retrieval -----------------------------------------------------------------
    vector_top_k: int = 30
    lexical_top_k: int = 30
    final_context_k: int = 5
    fusion_method: Literal["rrf", "linear"] = "linear"  # chosen by benchmark (README §10)
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    rrf_k: int = 60
    # Multiplicative boost scaled by source authority (0-100). RRF scores are compressed (rank 1
    # vs rank 3 differ by ~3%), so keep this small: authority breaks ties between comparably
    # relevant results; it must not override relevance. The LLM additionally sees authority
    # labels and is instructed to prefer official law.
    authority_boost: float = 0.02
    section_match_boost: float = 0.25  # boost when a query's "ধারা N" matches the chunk
    hnsw_ef_search: int = 80
    # Evidence thresholds. Below these the system refuses instead of answering.
    min_vector_similarity: float = 0.50  # calibrated: README §10
    min_rerank_score: float = 0.05
    max_context_chars: int = 12_000
    # Detailed questions get more evidence: more chunks, whole provisions, larger budget.
    detail_context_k: int = 10
    detail_max_context_chars: int = 20_000
    expand_sections_max: int = 3  # provisions completed with their sibling chunks

    # --- LLM (Groq) ----------------------------------------------------------------
    llm_provider: Literal["groq"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = ""
    # Used when the primary model is rate limited / unavailable (Groq limits are per model).
    groq_fallback_model: str = ""
    # Provider per-request token limits (0 = unlimited). Groq's free tier allows ~7000 input
    # tokens per minute and ~8000 total (input + max_tokens) per model; set e.g. 6000 / 7800.
    # Requests are sized to fit *before* sending: context is trimmed, output budget capped.
    llm_max_input_tokens: int = 0
    llm_max_total_tokens: int = 0
    llm_chars_per_token: float = 2.0  # measured ≈2.06 for Bengali on gpt-oss/qwen; conservative
    groq_temperature: float = 0.1
    # Output budgets. gpt-oss models spend part of this on hidden reasoning, so keep headroom.
    groq_max_tokens: int = 2048
    groq_max_tokens_detailed: int = 4096  # when the user asks for details / all sections
    groq_timeout_seconds: float = 45.0
    groq_max_retries: int = 3

    # --- conversation history ---
    conversation_enabled: bool = True
    conversation_history_turns: int = 6  # messages (user+assistant) shown to the LLM
    conversation_history_chars: int = 2500  # budget for the history block in the prompt
    conversation_retention_days: float = 30
    condense_followups: bool = True  # rewrite dependent follow-ups into standalone questions
    groq_condense_model: str = ""  # model for the rewrite ("" = GROQ_MODEL); a small one saves quota

    # --- API / security --------------------------------------------------------------
    admin_api_key: SecretStr | None = None
    public_api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)  # empty → chat is public
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 20
    admin_rate_limit_per_minute: int = 10
    max_message_chars: int = 1000
    suggest_limit: int = 5  # type-ahead suggestions returned
    suggest_min_similarity: float = 0.55  # BGE-M3 cosine floor for semantic suggestions
    suggest_rate_limit_per_minute: int = 120  # fires per keystroke (debounced client-side)
    request_timeout_seconds: float = 90.0
    trust_forwarded_for: bool = False  # only behind a trusted reverse proxy

    @field_validator("sync_source_types", "cors_origins", "public_api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("sync_source_types")
    @classmethod
    def _check_types(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(SOURCE_TYPES)
        if bad:
            raise ValueError(f"unknown source types: {sorted(bad)}; allowed: {SOURCE_TYPES}")
        return v

    @field_validator("database_url")
    @classmethod
    def _normalise_db_url(cls, v: str) -> str:
        # Accept plain postgresql:// URLs and force the psycopg3 driver.
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must be <= CHUNK_MAX_TOKENS")
        if self.app_env == "production" and not self.admin_api_key:
            raise ValueError("ADMIN_API_KEY is required in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.get_secret_value() and self.groq_model)


def export_runtime_env(settings: Settings) -> None:
    """Export settings consumed by third-party libraries via environment variables.

    Explicit process env always wins over values from `.env`."""
    import os

    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if settings.hf_api_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_api_token.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    export_runtime_env(settings)
    return settings
