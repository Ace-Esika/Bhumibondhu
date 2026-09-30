"""Public request/response models (request validation happens here)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import SOURCE_TYPES
from app.retrieval.query import SearchFilters


class ChatRequest(BaseModel):
    # Explicit examples so Swagger's "Try it out" pre-fills a valid body instead of
    # placeholder filter values ("string", year 1700) that would be rejected or match nothing.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"message": "নামজারি করতে কী কী কাগজপত্র লাগে?"},
        {"message": "জাল দলিল করলে কী শাস্তি?", "filters": {"doc_types": ["আইন"]}},
        {"message": "এর জন্য কোন আদালতে মামলা করতে হয়?", "conversation_id": "<id from the previous response>"},
    ]})

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_\-:.]+$")
    filters: SearchFilters | None = None


class SourceOut(BaseModel):
    index: int
    title: str | None
    source_type: str | None
    element_type: str | None = None
    source_id: str | None = None
    element_id: str | None = None
    doc_type: str | None = None
    section: str | None = None
    year: int | None = None
    authority: str | None = None
    url: str | None = None
    snippet: str | None = None
    cited: bool = True


class RetrievalInfo(BaseModel):
    query: str
    results_count: int
    route: str
    grounded: bool
    cached: bool = False
    reason: str | None = None
    standalone_query: str | None = Field(default=None, description="follow-up rewritten using the conversation")
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    retrieval: RetrievalInfo
    conversation_id: str | None = None
    request_id: str | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"query": "মৌজা ম্যাপ কিভাবে পাব", "top_k": 5},
    ]})

    query: str = Field(min_length=1, max_length=4000)
    filters: SearchFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True


class SearchHit(BaseModel):
    rank: int
    chunk_id: int
    score: dict[str, float | None]
    source: SourceOut
    context: str
    content: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    timings_ms: dict[str, float]
    reranked: bool


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_types: list[str] | None = None
    force: bool = False
    embed: bool = True
    reindex_full: bool = False
    allow_mass_delete: bool = False

    @field_validator("source_types")
    @classmethod
    def _check(cls, v):
        if v:
            bad = set(v) - set(SOURCE_TYPES)
            if bad:
                raise ValueError(f"unknown source types {sorted(bad)}")
        return v


class SyncQueued(BaseModel):
    run_id: int
    status: str


class IngestionStatus(BaseModel):
    runs: list[dict[str, Any]]
    embedding_jobs: list[dict[str, Any]]
    counts: dict[str, Any]


class ConversationTurnOut(BaseModel):
    role: str
    content: str
    standalone_query: str | None = None
    sources: list[str] = Field(default_factory=list)
    created_at: str | None = None


class ConversationOut(BaseModel):
    conversation_id: str
    turns: list[ConversationTurnOut]
