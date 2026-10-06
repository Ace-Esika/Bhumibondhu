from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.core.security import rate_limit, require_public_key

router = APIRouter(prefix="/api", tags=["suggest"])


class SuggestResponse(BaseModel):
    query: str
    suggestions: list[str]


@router.get("/suggest", response_model=SuggestResponse,
            dependencies=[Depends(require_public_key),
                          Depends(rate_limit("suggest", "suggest_rate_limit_per_minute"))])
async def suggest(request: Request, q: str = Query(..., max_length=200)) -> SuggestResponse:
    """Related questions from the curated list while the user types (no LLM call)."""
    index = request.app.state.suggestions
    results = await asyncio.to_thread(index.suggest, q)
    return SuggestResponse(query=q, suggestions=results)
