from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from app.api.deps import get_pipeline
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationTurnOut,
    RetrievalInfo,
    SourceOut,
)
from app.core.security import rate_limit, require_public_key
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse,
             dependencies=[Depends(require_public_key), Depends(rate_limit("chat"))])
async def chat(body: ChatRequest, request: Request, pipeline: RAGPipeline = Depends(get_pipeline)) -> ChatResponse:
    r = await pipeline.answer(body.message, body.conversation_id, body.filters)
    return ChatResponse(
        answer=r.answer,
        sources=[SourceOut(**s) for s in r.sources],
        retrieval=RetrievalInfo(query=r.query, results_count=r.results_count, route=r.route, grounded=r.grounded,
                                cached=r.cached, reason=r.reason, timings_ms=r.timings_ms,
                                standalone_query=r.standalone_query),
        conversation_id=r.conversation_id,
        request_id=getattr(request.state, "request_id", None),
    )


_CONV_ID = Path(max_length=100, pattern=r"^[A-Za-z0-9_\-:.]+$")


@router.get("/conversations/{conversation_id}", response_model=ConversationOut,
            dependencies=[Depends(require_public_key), Depends(rate_limit("conversation"))])
async def get_conversation(conversation_id: str = _CONV_ID,
                           pipeline: RAGPipeline = Depends(get_pipeline)) -> ConversationOut:
    """History of one conversation (the id acts as its access token)."""
    turns = await pipeline.store.recent(conversation_id, limit=200)
    if not turns:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return ConversationOut(conversation_id=conversation_id, turns=[
        ConversationTurnOut(role=t.role, content=t.content, standalone_query=t.metadata.get("standalone"),
                            sources=t.metadata.get("sources", []),
                            created_at=t.created_at.isoformat() if t.created_at else None) for t in turns])


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(require_public_key), Depends(rate_limit("conversation"))])
async def delete_conversation(conversation_id: str = _CONV_ID,
                              pipeline: RAGPipeline = Depends(get_pipeline)) -> None:
    if not await pipeline.store.delete(conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
