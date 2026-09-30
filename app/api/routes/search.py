from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline
from app.api.schemas import SearchHit, SearchRequest, SearchResponse, SourceOut
from app.core.config import get_settings
from app.core.security import rate_limit, require_public_key
from app.rag.citation import build_source
from app.rag.guardrails import clean_user_message
from app.rag.pipeline import RAGPipeline, _timings

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse,
             dependencies=[Depends(require_public_key), Depends(rate_limit("search"))])
async def search(body: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
    """Retrieval only (no LLM): useful for debugging relevance and for UI source browsing."""
    query = clean_user_message(body.query, get_settings().max_message_chars)
    docs, trace = await pipeline.searcher.search(query, body.filters, final_k=body.top_k, rerank=body.rerank)
    hits = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        hits.append(SearchHit(
            rank=i, chunk_id=m["chunk_id"],
            score={"fused": m["fused_score"], "vector": m["vector_score"], "lexical": m["lexical_score"],
                   "rerank": m["rerank_score"]},
            source=SourceOut(**build_source(i, d)), context=m["context"], content=m["content"],
        ))
    return SearchResponse(query=query, results=hits, timings_ms=_timings(trace), reranked=trace.reranked)
