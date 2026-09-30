"""Shared singletons, created once per process at startup."""

from __future__ import annotations

from fastapi import Request

from app.rag.pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline
