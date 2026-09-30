"""BGE reranker (BAAI/bge-reranker-v2-m3) as a LangChain document compressor.

Implemented on `langchain_core.documents.BaseDocumentCompressor` directly (rather than the
now-sunset `langchain_community` CrossEncoderReranker) over Hugging Face
sentence-transformers' `CrossEncoder`. Scores are sigmoid-activated relevance in [0, 1].

Only the fused top `RERANKER_TOP_K` candidates are ever scored.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, PrivateAttr

from app.core.config import Settings, get_settings
from app.retrieval.embeddings import resolve_device

log = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    pass


class BGEReranker(BaseDocumentCompressor):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    top_n: int = 5
    batch_size: int = 8
    max_length: int = 1024

    _model: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        self.model_name, device=resolve_device(self.device), max_length=self.max_length,
                        activation_fn=torch.nn.Sigmoid(),
                    )
                    log.info("reranker loaded", extra={"model": self.model_name})
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        with self._lock:
            try:
                scores = model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                       show_progress_bar=False)
            except Exception as e:
                raise RerankerError(f"rerank failed: {type(e).__name__}") from e
        return [float(s) for s in scores]

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        scores = self.score(query, [d.page_content for d in documents])
        ranked = sorted(zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True)
        out = []
        for doc, s in ranked[: self.top_n]:
            doc.metadata["rerank_score"] = s
            out.append(doc)
        return out


def build_reranker(settings: Settings) -> BGEReranker:
    return BGEReranker(
        model_name=settings.reranker_model, device=settings.reranker_device, top_n=settings.final_context_k,
        batch_size=settings.reranker_batch_size, max_length=settings.reranker_max_length,
    )


@lru_cache
def get_reranker() -> BGEReranker | None:
    s = get_settings()
    return build_reranker(s) if s.reranker_enabled else None
