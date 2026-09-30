"""BGE-M3 embeddings through LangChain's Hugging Face integration.

Providers (EMBEDDING_PROVIDER):
- `local` (default): `langchain_huggingface.HuggingFaceEmbeddings` running BAAI/bge-m3
  in-process via sentence-transformers. Self-hosted, free, CPU or CUDA. Weights are cached
  in HF_HOME (a Docker volume), so they are downloaded once.
- `hf_endpoint`: `HuggingFaceEndpointEmbeddings` (Hugging Face Inference API / TEI). No local
  weights, but text leaves the machine and needs HF_API_TOKEN. Opt-in only.

Vectors are L2-normalised (cosine similarity == dot product) and dimension-checked before
they ever reach pgvector. BGE-M3 needs no query instruction prefix.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


def resolve_device(device: str) -> str:
    if device != "auto":
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                log.warning("cuda requested but not available; falling back to cpu")
                return "cpu"
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_langchain_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "hf_endpoint":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        if not settings.hf_api_token:
            raise EmbeddingError("EMBEDDING_PROVIDER=hf_endpoint requires HF_API_TOKEN")
        return HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model,
            huggingfacehub_api_token=settings.hf_api_token.get_secret_value(),
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    device = resolve_device(settings.embedding_device)
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
    )
    # Our chunks are <= ~750 tokens incl. context; cap sequence length to bound CPU cost.
    emb._client.max_seq_length = settings.embedding_max_seq_length
    log.info("embedding model loaded", extra={"model": settings.embedding_model, "device": device})
    return emb


class EmbeddingService:
    """Thread-safe wrapper adding normalisation, validation, timing and async helpers."""

    def __init__(self, settings: Settings | None = None, backend: Embeddings | None = None):
        self.settings = settings or get_settings()
        self._backend = backend
        self._load_lock = threading.Lock()
        # One inference at a time per process: concurrent torch calls on CPU just thrash.
        self._infer_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def backend(self) -> Embeddings:
        if self._backend is None:
            with self._load_lock:
                if self._backend is None:
                    self._backend = build_langchain_embeddings(self.settings)
        return self._backend

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _finalise(self, vectors: list[list[float]]) -> list[list[float]]:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.settings.embedding_dim:
            raise EmbeddingError(f"embedding dimension {arr.shape} != expected {self.settings.embedding_dim}")
        if not np.isfinite(arr).all():
            raise EmbeddingError("embedding contains NaN/inf")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._infer_lock:
            try:
                vectors = self.backend.embed_documents(texts)
            except EmbeddingError:
                raise
            except Exception as e:
                raise EmbeddingError(f"embedding failed: {type(e).__name__}: {e}") from e
        return self._finalise(vectors)

    def embed_query(self, text: str) -> list[float]:
        with self._infer_lock:
            try:
                vector = self.backend.embed_query(text)
            except Exception as e:
                raise EmbeddingError(f"query embedding failed: {type(e).__name__}") from e
        return self._finalise([vector])[0]

    async def aembed_query(self, text: str) -> tuple[list[float], float]:
        t0 = time.perf_counter()
        vec = await asyncio.to_thread(self.embed_query, text)
        return vec, (time.perf_counter() - t0) * 1000


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
