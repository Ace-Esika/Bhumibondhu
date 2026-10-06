"""Type-ahead question suggestions drawn only from a curated question list.

Ranking blends exact text matching (prefix / substring / shared words) with BGE-M3 semantic
similarity, so "fees" can surface "What is the cost of ...". Only questions from the list are
ever returned, and semantic hits must clear `suggest_min_similarity`.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.text import normalize_query
from app.retrieval.embeddings import EmbeddingService

log = logging.getLogger(__name__)

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "data" / "suggestions.txt"
_WORD_RE = re.compile(r"[\wঀ-৿]+", re.UNICODE)
_MIN_CHARS = 2
_SEMANTIC_MIN_CHARS = 3
_QUERY_CACHE_SIZE = 512


def load_questions(path: Path) -> list[str]:
    if not path.is_file():
        log.warning("suggestions file not found", extra={"path": str(path)})
        return []
    seen: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seen.setdefault(normalize_query(line), line)
    return list(seen.values())


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class SuggestionIndex:
    def __init__(self, settings: Settings, embedder: EmbeddingService, questions: list[str] | None = None,
                 path: Path | None = None):
        self.settings = settings
        self.embedder = embedder
        self.questions = questions if questions is not None else load_questions(path or DEFAULT_FILE)
        self._norm = [normalize_query(q).lower() for q in self.questions]
        self._words = [set(_words(n)) for n in self._norm]
        self._lengths = np.array([len(q) for q in self.questions])
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()

    def build(self, batch_size: int = 32) -> None:
        """Embed the question list (blocking; call from a thread).

        Done in small batches so the shared inference lock is released between them and
        chat/search queries are never stuck behind the whole list.
        """
        if not self.questions or self._matrix is not None:
            return
        with self._lock:
            if self._matrix is not None:
                return
            parts = [self.embedder.embed_documents(self.questions[i:i + batch_size])
                     for i in range(0, len(self.questions), batch_size)]
            self._matrix = np.asarray([v for part in parts for v in part], dtype=np.float32)
        log.info("suggestion index ready", extra={"questions": len(self.questions)})

    def _embed(self, text: str) -> np.ndarray | None:
        vec = self._qcache.get(text)
        if vec is not None:
            self._qcache.move_to_end(text)
            return vec
        try:
            vec = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
        except Exception:
            log.warning("suggestion embedding failed; using text matching only")
            return None
        self._qcache[text] = vec
        if len(self._qcache) > _QUERY_CACHE_SIZE:
            self._qcache.popitem(last=False)
        return vec

    def _lexical(self, norm: str, words: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.questions), dtype=np.float32)
        if not words:
            return scores
        last = words[-1]
        for i, (q, qwords) in enumerate(zip(self._norm, self._words, strict=True)):
            if q.startswith(norm):
                scores[i] = 1.0
            elif norm in q:
                scores[i] = 0.85
            else:
                # Completed words must match exactly; the word being typed may be a prefix.
                done = sum(1 for w in words[:-1] if w in qwords)
                partial = any(w.startswith(last) for w in qwords)
                total = len(words)
                hit = done + (1 if partial else 0)
                if hit and hit / total >= 0.5:
                    scores[i] = 0.6 * hit / total
        return scores

    def suggest(self, text: str, limit: int | None = None) -> list[str]:
        limit = limit or self.settings.suggest_limit
        norm = normalize_query(text).lower()
        if len(norm) < _MIN_CHARS or not self.questions:
            return []
        lex = self._lexical(norm, _words(norm))
        score = lex.copy()
        if len(norm) >= _SEMANTIC_MIN_CHARS and self._matrix is not None:
            qv = self._embed(norm)
            if qv is not None:
                sim = self._matrix @ qv
                semantic = np.where(sim >= self.settings.suggest_min_similarity, sim, 0.0)
                score = np.maximum(lex, semantic * 0.8)
        # Best score first; among equal scores prefer the shorter (more canonical) question.
        order = np.lexsort((self._lengths, -score))
        return [self.questions[i] for i in order[:limit] if score[i] > 0]
