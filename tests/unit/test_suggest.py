from __future__ import annotations

import numpy as np

from app.retrieval.suggest import SuggestionIndex
from tests.conftest import make_settings

QUESTIONS = ["নামজারি করতে কী কী কাগজপত্র লাগে?", "জাল দলিল করলে কী শাস্তি?", "What is the cost of land registration?"]


class FakeEmbedder:
    """Maps 'fees' and 'cost' to the same direction so semantic matching is testable offline."""

    def _vec(self, t: str) -> list[float]:
        t = t.lower()
        v = np.zeros(4, dtype=np.float32)
        if "fees" in t or "cost" in t:
            v[0] = 1
        elif "জাল" in t:
            v[1] = 1
        elif "zzzz" in t:
            v[3] = 1
        else:
            v[2] = 1
        return v.tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _index():
    idx = SuggestionIndex(make_settings(), FakeEmbedder(), questions=QUESTIONS)
    idx.build()
    return idx


def test_prefix_match_bangla():
    assert _index().suggest("নামজারি") == [QUESTIONS[0]]


def test_semantic_match_fees_to_cost():
    assert _index().suggest("registration fees") == [QUESTIONS[2]]


def test_unrelated_text_returns_nothing():
    assert _index().suggest("zzzz qqqq") == []


def test_too_short_returns_nothing():
    assert _index().suggest("a") == []
