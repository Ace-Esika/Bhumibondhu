"""Multi-turn conversation behaviour (fake retriever, fake LLM, in-memory store)."""

import pytest

from app.llm.base import LLMRateLimited
from app.rag.condense import fallback_standalone, looks_dependent, render_turns
from app.rag.memory import InMemoryConversationStore, Turn
from tests.unit.test_rag_pipeline import FakeSearcher, _doc, _pipeline


class RecordingSearcher(FakeSearcher):
    def __init__(self, docs):
        super().__init__(docs)
        self.queries = []

    async def search(self, q, filters=None, final_k=None, rerank=True):
        self.queries.append(q.text if hasattr(q, "text") else q)
        return await super().search(q, filters, final_k, rerank)


@pytest.mark.parametrize("q,dependent", [
    ("এর ফি কত?", True), ("উক্ত আইনের ধারা ৫ কী বলে?", True), ("এই আইনে শাস্তি কী?", True),
    ("আর অনলাইনে করা যায়?", True), ("What about its penalty?", True), ("কত টাকা?", True),
    ("নামজারি করতে কী কী কাগজপত্র লাগে?", False), ("ভূমি উন্নয়ন কর অনলাইনে কিভাবে দিব?", False),
])
def test_looks_dependent(q, dependent):
    assert looks_dependent(q) is dependent


def test_render_turns_keeps_newest_within_budget():
    turns = [Turn("user", f"প্রশ্ন {i} " + "ক" * 300) for i in range(10)]
    out = render_turns(turns, max_chars=800)
    assert "প্রশ্ন 9" in out and "প্রশ্ন 0" not in out and len(out) < 1300


def test_fallback_prefixes_previous_user_question():
    turns = [Turn("user", "নামজারির ফি কত?"), Turn("assistant", "১১৭০ টাকা [1]")]
    assert fallback_standalone("অনলাইনে দেওয়া যায়?", turns) == "নামজারির ফি কত? — অনলাইনে দেওয়া যায়?"


def _conv_pipeline(*script, store=None, **s):
    p, llm = _pipeline([_doc(1)], *script, store=store, **s)
    p.searcher = RecordingSearcher([_doc(1)])
    return p, llm


async def test_first_turn_issues_id_and_saves_both_turns():
    store = InMemoryConversationStore()
    p, _ = _conv_pipeline("নামজারির ফি ১১৭০ টাকা [1]", store=store)
    r = await p.answer("নামজারির ফি কত?")
    assert r.conversation_id and len(r.conversation_id) >= 20
    turns = store.data[r.conversation_id]
    assert [t.role for t in turns] == ["user", "assistant"] and turns[1].metadata["sources"]


async def test_followup_is_rewritten_and_history_reaches_the_prompt_as_non_evidence():
    store = InMemoryConversationStore()
    p, llm = _conv_pipeline("নামজারির ফি ১১৭০ টাকা [1]", store=store)
    first = await p.answer("নামজারির ফি কত?")
    llm.script = ["নামজারির ফি কি অনলাইনে পরিশোধ করা যায়?", "হ্যাঁ, অনলাইনে দেওয়া যায় [1]"]
    r = await p.answer("এটা কি অনলাইনে দেওয়া যায়?", conversation_id=first.conversation_id)
    assert r.standalone_query == "নামজারির ফি কি অনলাইনে পরিশোধ করা যায়?"
    assert p.searcher.queries[-1] == r.standalone_query  # retrieval used the standalone question
    prompt = llm.messages[-1].content
    assert "<conversation_history>" in prompt and "এগুলো প্রমাণ নয়" in prompt and "নামজারির ফি কত?" in prompt
    assert r.conversation_id == first.conversation_id and len(store.data[r.conversation_id]) == 4
    assert store.data[r.conversation_id][2].metadata["standalone"] == r.standalone_query


async def test_self_contained_followup_skips_the_rewrite_call():
    store = InMemoryConversationStore()
    p, llm = _conv_pipeline("উত্তর [1]", store=store)
    first = await p.answer("নামজারির ফি কত?")
    n_calls = len(llm.calls)
    r = await p.answer("ভূমি উন্নয়ন কর অনলাইনে কিভাবে দিব?", conversation_id=first.conversation_id)
    assert r.standalone_query is None and len(llm.calls) == n_calls + 1  # answer only, no rewrite


async def test_rewrite_failure_falls_back_to_previous_question():
    store = InMemoryConversationStore()
    p, llm = _conv_pipeline("উত্তর [1]", store=store)
    first = await p.answer("নামজারির ফি কত?")
    llm.script = [LLMRateLimited("x"), "উত্তর [1]"]
    r = await p.answer("এর জন্য কী কাগজ লাগে?", conversation_id=first.conversation_id)
    assert r.standalone_query == "নামজারির ফি কত? — এর জন্য কী কাগজ লাগে?" and r.grounded


async def test_store_failures_never_break_answers():
    class Broken(InMemoryConversationStore):
        async def recent(self, *a):
            raise RuntimeError("db down")

        async def append(self, *a):
            raise RuntimeError("db down")

    p, _ = _conv_pipeline("উত্তর [1]", store=Broken())
    r = await p.answer("খতিয়ান কী?", conversation_id="abc")
    assert r.grounded and r.conversation_id == "abc"


async def test_conversation_can_be_disabled():
    store = InMemoryConversationStore()
    p, _ = _conv_pipeline("উত্তর [1]", store=store, conversation_enabled=False)
    r = await p.answer("খতিয়ান কী?")
    assert r.conversation_id is None and store.data == {}


async def test_answers_with_history_bypass_the_response_cache(monkeypatch):
    calls = []

    async def fake_get(key):
        calls.append(key)
        return None
    monkeypatch.setattr("app.rag.pipeline.cache_get", fake_get)
    store = InMemoryConversationStore()
    p, llm = _conv_pipeline("উত্তর [1]", store=store, cache_enabled=True)
    first = await p.answer("খতিয়ান কী?")
    assert len(calls) == 1
    llm.script = ["খতিয়ান সংশোধন কীভাবে করা যায়?", "উত্তর [1]"]
    await p.answer("এটা সংশোধন কীভাবে করব?", conversation_id=first.conversation_id)
    assert len(calls) == 1  # no cache lookup for the history-dependent turn


