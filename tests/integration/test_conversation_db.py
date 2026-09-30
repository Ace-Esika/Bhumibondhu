import pytest
from sqlalchemy import text

from app.rag.memory import SQLConversationStore, Turn

pytestmark = pytest.mark.integration


async def test_sql_store_roundtrip_purge_and_cascade(sm):
    async with sm() as s, s.begin():
        await s.execute(text("TRUNCATE conversations CASCADE"))
    store = SQLConversationStore(sm)
    await store.append("c1", [Turn("user", "নামজারির ফি কত?"), Turn("assistant", "১১৭০ টাকা [1]", {"route": "rag"})])
    await store.append("c1", [Turn("user", "এটা কি অনলাইনে দেওয়া যায়?", {"standalone": "..."})])
    turns = await store.recent("c1", limit=10)
    assert [t.role for t in turns] == ["user", "assistant", "user"] and turns[1].metadata == {"route": "rag"}
    assert [t.content for t in await store.recent("c1", limit=2)] == ["১১৭০ টাকা [1]", "এটা কি অনলাইনে দেওয়া যায়?"]
    assert await store.purge_older_than(1) == 0
    async with sm() as s, s.begin():
        await s.execute(text("UPDATE conversations SET updated_at = now() - interval '40 days' WHERE id='c1'"))
    assert await store.purge_older_than(30) == 1
    async with sm() as s:
        assert (await s.execute(text("SELECT count(*) FROM conversation_messages"))).scalar() == 0  # cascade
    assert await store.delete("c1") is False
