"""Conversation history storage.

`conversation_id` works as a capability: the server issues unguessable ids
(`secrets.token_urlsafe`), and anyone holding one can read that conversation. Clients may
also supply their own id (used as-is, created on first use). History is retained for
CONVERSATION_RETENTION_DAYS and purged by the worker.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Conversation, ConversationMessage


def new_conversation_id() -> str:
    return secrets.token_urlsafe(18)


@dataclass
class Turn:
    role: str  # user | assistant
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


class ConversationStore(Protocol):
    async def recent(self, conversation_id: str, limit: int) -> list[Turn]: ...
    async def append(self, conversation_id: str, turns: list[Turn]) -> None: ...
    async def delete(self, conversation_id: str) -> bool: ...


class SQLConversationStore:
    def __init__(self, sessionmaker):
        self.sessionmaker = sessionmaker

    async def recent(self, conversation_id: str, limit: int) -> list[Turn]:
        async with self.sessionmaker() as s:
            rows = (await s.execute(
                select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.id.desc()).limit(limit)
            )).scalars().all()
        return [Turn(r.role, r.content, r.metadata_ or {}, r.created_at) for r in reversed(rows)]

    async def append(self, conversation_id: str, turns: list[Turn]) -> None:
        async with self.sessionmaker() as s, s.begin():
            await s.execute(pg_insert(Conversation).values(id=conversation_id)
                            .on_conflict_do_update(index_elements=["id"], set_={"updated_at": func.now()}))
            for t in turns:
                s.add(ConversationMessage(conversation_id=conversation_id, role=t.role, content=t.content,
                                          metadata_=t.metadata))

    async def delete(self, conversation_id: str) -> bool:
        async with self.sessionmaker() as s, s.begin():
            n = (await s.execute(delete(Conversation).where(Conversation.id == conversation_id))).rowcount
        return bool(n)

    async def purge_older_than(self, days: float) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        async with self.sessionmaker() as s, s.begin():
            return (await s.execute(delete(Conversation).where(Conversation.updated_at < cutoff))).rowcount


class InMemoryConversationStore:
    """For tests and database-less development."""

    def __init__(self):
        self.data: dict[str, list[Turn]] = {}

    async def recent(self, conversation_id: str, limit: int) -> list[Turn]:
        return list(self.data.get(conversation_id, []))[-limit:]

    async def append(self, conversation_id: str, turns: list[Turn]) -> None:
        self.data.setdefault(conversation_id, []).extend(turns)

    async def delete(self, conversation_id: str) -> bool:
        return self.data.pop(conversation_id, None) is not None
