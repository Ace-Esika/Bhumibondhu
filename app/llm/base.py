"""Provider-agnostic LLM interface. Swap providers by implementing `LLMProvider`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage


class LLMError(RuntimeError):
    """Base class; `user_message` is safe to show to end users."""

    user_message = "উত্তর তৈরির সেবাটি এই মুহূর্তে সাময়িকভাবে অনুপলব্ধ। অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।"
    http_status = 503


class LLMNotConfigured(LLMError):
    user_message = "উত্তর তৈরির সেবাটি কনফিগার করা হয়নি।"


class LLMRateLimited(LLMError):
    user_message = "অনুরোধের চাপ বেশি হওয়ায় এই মুহূর্তে উত্তর দেওয়া যাচ্ছে না। অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।"
    http_status = 503


class LLMTimeout(LLMError):
    user_message = "উত্তর তৈরিতে নির্ধারিত সময়ের বেশি লেগেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"
    http_status = 504


class LLMUnavailable(LLMError):
    pass


class LLMBadRequest(LLMError):
    http_status = 502


class LLMRequestTooLarge(LLMError):
    """Prompt exceeds the provider's per-request/token limits; retry with less context."""

    http_status = 502


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: float
    attempts: int = 1
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None  # "length" means the output budget cut the answer off

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMProvider(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def generate(self, messages: list[BaseMessage], max_tokens: int | None = None,
                       model: str | None = None) -> LLMResult:
        """`model` overrides the configured model for this call (e.g. a small model for
        follow-up rewriting)."""
