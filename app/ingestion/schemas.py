"""Validation schemas for Bhumipedia API records.

Derived from the project's API documentation and verified against live responses. The
upstream API omits any field whose value is null, so almost every field is optional;
only the identity/title fields needed to build a document are required. Unknown fields are
kept (extra="allow") so the raw record is preserved for auditing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("*", mode="before")
    @classmethod
    def _null_list_to_empty(cls, v: Any, info) -> Any:
        # The API omits nulls, but tolerate an explicit null for list-typed fields.
        field = cls.model_fields.get(info.field_name)
        if v is None and field is not None and field.default_factory is list:
            return []
        return v


class SubScheduleIn(_Lenient):
    id: int
    number: str | None = None
    heading: str | None = None
    content: str | None = None
    note: str | None = None


class ScheduleIn(_Lenient):
    id: int
    number: str | None = None
    heading: str | None = None
    content: str | None = None
    note: str | None = None
    subschedules: list[SubScheduleIn] = Field(default_factory=list)


class SubsectionIn(_Lenient):
    id: int
    number: str | None = None
    heading: str | None = None
    content: str | None = None
    note: str | None = None
    schedules: list[ScheduleIn] = Field(default_factory=list)


class SectionIn(_Lenient):
    id: int
    number: str | None = None
    heading: str | None = None
    content: str | None = None
    note: str | None = None
    subsections: list[SubsectionIn] = Field(default_factory=list)
    schedules: list[ScheduleIn] = Field(default_factory=list)


class EbookIn(_Lenient):
    id: int
    title_of_act: str | None = None  # empty for some circulars; a title is derived downstream
    ebooks_type: str | None = None
    act_year: str | None = None
    number: str | None = None
    publication_date: str | None = None
    publication_by: str | None = None
    proposal: str | None = None
    objective: str | None = None
    motto: str | None = None
    file: str | None = None
    merged_file: str | None = None
    system_generated_pdf: str | None = None
    schedules: str | None = None  # free-text field on the act itself
    heading: str | None = None
    footer: str | None = None
    created_at_bn: str | None = None
    created_at_en: str | None = None
    applicable_date_bn: str | None = None
    applicable_date_en: str | None = None
    branch: str | None = None
    sub_branch: str | None = None
    signature_position: str | None = None
    signature_by: str | None = None
    meta_keywords: list[str] = Field(default_factory=list)
    multiple_reference_link: list[str] = Field(default_factory=list)
    created_date: str | None = None
    sections: list[SectionIn] = Field(default_factory=list)


class BlogIn(_Lenient):
    id: int
    title_name: str
    author: str | None = None
    content: str | None = None
    cover: str | None = None
    featured: bool | None = None
    created_date: str | None = None


class TopicIn(_Lenient):
    id: str
    group: str | None = None
    title: str
    description: str | None = None
    status: str | None = None
    is_pinned: bool | None = None
    is_archived: bool | None = None
    created_date: str | None = None


class ForumIn(_Lenient):
    id: str
    name: str
    description: str | None = None
    badge: str | None = None
    group_type: str | None = None
    category: Any | None = None
    created_date: str | None = None
    topics: list[TopicIn] = Field(default_factory=list)


class QnaIn(_Lenient):
    id: int
    question: str
    answer: str
    category: str | None = None
    keyword: str | None = None

    @field_validator("question", "answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("empty question/answer")
        return v


SCHEMAS: dict[str, type[_Lenient]] = {
    "ebook": EbookIn,
    "blog": BlogIn,
    "forum": ForumIn,
    "qna_type1": QnaIn,
    "qna_type2": QnaIn,
}

ENDPOINTS: dict[str, str] = {
    "ebook": "/api/ebooks/full/",
    "blog": "/api/blogs/full/",
    "forum": "/api/forums/full/",
    "qna_type1": "/api/v1/qna/type1/",
    "qna_type2": "/api/v1/qna/type2/",
}
