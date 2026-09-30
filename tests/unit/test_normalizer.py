import pytest

from app.ingestion.normalizer import (
    AUTHORITY_FORUM,
    AUTHORITY_GAZETTE_LAW,
    AUTHORITY_OFFICIAL_QNA,
    RecordValidationError,
    normalize,
    normalize_category,
    parse_date,
)
from tests.conftest import load_snapshot, make_settings


def _ebook(eid):
    return next(e for e in load_snapshot("ebook") if e["id"] == eid)


def test_ebook_hierarchy_preserved():
    (doc,) = normalize("ebook", _ebook(164), make_settings())
    assert doc.source_type == "ebook" and doc.source_id == "164"
    sections = [u for u in doc.units if u.source_type == "section"]
    assert sections and all(u.parent_source_id == "164" for u in sections)
    subs = [c for s in sections for c in s.children if c.source_type == "subsection"]
    assert subs and all(c.parent_source_id in {s.source_id for s in sections} for c in subs)
    schedules = [g for s in sections for c in s.children for g in c.children if g.source_type == "schedule"]
    assert schedules
    subschedules = [x for g in schedules for x in g.children]
    assert subschedules and all(x.source_type == "subschedule" for x in subschedules)


def test_ebook_metadata_and_authority():
    raw = _ebook(241)
    (doc,) = normalize("ebook", raw, make_settings())
    assert doc.title == raw["title_of_act"].strip()
    assert doc.doc_type == raw["ebooks_type"]
    assert doc.file_url == raw.get("file")
    assert doc.url is None  # no page URL template configured → never invented
    if raw.get("publication_by") == "গেজেট":
        assert doc.authority == AUTHORITY_GAZETTE_LAW


def test_empty_title_is_derived_from_real_fields_and_flagged():
    (doc,) = normalize("ebook", _ebook(838), make_settings())
    assert doc.metadata["title_derived"] is True
    assert "পরিপত্র" in doc.title and "৩১.০০" in doc.title
    body = [u for u in doc.units if u.source_type == "ebook_text"]
    assert body and body[0].metadata["role"] == "body"


def test_url_template_is_used_when_configured():
    (doc,) = normalize("ebook", _ebook(241), make_settings(ebook_url_template="https://example.test/acts/{id}"))
    assert doc.url == "https://example.test/acts/241"


def test_placeholder_records_are_excluded():
    assert normalize("ebook", _ebook(633), make_settings()) == []
    forums = load_snapshot("forum")
    test_group = next(f for f in forums if f["name"].lower().startswith("test"))
    assert normalize("forum", test_group, make_settings()) == []
    # The filter is configurable.
    assert normalize("ebook", _ebook(633), make_settings(exclude_title_regex="")) != []


def test_qna_is_one_unit_with_category_normalised():
    raw = load_snapshot("qna_type2")[0]
    (doc,) = normalize("qna_type2", raw, make_settings())
    assert doc.authority == AUTHORITY_OFFICIAL_QNA
    assert len(doc.units) == 1
    assert raw["question"].strip()[:10] in doc.units[0].text
    assert "উত্তর:" in doc.units[0].text
    assert normalize_category(" Khotian ") == "khotian"
    assert normalize_category("") is None


def test_forum_topics_keep_parent_and_low_authority():
    group = {"id": "g1", "name": "ভূমি আলোচনা", "badge": "none", "group_type": "open",
             "topics": [{"id": "t1", "group": "g1", "title": "নামজারি প্রশ্ন", "description": "<p>বিস্তারিত</p>"}]}
    docs = normalize("forum", group, make_settings())
    (topic,) = [d for d in docs if d.source_type == "topic"]
    assert topic.parent_source_id == "g1" and topic.authority == AUTHORITY_FORUM
    assert topic.units[0].text == "বিস্তারিত"


def test_blog_html_is_cleaned():
    raw = load_snapshot("blog")[0]
    (doc,) = normalize("blog", raw, make_settings())
    assert "<p>" not in doc.units[0].text and doc.units[0].text


@pytest.mark.parametrize("raw", [{"question": "q", "answer": "a"}, {"id": 1, "question": " ", "answer": "a"}])
def test_invalid_records_raise(raw):
    with pytest.raises(RecordValidationError):
        normalize("qna_type2", raw, make_settings())


def test_missing_optional_fields_and_null_lists_tolerated():
    (doc,) = normalize("ebook", {"id": 1, "title_of_act": "ক আইন", "meta_keywords": None, "sections": None},
                       make_settings())
    assert doc.units == [] and doc.year is None


def test_parse_date_formats():
    assert str(parse_date("2026-09-13")) == "2026-09-13"
    assert str(parse_date("০১-০১-২০২৩")) == "2023-01-01"
    assert parse_date("") is None and parse_date("31-02-2020") is None
