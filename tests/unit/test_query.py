import pytest
from pydantic import ValidationError

from app.retrieval.query import QueryRoute, SearchFilters, analyze_query


@pytest.mark.parametrize("q,section", [
    ("ধারা ৫ এ কী বলা আছে", "5"), ("ধারা-১২ অনুযায়ী", "12"), ("৫ নং ধারা", "5"), ("বিধি ৩ক", "3ক"),
    ("section 7 of the act", "7"), ("নামজারি কী", None),
])
def test_section_extraction(q, section):
    assert analyze_query(q).section_number == section


def test_year_extraction():
    assert analyze_query("ভূমি আইন ২০২৩ এর ধারা ২").year == 2023
    assert analyze_query("নামজারি").year is None


def test_routes():
    assert analyze_query("নামজারি করতে কী কী লাগে?").route == QueryRoute.RAG
    q = analyze_query("কতটি আইন আছে?")
    assert q.route == QueryRoute.STRUCTURED and q.structured == {"kind": "count_ebooks", "doc_type": "আইন",
                                                                 "year": None}
    assert analyze_query("২০২৩ সালের কয়টি পরিপত্র আছে").structured["year"] == 2023
    assert analyze_query("How many circulars are there?").structured["doc_type"] == "পরিপত্র"
    assert analyze_query("হ্যালো").route == QueryRoute.SMALLTALK
    assert analyze_query("ধন্যবাদ!").route == QueryRoute.SMALLTALK
    for salam in ("আসসালামু আলাইকুম", "আসসালামু আলাইকুম!", "assalamu alaikum", "সালাম"):
        q = analyze_query(salam)
        assert q.route == QueryRoute.SMALLTALK and q.structured == {"greeting": "salam"}
    assert analyze_query("ধন্যবাদ").structured == {"greeting": "thanks"}
    assert analyze_query("হ্যালো").structured == {"greeting": "hello"}
    # a count question about something without structured data stays RAG
    assert analyze_query("নামজারিতে কত টাকা লাগে").route == QueryRoute.RAG


def test_filters_validation():
    assert SearchFilters().is_empty()
    assert not SearchFilters(source_types=["ebook"]).is_empty()
    with pytest.raises(ValidationError):
        SearchFilters(source_types=["pinecone"])
    with pytest.raises(ValidationError):
        SearchFilters(year=99)


def test_definition_intent_adds_legal_definition_terms():
    q = analyze_query("আইন অনুযায়ী কৃষিভূমি বলতে কী বোঝায়?")
    assert q.is_definition and "সংজ্ঞা:*" in q.terms and "কৃষিভূমি:*" in q.terms
    assert not analyze_query("নামজারি করতে কী কী লাগে?").is_definition


@pytest.mark.parametrize("q,detail,all_sections", [
    ("ভূমি অপরাধ প্রতিরোধ আইনের ধারা ৪ বিস্তারিত বলুন", True, False),
    ("অনলাইনে নামজারির পুরো প্রক্রিয়া ধাপে ধাপে বলুন", True, False),
    ("ভূমি উন্নয়ন কর আইন, ২০২৩ এর সব ধারা ব্যাখ্যা করুন", True, True),
    ("ভূমি-খাতয়ান (পাবত্য চট্টগ্রাম) অধ্যাদেশ, ১৯৮৪ ধারাসমূহ?", True, True),
    ("Explain the land crime act in detail", True, False),
    ("নামজারি করতে কী কী লাগে?", False, False),
    ("দাখিলা কি?", False, False),
])
def test_detail_and_all_sections_intent(q, detail, all_sections):
    a = analyze_query(q)
    assert (a.wants_detail, a.wants_all_sections) == (detail, all_sections)
