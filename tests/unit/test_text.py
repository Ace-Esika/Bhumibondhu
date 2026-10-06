from app.core.text import (
    clean_unicode,
    html_to_text,
    lexical_text,
    light_stem,
    normalize_section_number,
    parse_year,
    query_terms,
    tokenize,
)


def test_html_to_text_preserves_structure_and_drops_active_content():
    html = ("<p><strong>সূচিপত্র</strong></p><p>ভূমিকা&nbsp;...</p><ul><li>এক</li><li>দুই</li></ul>"
            "<table><tr><td>ক্রমিক</td><td>পদ</td></tr><tr><td>১</td><td>অফিসার</td></tr></table>"
            "<script>alert(1)</script><img src=x onerror=alert(1)>")
    text = html_to_text(html)
    assert "সূচিপত্র" in text and "ভূমিকা ..." in text
    assert "- এক" in text and "- দুই" in text
    assert "ক্রমিক | পদ" in text and "১ | অফিসার" in text
    assert "alert" not in text and "<" not in text


def test_html_to_text_plain_and_empty():
    assert html_to_text(None) == ""
    assert html_to_text("") == ""
    assert html_to_text("সরল &amp; পাঠ") == "সরল & পাঠ"


def test_parse_year_bengali_and_ascii():
    assert parse_year("২০২৩") == 2023
    assert parse_year("1972") == 1972
    assert parse_year("") is None
    assert parse_year("abc") is None
    assert parse_year("0001") is None


def test_normalize_section_number():
    assert normalize_section_number("১। ") == "1"
    assert normalize_section_number("(২) ") == "2"
    assert normalize_section_number("৫ক") == "5ক"
    assert normalize_section_number("0") is None
    assert normalize_section_number("") is None
    assert normalize_section_number(None) is None


def test_normalize_section_number_upstream_quirks():
    # U+09F7 is used by the upstream API as a separator ("৩৷"); it must not leak into the number.
    assert normalize_section_number("৩৷") == "3"
    assert normalize_section_number("১০৷ ") == "10"
    assert normalize_section_number("9a") == "9A" == normalize_section_number("9A")
    assert normalize_section_number("26[86") == "26"
    assert normalize_section_number("১৪৫ঝ") == "145ঝ"


def test_lexical_coverage_helper():
    from app.core.text import lexical_coverage
    assert lexical_coverage(["নামজারি:*", "ফি"], "নামজারির ফি") == 1.0
    assert lexical_coverage(["নামজারি:*", "ফি"], "অন্য") == 0.0
    assert lexical_coverage([], "x") == 0.0


def test_tokenize_normalises_digits_and_zero_width():
    assert tokenize("ধারা ৫।") == ["ধারা", "5"]
    assert tokenize("নাম‌জারি") == ["নামজারি"]


def test_precomposed_ya_matches_decomposed():
    # U+09DF (precomposed য়) and য + nukta must normalise identically.
    assert clean_unicode("য়") == clean_unicode("য়")
    assert query_terms("কয়টি") == query_terms("কয়টি")


def test_query_terms_stopwords_and_prefixes():
    assert query_terms("নামজারি করতে কী কী লাগে?") == ["নামজারি:*"]
    terms = query_terms("ডিসিআর ফি কি অনলাইনে দেয়া যাবে")
    assert "ডিসিআর:*" in terms and "ফি" in terms and "অনলাইন:*" in terms


def test_light_stem_genitive_only_after_vowel_sign():
    assert light_stem("নামজারির") == "নামজারি"
    assert light_stem("খতিয়ানের") == "খতিয়ান"
    assert light_stem("ডিসিআর") == "ডিসিআর"  # র after independent vowel is part of the word
    assert light_stem("fee") == "fee"


def test_query_terms_are_tsquery_safe():
    import re

    terms = query_terms("x' | !(drop) & table:* <-> ধারা")
    assert terms
    for t in terms:
        assert re.fullmatch(r"[\w\u0980-\u09FF]+(:\*)?", t), t
        assert not set(t) & set("'|!&()<>"), t


def test_lexical_text():
    assert lexical_text("ভূমি উন্নয়ন কর, ২০২৩!") == "ভূমি উন্নয়ন কর 2023"


def test_spelling_variants_match_across_index_and_query():
    assert "নম্বর:*" in query_terms("হোল্ডিং নম্বর কী?")
    assert "নম্বর" in lexical_text("হোল্ডিং নাম্বার হল").split()
    assert lexical_text("জরীপ") == lexical_text("জরিপ")


def test_filler_words_are_stopwords_and_short_stems_expand():
    assert query_terms("দাখিলা মানে কী জিনিস?") == ["দাখিলা:*"]
    assert query_terms("জমির") == ["জমির", "জমি"]
