import pytest

from app.retrieval.hybrid import Candidate, apply_boosts, fuse, lexical_coverage
from app.retrieval.query import SearchFilters, analyze_query
from app.retrieval.search import Hit, filter_clause
from tests.conftest import make_settings


def test_rrf_fusion_math():
    v = [Hit(1, 0.9, 1), Hit(2, 0.8, 2)]
    lx = [Hit(2, 0.5, 1), Hit(3, 0.1, 2)]
    c = fuse(v, lx, "rrf", 0.6, 0.4, rrf_k=60)
    assert c[1].fused == pytest.approx(0.6 / 61)
    assert c[2].fused == pytest.approx(0.6 / 62 + 0.4 / 61)
    assert c[3].fused == pytest.approx(0.4 / 62)
    assert max(c.values(), key=lambda x: x.fused).chunk_id == 2  # found by both → wins


def test_linear_fusion_minmax():
    v = [Hit(1, 0.9, 1), Hit(2, 0.5, 2)]
    lx = [Hit(2, 0.3, 1)]
    c = fuse(v, lx, "linear", 0.5, 0.5)
    assert c[1].fused == pytest.approx(0.5)
    assert c[2].fused == pytest.approx(0.5)  # vec normalised 0 + lex normalised 1 (single hit)


def test_weights_are_respected():
    v, lx = [Hit(1, 0.9, 1)], [Hit(2, 0.9, 1)]
    c = fuse(v, lx, "rrf", 1.0, 0.0)
    assert c[1].fused > c[2].fused == 0


def test_unknown_fusion_rejected():
    with pytest.raises(ValueError):
        fuse([], [], "magic", 1, 1)


def test_boosts_authority_and_section():
    s = make_settings(authority_boost=0.2, section_match_boost=0.5)
    q = analyze_query("ধারা ৫ কী বলে")
    law = Candidate(1, fused=1.0, row={"authority": 100, "section_number": "5", "year": 2020})
    forum = Candidate(2, fused=1.0, row={"authority": 10, "section_number": None})
    apply_boosts([law, forum], q, s)
    assert law.fused == pytest.approx(1.2 * 1.5)
    assert forum.fused == pytest.approx(1.02)


def test_lexical_coverage_prefix_terms():
    assert lexical_coverage(["নামজারি:*", "ফি"], "নামজারির আবেদন ফি") == 1.0
    assert lexical_coverage(["নামজারি:*", "দাখিলা"], "নামজারির আবেদন") == 0.5
    assert lexical_coverage([], "x") == 0.0


def test_filter_clause_uses_bind_params_only():
    params = {}
    sql = filter_clause(SearchFilters(source_types=["ebook"], doc_types=["আইন'; DROP TABLE x;--"], year=2020),
                        params)
    assert "DROP" not in sql and ":f_doc_types" in sql and ":f_year" in sql
    assert params["f_doc_types"] == ["আইন'; DROP TABLE x;--"]
    assert filter_clause(None, {}) == ""


# --- structural lookup (pure ranking) ------------------------------------------------------
import json  # noqa: E402

from app.core.text import query_terms  # noqa: E402
from app.retrieval.search import rank_section_rows  # noqa: E402


def _row(cid, doc, idx, title, meta=None, content="x"):
    return {"id": cid, "document_id": doc, "chunk_index": idx, "title": title, "content": content,
            "metadata": json.dumps(meta or {}, ensure_ascii=False)}


def test_lookup_picks_act_by_document_title_not_heading():
    rows = [_row(1, 1, 5, "রাষ্ট্রীয় অধিগ্রহণ ও প্রজাস্বত্ব আইন, ১৯৫০"),
            _row(2, 2, 3, "অর্পিত সম্পত্তি প্রত্যর্পণ আইন, ২০০১"),
            _row(3, 3, 1, "রাষ্ট্রীয় অধিগ্রহণ ও প্রজাস্বত্ব (সংশোধন) আইন, ১৯৫১ এর ব্যাখ্যা")]
    terms = [t for t in query_terms("রাষ্ট্রীয় অধিগ্রহণ ও প্রজাস্বত্ব আইন, ১৯৫০") if not t.isdigit()]
    hits = rank_section_rows(rows, terms, 6, year=1950)
    assert [h.chunk_id for h in hits][:1] == [1]


def test_lookup_year_is_only_a_tiebreaker():
    # same title coverage, the stored `year` field is irrelevant: the title year decides
    rows = [_row(1, 1, 1, "ভূমি উন্নয়ন কর আইন, ২০২৩"), _row(2, 2, 1, "ভূমি উন্নয়ন কর আইন, ১৯৭৬")]
    terms = [t for t in query_terms("ভূমি উন্নয়ন কর আইন") if not t.isdigit()]
    assert rank_section_rows(rows, terms, 6, year=1976)[0].chunk_id == 2
    # no act is excluded just because the query's year is absent from its title
    assert rank_section_rows([rows[0]], terms, 6, year=1999)[0].chunk_id == 1


def test_lookup_ambiguous_title_returns_nothing():
    rows = [_row(i, i, 1, f"ভূমি আইন {i}") for i in range(1, 5)]
    assert rank_section_rows(rows, ["ভূমি:*", "আইন"], 6) == []


def test_lookup_subsection_first_and_whole_provision_returned():
    t = "ভূমি আইন, ২০২৩"
    rows = [_row(10, 1, 4, t, {"subsection_numbers": ["(১)", "(২)"]}, "(১) ক\n(২) খ"),
            _row(11, 1, 5, t, {"subsection_numbers": ["(৩)", "(৪)"]}, "(৩) গ\n(৪) ঘ"),
            _row(12, 1, 6, t, {"subsection_numbers": ["(৫)"]}, "(৫) ঙ")]
    terms = ["ভূমি:*", "আইন"]
    assert [h.chunk_id for h in rank_section_rows(rows, terms, 6, subsection="4")] == [11, 10, 12]
    assert [h.chunk_id for h in rank_section_rows(rows, terms, 6, subsection="5")][0] == 12
    # no subsection requested: document order
    assert [h.chunk_id for h in rank_section_rows(rows, terms, 6)] == [10, 11, 12]


def test_lookup_clause_found_in_content():
    t = "ভূমি আইন, ২০২৩"
    rows = [_row(1, 1, 1, t, {}, "(ক) প্রথম"), _row(2, 1, 2, t, {}, "দফা (খ) দ্বিতীয়\n(গ) তৃতীয়")]
    assert rank_section_rows(rows, ["ভূমি:*", "আইন"], 6, clause="গ")[0].chunk_id == 2


def test_overview_chunks_are_demoted_unless_query_is_about_the_act():
    from app.retrieval.query import analyze_query
    s = make_settings()
    ov = Candidate(1, fused=1.0, row={"authority": 0, "chunk_metadata": {"role": "overview"}})
    sec = Candidate(2, fused=0.9, row={"authority": 0, "chunk_metadata": {}})
    apply_boosts([ov, sec], analyze_query("ভূমি আইনের ধারা ৩ কী"), s)
    assert sec.fused > ov.fused
    ov2 = Candidate(1, fused=1.0, row={"authority": 0, "chunk_metadata": {"role": "overview"}})
    apply_boosts([ov2], analyze_query("এই আইনের সূচি দেখাও"), s)
    assert ov2.fused == 1.0


def test_year_in_query_does_not_gate_the_section_boost():
    from app.retrieval.query import analyze_query
    s = make_settings()
    c = Candidate(1, fused=1.0, row={"authority": 0, "section_number": "5", "year": 1951, "chunk_metadata": {}})
    apply_boosts([c], analyze_query("আইন, ১৯৫০ এর ধারা ৫"), s)
    assert c.fused > 1.0
