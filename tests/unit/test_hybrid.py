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
