import json

import pytest
from langchain_core.documents import Document

from app.evaluation.benchmark import load_dataset, retrieval_metrics


def _d(doc_type, doc_id, el_type=None, el_id=None):
    return Document(page_content="x", metadata={"doc_source_type": doc_type, "doc_source_id": doc_id,
                                                "source_type": el_type or doc_type, "source_id": el_id or doc_id})


def test_metrics_doc_and_element_level():
    docs = [_d("qna_type2", "1"), _d("ebook", "9", "section", "55"), _d("qna_type2", "2")]
    m = retrieval_metrics(docs, {"section:55", "qna_type2:2"}, ks=[1, 3])
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@3"] == 1.0 and m["recall@1"] == 0.0
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx(0.5)


def test_no_relevant_results():
    m = retrieval_metrics([_d("blog", "1")], {"qna_type2:9"}, ks=[5])
    assert m["mrr"] == 0.0 and m["hit@5"] == 0.0 and m["precision@5"] == 0.0


def test_curated_dataset_is_well_formed():
    rows = load_dataset("evaluation/curated.jsonl")
    assert len(rows) >= 20 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        for sid in r["expected_source_ids"]:
            t, _, i = sid.partition(":")
            assert t in {"ebook", "qna_type1", "qna_type2", "blog", "topic", "section", "subsection", "schedule",
                         "subschedule"} and i, sid


def test_load_json_array(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"question": "q", "expected_source_ids": []}]))
    assert load_dataset(p)[0]["id"] == "ex0"
