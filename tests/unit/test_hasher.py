from app.ingestion.hasher import source_hash


def test_volatile_counters_do_not_change_hash():
    a = {"id": 1, "content": "x", "viewer_counter": 1, "like_user_counter": 2}
    b = {"id": 1, "content": "x", "viewer_counter": 999, "like_user_counter": 5}
    assert source_hash(a) == source_hash(b)


def test_content_and_config_change_hash():
    a = {"id": 1, "content": "x"}
    assert source_hash(a) != source_hash({"id": 1, "content": "y"})
    assert source_hash(a, "t500") != source_hash(a, "t400")


def test_key_order_is_irrelevant():
    assert source_hash({"a": 1, "b": [1, {"c": 2, "d": 3}]}) == source_hash({"b": [1, {"d": 3, "c": 2}], "a": 1})
