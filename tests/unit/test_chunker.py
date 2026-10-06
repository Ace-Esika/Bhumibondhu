from app.ingestion.chunker import Chunker, ChunkingConfig
from app.ingestion.normalizer import NormalizedDocument, Unit, normalize
from tests.conftest import load_snapshot, make_settings


def _chunker(word_tokens, **cfg):
    return Chunker(word_tokens, ChunkingConfig(**{"target_tokens": 60, "max_tokens": 90, "overlap_tokens": 10,
                                                  "min_tokens": 5, **cfg}))


def _act(sections):
    return NormalizedDocument(source_type="ebook", source_id="9", title="পরীক্ষা ভূমি আইন, ২০২৫", doc_type="আইন",
                              year=2025, metadata={"act_year_raw": "২০২৫"}, units=sections, section_label="ধারা")


def test_small_section_is_one_chunk_with_parent_context(word_tokens):
    sec = Unit("section", "100", "9", "এই আইন পরীক্ষা ভূমি আইন নামে অভিহিত হইবে।", number="১। ", heading="সংক্ষিপ্ত শিরোনাম")
    chunks = _chunker(word_tokens).chunk(_act([sec]))
    overview, c = chunks
    assert overview.metadata["role"] == "overview"
    assert c.source_type == "section" and c.source_id == "100" and c.parent_source_id == "9"
    assert c.section_number == "1" and c.section_heading == "সংক্ষিপ্ত শিরোনাম"
    assert "আইনের নাম: পরীক্ষা ভূমি আইন, ২০২৫" in c.context
    assert "সাল: ২০২৫" in c.context and "ধারা: ১ — সংক্ষিপ্ত শিরোনাম" in c.context
    assert [x.chunk_index for x in chunks] == [0, 1]


def test_oversized_section_splits_on_subsection_boundaries(word_tokens):
    subs = [Unit("subsection", f"s{i}", "100", " ".join(["শব্দ"] * 40) + f" শেষ{i}", number=f"({i})")
            for i in range(1, 5)]
    sec = Unit("section", "100", "9", "এই আইনে—", number="২।", heading="সংজ্ঞা", children=subs)
    chunks = [c for c in _chunker(word_tokens).chunk(_act([sec])) if c.metadata.get("role") != "overview"]
    assert len(chunks) >= 2
    # No subsection is cut in half: each "শেষi" marker appears in exactly one chunk.
    for i in range(1, 5):
        assert sum(f"শেষ{i}" in c.content for c in chunks) == 1
    for c in chunks:
        assert "ধারা: ২ — সংজ্ঞা" in c.context and c.section_number == "2"
        assert c.token_count <= 90 + 5
    # The short lead-in gives child chunks their meaning.
    assert any("প্রারম্ভিক অংশ: এই আইনে—" in c.context for c in chunks)


def test_long_unstructured_text_respects_limits_and_overlaps(word_tokens):
    paras = [" ".join([f"w{p}_{i}" for i in range(25)]) + "।" for p in range(12)]
    doc = NormalizedDocument(source_type="ebook", source_id="7", title="পরিপত্র", doc_type="পরিপত্র",
                             units=[Unit("ebook_text", "7", None, "\n".join(paras), metadata={"role": "body"})])
    chunks = [c for c in _chunker(word_tokens).chunk(doc) if c.metadata.get("role") == "body"]
    assert len(chunks) > 2
    assert all(word_tokens(c.content) <= 90 for c in chunks)
    # Paragraphs (26 tokens) exceed the overlap budget (10): the trailing words of the
    # previous piece are carried over instead.
    carried = chunks[1].content.split("\n")[0]
    assert 0 < word_tokens(carried) <= 10
    assert chunks[0].content.endswith(carried)


def test_whole_short_paragraphs_are_carried_as_overlap(word_tokens):
    paras = [" ".join([f"w{p}_{i}" for i in range(7)]) for p in range(30)]
    doc = NormalizedDocument(source_type="blog", source_id="b", title="ব্লগ",
                             units=[Unit("blog", "b", None, "\n".join(paras))])
    chunks = _chunker(word_tokens).chunk(doc)
    assert len(chunks) > 2
    assert chunks[1].content.split("\n")[0] in chunks[0].content.split("\n")


def test_qna_stays_one_unit(word_tokens):
    doc = normalize("qna_type2", load_snapshot("qna_type2")[0], make_settings())[0]
    (chunk,) = Chunker(word_tokens, ChunkingConfig()).chunk(doc)
    assert chunk.source_type == "qna_type2" and chunk.content.startswith("প্রশ্ন:")
    assert "সরকারি প্রশ্নোত্তর" in chunk.context and "বিষয়শ্রেণি: namjari" in chunk.context


def test_real_ebook_all_chunks_within_limits(word_tokens):
    cfg = ChunkingConfig(target_tokens=120, max_tokens=180, overlap_tokens=20, min_tokens=10)
    ch = Chunker(word_tokens, cfg)
    for raw in load_snapshot("ebook"):
        for doc in normalize("ebook", raw, make_settings()):
            chunks = ch.chunk(doc)
            assert chunks, raw["id"]
            for c in chunks:
                assert c.content.strip()
                assert word_tokens(c.content) <= cfg.max_tokens, (raw["id"], c.source_type)


def test_hashes_are_deterministic(word_tokens):
    doc = normalize("ebook", load_snapshot("ebook")[0], make_settings())[0]
    a = [c.content_hash for c in _chunker(word_tokens).chunk(doc)]
    b = [c.content_hash for c in _chunker(word_tokens).chunk(doc)]
    assert a == b and len(set(a)) == len(a)


def test_long_preamble_never_pushes_the_table_of_contents_out(word_tokens):
    sections = [Unit("section", str(100 + i), "9", f"বিধান {i}", number=f"{i}।", heading=f"শিরোনাম {i}")
                for i in range(1, 41)]
    doc = _act(sections)
    doc.metadata["proposal"] = " ".join(["প্রস্তাবনার-শব্দ"] * 400)
    chunks = _chunker(word_tokens).chunk(doc)
    toc_text = "\n".join(c.content for c in chunks if c.metadata.get("role") in ("overview", "toc"))
    for i in range(1, 41):
        assert f"শিরোনাম {i}" in toc_text  # every heading survives, across continuation chunks
    assert "মোট ৪০টি" in toc_text
    assert any(c.metadata.get("role") == "preamble" for c in chunks)
    assert "প্রস্তাবনার-শব্দ" not in toc_text


def test_chunks_record_every_subsection_number(word_tokens):
    subs = [Unit("subsection", f"s{i}", "100", " ".join(["শব্দ"] * 40) + f" শেষ{i}", number=f"({i})")
            for i in range(1, 5)]
    sec = Unit("section", "100", "9", "এই আইনে—", number="২।", heading="সংজ্ঞা", children=subs)
    chunks = [c for c in _chunker(word_tokens).chunk(_act([sec])) if c.metadata.get("role") != "overview"]
    seen = [n for c in chunks for n in c.metadata["subsection_numbers"]]
    assert seen == ["(1)", "(2)", "(3)", "(4)"]
    small = Unit("section", "101", "9", "", number="৩।", heading="ক", children=[
        Unit("subsection", "t1", "101", "প্রথম", number="(১)"), Unit("subsection", "t2", "101", "দ্বিতীয়", number="(২)")])
    (c,) = [c for c in _chunker(word_tokens).chunk(_act([small])) if c.metadata.get("role") != "overview"]
    assert c.metadata["subsection_numbers"] == ["(১)", "(২)"]


def _manual(provisions, filler=30):
    lines = ["ভূমি আপীল বোর্ড আইন", *[f"{n}। শিরোনাম{n}ঃ " + " ".join([f"ক{n}_{i}" for i in range(filler)])
                                       for n in "১২৩৪৫৬৭৮৯"[:provisions]]]
    return NormalizedDocument(source_type="ebook", source_id="m", title="ম্যানুয়াল", doc_type="ম্যানুয়াল",
                              units=[Unit("ebook_text", "m", None, "\n".join(lines), metadata={"role": "body"})])


def test_manual_text_is_packed_on_provision_boundaries(word_tokens):
    chunks = [c for c in _chunker(word_tokens).chunk(_manual(9)) if c.metadata.get("role") == "body"]
    assert len(chunks) >= 3
    for c in chunks:
        # every provision that starts in a chunk is complete in it: each marker of a provision
        # appears exactly once across all chunks
        assert word_tokens(c.content) <= 90
    for n in "১২৩৪৫৬৭৮৯":
        assert sum(f"ক{n}_29" in c.content for c in chunks) == 1
        owner = next(c for c in chunks if f"ক{n}_0" in c.content)
        assert f"ক{n}_29" in owner.content  # not cut in half
    assert chunks[0].metadata["provisions"][0] == "১"
    for c in chunks:  # each chunk is labelled with the provision it starts in (or the heading before it)
        first = c.content.split("\n", 1)[0]
        if first[:2] in {f"{n}।" for n in "১২৩৪৫৬৭৮৯"}:
            assert f"প্রসঙ্গ: {first[:2]} শিরোনাম{first[0]}" in c.context
    assert all(c.section_number is None for c in chunks)  # manuals embed several acts: no false exact keys


def test_oversized_provision_is_split_and_keeps_its_title(word_tokens):
    chunks = [c for c in _chunker(word_tokens).chunk(_manual(4, filler=70)) if c.metadata.get("role") == "body"]
    parts = [c for c in chunks if "ক২_" in c.content]
    assert len(parts) >= 2
    assert all("প্রসঙ্গ: ২। শিরোনাম২" in c.context for c in parts)


def test_heading_heuristic_rejects_clauses_and_signatures():
    h = Chunker._is_heading
    assert h("ভূমি ব্যবস্থাপনা") and h("সংযোজনী 'খ'")
    assert not h("(ঘ) বোর্ডের চেয়ারম্যান ও সদস্যগণের নিয়োগ এবং তাহাদের অপসারণ:")
    assert not h("আদেশক্রমে") and not h("স্বা/- (মোঃ ফয়জুর রহমান)")
    assert not h("২৩। শাস্তি") and not h("৩০ জুন, ২০২৬\"")
