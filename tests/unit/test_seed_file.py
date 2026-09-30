"""Guard: the committed prebuilt index must match the code that will load it.

If this fails you changed processing (PIPELINE_VERSION / chunking / exclusion settings)
without refreshing the seed. Re-sync, then run:  python -m app.cli export-index
Otherwise every fresh clone/server would reprocess the corpus on first start.
"""

from pathlib import Path

import pytest

from app.ingestion.hasher import PIPELINE_VERSION
from app.ingestion.seed import FORMAT_VERSION, processing_signature, read_manifest
from tests.conftest import make_settings

SEED = Path(__file__).resolve().parents[2] / "seed" / "bhumipedia-index.tar.gz"


@pytest.mark.skipif(not SEED.exists(), reason="no committed seed")
def test_committed_seed_matches_current_code():
    m = read_manifest(SEED)
    s = make_settings()
    assert m["format_version"] == FORMAT_VERSION
    assert m["pipeline_version"] == PIPELINE_VERSION, "seed is stale: run `python -m app.cli export-index`"
    assert m["processing_signature"] == processing_signature(s), "seed built with other chunking settings"
    assert m["embedding_model"] == s.embedding_model and m["embedding_dim"] == s.embedding_dim
    assert m["rows"]["document_chunks"] > 0
