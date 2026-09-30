import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db.database import dispose_engine
from app.ingestion.runner import claim_next_queued_run
from app.main import create_app
from tests.conftest import make_settings
from tests.integration.conftest import TEST_DB_URL
from tests.unit.test_api import FakePipeline

pytestmark = pytest.mark.integration
ADMIN = {"X-Admin-API-Key": "test-admin-key"}


async def test_admin_queue_and_status(sm):
    await dispose_engine()  # bind the global engine to this test's event loop
    app = create_app(pipeline=FakePipeline(), warmup=False)
    s = make_settings(database_url=TEST_DB_URL)
    app.dependency_overrides[get_settings] = lambda: s
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/admin/sync", json={"source_types": ["qna_type2"], "force": True}, headers=ADMIN)
        assert r.status_code == 202 and r.json()["status"] == "queued"
        run_id = r.json()["run_id"]
        assert await claim_next_queued_run() == run_id  # what the worker does
        assert await claim_next_queued_run() is None  # claimed exactly once
        r = await c.get("/api/admin/ingestion-status", headers=ADMIN)
        assert r.status_code == 200
        run = r.json()["runs"][0]
        assert run["id"] == run_id and run["status"] == "running" and run["source_types"] == ["qna_type2"]
        assert "documents" in r.json()["counts"]
    await dispose_engine()
