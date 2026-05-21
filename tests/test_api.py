from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from scripts.make_sample import make_sample

client = TestClient(app)


def test_analyze_endpoint(tmp_path):
    sample = make_sample(tmp_path / "api.tif", size=160)
    with open(sample, "rb") as fh:
        resp = client.post(
            "/v1/analyze",
            files={"file": ("api.tif", fh, "image/tiff")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_version"] == "ndvi-deterministic-v0"
    assert "ndvi_mean" in body["metrics"]
    assert body["overlay_url"].endswith("ndvi_overlay.png")
    assert isinstance(body["stress_zones"], list)


def test_index_page_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AeroAtlas" in resp.text
