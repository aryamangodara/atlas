from __future__ import annotations

import numpy as np
import rasterio
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.segmentation import (
    BandContext,
    VegetationMaskBackend,
    run_tiled,
    segment_raster,
)
from scripts.make_sample import make_sample

client = TestClient(app)


def _ctx_from(path) -> BandContext:
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
    return BandContext(
        arr, config.DEFAULT_RED_BAND, config.DEFAULT_NIR_BAND,
        config.GREEN_BAND, config.BLUE_BAND,
    )


def test_fallback_backend_classes_and_shape(tmp_path):
    sample = make_sample(tmp_path / "s.tif", size=128)
    ctx = _ctx_from(sample)
    labels = VegetationMaskBackend().predict(ctx)
    assert labels.shape == ctx.array.shape[1:]
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert (labels == 2).any()   # healthy field -> dense canopy present


def test_tiled_equals_single_pass(tmp_path):
    # For a per-pixel backend, tiled inference must EXACTLY equal a single pass:
    # proves the stitcher covers every pixel with no gaps or seams.
    sample = make_sample(tmp_path / "s.tif", size=200)
    ctx = _ctx_from(sample)
    backend = VegetationMaskBackend()
    single = backend.predict(ctx).astype("int16")
    tiled = run_tiled(backend, ctx, tile=64, overlap=16)
    assert np.array_equal(single, tiled)


def test_segment_raster_coverage_and_nodata(tmp_path):
    sample = make_sample(tmp_path / "s.tif", size=160)
    res = segment_raster(sample)
    assert res.backend == "VegetationMaskBackend"
    assert res.model_version.startswith("vegmask")
    assert -1 not in res.classes                       # nodata is not a class
    assert abs(sum(res.coverage_pct.values()) - 100.0) < 1.0  # ~all valid area covered


def test_api_includes_segmentation(tmp_path):
    sample = make_sample(tmp_path / "api.tif", size=160)
    with open(sample, "rb") as fh:
        resp = client.post("/v1/analyze", files={"file": ("api.tif", fh, "image/tiff")})
    assert resp.status_code == 200, resp.text
    seg = resp.json()["segmentation"]
    assert seg is not None
    assert seg["mask_url"].endswith("segmentation.png")
    assert seg["coverage_pct"]
