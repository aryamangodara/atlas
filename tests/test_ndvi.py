from __future__ import annotations

import numpy as np

from app.ndvi import analyze_geotiff, compute_ndvi
from scripts.make_sample import make_sample


def test_compute_ndvi_basic():
    red = np.array([[0.1, 0.2]], dtype="float32")
    nir = np.array([[0.5, 0.2]], dtype="float32")
    valid = np.ones((1, 2), dtype=bool)
    ndvi = compute_ndvi(red, nir, valid)
    assert abs(float(ndvi[0, 0]) - (0.4 / 0.6)) < 1e-4
    assert abs(float(ndvi[0, 1]) - 0.0) < 1e-6


def test_compute_ndvi_zero_denominator_is_nan():
    red = np.zeros((1, 1), dtype="float32")
    nir = np.zeros((1, 1), dtype="float32")
    valid = np.ones((1, 1), dtype=bool)
    ndvi = compute_ndvi(red, nir, valid)
    assert np.isnan(ndvi[0, 0])


def test_analyze_sample_end_to_end(tmp_path):
    sample = make_sample(tmp_path / "s.tif", size=160)
    res = analyze_geotiff(sample)
    assert res.ndvi_mean > 0.3                 # mostly healthy field
    assert 0.0 <= res.stressed_area_pct <= 100.0
    assert res.stressed_area_pct > 0.0         # the stress patches register
    assert len(res.zones) >= 1                 # at least one stress polygon
    assert res.crs and "32643" in res.crs      # georeferencing preserved
