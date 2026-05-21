"""Generate a synthetic georeferenced 5-band multispectral GeoTIFF for testing.

Band order matches the POC defaults: 1=Blue 2=Green 3=Red 4=NIR 5=RedEdge.
Produces a mostly-healthy field with one 'medium' and one 'high' stress patch,
so /v1/analyze returns a clear, recognisable result out of the box.

Usage:
    python scripts/make_sample.py [output.tif]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def make_sample(out_path, size: int = 512) -> Path:
    out_path = Path(out_path)
    rng = np.random.default_rng(42)
    h = w = int(size)

    # Healthy vegetation: low red, high NIR -> NDVI ~0.7
    red = np.full((h, w), 0.08, dtype="float32") + rng.normal(0, 0.008, (h, w)).astype("float32")
    nir = np.full((h, w), 0.45, dtype="float32") + rng.normal(0, 0.015, (h, w)).astype("float32")

    ys, xs = np.ogrid[:h, :w]

    # Medium-stress patch: NDVI ~0.25
    cx, cy, r = w * 0.35, h * 0.40, size * 0.12
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    red[m], nir[m] = 0.13, 0.22

    # High-stress patch: NDVI ~ -0.17
    cx2, cy2, r2 = w * 0.70, h * 0.65, size * 0.07
    m2 = (xs - cx2) ** 2 + (ys - cy2) ** 2 <= r2 * r2
    red[m2], nir[m2] = 0.14, 0.10

    blue = np.full((h, w), 0.05, dtype="float32")
    green = np.full((h, w), 0.10, dtype="float32")
    rededge = (nir + red) / 2.0

    bands = np.clip(np.stack([blue, green, red, nir, rededge]), 0.0, 1.0).astype("float32")

    # Georeference: UTM 43N (EPSG:32643), 0.1 m/px, over central India.
    transform = from_origin(700000.0, 2_000_000.0, 0.1, 0.1)
    profile = dict(
        driver="GTiff", height=h, width=w, count=5, dtype="float32",
        crs="EPSG:32643", transform=transform, compress="deflate",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands)
    print(f"wrote {out_path}  ({w}x{h}px, 5 bands, EPSG:32643)")
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_field.tif")
    make_sample(target)
