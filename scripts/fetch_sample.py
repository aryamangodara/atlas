"""Download a real Sentinel-2 clip over an Indian farm and stack it into a
ready-to-upload multispectral GeoTIFF for testing /v1/analyze.

No API key required: queries the public Element84 Earth Search STAC API and
reads the public Sentinel-2 L2A Cloud-Optimised GeoTIFFs on AWS directly (only
the requested window is fetched, via COG range requests).

Output band order matches the POC defaults (1=Blue 2=Green 3=Red 4=NIR), so the
file works with /v1/analyze using the default red_band=3 / nir_band=4.

Usage:
    python scripts/fetch_sample.py                       # default: Punjab, India
    python scripts/fetch_sample.py --lat 30.9 --lon 75.85 --out data/s2_field.tif
    python scripts/fetch_sample.py --size 256 --max-cloud 5 --months 12

Falls back gracefully (with a clear message) if there is no internet or no
low-cloud scene is found — in which case use scripts/make_sample.py instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# Assets to pull, in output order -> 1=Blue 2=Green 3=Red 4=NIR. Each entry
# lists candidate STAC asset keys (Earth Search v1 names, with B0x fallbacks).
WANT = [
    ("blue", ["blue", "B02"]),
    ("green", ["green", "B03"]),
    ("red", ["red", "B04"]),
    ("nir", ["nir", "B08"]),
]


def _search(lat, lon, max_cloud, months, limit=20):
    end = dt.date.today()
    start = end - dt.timedelta(days=30 * months)
    d = 0.02  # ~2 km search box around the point
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon - d, lat - d, lon + d, lat + d],
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
    }
    req = urllib.request.Request(
        STAC_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        feats = json.load(resp).get("features", [])
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100.0))
    return feats


def _asset_href(item, candidates):
    assets = item["assets"]
    for key in candidates:
        if key in assets and "href" in assets[key]:
            return assets[key]["href"]
    raise KeyError(f"none of {candidates} found in assets {list(assets)[:10]}")


def fetch_sample(lat, lon, out_path, size=256, max_cloud=10.0, months=12) -> Path:
    out_path = Path(out_path)
    feats = _search(lat, lon, max_cloud, months)
    if not feats:
        raise RuntimeError(
            f"No Sentinel-2 scene with <{max_cloud}% cloud near ({lat}, {lon}) in the "
            f"last {months} months. Try a higher --max-cloud or a different point."
        )
    item = feats[0]
    cc = item["properties"].get("eo:cloud_cover")
    date = item["properties"].get("datetime", "")[:10]
    print(f"scene: {item['id']}  date={date}  cloud={cc}%")

    hrefs = {name: _asset_href(item, cands) for name, cands in WANT}

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        # All four bands are native 10 m on the same grid; define the window from
        # the red band, then read the identical window from each.
        with rasterio.open(hrefs["red"]) as ref:
            xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
            row, col = ref.index(xs[0], ys[0])
            size = min(size, ref.width, ref.height)
            half = size // 2
            col_off = min(max(0, col - half), ref.width - size)
            row_off = min(max(0, row - half), ref.height - size)
            win = Window(col_off, row_off, size, size)
            out_transform = ref.window_transform(win)
            out_crs = ref.crs

        bands = []
        for name, _ in WANT:
            with rasterio.open(hrefs[name]) as src:
                bands.append(src.read(1, window=win).astype("float32"))

    # Sentinel-2 L2A (processing baseline >= 04.00): reflectance = (DN - 1000) / 10000
    stack = np.clip((np.stack(bands) - 1000.0) / 10000.0, 0.0, 1.0).astype("float32")

    profile = dict(
        driver="GTiff", height=stack.shape[1], width=stack.shape[2], count=4,
        dtype="float32", crs=out_crs, transform=out_transform, compress="deflate",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        for i, desc in enumerate(["blue (B02)", "green (B03)", "red (B04)", "nir (B08)"], start=1):
            dst.set_band_description(i, desc)

    print(f"wrote {out_path}  ({stack.shape[2]}x{stack.shape[1]}px, 4 bands, {out_crs})")
    print("upload with the default bands (red_band=3, nir_band=4):")
    print(f'  curl.exe -X POST http://127.0.0.1:8000/v1/analyze -F "file=@{out_path.as_posix()}"')
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch a Sentinel-2 test clip as a stacked GeoTIFF.")
    p.add_argument("--lat", type=float, default=30.90, help="latitude (default: Punjab, India)")
    p.add_argument("--lon", type=float, default=75.85, help="longitude")
    p.add_argument("--out", default="data/sentinel_sample.tif", help="output GeoTIFF path")
    p.add_argument("--size", type=int, default=256, help="output size in pixels (10 m/px)")
    p.add_argument("--max-cloud", type=float, default=10.0, help="max scene cloud cover %%")
    p.add_argument("--months", type=int, default=12, help="how far back to search")
    a = p.parse_args(argv)
    try:
        fetch_sample(a.lat, a.lon, a.out, a.size, a.max_cloud, a.months)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Tip: no internet or no clear scene? Use the synthetic sample instead:\n"
            "  python scripts/make_sample.py data/sample_field.tif",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
