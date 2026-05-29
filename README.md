# AeroAtlas POC — Crop Intelligence API (NDVI + segmentation)

The fastest slice of the [architecture plan](ARCHITECTURE_PLAN.md) (§13): prove
*raw multispectral imagery in → refined crop-stress intelligence out, via an API*.

**NDVI runs as deterministic math — no ML model, no GPU, no training data** (Principle 6:
NDVI is band arithmetic). A *pluggable* crop-mask segmentation layer runs alongside it,
defaulting to a heuristic backend so the core stays dependency-light.

## What it does
`POST /v1/analyze` with a multispectral GeoTIFF returns:
```json
{
  "flight_id": "flight-ab12cd34",
  "metrics": { "ndvi_mean": 0.62, "stressed_area_pct": 6.1 },
  "stress_zones": [
    { "geometry": { "type": "Polygon", "coordinates": [ ... ] }, "severity": "high", "area_pixels": 393 }
  ],
  "overlay_url": "/results/flight-ab12cd34/ndvi_overlay.png",
  "model_version": "ndvi-deterministic-v0",
  "crs": "EPSG:32643",
  "segmentation": {
    "backend": "VegetationMaskBackend",
    "model_version": "vegmask-ndvi-heuristic-v0",
    "classes": { "0": "non_crop", "1": "crop", "2": "dense_canopy" },
    "coverage_pct": { "non_crop": 1.5, "crop": 4.5, "dense_canopy": 94.0 },
    "mask_url": "/results/flight-ab12cd34/segmentation.png"
  }
}
```
…plus a colour-coded NDVI overlay PNG (red = stressed, green = healthy) and a class-coloured crop-mask PNG.

## Quick start (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. make a sample multispectral GeoTIFF (or bring your own)
python scripts\make_sample.py data\sample_field.tif

# 2. run the API
uvicorn app.main:app --reload

# 3a. open the browser demo:  http://127.0.0.1:8000/
# 3b. or call it directly:
curl.exe -X POST http://127.0.0.1:8000/v1/analyze -F "file=@data/sample_field.tif"
```
macOS/Linux: `source .venv/bin/activate` and use forward slashes.

## Test data
- **Synthetic (offline, instant):** `python scripts\make_sample.py data\sample_field.tif`
- **Real Sentinel-2 over an Indian farm (needs internet, no API key):**
  ```powershell
  python scripts\fetch_sample.py --out data\sentinel_sample.tif       # default: Punjab
  python scripts\fetch_sample.py --lat 17.4 --lon 78.5 --max-cloud 5   # anywhere
  ```
  Pulls a low-cloud clip from the public Earth Search STAC + AWS open COGs and
  stacks Blue/Green/Red/NIR, so it uploads with the default bands. Ordinary RGB
  photos won't work — NDVI needs a near-infrared band.

## Band layout
Defaults assume a Micasense-style 5-band stack: `1=Blue 2=Green 3=Red 4=NIR 5=RedEdge`.
Override per request: `-F red_band=3 -F nir_band=4 -F stress_threshold=0.30`.

## Tests
```powershell
pytest -q
```

## Crop-mask segmentation (pluggable)
A `segmentation` section runs alongside NDVI. The backend is pluggable: by default a
deterministic NDVI-heuristic classifier (no ML deps); set `SEGFORMER_CHECKPOINT` in
`app/config.py` and `pip install -r requirements-ml.txt` (torch + transformers) to
activate a real SegFormer. A fine-tuned crop checkpoint drops into the same slot later.

## Deliberately out of scope (see §13 "POC → MVP path")
OpenDroneMap / SfM, async jobs + webhooks, S3, multi-tenancy / auth, billing — all
deferred. The next modeling step is a SegFormer **fine-tuned on labeled Indian crop
data** (Month 1–3), which replaces the checkpoint behind the same contract.

## How this maps to the target architecture
| POC file | Target home | Layer |
|---|---|---|
| `app/ndvi.py` | `packages/inference` (deterministic-math path) | Layer 2 |
| `app/overlay.py`, `app/main.py` | `apps/api` | Layer 4 |
| `scripts/make_sample.py` | `ml/datasets` (test fixtures) | — |

The `model_version` field is already in the contract (Principle 2), so adding
real models later won't break clients.
