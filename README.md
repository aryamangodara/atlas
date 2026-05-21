# AeroAtlas POC — NDVI Crop-Stress API

The fastest slice of the [architecture plan](ARCHITECTURE_PLAN.md) (§13): prove
*raw multispectral imagery in → refined crop-stress intelligence out, via an API*.

**Deterministic NDVI — no ML model, no GPU, no training data.** This is the
"intelligence needs no model" insight from Principle 6: NDVI is band arithmetic.

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
  "crs": "EPSG:32643"
}
```
…plus a colour-coded NDVI overlay PNG (red = stressed, green = healthy).

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

## Band layout
Defaults assume a Micasense-style 5-band stack: `1=Blue 2=Green 3=Red 4=NIR 5=RedEdge`.
Override per request: `-F red_band=3 -F nir_band=4 -F stress_threshold=0.30`.

## Tests
```powershell
pytest -q
```

## Deliberately out of scope (see §13 "POC → MVP path")
OpenDroneMap / SfM, async jobs + webhooks, S3, multi-tenancy / auth, billing —
all deferred. The next step is the first fine-tuned **SegFormer** crop mask
running alongside NDVI.

## How this maps to the target architecture
| POC file | Target home | Layer |
|---|---|---|
| `app/ndvi.py` | `packages/inference` (deterministic-math path) | Layer 2 |
| `app/overlay.py`, `app/main.py` | `apps/api` | Layer 4 |
| `scripts/make_sample.py` | `ml/datasets` (test fixtures) | — |

The `model_version` field is already in the contract (Principle 2), so adding
real models later won't break clients.
