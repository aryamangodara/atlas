# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is (read this first)

This repo is **two things at different scopes**, and conflating them causes mistakes:

1. **`ARCHITECTURE_PLAN.md`** — the authoritative design for the full AeroAtlas platform: a 4-layer, 3-tier B2B SaaS for drone-data intelligence. **Most of it is intentionally NOT built yet.** Treat it as the source of truth for design decisions, tech-stack choices, and *what is deliberately deferred* (see its §6 defer list and §13 "POC → MVP path").
2. **The actual code (`app/`, `scripts/`, `tests/`)** — a minimal, working POC implementing only the **§13 "fastest path" slice**: deterministic NDVI crop-stress analysis exposed as one API endpoint. No database, no async jobs, no auth, no ML model.

When asked to extend the system, first check `ARCHITECTURE_PLAN.md` to see whether the feature is a planned layer and how it's meant to fit — don't invent architecture that contradicts the plan.

## Environment

Windows, Python **3.14**, virtualenv at `.venv` (already created and populated). Geospatial wheels (rasterio 1.5, numpy 2.4, shapely 2.1, Pillow 12) install from PyPI — no conda / system GDAL needed.

## Commands

```powershell
# setup (already done once; redo only if .venv is missing)
python -m venv .venv
.venv\Scripts\Activate.ps1            # bash: source .venv/Scripts/activate
pip install -r requirements.txt

python scripts\make_sample.py data\sample_field.tif   # synthetic test GeoTIFF
uvicorn app.main:app --reload                          # serve API + demo page at http://127.0.0.1:8000/
pytest -q                                              # full suite
pytest tests/test_ndvi.py::test_compute_ndvi_basic     # single test
```

Manual endpoint check: `curl.exe -X POST http://127.0.0.1:8000/v1/analyze -F "file=@data/sample_field.tif"`

## POC request flow

`POST /v1/analyze` (multipart GeoTIFF) is the whole product surface. The path across files:

- **`app/main.py`** — saves the upload to `uploads/`, orchestrates the pipeline, serves the JSON response and the browser demo page (`/`). Overlays are written to `results/{flight_id}/` and served via a `StaticFiles` mount at `/results`.
- **`app/ndvi.py`** (`analyze_geotiff`) — the "intelligence" engine. rasterio reads the Red/NIR bands → `NDVI = (NIR-Red)/(NIR+Red)` → thresholds into high/medium stress masks → vectorises to GeoJSON polygons (`rasterio.features.shapes` + shapely, reprojected to EPSG:4326). **Deterministic math, no model** (architecture plan, Principle 6).
- **`app/overlay.py`** — renders NDVI as an RdYlGn PNG (Pillow + numpy, transparent nodata).
- **`app/schemas.py`** — the Pydantic response contract returned to clients.
- **`app/config.py`** — band defaults, stress thresholds, paths, `MODEL_VERSION`.

## Conventions and non-obvious invariants

- **The response always carries `model_version`** (currently `"ndvi-deterministic-v0"`) even though v0 is pure math. This is deliberate (Principle 2): when real models are added later, the client contract must not break. Keep this field; version it.
- **Deterministic computations stay model-free.** NDVI (and, later, volumetrics / change detection) are arithmetic — do not route them through an ML model. ML is reserved for *perception* tasks (segmentation/detection); see the architecture plan §3. The planned next step is a fine-tuned SegFormer crop mask running *alongside* NDVI behind the same contract.
- **Band layout assumption:** defaults are a Micasense-style 5-band stack `1=Blue 2=Green 3=Red 4=NIR 5=RedEdge`. Callers override per request via the `red_band` / `nir_band` / `stress_threshold` form fields.
- **`conftest.py` at the repo root** exists solely to put the root on `sys.path` so `import app` / `import scripts` resolve under pytest.
- **`scripts/make_sample.py`** generates a *seeded* (rng=42) synthetic field — healthy base plus one medium and one high stress patch — so `/v1/analyze` yields a recognisable result (≈0.66 mean NDVI, ≈6% stressed, 2 zones) with no real data on hand. `tests/test_api.py` uses FastAPI's `TestClient` (requires `httpx`).
- **`results/`, `uploads/`, and `data/*.tif` are gitignored** runtime artifacts.
