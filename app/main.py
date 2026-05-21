"""AeroAtlas POC — FastAPI service.

POST /v1/analyze : multispectral GeoTIFF in -> crop-stress intelligence JSON out.
GET  /           : minimal browser demo (upload form).
GET  /results/...: serves generated overlay PNGs.
"""
from __future__ import annotations

import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .ndvi import analyze_geotiff
from .overlay import render_ndvi_overlay
from .schemas import AnalyzeResponse, Metrics, StressZone

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AeroAtlas POC — NDVI Crop-Stress API", version="0.1.0")
app.mount("/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    flight_id: str = Form(default=""),
    red_band: int = Form(default=config.DEFAULT_RED_BAND),
    nir_band: int = Form(default=config.DEFAULT_NIR_BAND),
    stress_threshold: float = Form(default=config.MEDIUM_STRESS_MAX),
):
    fid = flight_id.strip() or f"flight-{uuid.uuid4().hex[:8]}"
    safe_name = (file.filename or "upload.tif").replace("/", "_").replace("\\", "_")
    upload_path = config.UPLOADS_DIR / f"{fid}_{safe_name}"
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = analyze_geotiff(
            upload_path,
            red_band=red_band,
            nir_band=nir_band,
            medium_max=stress_threshold,
        )
    except Exception as exc:  # surface a clean 422 to the client
        raise HTTPException(status_code=422, detail=f"Could not analyze GeoTIFF: {exc}")

    overlay_path = config.RESULTS_DIR / fid / "ndvi_overlay.png"
    render_ndvi_overlay(result.ndvi, result.valid_mask, overlay_path)

    return AnalyzeResponse(
        flight_id=fid,
        metrics=Metrics(
            ndvi_mean=result.ndvi_mean,
            stressed_area_pct=result.stressed_area_pct,
        ),
        stress_zones=[
            StressZone(geometry=z.geometry, severity=z.severity, area_pixels=z.area_pixels)
            for z in result.zones
        ],
        overlay_url=f"/results/{fid}/ndvi_overlay.png",
        model_version=config.MODEL_VERSION,
        crs=result.crs,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AeroAtlas POC - NDVI Crop-Stress</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
  h1{font-size:1.4rem;margin-bottom:.2rem} .sub{color:#666;margin-top:0}
  form{margin:1.5rem 0;padding:1rem;border:1px solid #ddd;border-radius:8px}
  button{background:#0a7d3f;color:#fff;border:0;padding:.5rem 1rem;border-radius:6px;cursor:pointer;font-size:1rem}
  .row{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:1rem;align-items:flex-start}
  pre{background:#0f172a;color:#e2e8f0;padding:1rem;border-radius:8px;overflow:auto;flex:1;min-width:320px;font-size:.8rem}
  img{max-width:360px;border:1px solid #ccc;border-radius:8px}
  label{font-size:.8rem;color:#444;display:block;margin:.4rem 0 .1rem}
  input[type=number]{width:5rem}
</style>
</head>
<body>
  <h1>AeroAtlas - NDVI Crop-Stress (POC)</h1>
  <p class="sub">Upload a multispectral GeoTIFF, get crop-stress intelligence + an overlay. Deterministic NDVI, no model.</p>
  <form id="f">
    <input type="file" name="file" accept=".tif,.tiff" required>
    <div class="row">
      <div><label>Red band</label><input type="number" name="red_band" value="3"></div>
      <div><label>NIR band</label><input type="number" name="nir_band" value="4"></div>
      <div><label>Stress NDVI &lt;</label><input type="number" step="0.01" name="stress_threshold" value="0.30"></div>
    </div>
    <p><button type="submit">Analyze</button> <span id="status"></span></p>
  </form>
  <div class="row">
    <pre id="out">Awaiting upload...</pre>
    <div><img id="overlay" alt="" style="display:none"></div>
  </div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),
      img=document.getElementById('overlay'),st=document.getElementById('status');
f.addEventListener('submit',async e=>{
  e.preventDefault();st.textContent='analyzing...';img.style.display='none';
  try{
    const r=await fetch('/v1/analyze',{method:'POST',body:new FormData(f)});
    const j=await r.json();
    out.textContent=JSON.stringify(j,null,2);
    if(r.ok && j.overlay_url){img.src=j.overlay_url+'?t='+Date.now();img.style.display='block';}
    st.textContent=r.ok?'done':('error '+r.status);
  }catch(err){out.textContent=String(err);st.textContent='failed';}
});
</script>
</body>
</html>
"""
