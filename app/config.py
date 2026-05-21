"""Configuration for the AeroAtlas NDVI POC.

Deliberately simple module-level constants. The MVP replaces this with proper
settings + per-tenant config; for the POC, constants are enough.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"

# --- Band layout -----------------------------------------------------------
# Defaults assume a Micasense-style 5-band stack (1-indexed):
#   1=Blue  2=Green  3=Red  4=NIR  5=RedEdge
# Override per request via the /v1/analyze form fields.
DEFAULT_RED_BAND = 3
DEFAULT_NIR_BAND = 4

# --- NDVI stress thresholds ------------------------------------------------
# Typical vegetation NDVI: <0.2 bare/none, 0.2-0.4 sparse/stressed, >0.6 healthy.
HIGH_STRESS_MAX = 0.15    # NDVI below this        -> "high" severity
MEDIUM_STRESS_MAX = 0.30  # NDVI in [HIGH, MEDIUM) -> "medium" severity
# stressed_area_pct = % of valid pixels with NDVI < MEDIUM_STRESS_MAX

# --- Vectorisation limits --------------------------------------------------
MAX_ZONES = 50         # cap polygons returned in the payload
MIN_ZONE_PIXELS = 25   # drop speckle smaller than this

# --- Provenance ------------------------------------------------------------
# Present in every response from day zero (Principle 2), even though v0 is pure
# deterministic math, so the client contract never breaks when models arrive.
MODEL_VERSION = "ndvi-deterministic-v0"
