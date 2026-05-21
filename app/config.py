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

# --- Segmentation (crop-mask) backend --------------------------------------
# The segmentation layer is pluggable (see app/segmentation.py):
#   * SEGFORMER_CHECKPOINT empty -> deterministic NDVI-heuristic fallback (runs today)
#   * SEGFORMER_CHECKPOINT set    -> real SegFormer via transformers, IF torch is installed
#     ("nvidia/segformer-b0-finetuned-ade-512-512" smoke-tests the plumbing; a
#      fine-tuned crop checkpoint drops in once labeled Indian data exists, Month 1-3).
ENABLE_SEGMENTATION = True
SEGFORMER_CHECKPOINT = ""
GREEN_BAND = 2
BLUE_BAND = 1

# Tiled inference: orthomosaics are too large for a single forward pass.
TILE_SIZE = 512
TILE_OVERLAP = 64

# Colours for the heuristic fallback's classes (RGB); SegFormer uses an auto palette.
SEG_CLASS_COLORS = {
    0: (150, 110, 70),    # non_crop  (bare soil / water / built)
    1: (140, 200, 120),   # crop      (vegetation)
    2: (35, 132, 67),     # dense_canopy
}
