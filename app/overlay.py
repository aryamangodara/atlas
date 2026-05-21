"""Render NDVI as a colour-mapped PNG overlay (RdYlGn-style ramp).

Low NDVI (stress) -> red, high NDVI (healthy) -> green. Invalid pixels are
fully transparent so the overlay can sit on a basemap. Uses Pillow + numpy
(no matplotlib) to keep dependencies light.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# RdYlGn-style colour stops, NDVI low -> high
_STOPS = [
    (0.00, (165, 0, 38)),
    (0.25, (244, 109, 67)),
    (0.50, (255, 255, 191)),
    (0.75, (166, 217, 106)),
    (1.00, (26, 152, 80)),
]


def _colormap(t: np.ndarray) -> np.ndarray:
    """Map normalised values t in [0,1] -> (H,W,3) uint8 via a piecewise-linear ramp."""
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros(t.shape + (3,), dtype="float32")
    for (p0, c0), (p1, c1) in zip(_STOPS[:-1], _STOPS[1:]):
        seg = (t >= p0) & (t <= p1)
        frac = (t[seg] - p0) / ((p1 - p0) or 1.0)
        for k in range(3):
            out[seg, k] = c0[k] + frac * (c1[k] - c0[k])
    return np.clip(out, 0, 255).astype("uint8")


def render_ndvi_overlay(
    ndvi: np.ndarray,
    valid_mask: np.ndarray,
    out_path,
    vmin: float = -0.1,
    vmax: float = 0.8,
) -> Path:
    """Write a colour-mapped NDVI PNG with transparent invalid pixels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm = (ndvi - vmin) / ((vmax - vmin) or 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    rgb = _colormap(norm)

    alpha = np.where(valid_mask, 255, 0).astype("uint8")
    rgba = np.dstack([rgb, alpha])

    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return out_path
