"""Deterministic NDVI crop-stress analysis — the POC's 'intelligence' engine.

No ML model: NDVI is band arithmetic (architecture plan, Principle 6). This
module reads a multispectral GeoTIFF, computes NDVI, and derives crop-stress
metrics plus vector stress zones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from shapely.geometry import shape

from . import config


@dataclass
class StressZone:
    geometry: Dict[str, Any]   # GeoJSON geometry
    severity: str              # "high" | "medium"
    area_pixels: int


@dataclass
class NDVIResult:
    ndvi: np.ndarray
    valid_mask: np.ndarray
    ndvi_mean: float
    stressed_area_pct: float
    zones: List[StressZone]
    crs: Optional[str]
    width: int
    height: int


def compute_ndvi(red: np.ndarray, nir: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red); NaN where invalid or denominator == 0."""
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    ndvi = np.full(red.shape, np.nan, dtype="float32")
    ok = valid & (denom != 0)
    ndvi[ok] = (nir[ok] - red[ok]) / denom[ok]
    return ndvi


def _polygonise(mask, severity, transform, crs, pixel_area) -> List[StressZone]:
    """Vectorise a boolean stress mask into GeoJSON stress zones."""
    zones: List[StressZone] = []
    if not mask.any():
        return zones
    mask_u8 = mask.astype("uint8")
    for geom, val in rio_shapes(mask_u8, mask=mask, transform=transform):
        if val != 1:
            continue
        area_px = int(round(shape(geom).area / pixel_area)) if pixel_area else 0
        if area_px < config.MIN_ZONE_PIXELS:
            continue
        out_geom = transform_geom(crs, "EPSG:4326", geom) if crs else geom
        zones.append(StressZone(geometry=out_geom, severity=severity, area_pixels=area_px))
    return zones


def analyze_geotiff(
    path,
    red_band: Optional[int] = None,
    nir_band: Optional[int] = None,
    high_max: Optional[float] = None,
    medium_max: Optional[float] = None,
) -> NDVIResult:
    """Run the full NDVI crop-stress pipeline on a multispectral GeoTIFF."""
    red_band = red_band or config.DEFAULT_RED_BAND
    nir_band = nir_band or config.DEFAULT_NIR_BAND
    high_max = config.HIGH_STRESS_MAX if high_max is None else high_max
    medium_max = config.MEDIUM_STRESS_MAX if medium_max is None else medium_max

    with rasterio.open(path) as src:
        need = max(red_band, nir_band)
        if src.count < need:
            raise ValueError(
                f"GeoTIFF has {src.count} band(s) but Red={red_band}/NIR={nir_band} "
                f"were requested. Pass red_band/nir_band matching your sensor."
            )
        red = src.read(red_band)
        nir = src.read(nir_band)
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
        nodata = src.nodata
        pixel_area = abs(transform.a * transform.e) or 1.0

    valid = np.ones(red.shape, dtype=bool)
    if nodata is not None:
        valid &= (red != nodata) & (nir != nodata)
    valid &= ~((red == 0) & (nir == 0))  # common implicit nodata

    ndvi = compute_ndvi(red, nir, valid)
    valid_ndvi = valid & ~np.isnan(ndvi)
    n_valid = int(valid_ndvi.sum())

    ndvi_mean = float(np.mean(ndvi[valid_ndvi])) if n_valid else 0.0
    stressed = valid_ndvi & (ndvi < medium_max)
    stressed_area_pct = (100.0 * int(stressed.sum()) / n_valid) if n_valid else 0.0

    high_mask = valid_ndvi & (ndvi < high_max)
    medium_mask = valid_ndvi & (ndvi >= high_max) & (ndvi < medium_max)

    zones = _polygonise(high_mask, "high", transform, crs, pixel_area)
    zones += _polygonise(medium_mask, "medium", transform, crs, pixel_area)
    zones.sort(key=lambda z: z.area_pixels, reverse=True)
    zones = zones[: config.MAX_ZONES]

    return NDVIResult(
        ndvi=ndvi,
        valid_mask=valid_ndvi,
        ndvi_mean=round(ndvi_mean, 4),
        stressed_area_pct=round(stressed_area_pct, 2),
        zones=zones,
        crs=str(crs) if crs else None,
        width=width,
        height=height,
    )
