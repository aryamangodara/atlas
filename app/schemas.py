"""API response models — the client contract (architecture plan, §13)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Metrics(BaseModel):
    ndvi_mean: float = Field(..., description="Mean NDVI over valid pixels")
    stressed_area_pct: float = Field(
        ..., description="Percent of valid area below the stress NDVI threshold"
    )


class StressZone(BaseModel):
    geometry: Dict[str, Any] = Field(
        ..., description="GeoJSON geometry (EPSG:4326 when the source is georeferenced)"
    )
    severity: str = Field(..., description='"high" or "medium"')
    area_pixels: int


class CropSegmentation(BaseModel):
    model_version: str = Field(..., description="Provenance of the segmentation model/backend")
    backend: str = Field(..., description="Backend class that produced the mask")
    classes: Dict[int, str] = Field(..., description="Class id -> name for classes present in the scene")
    coverage_pct: Dict[str, float] = Field(..., description="Percent of valid area per class")
    mask_url: str = Field(..., description="URL of the class-coloured mask PNG")


class AnalyzeResponse(BaseModel):
    flight_id: str
    metrics: Metrics
    stress_zones: List[StressZone]
    overlay_url: str
    model_version: str
    crs: Optional[str] = Field(None, description="Source coordinate reference system")
    segmentation: Optional[CropSegmentation] = Field(
        None, description="Crop-mask segmentation (alongside NDVI); omitted if disabled/unavailable"
    )
