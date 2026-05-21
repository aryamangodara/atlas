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


class AnalyzeResponse(BaseModel):
    flight_id: str
    metrics: Metrics
    stress_zones: List[StressZone]
    overlay_url: str
    model_version: str
    crs: Optional[str] = Field(None, description="Source coordinate reference system")
