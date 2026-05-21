"""Crop-mask segmentation — the pluggable inference layer (architecture plan, Layer 2).

Two backends behind one interface:
  * VegetationMaskBackend — deterministic NDVI-heuristic crop/non-crop classifier.
    The default; needs no ML deps so the POC runs today.
  * SegFormerBackend      — a real, fine-tunable SegFormer via HuggingFace
    `transformers`. Lazy-imports torch/transformers; activates only when a
    checkpoint is configured AND torch is installed. This is the slot a
    fine-tuned crop model drops into later, with no client-contract change —
    each result carries its own model_version (Principle 2).

Large orthomosaics are processed with tiled inference (`run_tiled`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np
import rasterio

from . import config
from .ndvi import compute_ndvi

logger = logging.getLogger("aeroatlas.segmentation")


@dataclass
class BandContext:
    """A (multi-band) image tile plus the 1-indexed bands each backend needs."""

    array: np.ndarray   # (C, H, W) float32
    red: int
    nir: int
    green: int
    blue: int

    def band(self, idx: int) -> np.ndarray:
        return self.array[idx - 1]


class SegmentationBackend(Protocol):
    model_version: str
    classes: Dict[int, str]

    def predict(self, ctx: BandContext) -> np.ndarray:
        """Return a 2-D int label mask (H, W) for the given tile."""
        ...


class VegetationMaskBackend:
    """Deterministic crop/non-crop classifier from NDVI thresholds.

    A stand-in for the 'crop segmentation / class' task until a fine-tuned
    SegFormer is available; honest about being a heuristic via its model_version.
    """

    model_version = "vegmask-ndvi-heuristic-v0"
    classes = {0: "non_crop", 1: "crop", 2: "dense_canopy"}

    crop_min = 0.20     # NDVI >= this -> at least 'crop'
    dense_min = 0.50    # NDVI >= this -> 'dense_canopy'

    def predict(self, ctx: BandContext) -> np.ndarray:
        red = ctx.band(ctx.red).astype("float32")
        nir = ctx.band(ctx.nir).astype("float32")
        ndvi = compute_ndvi(red, nir, np.ones(red.shape, dtype=bool))
        labels = np.zeros(red.shape, dtype="int16")   # 0 = non_crop
        labels[ndvi >= self.crop_min] = 1             # crop
        labels[ndvi >= self.dense_min] = 2            # dense canopy
        return labels


class SegFormerBackend:
    """Real SegFormer backend via HuggingFace transformers (lazy-loaded).

    Raises at construction if torch/transformers/the checkpoint are unavailable,
    so the factory can fall back. Operates on a pseudo-RGB built from the
    blue/green/red bands.
    """

    def __init__(self, checkpoint: str):
        # Lazy imports keep torch/transformers optional for the core POC.
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self._torch = torch
        self.checkpoint = checkpoint
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).eval()
        self.model_version = f"segformer:{checkpoint}"
        self.classes = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, ctx: BandContext) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        r, g, b = ctx.band(ctx.red), ctx.band(ctx.green), ctx.band(ctx.blue)
        rgb = np.stack([r, g, b], axis=-1).astype("float32")
        if float(np.nanmax(rgb)) <= 1.5:   # reflectance in [0,1] -> 0-255
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype("uint8")

        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1)[0].cpu().numpy().astype("int16")


def get_segmentation_backend() -> SegmentationBackend:
    """Pick SegFormer if configured + importable, else the heuristic fallback."""
    ckpt = config.SEGFORMER_CHECKPOINT
    if ckpt:
        try:
            backend = SegFormerBackend(ckpt)
            logger.info("Using SegFormer backend: %s", ckpt)
            return backend
        except Exception as exc:  # torch/transformers missing or load failed
            logger.warning("SegFormer unavailable (%s); falling back to NDVI heuristic", exc)
    return VegetationMaskBackend()


def run_tiled(
    backend: SegmentationBackend,
    ctx: BandContext,
    tile: int,
    overlap: int,
) -> np.ndarray:
    """Tiled inference over a large image, with overlap and full coverage.

    Stitching is last-wins on overlaps (exact for per-pixel backends; for a
    learned model it is the simple baseline — probability blending is the
    scale-up improvement).
    """
    _, H, W = ctx.array.shape
    if tile <= 0 or (H <= tile and W <= tile):
        return backend.predict(ctx).astype("int16")

    stride = max(1, tile - overlap)
    rows = sorted(set(list(range(0, max(1, H - tile + 1), stride)) + [max(0, H - tile)]))
    cols = sorted(set(list(range(0, max(1, W - tile + 1), stride)) + [max(0, W - tile)]))

    out = np.full((H, W), -1, dtype="int16")
    for r0 in rows:
        for c0 in cols:
            r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
            sub = BandContext(ctx.array[:, r0:r1, c0:c1], ctx.red, ctx.nir, ctx.green, ctx.blue)
            out[r0:r1, c0:c1] = backend.predict(sub).astype("int16")
    return out


@dataclass
class SegmentationResult:
    labels: np.ndarray
    classes: Dict[int, str]
    coverage_pct: Dict[str, float]
    model_version: str
    backend: str


def segment_raster(
    path,
    backend: Optional[SegmentationBackend] = None,
    red_band: Optional[int] = None,
    nir_band: Optional[int] = None,
) -> SegmentationResult:
    """Run tiled segmentation on a GeoTIFF and summarise per-class coverage."""
    backend = backend or get_segmentation_backend()
    red_band = red_band or config.DEFAULT_RED_BAND
    nir_band = nir_band or config.DEFAULT_NIR_BAND

    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        nodata = src.nodata

    ctx = BandContext(arr, red_band, nir_band, config.GREEN_BAND, config.BLUE_BAND)
    labels = run_tiled(backend, ctx, config.TILE_SIZE, config.TILE_OVERLAP)

    red, nir = arr[red_band - 1], arr[nir_band - 1]
    valid = ~((red == 0) & (nir == 0))
    if nodata is not None:
        valid &= (red != nodata) & (nir != nodata)
    labels[~valid] = -1

    n_valid = int(valid.sum())
    present = [int(c) for c in np.unique(labels) if c >= 0]
    classes = {c: backend.classes.get(c, str(c)) for c in present}
    coverage = (
        {classes[c]: round(100.0 * int((labels == c).sum()) / n_valid, 2) for c in present}
        if n_valid
        else {}
    )

    return SegmentationResult(
        labels=labels,
        classes=classes,
        coverage_pct=coverage,
        model_version=backend.model_version,
        backend=backend.__class__.__name__,
    )
