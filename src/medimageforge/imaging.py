"""Image inspection utilities: what a model will eventually see.

Why this module exists:
    Numbers in a report (Step 2) tell you files exist. Looking at pixels
    tells you what they *mean*. Three concepts live here:

      - CT windowing: the SAME slice saved twice — "brain" window stretches
        contrast over soft-tissue densities (hemorrhage is visible), "bone"
        window stretches it over dense bone (fractures are visible). One
        scan, two views optimized for different tissue.
      - Segmentation masks: a _HGE_Seg.jpg is a same-size image where the
        hemorrhage region is white and everything else is black. Saved as
        JPEG it is NOT perfectly binary — we threshold to recover it.
      - Pixel statistics: uint8 grayscale 0-255. Knowing the dtype and
        intensity range is required before any normalization in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    """Load an image as a uint8 grayscale array (H, W). Values 0-255."""
    return np.asarray(Image.open(path).convert("L"))


def mask_from_jpeg(mask_array: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Recover a boolean mask from a JPEG-compressed mask image.

    JPEG saves the white region with ringing artifacts at the edges, so
    the stored values are not exactly {0, 255}. Thresholding at the
    midpoint keeps the intended region and discards the artifacts.
    """
    return mask_array > threshold


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Return an RGB array with `color` blended over `image` where mask is True.

    The slice stays grayscale except inside the mask, so a correctly aligned
    mask appears as a red wash over the hemorrhage — the visual check that
    labels and pixels agree.
    """
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)
    return rgb.astype(np.uint8)


def side_by_side(*arrays: np.ndarray) -> np.ndarray:
    """Concatenate same-height images horizontally with a thin separator.

    Handles mixing grayscale (H, W) and RGB (H, W, 3) by promoting to RGB.
    """
    rgbs = [a if a.ndim == 3 else np.stack([a] * 3, axis=-1) for a in arrays]
    height = rgbs[0].shape[0]
    sep = np.full((height, 2, 3), 255, dtype=np.uint8)  # 2px white divider
    parts = []
    for i, a in enumerate(rgbs):
        if i:
            parts.append(sep)
        parts.append(a)
    return np.concatenate(parts, axis=1)


def pixel_stats(array: np.ndarray) -> dict:
    """Basic intensity statistics — the sanity check before normalization."""
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 2),
        "std": round(float(array.std()), 2),
    }


def find_masks(patient_brain_dir: Path, mask_suffix: str) -> list[Path]:
    """All mask files in a patient's brain/ folder, sorted by slice number."""
    masks = [p for p in patient_brain_dir.glob(f"*{mask_suffix}.jpg")]
    return sorted(masks, key=lambda p: int(p.name.split("_")[0]))
