from __future__ import annotations

import numpy as np


def build_vignette_mask(width: int, height: int, fraction: float) -> np.ndarray:
    """Precompute once per render — depends only on width/height/fraction."""
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    max_r = np.hypot(cx, cy)
    r = np.hypot(xx - cx, yy - cy) / max_r
    mask = 1.0 - fraction * (r**2)
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def apply_vignette(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def apply_grain(frame: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return frame
    noise = rng.normal(0.0, strength, size=frame.shape[:2])[:, :, None]
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
