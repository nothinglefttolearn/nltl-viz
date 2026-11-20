"""The Image Subject: a user-supplied raster picture animated in place of a
Shape (see ADR 0004). Loading and per-size resampling live here; drawing
lives in `render.py` next to the Shape drawing it mirrors."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cairo
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps


def premultiplied_rgba_to_surface(rgba: np.ndarray) -> cairo.ImageSurface:
    """Copies an (h, w, 4) premultiplied RGBA uint8 array into a new
    cairo ARGB32 surface (premultiplied, BGRA in memory on little-endian,
    row stride possibly padded beyond w*4)."""
    h, w = rgba.shape[:2]
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(w, 1), max(h, 1))
    if w == 0 or h == 0:
        return surface
    stride = surface.get_stride()
    buf = np.ndarray(shape=(h, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data())
    buf[:, :w, :] = rgba[:, :, [2, 1, 0, 3]]
    surface.mark_dirty()
    return surface


class Image:
    """A loaded Image Subject. Holds the decoded picture once (EXIF
    orientation applied, first frame only for animated files, alpha kept) and
    hands out copies resampled to the contain-box size each render needs —
    cached, since a render reuses one or (under `Layout.grid`) two sizes for
    every frame."""

    def __init__(self, picture: PILImage.Image, path: Path | None = None) -> None:
        self.path = path
        self._picture = picture.convert("RGBA")
        self.width, self.height = self._picture.size
        if self.width == 0 or self.height == 0:
            raise ValueError(f"Image has no pixels: {path}")
        self._resized = lru_cache(maxsize=8)(self._resize)
        self._surfaces = lru_cache(maxsize=8)(self._surface)

    @classmethod
    def load(cls, path: Path) -> Image:
        try:
            with PILImage.open(path) as opened:
                opened.seek(0)
                picture = ImageOps.exif_transpose(opened).convert("RGBA")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read image {path}: {exc}") from exc
        return cls(picture, path)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def contain_box(self, side: float) -> tuple[int, int]:
        """Pixel size of this Image scaled to fit inside a `side`-square,
        aspect preserved (letterboxed along the shorter dimension)."""
        if self.aspect >= 1.0:
            return max(1, round(side)), max(1, round(side / self.aspect))
        return max(1, round(side * self.aspect)), max(1, round(side))

    def premultiplied(self, box_w: int, box_h: int) -> np.ndarray:
        """(box_h, box_w, 4) premultiplied RGBA uint8, resampled to exactly
        that size. Read-only — cached and shared between frames."""
        return self._resized(box_w, box_h)

    def surface(self, box_w: int, box_h: int) -> cairo.ImageSurface:
        """`premultiplied(box_w, box_h)` as a cairo surface, cached."""
        return self._surfaces(box_w, box_h)

    def _resize(self, box_w: int, box_h: int) -> np.ndarray:
        resized = self._picture.resize((box_w, box_h), PILImage.Resampling.LANCZOS)
        rgba = np.asarray(resized, dtype=np.uint16)
        alpha = rgba[:, :, 3:4]
        rgb = (rgba[:, :, :3] * alpha + 127) // 255
        out = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
        out.flags.writeable = False
        return out

    def _surface(self, box_w: int, box_h: int) -> cairo.ImageSurface:
        return premultiplied_rgba_to_surface(self.premultiplied(box_w, box_h))
