from __future__ import annotations

from enum import Enum

import cairo
import numpy as np

from nltl_viz.preset import Preset

_BASE_HALF_FRACTION = 0.35
_PULSE_HALF_FRACTION = 0.5  # edge-to-edge in the frame's shorter dimension — see pulse_polygon_points
_FLASH_VISIBLE_THRESHOLD = 0.02


class Shape(str, Enum):
    face = "face"
    space = "space"


class Motion(str, Enum):
    deform = "deform"
    rigid = "rigid"
    pulse = "pulse"


def cosine_interp_cyclic(values: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Cyclic cosine interpolation: monotonic between control points, no overshoot,
    passes exactly through every control point at t = i/len(values)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    t_arr = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    idx_f = t_arr * n
    i0 = np.floor(idx_f).astype(int) % n
    i1 = (i0 + 1) % n
    u = idx_f - np.floor(idx_f)
    blend = (1.0 - np.cos(u * np.pi)) / 2.0
    return values[i0] * (1.0 - blend) + values[i1] * blend


def _face_vertices(size: int, half_fraction: float = _BASE_HALF_FRACTION) -> np.ndarray:
    """The NLTL face: inscribed in a bounding square, top-left and bottom-left
    at the box's own corners, bottom-right at 2/3 width along the bottom edge,
    top-right at 2/3 width x 1/3 height (an interior point of the box, not on
    any box edge). Clockwise from top-left."""
    cx = cy = size / 2.0
    half = half_fraction * size
    x0, y0 = cx - half, cy - half
    side = 2.0 * half
    return np.array(
        [
            (x0, y0),
            (x0 + (2.0 / 3.0) * side, y0 + (1.0 / 3.0) * side),
            (x0 + (2.0 / 3.0) * side, y0 + side),
            (x0, y0 + side),
        ],
        dtype=np.float64,
    )


def _space_vertices(size: int, half_fraction: float = _BASE_HALF_FRACTION) -> np.ndarray:
    """NLTL space: the same bounding square minus the NLTL face — the
    complementary pentagon. Shares the face's short vertical edge and
    diagonal as its own boundary, traversed in the opposite direction, so
    the two shapes are exact complements with no gap or overlap. Not convex
    — there's a reflex vertex where the face's silhouette cuts in. Clockwise
    from top-left."""
    cx = cy = size / 2.0
    half = half_fraction * size
    x0, y0 = cx - half, cy - half
    side = 2.0 * half
    return np.array(
        [
            (x0, y0),
            (x0 + side, y0),
            (x0 + side, y0 + side),
            (x0 + (2.0 / 3.0) * side, y0 + side),
            (x0 + (2.0 / 3.0) * side, y0 + (1.0 / 3.0) * side),
        ],
        dtype=np.float64,
    )


def _polygon_vertices(size: int, shape: Shape = Shape.face, half_fraction: float = _BASE_HALF_FRACTION) -> np.ndarray:
    if shape == Shape.space:
        return _space_vertices(size, half_fraction)
    return _face_vertices(size, half_fraction)


def _polygon_centroid(vertices: np.ndarray) -> tuple[float, float]:
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def _polygon_perimeter_points(
    t: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length-proportional position on the polygon boundary at parameter
    t in [0,1) — so a control point index maps to actual distance traveled
    around the perimeter, not an equal share per edge."""
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    edge_vectors = ends - starts
    edge_lengths = np.hypot(edge_vectors[:, 0], edge_vectors[:, 1])
    cum_lengths = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    total_length = cum_lengths[-1]

    t = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    target = t * total_length
    seg_idx = np.clip(
        np.searchsorted(cum_lengths, target, side="right") - 1, 0, len(vertices) - 1
    )
    local_frac = (target - cum_lengths[seg_idx]) / edge_lengths[seg_idx]

    points = starts[seg_idx] + edge_vectors[seg_idx] * local_frac[:, None]
    return points[:, 0], points[:, 1]


def deformed_polygon_points(
    band_values: np.ndarray,
    size: int,
    amplitude: float,
    shape: Shape = Shape.face,
    n_samples: int = 256,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, tuple[float, float]]:
    vertices = _polygon_vertices(size, shape) + np.array(offset)
    centroid = _polygon_centroid(vertices)
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    bx, by = _polygon_perimeter_points(t, vertices)
    d = cosine_interp_cyclic(band_values, t)
    scale = 1.0 + amplitude * d
    x = centroid[0] + (bx - centroid[0]) * scale
    y = centroid[1] + (by - centroid[1]) * scale
    return np.stack([x, y], axis=1), centroid


def scaled_polygon_points(
    size: int, shape: Shape, scale: float, *, offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, tuple[float, float]]:
    """The plain, undistorted shape's own vertices, uniformly scaled from its
    centroid — used by `Motion.rigid`, where the perimeter stays in
    proportion rather than deforming per-band."""
    vertices = _polygon_vertices(size, shape) + np.array(offset)
    centroid = _polygon_centroid(vertices)
    centroid_arr = np.array(centroid)
    points = centroid_arr + (vertices - centroid_arr) * scale
    return points, centroid


def pulse_polygon_points(
    size: int, shape: Shape, *, offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, tuple[float, float]]:
    """`Motion.pulse`'s shape: truly constant size and proportions, inscribed
    edge-to-edge in the frame's shorter dimension (`size` is already
    `min(width, height)` by the time it reaches here — see `render_frame`) —
    unlike `deform`/`rigid`, which stay at `_BASE_HALF_FRACTION`'s small
    centered scale. Reactivity comes entirely from the fill's opacity, not
    from any per-frame change to these points."""
    vertices = _polygon_vertices(size, shape, half_fraction=_PULSE_HALF_FRACTION) + np.array(offset)
    centroid = _polygon_centroid(vertices)
    return vertices, centroid


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def draw_flash(
    ctx: cairo.Context,
    brightness: float,
    color: tuple[float, float, float],
    preset: Preset,
    size: int,
    center: tuple[float, float],
) -> None:
    if brightness <= _FLASH_VISIBLE_THRESHOLD:
        return
    cx, cy = center
    radius = preset.flash_size * 0.5 * size * (0.4 + 0.6 * min(brightness, 1.0))
    if radius <= 0:
        return
    r, g, b = (c / 255.0 for c in color)
    alpha = min(brightness, 1.0)
    gradient = cairo.RadialGradient(cx, cy, 0.0, cx, cy, radius)
    gradient.add_color_stop_rgba(0.0, r, g, b, alpha)
    gradient.add_color_stop_rgba(1.0, r, g, b, 0.0)
    ctx.set_source(gradient)
    ctx.paint()


def _fill_polygon(
    ctx: cairo.Context, points: np.ndarray, color: tuple[float, float, float], alpha: float = 1.0
) -> None:
    ctx.set_source_rgba(*color, alpha)
    ctx.move_to(points[0, 0], points[0, 1])
    for px, py in points[1:]:
        ctx.line_to(px, py)
    ctx.close_path()
    ctx.fill()


def draw_inner_shape(
    ctx: cairo.Context,
    brightness: float,
    color: tuple[float, float, float],
    preset: Preset,
    size: int,
    center: tuple[float, float],
    shape: Shape,
) -> None:
    """`Motion.rigid`'s replacement for `draw_flash`: a smaller, solid-filled,
    concentric copy of the outer shape instead of a soft radial-gradient
    blob — but driven by the exact same onset/decay/color mechanic."""
    if brightness <= _FLASH_VISIBLE_THRESHOLD:
        return
    inner_scale = preset.flash_size * (0.4 + 0.6 * min(brightness, 1.0))
    if inner_scale <= 0:
        return
    points, _ = scaled_polygon_points(size, shape, inner_scale)
    # scaled_polygon_points scales from the shape's own centroid, which
    # already equals `center` here, so points are already correctly placed.
    rgb01 = tuple(c / 255.0 for c in color)
    _fill_polygon(ctx, points, rgb01)


def draw_frame(
    ctx: cairo.Context,
    points: np.ndarray,
    flash_brightness: float,
    flash_color: tuple[float, float, float],
    preset: Preset,
    size: int,
    center: tuple[float, float],
    shape: Shape = Shape.face,
    motion: Motion = Motion.deform,
    *,
    transparent_background: bool = False,
    opacity: float = 0.0,
) -> None:
    if not transparent_background:
        bg = _hex_to_rgb01(preset.background_color)
        ctx.set_source_rgb(*bg)
        ctx.paint()

    if motion == Motion.pulse:
        # No flash, no bass/treble centroid color — the whole shape's alpha
        # is the only reactive element in this mode.
        outline_rgb = _hex_to_rgb01(preset.outline_color)
        _fill_polygon(ctx, points, outline_rgb, opacity)
        return

    if motion == Motion.rigid:
        outline_rgb = _hex_to_rgb01(preset.outline_color)
        _fill_polygon(ctx, points, outline_rgb)
        draw_inner_shape(ctx, flash_brightness, flash_color, preset, size, center, shape)
        return

    draw_flash(ctx, flash_brightness, flash_color, preset, size, center)

    outline_rgb = _hex_to_rgb01(preset.outline_color)
    ctx.set_source_rgb(*outline_rgb)
    ctx.set_line_width(max(1.0, size * 0.004))
    ctx.move_to(points[0, 0], points[0, 1])
    for px, py in points[1:]:
        ctx.line_to(px, py)
    ctx.close_path()
    ctx.stroke()


def surface_to_rgb24(
    surface: cairo.ImageSurface, width: int, height: int
) -> np.ndarray:
    """cairo.FORMAT_ARGB32 is premultiplied, native-endian — BGRA in memory on
    little-endian — with row stride possibly padded beyond width*4."""
    surface.flush()
    stride = surface.get_stride()
    buf = np.ndarray(
        shape=(height, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data()
    )
    buf = buf[:, :width, :]
    rgb = buf[:, :, [2, 1, 0]]
    return np.ascontiguousarray(rgb)


def surface_to_rgba_premultiplied(
    surface: cairo.ImageSurface, width: int, height: int
) -> np.ndarray:
    """Like `surface_to_rgb24` but keeps the alpha channel, and leaves RGB
    premultiplied by alpha rather than un-premultiplying it — premultiplied
    "over" compositing (`out = fg_premult + bg * (1 - alpha)`) needs no
    unpremultiply/divide step, so this is both simpler and avoids the
    precision loss un-premultiplying would cost at partially-covered
    (anti-aliased) edge pixels."""
    surface.flush()
    stride = surface.get_stride()
    buf = np.ndarray(
        shape=(height, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data()
    )
    buf = buf[:, :width, :]
    rgba = buf[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(rgba)


def render_frame(
    band_values: np.ndarray,
    flash_brightness: float,
    flash_color: tuple[float, float, float],
    preset: Preset,
    size: int,
    shape: Shape = Shape.face,
    motion: Motion = Motion.deform,
    scale_value: float = 0.5,
    *,
    width: int | None = None,
    height: int | None = None,
    transparent_background: bool = False,
) -> np.ndarray:
    """Renders one frame onto a `width`x`height` canvas (defaulting to a
    `size`x`size` square when unset), with the shape itself always sized and
    proportioned off `size` and centered in the canvas — so a wider canvas
    just extends the (transparent, in overlay mode) area around the shape
    rather than stretching it."""
    w = width if width is not None else size
    h = height if height is not None else size
    offset = ((w - size) / 2.0, (h - size) / 2.0)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)

    opacity = 0.0
    if motion == Motion.pulse:
        points, centroid = pulse_polygon_points(size, shape, offset=offset)
        opacity = preset.pulse_max_opacity * scale_value
    elif motion == Motion.rigid:
        scale = 1.0 + preset.scale_amplitude * (scale_value - 0.5) * 2.0
        points, centroid = scaled_polygon_points(size, shape, scale, offset=offset)
    else:
        points, centroid = deformed_polygon_points(
            band_values, size, preset.deform_amplitude, shape, offset=offset
        )

    draw_frame(
        ctx, points, flash_brightness, flash_color, preset, size, centroid, shape, motion,
        transparent_background=transparent_background, opacity=opacity,
    )
    if transparent_background:
        return surface_to_rgba_premultiplied(surface, w, h)
    return surface_to_rgb24(surface, w, h)
