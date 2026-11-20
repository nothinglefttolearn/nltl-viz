from __future__ import annotations

from enum import Enum
from functools import cache

import cairo
import nltl_values
import numpy as np

from nltl_viz.image import Image, premultiplied_rgba_to_surface
from nltl_viz.palette import Palette
from nltl_viz.preset import Preset

_BASE_HALF_FRACTION = 0.35
_PULSE_HALF_FRACTION = (
    0.5  # edge-to-edge in the frame's shorter dimension — see pulse_polygon_points
)
_FLASH_VISIBLE_THRESHOLD = 0.02


class Shape(str, Enum):
    """Shapes deliberately exposed as `--shape` choices. Each value must name
    a shape in nltl-values; a shape that is new in nltl-values is only added
    here after it has been checked rendered under every Motion."""

    face = "face"
    space = "space"
    face2 = "face2"
    space2 = "space2"
    face3 = "face3"
    space3 = "space3"


class Motion(str, Enum):
    deform = "deform"
    rigid = "rigid"
    pulse = "pulse"
    split = "split"


class Layout(str, Enum):
    single = "single"
    grid = "grid"


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


@cache
def _unit_points(shape: Shape) -> np.ndarray:
    """The shape's points from nltl-values, in its unit square (screen
    orientation: origin top-left, y down — same as Cairo, so no flip)."""
    points = np.array(nltl_values.shape(shape.value).points, dtype=np.float64)
    points.flags.writeable = False  # cached and shared — callers must not mutate it
    return points


def _polygon_vertices(
    size: int, shape: Shape = Shape.face, half_fraction: float = _BASE_HALF_FRACTION
) -> np.ndarray:
    """The shape's nltl-values unit-square points mapped onto a bounding
    square of side `2 * half_fraction * size`, centered in a `size`-square
    canvas. The bounding square — not the shape's own extent — is what's
    centered, so `face` and `space` stay exact complements on screen."""
    half = half_fraction * size
    origin = size / 2.0 - half
    return origin + _unit_points(shape) * (2.0 * half)


def _polygon_centroid(vertices: np.ndarray) -> tuple[float, float]:
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def _polygon_perimeter_points(t: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    seg_idx = np.clip(np.searchsorted(cum_lengths, target, side="right") - 1, 0, len(vertices) - 1)
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


def _intersect_horizontal(p0: np.ndarray, p1: np.ndarray, y: float) -> np.ndarray:
    t = (y - p0[1]) / (p1[1] - p0[1])
    return np.array([p0[0] + t * (p1[0] - p0[0]), y])


def _clip_polygon_half_plane(vertices: np.ndarray, y: float, *, keep_below: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of `vertices` against the horizontal
    half-plane y <= `y` (keep_below=True) or y >= `y` (keep_below=False).
    Clipping against a single half-plane is valid for concave subject
    polygons too (e.g. Shape.space's reflex vertex) — the artifacts
    Sutherland-Hodgman can produce on concave *subjects* only show up when
    clipping against a multi-edge convex window, not a single straight
    line."""
    output: list[np.ndarray] = []
    n = len(vertices)
    for i in range(n):
        curr, prev = vertices[i], vertices[i - 1]
        curr_in = (curr[1] <= y) if keep_below else (curr[1] >= y)
        prev_in = (prev[1] <= y) if keep_below else (prev[1] >= y)
        if curr_in:
            if not prev_in:
                output.append(_intersect_horizontal(prev, curr, y))
            output.append(curr)
        elif prev_in:
            output.append(_intersect_horizontal(prev, curr, y))
    return np.array(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)


def split_polygon_points(
    size: int, shape: Shape = Shape.face, *, offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    """`Motion.split`'s geometry: the plain, undistorted shape cut along the
    horizontal midline of its own bounding box (symmetric around `size / 2`,
    the same basis `_BASE_HALF_FRACTION` centers on) into a top half and a
    bottom half — each a standalone closed polygon, ready to be translated
    independently and stroked. No deformation, no scaling; the only motion
    is each half's rigid horizontal offset, applied by the caller."""
    vertices = _polygon_vertices(size, shape) + np.array(offset)
    y_mid = offset[1] + size / 2.0
    top = _clip_polygon_half_plane(vertices, y_mid, keep_below=True)
    bottom = _clip_polygon_half_plane(vertices, y_mid, keep_below=False)
    return top, bottom


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
    # Scale the plain shape about its own centroid, then place that centroid
    # on `center` — the outer shape's centroid, which already includes the
    # Layout/overlay offset that `size` alone doesn't carry.
    vertices = _polygon_vertices(size, shape)
    own_centroid = np.array(_polygon_centroid(vertices))
    points = np.array(center) + (vertices - own_centroid) * inner_scale
    rgb01 = tuple(c / 255.0 for c in color)
    _fill_polygon(ctx, points, rgb01)


def _paint_background(ctx: cairo.Context, palette: Palette, transparent_background: bool) -> None:
    if transparent_background:
        return
    bg = _hex_to_rgb01(palette.background_color)
    ctx.set_source_rgb(*bg)
    ctx.paint()


def draw_shape_instance(
    ctx: cairo.Context,
    points: np.ndarray,
    flash_brightness: float,
    flash_color: tuple[float, float, float],
    preset: Preset,
    palette: Palette,
    size: int,
    center: tuple[float, float],
    shape: Shape = Shape.face,
    motion: Motion = Motion.deform,
    *,
    opacity: float = 0.0,
) -> None:
    """Draws one shape copy (fill/stroke + its flash) onto `ctx`. Background
    painting is a separate, canvas-wide step (`_paint_background`) so
    `Layout.grid` can paint once and call this once per quadrant."""
    if motion == Motion.pulse:
        # No flash, no bass/treble centroid color — the whole shape's alpha
        # is the only reactive element in this mode.
        outline_rgb = _hex_to_rgb01(palette.outline_color)
        _fill_polygon(ctx, points, outline_rgb, opacity)
        return

    if motion == Motion.rigid:
        outline_rgb = _hex_to_rgb01(palette.outline_color)
        _fill_polygon(ctx, points, outline_rgb)
        draw_inner_shape(ctx, flash_brightness, flash_color, preset, size, center, shape)
        return

    draw_flash(ctx, flash_brightness, flash_color, preset, size, center)

    outline_rgb = _hex_to_rgb01(palette.outline_color)
    _stroke_polygon(ctx, points, outline_rgb, max(1.0, size * 0.004))


def _stroke_polygon(
    ctx: cairo.Context, points: np.ndarray, color: tuple[float, float, float], line_width: float
) -> None:
    ctx.set_source_rgb(*color)
    ctx.set_line_width(line_width)
    ctx.move_to(points[0, 0], points[0, 1])
    for px, py in points[1:]:
        ctx.line_to(px, py)
    ctx.close_path()
    ctx.stroke()


def draw_split_instance(
    ctx: cairo.Context,
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    dx: float,
    palette: Palette,
    size: int,
) -> None:
    """`Motion.split`: strokes the top half shifted `-dx` in x and the
    bottom half shifted `+dx`, so the two halves — flush at dx=0 — slide
    apart horizontally as `dx` grows. No flash, no per-band color — the
    slide distance is the only reactive element, deliberately as minimal as
    `Motion.pulse`'s single opacity mechanic."""
    outline_rgb = _hex_to_rgb01(palette.outline_color)
    line_width = max(1.0, size * 0.004)
    for points, shift in ((top_points, -dx), (bottom_points, dx)):
        if len(points) == 0:
            continue
        _stroke_polygon(ctx, points + np.array([shift, 0.0]), outline_rgb, line_width)


def surface_to_rgb24(surface: cairo.ImageSurface, width: int, height: int) -> np.ndarray:
    """cairo.FORMAT_ARGB32 is premultiplied, native-endian — BGRA in memory on
    little-endian — with row stride possibly padded beyond width*4."""
    surface.flush()
    stride = surface.get_stride()
    buf = np.ndarray(shape=(height, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data())
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
    buf = np.ndarray(shape=(height, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data())
    buf = buf[:, :width, :]
    rgba = buf[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(rgba)


def _layout_instances(
    layout: Layout, w: int, h: int, size: int
) -> list[tuple[int, tuple[float, float]]]:
    """One (instance_size, offset) pair per shape copy to draw.

    `Layout.single` reuses the caller's own `size` (already `min(w, h)` by
    convention — see `_frame_generator`/`_overlay_frame_generator`), centered
    in the full canvas.

    `Layout.grid` ignores that `size` and re-derives a quadrant-fit size
    directly from the canvas dimensions, since each quadrant's own space —
    not the single-shape basis size the caller computed — is what should
    drive its scale. Fixed at a 2x2 grid (4 identical copies)."""
    if layout == Layout.single:
        offset = ((w - size) / 2.0, (h - size) / 2.0)
        return [(size, offset)]

    qw, qh = w / 2.0, h / 2.0
    instance_size = int(min(qw, qh))
    local_offset = ((qw - instance_size) / 2.0, (qh - instance_size) / 2.0)
    return [
        (instance_size, (col * qw + local_offset[0], row * qh + local_offset[1]))
        for row in range(2)
        for col in range(2)
    ]


def image_box(
    image: Image, size: int, half_fraction: float, offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float, int, int]:
    """The Image's contain box: `(x0, y0, box_w, box_h)` — the Image scaled
    to fit, aspect preserved, inside the same bounding square a Shape would
    occupy at `half_fraction` (see `_polygon_vertices`), centered in it. So
    every Preset amplitude means the same for either Subject (ADR 0004)."""
    side = 2.0 * half_fraction * size
    box_w, box_h = image.contain_box(side)
    x0 = offset[0] + (size - box_w) / 2.0
    y0 = offset[1] + (size - box_h) / 2.0
    return x0, y0, box_w, box_h


def _rect_perimeter_t(dx: np.ndarray, dy: np.ndarray, half_w: float, half_h: float) -> np.ndarray:
    """For directions `(dx, dy)` from a rectangle's center, the arc-length
    parameter t in [0, 1) of where each ray meets the rectangle's edge —
    starting at the top-left corner and running clockwise on screen (y down),
    the same convention `_polygon_perimeter_points` gives a Shape whose first
    vertex is top-left."""
    w, h = 2.0 * half_w, 2.0 * half_h
    perimeter = 2.0 * (w + h)
    with np.errstate(divide="ignore", invalid="ignore"):
        kx = np.where(dx != 0, half_w / np.abs(dx), np.inf)
        ky = np.where(dy != 0, half_h / np.abs(dy), np.inf)
    k = np.minimum(kx, ky)
    k = np.where(np.isfinite(k), k, 0.0)
    # Hit point relative to the top-left corner.
    bx = np.clip(half_w + dx * k, 0.0, w)
    by = np.clip(half_h + dy * k, 0.0, h)
    hits_side = kx < ky
    s = np.where(
        hits_side,
        np.where(dx > 0, w + by, 2.0 * w + h + (h - by)),  # right edge / left edge
        np.where(dy < 0, bx, w + h + (w - bx)),  # top edge / bottom edge
    )
    return np.mod(s / perimeter, 1.0)


def _sample_bilinear(src: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Bilinear lookup of premultiplied RGBA `src` (h, w, 4) at pixel-center
    coordinates `(sx, sy)`; anything outside `src` reads as transparent. Done
    on premultiplied values so anti-aliased edges don't fringe."""
    h, w = src.shape[:2]
    padded = np.zeros((h + 2, w + 2, 4), dtype=np.float32)
    padded[1:-1, 1:-1] = src
    # +1 for the padding border, -0.5 to go from pixel-center to index space.
    fx = np.clip(sx + 0.5, 0.0, w + 1.0)
    fy = np.clip(sy + 0.5, 0.0, h + 1.0)
    x0 = np.clip(np.floor(fx).astype(np.intp), 0, w)
    y0 = np.clip(np.floor(fy).astype(np.intp), 0, h)
    ux = (fx - x0)[..., None]
    uy = (fy - y0)[..., None]
    top = padded[y0, x0] * (1.0 - ux) + padded[y0, x0 + 1] * ux
    bottom = padded[y0 + 1, x0] * (1.0 - ux) + padded[y0 + 1, x0 + 1] * ux
    return top * (1.0 - uy) + bottom * uy


def deformed_image(
    image: Image,
    band_values: np.ndarray,
    size: int,
    amplitude: float,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    canvas: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """`Motion.deform` for an Image: the contain-box rectangle stands in for
    the polygon perimeter (ADR 0004). Bands are spread around it by arc
    length, and every pixel is pushed radially from the box's center by
    `1 + amplitude * band(t)`, t being where its ray meets the edge.

    Because that warp is purely radial — a pixel's scale depends only on its
    angle — it inverts exactly: each output pixel `q` reads the source at
    `c + (q - c) / scale(t(q))`. Returns the warped premultiplied RGBA
    region, its integer top-left position on the canvas, and the center."""
    x0, y0, box_w, box_h = image_box(image, size, _BASE_HALF_FRACTION, offset)
    cx, cy = x0 + box_w / 2.0, y0 + box_h / 2.0
    half_w, half_h = box_w / 2.0, box_h / 2.0
    reach = 1.0 + max(amplitude, 0.0) * max(float(np.max(band_values, initial=0.0)), 0.0)

    left, top = int(np.floor(cx - half_w * reach)), int(np.floor(cy - half_h * reach))
    right, bottom = int(np.ceil(cx + half_w * reach)), int(np.ceil(cy + half_h * reach))
    if canvas is not None:
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, canvas[0]), min(bottom, canvas[1])
    if right <= left or bottom <= top:
        return np.zeros((0, 0, 4), dtype=np.uint8), (left, top), (cx, cy)

    qx, qy = np.meshgrid(
        np.arange(left, right, dtype=np.float64) + 0.5,
        np.arange(top, bottom, dtype=np.float64) + 0.5,
    )
    dx, dy = qx - cx, qy - cy
    t = _rect_perimeter_t(dx, dy, half_w, half_h)
    scale = 1.0 + amplitude * cosine_interp_cyclic(band_values, t)
    sx = dx / scale + half_w - 0.5
    sy = dy / scale + half_h - 0.5
    warped = _sample_bilinear(image.premultiplied(box_w, box_h), sx, sy)
    region = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
    return region, (left, top), (cx, cy)


def _paint_image(
    ctx: cairo.Context,
    image: Image,
    box: tuple[float, float, int, int],
    scale: float = 1.0,
    *,
    alpha: float = 1.0,
    silhouette_rgb: tuple[float, float, float] | None = None,
) -> None:
    """Paints the Image's contain box scaled by `scale` about its own center.
    With `silhouette_rgb`, paints that flat color through the Image's alpha
    instead of its pixels (`Motion.rigid`'s inner flash copy)."""
    x0, y0, box_w, box_h = box
    if scale <= 0:
        return
    cx, cy = x0 + box_w / 2.0, y0 + box_h / 2.0
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)
    ctx.translate(-box_w / 2.0, -box_h / 2.0)
    surface = image.surface(box_w, box_h)
    if silhouette_rgb is None:
        ctx.set_source_surface(surface, 0.0, 0.0)
        ctx.paint_with_alpha(alpha)
    else:
        ctx.set_source_rgba(*silhouette_rgb, alpha)
        ctx.mask_surface(surface, 0.0, 0.0)
    ctx.restore()


def draw_image_instance(
    ctx: cairo.Context,
    image: Image,
    band_values: np.ndarray,
    flash_brightness: float,
    flash_color: tuple[float, float, float],
    preset: Preset,
    size: int,
    motion: Motion,
    scale_value: float,
    offset: tuple[float, float],
    canvas: tuple[int, int],
) -> None:
    """One Image Subject copy — the Image counterpart of
    `draw_shape_instance`/`draw_split_instance`. Each Motion keeps its Shape
    meaning; the Image's own pixels take the place of `outline_color`, and
    the flash (deform/rigid only) is drawn on top, confined to the Image's
    alpha, since an opaque Image would hide a flash drawn behind it."""
    if motion == Motion.pulse:
        box = image_box(image, size, _PULSE_HALF_FRACTION, offset)
        _paint_image(ctx, image, box, alpha=preset.pulse_max_opacity * scale_value)
        return

    box = image_box(image, size, _BASE_HALF_FRACTION, offset)
    x0, y0, box_w, box_h = box

    if motion == Motion.split:
        dx = preset.split_amplitude * size * scale_value
        y_mid = offset[1] + size / 2.0
        for clip_top, clip_bottom, shift in ((y0, y_mid, -dx), (y_mid, y0 + box_h, dx)):
            if clip_bottom <= clip_top:
                continue
            ctx.save()
            ctx.translate(shift, 0.0)
            ctx.rectangle(x0, clip_top, box_w, clip_bottom - clip_top)
            ctx.clip()
            _paint_image(ctx, image, box)
            ctx.restore()
        return

    if motion == Motion.rigid:
        scale = 1.0 + preset.scale_amplitude * (scale_value - 0.5) * 2.0
        _paint_image(ctx, image, box, scale)
        if flash_brightness > _FLASH_VISIBLE_THRESHOLD:
            inner_scale = preset.flash_size * (0.4 + 0.6 * min(flash_brightness, 1.0))
            rgb01 = tuple(c / 255.0 for c in flash_color)
            _paint_image(ctx, image, box, inner_scale, silhouette_rgb=rgb01)
        return

    region, (left, top), center = deformed_image(
        image, band_values, size, preset.deform_amplitude, offset=offset, canvas=canvas
    )
    if region.size == 0:
        return
    warped = premultiplied_rgba_to_surface(region)
    ctx.set_source_surface(warped, left, top)
    ctx.paint()

    if flash_brightness <= _FLASH_VISIBLE_THRESHOLD:
        return
    radius = preset.flash_size * 0.5 * size * (0.4 + 0.6 * min(flash_brightness, 1.0))
    if radius <= 0:
        return
    r, g, b = (c / 255.0 for c in flash_color)
    gradient = cairo.RadialGradient(*center, 0.0, *center, radius)
    gradient.add_color_stop_rgba(0.0, r, g, b, min(flash_brightness, 1.0))
    gradient.add_color_stop_rgba(1.0, r, g, b, 0.0)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_source(gradient)
    ctx.mask_surface(warped, left, top)
    ctx.restore()


def render_frame(
    band_values: np.ndarray,
    flash_brightness: float,
    flash_color: tuple[float, float, float],
    preset: Preset,
    palette: Palette,
    size: int,
    subject: Shape | Image = Shape.face,
    motion: Motion = Motion.deform,
    scale_value: float = 0.5,
    layout: Layout = Layout.single,
    *,
    width: int | None = None,
    height: int | None = None,
    transparent_background: bool = False,
) -> np.ndarray:
    """Renders one frame onto a `width`x`height` canvas (defaulting to a
    `size`x`size` square when unset). Under `Layout.single` the Subject is
    sized and proportioned off `size` and centered in the canvas — so a
    wider canvas just extends the (transparent, in overlay mode) area around
    the Subject rather than stretching it. Under `Layout.grid`, 4 identical
    copies are drawn, one per quadrant, all driven by the same
    `band_values`/`flash_brightness`/`flash_color`/`scale_value`."""
    w = width if width is not None else size
    h = height if height is not None else size

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    _paint_background(ctx, palette, transparent_background)

    for instance_size, offset in _layout_instances(layout, w, h, size):
        if isinstance(subject, Image):
            draw_image_instance(
                ctx,
                subject,
                band_values,
                flash_brightness,
                flash_color,
                preset,
                instance_size,
                motion,
                scale_value,
                offset,
                (w, h),
            )
            continue

        shape = subject
        if motion == Motion.split:
            top_points, bottom_points = split_polygon_points(instance_size, shape, offset=offset)
            dx = preset.split_amplitude * instance_size * scale_value
            draw_split_instance(ctx, top_points, bottom_points, dx, palette, instance_size)
            continue

        opacity = 0.0
        if motion == Motion.pulse:
            points, centroid = pulse_polygon_points(instance_size, shape, offset=offset)
            opacity = preset.pulse_max_opacity * scale_value
        elif motion == Motion.rigid:
            scale = 1.0 + preset.scale_amplitude * (scale_value - 0.5) * 2.0
            points, centroid = scaled_polygon_points(instance_size, shape, scale, offset=offset)
        else:
            points, centroid = deformed_polygon_points(
                band_values, instance_size, preset.deform_amplitude, shape, offset=offset
            )

        draw_shape_instance(
            ctx,
            points,
            flash_brightness,
            flash_color,
            preset,
            palette,
            instance_size,
            centroid,
            shape,
            motion,
            opacity=opacity,
        )

    if transparent_background:
        return surface_to_rgba_premultiplied(surface, w, h)
    return surface_to_rgb24(surface, w, h)
