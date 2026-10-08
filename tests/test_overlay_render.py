import numpy as np

from nltl_viz.palette import BUILTIN as PALETTE_BUILTIN
from nltl_viz.preset import BUILTIN
from nltl_viz.render import (
    Layout,
    Motion,
    Shape,
    _layout_instances,
    _polygon_centroid,
    _polygon_vertices,
    deformed_polygon_points,
    render_frame,
    scaled_polygon_points,
)

PALETTE = PALETTE_BUILTIN["black"]


def test_offset_translates_deformed_points_and_centroid():
    band_values = np.random.default_rng(0).uniform(0, 1, size=32)
    points, centroid = deformed_polygon_points(band_values, 300, amplitude=0.3)
    offset_points, offset_centroid = deformed_polygon_points(
        band_values, 300, amplitude=0.3, offset=(420.0, 10.0)
    )
    assert np.allclose(offset_points, points + np.array([420.0, 10.0]))
    assert offset_centroid == (centroid[0] + 420.0, centroid[1] + 10.0)


def test_offset_translates_scaled_points_and_centroid():
    points, centroid = scaled_polygon_points(300, Shape.face, scale=1.2)
    offset_points, offset_centroid = scaled_polygon_points(300, Shape.face, scale=1.2, offset=(420.0, 10.0))
    assert np.allclose(offset_points, points + np.array([420.0, 10.0]))
    assert offset_centroid == (centroid[0] + 420.0, centroid[1] + 10.0)


def test_render_frame_default_matches_prior_opaque_behavior():
    preset = BUILTIN["industrial"]
    frame = render_frame(
        np.zeros(preset.band_count), 0.0, (0, 0, 0), preset, PALETTE, size=100, shape=Shape.face, motion=Motion.rigid
    )
    assert frame.shape == (100, 100, 3)


def test_render_frame_transparent_background_is_rgba_with_transparent_corners():
    preset = BUILTIN["industrial"]
    size, width, height = 100, 300, 100
    frame = render_frame(
        np.zeros(preset.band_count),
        0.0,
        (0, 0, 0),
        preset,
        PALETTE,
        size=size,
        shape=Shape.face,
        motion=Motion.rigid,
        scale_value=0.5,
        width=width,
        height=height,
        transparent_background=True,
    )
    assert frame.shape == (height, width, 4)
    # corners are untouched background -> fully transparent
    assert frame[0, 0, 3] == 0
    assert frame[height - 1, width - 1, 3] == 0


def test_render_frame_centers_shape_within_wider_canvas():
    preset = BUILTIN["industrial"]
    size, width, height = 100, 300, 100
    offset = ((width - size) / 2.0, (height - size) / 2.0)
    vertices = _polygon_vertices(size, Shape.face) + np.array(offset)
    cx, cy = _polygon_centroid(vertices)

    frame = render_frame(
        np.zeros(preset.band_count),
        0.0,
        (0, 0, 0),
        preset,
        PALETTE,
        size=size,
        shape=Shape.face,
        motion=Motion.rigid,
        scale_value=0.5,  # scale == 1.0 for the "industrial" preset's scale_amplitude
        width=width,
        height=height,
        transparent_background=True,
    )
    # centroid of a solid-filled shape in rigid motion must be fully opaque
    assert frame[int(cy), int(cx), 3] == 255


def test_grid_layout_draws_four_opaque_quadrant_centers():
    preset = BUILTIN["industrial"]
    width = height = 400
    frame = render_frame(
        np.zeros(preset.band_count),
        0.0,
        (0, 0, 0),
        preset,
        PALETTE,
        size=width,  # ignored for grid — see _layout_instances
        shape=Shape.face,
        motion=Motion.rigid,
        scale_value=0.5,
        layout=Layout.grid,
        width=width,
        height=height,
        transparent_background=True,
    )
    assert frame.shape == (height, width, 4)
    for instance_size, offset in _layout_instances(Layout.grid, width, height, width):
        vertices = _polygon_vertices(instance_size, Shape.face) + np.array(offset)
        cx, cy = _polygon_centroid(vertices)
        # centroid of each quadrant's solid-filled shape must be fully opaque
        assert frame[int(cy), int(cx), 3] == 255
    # nothing painted at the seam between quadrants
    assert frame[200, 200, 3] == 0


def test_single_layout_ignores_grid_and_draws_once():
    preset = BUILTIN["industrial"]
    frame_single = render_frame(
        np.zeros(preset.band_count), 0.0, (0, 0, 0), preset, PALETTE,
        size=100, shape=Shape.face, motion=Motion.rigid, layout=Layout.single,
    )
    frame_default = render_frame(
        np.zeros(preset.band_count), 0.0, (0, 0, 0), preset, PALETTE,
        size=100, shape=Shape.face, motion=Motion.rigid,
    )
    assert np.array_equal(frame_single, frame_default)
