import numpy as np

from nltl_viz.render import (
    Shape,
    _face_vertices,
    _polygon_centroid,
    _polygon_perimeter_points,
    _polygon_vertices,
    _space_vertices,
    deformed_polygon_points,
    scaled_polygon_points,
)


def test_vertices_match_expected_geometry():
    size = 300
    vertices = _polygon_vertices(size)
    cx = cy = size / 2.0
    half = 0.35 * size
    x0, y0 = cx - half, cy - half
    side = 2.0 * half

    expected = np.array(
        [
            (x0, y0),
            (x0 + (2.0 / 3.0) * side, y0 + (1.0 / 3.0) * side),
            (x0 + (2.0 / 3.0) * side, y0 + side),
            (x0, y0 + side),
        ]
    )
    assert np.allclose(vertices, expected)


def test_centroid_is_within_bounding_extent():
    vertices = _polygon_vertices(300)
    cx, cy = _polygon_centroid(vertices)
    assert vertices[:, 0].min() <= cx <= vertices[:, 0].max()
    assert vertices[:, 1].min() <= cy <= vertices[:, 1].max()


def test_perimeter_point_at_t0_is_first_vertex():
    vertices = _polygon_vertices(300)
    x, y = _polygon_perimeter_points(np.array([0.0]), vertices)
    assert np.allclose([x[0], y[0]], vertices[0])


def test_perimeter_point_at_vertex_arc_length_fraction():
    vertices = _polygon_vertices(300)
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    edge_lengths = np.hypot(*(ends - starts).T)
    total = edge_lengths.sum()

    # fraction of perimeter traveled to reach vertex index 2
    frac_to_vertex_2 = edge_lengths[:2].sum() / total
    x, y = _polygon_perimeter_points(np.array([frac_to_vertex_2]), vertices)
    assert np.allclose([x[0], y[0]], vertices[2], atol=1e-6)


def test_zero_amplitude_matches_base_perimeter():
    vertices = _polygon_vertices(300)
    band_values = np.random.default_rng(0).uniform(0, 1, size=32)
    points, _ = deformed_polygon_points(band_values, 300, amplitude=0.0)

    n_samples = 256
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    bx, by = _polygon_perimeter_points(t, vertices)
    base_points = np.stack([bx, by], axis=1)

    assert np.allclose(points, base_points)


def test_deformation_scales_from_returned_centroid():
    band_values = np.full(32, 1.0)  # max deformation everywhere
    points, centroid = deformed_polygon_points(band_values, 300, amplitude=0.5)
    vertices = _polygon_vertices(300)
    n_samples = 256
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    bx, by = _polygon_perimeter_points(t, vertices)

    expected_x = centroid[0] + (bx - centroid[0]) * 1.5
    expected_y = centroid[1] + (by - centroid[1]) * 1.5
    assert np.allclose(points[:, 0], expected_x)
    assert np.allclose(points[:, 1], expected_y)


def test_polygon_vertices_dispatches_on_shape():
    size = 300
    assert np.allclose(_polygon_vertices(size, Shape.face), _face_vertices(size))
    assert np.allclose(_polygon_vertices(size, Shape.space), _space_vertices(size))
    # default (no shape arg) is face, for backwards compatibility
    assert np.allclose(_polygon_vertices(size), _face_vertices(size))


def test_space_is_pentagon_matching_square_minus_face():
    size = 300
    cx = cy = size / 2.0
    half = 0.35 * size
    x0, y0 = cx - half, cy - half
    side = 2.0 * half

    expected = np.array(
        [
            (x0, y0),
            (x0 + side, y0),
            (x0 + side, y0 + side),
            (x0 + (2.0 / 3.0) * side, y0 + side),
            (x0 + (2.0 / 3.0) * side, y0 + (1.0 / 3.0) * side),
        ]
    )
    assert np.allclose(_space_vertices(size), expected)


def test_face_and_space_share_the_cut_boundary_exactly():
    """The two shapes must be exact complements: the face's short vertical
    edge and diagonal are literally the same segments as the space's,
    just traversed in the opposite direction."""
    size = 300
    face = _face_vertices(size)
    space = _space_vertices(size)

    # face's top-right (index 1) and bottom-right (index 2) vertices
    # are the space's last two vertices (index 4 and 3), respectively
    assert np.allclose(face[1], space[4])
    assert np.allclose(face[2], space[3])


def test_space_centroid_is_within_bounding_extent():
    vertices = _space_vertices(300)
    cx, cy = _polygon_centroid(vertices)
    assert vertices[:, 0].min() <= cx <= vertices[:, 0].max()
    assert vertices[:, 1].min() <= cy <= vertices[:, 1].max()


def test_deformed_polygon_points_differ_between_shapes():
    band_values = np.random.default_rng(0).uniform(0, 1, size=32)
    face_points, face_centroid = deformed_polygon_points(band_values, 300, 0.3, Shape.face)
    space_points, space_centroid = deformed_polygon_points(band_values, 300, 0.3, Shape.space)

    assert face_points.shape == space_points.shape  # same n_samples regardless of shape
    assert not np.allclose(face_points, space_points)
    assert face_centroid != space_centroid


def test_scaled_polygon_points_at_scale_one_matches_plain_vertices():
    points, centroid = scaled_polygon_points(300, Shape.face, scale=1.0)
    assert np.allclose(points, _face_vertices(300))
    assert centroid == _polygon_centroid(_face_vertices(300))


def test_scaled_polygon_points_scales_from_centroid():
    vertices = _space_vertices(300)
    centroid = _polygon_centroid(vertices)
    points, returned_centroid = scaled_polygon_points(300, Shape.space, scale=1.5)

    expected = np.array(centroid) + (vertices - np.array(centroid)) * 1.5
    assert np.allclose(points, expected)
    assert returned_centroid == centroid


def test_scaled_polygon_points_below_one_shrinks_toward_centroid():
    points, centroid = scaled_polygon_points(300, Shape.face, scale=0.5)
    vertices = _face_vertices(300)
    # every scaled point should be strictly closer to the centroid than the
    # corresponding plain vertex (shape shrinks toward its own centroid)
    centroid_arr = np.array(centroid)
    plain_dist = np.hypot(*(vertices - centroid_arr).T)
    scaled_dist = np.hypot(*(points - centroid_arr).T)
    assert np.all(scaled_dist < plain_dist)
