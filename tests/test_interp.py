import numpy as np

from nltl_viz.render import cosine_interp_cyclic


def test_passes_through_control_points():
    values = np.array([0.0, 1.0, 0.3, 0.8, 0.5])
    n = len(values)
    t = np.arange(n) / n
    out = cosine_interp_cyclic(values, t)
    assert np.allclose(out, values, atol=1e-9)


def test_constant_input_yields_constant_output():
    values = np.full(10, 0.42)
    t = np.linspace(0, 1, 100, endpoint=False)
    out = cosine_interp_cyclic(values, t)
    assert np.allclose(out, 0.42)


def test_smooth_wraparound_near_boundary():
    values = np.array([0.1, 0.9, 0.2, 0.7])
    just_before = cosine_interp_cyclic(values, 0.9999)
    at_zero = cosine_interp_cyclic(values, 0.0)
    assert abs(just_before - at_zero) < 0.05
