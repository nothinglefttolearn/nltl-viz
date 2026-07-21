import numpy as np

from nltl_viz.audio import lerp_color


def test_exact_bass_at_t0():
    out = lerp_color("#6E5470", "#2E8C8A", 0.0)
    assert np.allclose(out, [0x6E, 0x54, 0x70])


def test_exact_treble_at_t1():
    out = lerp_color("#6E5470", "#2E8C8A", 1.0)
    assert np.allclose(out, [0x2E, 0x8C, 0x8A])


def test_midpoint_is_channelwise_average():
    out = lerp_color("#6E5470", "#2E8C8A", 0.5)
    expected = (np.array([0x6E, 0x54, 0x70]) + np.array([0x2E, 0x8C, 0x8A])) / 2.0
    assert np.allclose(out, expected)


def test_out_of_range_is_clamped():
    low = lerp_color("#6E5470", "#2E8C8A", -5.0)
    high = lerp_color("#6E5470", "#2E8C8A", 5.0)
    assert np.allclose(low, [0x6E, 0x54, 0x70])
    assert np.allclose(high, [0x2E, 0x8C, 0x8A])


def test_array_input():
    out = lerp_color("#6E5470", "#2E8C8A", np.array([0.0, 1.0]))
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [0x6E, 0x54, 0x70])
    assert np.allclose(out[1], [0x2E, 0x8C, 0x8A])
