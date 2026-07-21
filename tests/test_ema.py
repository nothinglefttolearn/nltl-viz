import numpy as np

from nltl_viz.audio import _ema_smooth


def test_constant_input_is_steady_state():
    raw = np.full(50, 0.7)
    out = _ema_smooth(raw, attack_sec=0.05, release_sec=0.3, fps=30)
    assert np.allclose(out, 0.7)


def test_attack_faster_than_release():
    raw = np.concatenate([np.zeros(30), np.ones(60), np.zeros(60)])
    out = _ema_smooth(raw, attack_sec=0.02, release_sec=0.3, fps=30)

    # a few frames after the step up, attack should have closed most of the gap
    rise_gap = 1.0 - out[35]
    # a few frames after the step down, release should have closed far less
    fall_gap = out[95]

    assert rise_gap < 0.3
    assert fall_gap > 0.5


def test_output_never_exceeds_input_bounds():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=200)
    out = _ema_smooth(raw, attack_sec=0.05, release_sec=0.2, fps=30)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
