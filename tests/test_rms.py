import numpy as np

from nltl_viz.audio import _rms_envelope_normalized


def test_rms_envelope_is_bounded_zero_to_one():
    rng = np.random.default_rng(0)
    y = rng.uniform(-1, 1, size=44100 * 3).astype(np.float64)
    envelope = _rms_envelope_normalized(y, hop_length=512)
    assert envelope.min() >= 0.0
    assert envelope.max() <= 1.0


def test_louder_segment_has_higher_normalized_value():
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    # quiet for the first half, loud for the second half
    quiet = 0.01 * np.sin(2 * np.pi * 220 * t[: len(t) // 2])
    loud = 0.9 * np.sin(2 * np.pi * 220 * t[len(t) // 2 :])
    y = np.concatenate([quiet, loud])

    envelope = _rms_envelope_normalized(y, hop_length=512)
    midpoint = len(envelope) // 2
    assert envelope[:midpoint].mean() < envelope[midpoint:].mean()
