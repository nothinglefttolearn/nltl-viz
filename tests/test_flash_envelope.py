import numpy as np
import pytest

from nltl_viz.audio import OnsetEvent, precompute_flash_signal
from nltl_viz.preset import Preset


def _preset(**overrides):
    return Preset(name="t", flash_decay_ms=200.0, flash_intensity_scale=1.0, **overrides)


def test_envelope_at_exactly_decay_time():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(onsets, centroid, n_frames=60, fps=30, preset=_preset())

    frame_at_200ms = round(200.0 / (1000.0 / 30))  # 6 frames
    assert brightness[frame_at_200ms] == pytest.approx(0.05, rel=0.05)


def test_monotonically_decreasing_after_onset():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(onsets, centroid, n_frames=60, fps=30, preset=_preset())

    window = brightness[0:20]
    assert np.all(np.diff(window) <= 1e-9)


def test_overlapping_onsets_combine_via_max_not_sum():
    onsets = [OnsetEvent(frame_index=0, strength=1.0), OnsetEvent(frame_index=1, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(onsets, centroid, n_frames=60, fps=30, preset=_preset())

    # if combined via sum, frame 1 would be close to 2.0 (peak + near-peak decay); max keeps it near 1.0
    assert brightness[1] < 1.1
