import numpy as np
import pytest

from nltl_viz.audio import OnsetEvent, precompute_flash_signal
from nltl_viz.palette import Palette
from nltl_viz.preset import Preset


def _preset(**overrides):
    return Preset(name="t", flash_decay_ms=200.0, flash_intensity_scale=1.0, **overrides)


def _palette(**overrides):
    return Palette(name="t", **overrides)


def test_envelope_at_exactly_decay_time():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(
        onsets, centroid, n_frames=60, fps=30, preset=_preset(), palette=_palette()
    )

    frame_at_200ms = round(200.0 / (1000.0 / 30))  # 6 frames
    assert brightness[frame_at_200ms] == pytest.approx(0.05, rel=0.05)


def test_monotonically_decreasing_after_onset():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(
        onsets, centroid, n_frames=60, fps=30, preset=_preset(), palette=_palette()
    )

    window = brightness[0:20]
    assert np.all(np.diff(window) <= 1e-9)


def test_overlapping_onsets_combine_via_max_not_sum():
    onsets = [OnsetEvent(frame_index=0, strength=1.0), OnsetEvent(frame_index=1, strength=1.0)]
    centroid = np.zeros(60)
    brightness, _ = precompute_flash_signal(
        onsets, centroid, n_frames=60, fps=30, preset=_preset(), palette=_palette()
    )

    # if combined via sum, frame 1 would be close to 2.0 (peak + near-peak decay); max keeps it near 1.0
    assert brightness[1] < 1.1


def test_monochrome_palette_color_is_constant_regardless_of_centroid():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.array([0.0, 0.5, 1.0])
    _, color = precompute_flash_signal(
        onsets, centroid, n_frames=3, fps=30, preset=_preset(),
        palette=_palette(flash_primary_color="#112233"),
    )
    assert np.allclose(color, [0x11, 0x22, 0x33])


def test_duotone_palette_lerps_by_centroid():
    onsets = [OnsetEvent(frame_index=0, strength=1.0)]
    centroid = np.array([0.0, 1.0])
    _, color = precompute_flash_signal(
        onsets, centroid, n_frames=2, fps=30, preset=_preset(),
        palette=_palette(flash_primary_color="#6E5470", flash_secondary_color="#2E8C8A"),
    )
    assert np.allclose(color[0], [0x6E, 0x54, 0x70])
    assert np.allclose(color[1], [0x2E, 0x8C, 0x8A])
