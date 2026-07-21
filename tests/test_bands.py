import numpy as np

from nltl_viz.audio import _band_energies


def test_energy_attributed_to_correct_band():
    band_count = 8
    fmin, fmax = 30.0, 8000.0
    n_bins = 1025
    n_frames = 5
    freqs = np.linspace(0, 22050, n_bins)

    edges = np.geomspace(fmin, fmax, band_count + 1)
    target_band = 3
    target_freq = (edges[target_band] + edges[target_band + 1]) / 2.0
    target_bin = int(np.argmin(np.abs(freqs - target_freq)))

    S_power = np.full((n_bins, n_frames), 1e-9)
    S_power[target_bin, :] = 1.0

    energies = _band_energies(S_power, freqs, band_count, fmin, fmax)

    assert energies.shape == (n_frames, band_count)
    dominant_band = int(np.argmax(energies[0]))
    assert dominant_band == target_band


def test_no_crash_at_lowest_and_highest_bands():
    band_count = 32
    n_bins = 1025
    freqs = np.linspace(0, 22050, n_bins)
    S_power = np.random.default_rng(0).uniform(0, 1, size=(n_bins, 3))

    energies = _band_energies(S_power, freqs, band_count, 30.0, 8000.0)

    assert energies.shape == (3, band_count)
    assert np.all(np.isfinite(energies))
