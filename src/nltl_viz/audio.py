from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from nltl_viz.preset import Preset

_N_FFT = 2048
_BAND_FMIN = 30.0
_BAND_FMAX = 8000.0
_CENTROID_LO_HZ = 300.0
_CENTROID_HI_HZ = 4000.0


@dataclass
class OnsetEvent:
    frame_index: int
    strength: float


@dataclass
class AudioAnalysis:
    sample_rate: int
    duration_sec: float
    n_frames: int
    band_energy: np.ndarray
    scale_envelope: np.ndarray
    flash_brightness: np.ndarray
    flash_color: np.ndarray


def analyze(audio_path: Path, preset: Preset, fps: int = 30) -> AudioAnalysis:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    n_frames = max(1, math.ceil(duration_sec * fps))
    hop_length = max(1, round(sr / fps))

    S_power = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)

    fmax = min(_BAND_FMAX, sr / 2.0)
    band_energy_raw = _band_energies(S_power, freqs, preset.band_count, _BAND_FMIN, fmax)
    band_energy_raw = _resize_to_frames(band_energy_raw, n_frames)
    band_db = 10.0 * np.log10(band_energy_raw + 1e-12)
    band_norm = _normalize_bands(band_db)
    band_smoothed = _ema_smooth(band_norm, preset.smoothing_attack, preset.smoothing_release, fps)

    onsets = _detect_onsets(y, sr, hop_length, preset.onset_sensitivity, n_frames)

    centroid = _centroid_normalized(y, sr, _N_FFT, hop_length)
    centroid = _resize_to_frames(centroid, n_frames)

    rms_norm = _rms_envelope_normalized(y, hop_length)
    rms_norm = _resize_to_frames(rms_norm, n_frames)
    scale_envelope = _ema_smooth(rms_norm, preset.smoothing_attack, preset.smoothing_release, fps)

    flash_brightness, flash_color = precompute_flash_signal(onsets, centroid, n_frames, fps, preset)

    return AudioAnalysis(
        sample_rate=sr,
        duration_sec=duration_sec,
        n_frames=n_frames,
        band_energy=band_smoothed,
        scale_envelope=scale_envelope,
        flash_brightness=flash_brightness,
        flash_color=flash_color,
    )


def _resize_to_frames(arr: np.ndarray, n_frames: int) -> np.ndarray:
    t = arr.shape[0]
    if t == n_frames:
        return arr
    if t > n_frames:
        return arr[:n_frames]
    pad_width = n_frames - t
    if arr.ndim == 1:
        return np.pad(arr, (0, pad_width), mode="edge")
    return np.pad(arr, ((0, pad_width), (0, 0)), mode="edge")


def _band_energies(
    S_power: np.ndarray, freqs: np.ndarray, band_count: int, fmin: float, fmax: float
) -> np.ndarray:
    edges = np.geomspace(fmin, fmax, band_count + 1)
    n_stft_frames = S_power.shape[1]
    n_bins = S_power.shape[0]
    energies = np.zeros((n_stft_frames, band_count), dtype=np.float64)
    for i in range(band_count):
        bin_lo = int(np.searchsorted(freqs, edges[i]))
        bin_hi = int(np.searchsorted(freqs, edges[i + 1]))
        bin_hi = max(bin_hi, bin_lo + 1)
        bin_hi = min(bin_hi, n_bins)
        bin_lo = min(bin_lo, bin_hi - 1)
        energies[:, i] = S_power[bin_lo:bin_hi, :].mean(axis=0)
    return energies


def _normalize_bands(band_db: np.ndarray) -> np.ndarray:
    floor = np.percentile(band_db, 5, axis=0, keepdims=True)
    ceiling = np.percentile(band_db, 95, axis=0, keepdims=True)
    span = np.maximum(ceiling - floor, 1e-6)
    return np.clip((band_db - floor) / span, 0.0, 1.0)


def _ema_smooth(raw: np.ndarray, attack_sec: float, release_sec: float, fps: int) -> np.ndarray:
    alpha_attack = 1.0 - math.exp(-1.0 / max(attack_sec * fps, 1e-6))
    alpha_release = 1.0 - math.exp(-1.0 / max(release_sec * fps, 1e-6))
    out = np.empty_like(raw, dtype=np.float64)
    out[0] = raw[0]
    for i in range(1, len(raw)):
        rising = raw[i] > out[i - 1]
        alpha = np.where(rising, alpha_attack, alpha_release)
        out[i] = out[i - 1] + alpha * (raw[i] - out[i - 1])
    return out


def _detect_onsets(
    y: np.ndarray, sr: int, hop_length: int, sensitivity: float, n_frames: int
) -> list[OnsetEvent]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    delta = 0.07 / max(sensitivity, 1e-6)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, delta=delta, backtrack=False
    )
    ref = float(np.percentile(onset_env, 95)) if len(onset_env) else 1.0
    ref = max(ref, 1e-6)

    events = []
    for frame in onset_frames:
        if frame >= n_frames:
            continue
        strength = float(np.clip(onset_env[frame] / ref, 0.0, 1.0))
        events.append(OnsetEvent(frame_index=int(frame), strength=strength))
    return events


def _rms_envelope_normalized(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Overall loudness per frame — log-compressed (RMS is amplitude-domain,
    so 20*log10 rather than the 10*log10 used for the power-domain bands)
    and percentile-normalized the same way as band energy, so a quiet track
    and a loud track both span a comparable [0,1] range."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = 20.0 * np.log10(rms + 1e-12)
    floor = np.percentile(rms_db, 5)
    ceiling = np.percentile(rms_db, 95)
    span = max(ceiling - floor, 1e-6)
    return np.clip((rms_db - floor) / span, 0.0, 1.0)


def _centroid_normalized(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    return np.clip((centroid - _CENTROID_LO_HZ) / (_CENTROID_HI_HZ - _CENTROID_LO_HZ), 0.0, 1.0)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def lerp_color(bass_hex: str, treble_hex: str, t: np.ndarray | float) -> np.ndarray:
    bass_rgb = np.array(_hex_to_rgb(bass_hex), dtype=np.float64)
    treble_rgb = np.array(_hex_to_rgb(treble_hex), dtype=np.float64)
    t_arr = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    if t_arr.ndim == 0:
        return bass_rgb * (1 - t_arr) + treble_rgb * t_arr
    t_col = t_arr[:, None]
    return bass_rgb[None, :] * (1 - t_col) + treble_rgb[None, :] * t_col


def precompute_flash_signal(
    onsets: list[OnsetEvent], centroid: np.ndarray, n_frames: int, fps: int, preset: Preset
) -> tuple[np.ndarray, np.ndarray]:
    brightness = np.zeros(n_frames, dtype=np.float64)
    decay_rate = math.log(20.0) / max(preset.flash_decay_ms, 1e-6)
    window_frames = max(1, math.ceil(3.0 * preset.flash_decay_ms / 1000.0 * fps))
    ms_per_frame = 1000.0 / fps

    for onset in onsets:
        peak = float(np.clip(preset.flash_intensity_scale * onset.strength, 0.0, 1.4))
        end = min(n_frames, onset.frame_index + window_frames)
        for j in range(onset.frame_index, end):
            t_ms = (j - onset.frame_index) * ms_per_frame
            envelope = math.exp(-t_ms * decay_rate)
            contribution = peak * envelope
            if contribution > brightness[j]:
                brightness[j] = contribution

    color = lerp_color(preset.bass_color, preset.treble_color, centroid)
    return brightness, color
