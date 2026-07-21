import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nltl_viz import video


def test_is_video_file_detects_known_containers():
    assert video.is_video_file(Path("clip.mp4"))
    assert video.is_video_file(Path("clip.MOV"))
    assert not video.is_video_file(Path("track.wav"))
    assert not video.is_video_file(Path("track.mp3"))


def _fake_ffprobe_result(width, height, r_frame_rate, duration):
    payload = {
        "streams": [{"width": width, "height": height, "r_frame_rate": r_frame_rate}],
        "format": {"duration": str(duration)},
    }
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")


def test_probe_parses_non_integer_frame_rate(monkeypatch):
    # GoPro-style 30000/1001 (~29.97fps) rational frame rate.
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_ffprobe_result(1920, 1080, "30000/1001", 30.5)
    )
    result = video.probe(Path("clip.mp4"))
    assert result.width == 1920
    assert result.height == 1080
    assert result.fps == pytest.approx(30000 / 1001)
    assert result.fps_rational == "30000/1001"
    assert result.duration_sec == pytest.approx(30.5)


def test_probe_reduces_integer_frame_rate_to_bare_number(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_ffprobe_result(1920, 1080, "30/1", 12.0)
    )
    result = video.probe(Path("clip.mp4"))
    assert result.fps == pytest.approx(30.0)
    assert result.fps_rational == "30"


def test_probe_raises_on_ffprobe_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    )
    with pytest.raises(RuntimeError, match="boom"):
        video.probe(Path("missing.mp4"))


def test_composite_over_is_noop_at_zero_alpha():
    video_frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    viz_rgba = np.zeros((4, 4, 4), dtype=np.uint8)  # fully transparent
    out = video.composite_over(video_frame, viz_rgba)
    assert np.array_equal(out, video_frame)


def test_composite_over_replaces_at_full_alpha():
    video_frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    viz_rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    viz_rgba[:, :, 0] = 200  # premultiplied red, alpha = 255 -> fully opaque
    viz_rgba[:, :, 3] = 255
    out = video.composite_over(video_frame, viz_rgba)
    assert np.all(out[:, :, 0] == 200)
    assert np.all(out[:, :, 1] == 0)


def test_composite_over_blends_at_partial_alpha():
    video_frame = np.full((1, 1, 3), 100, dtype=np.uint8)
    viz_rgba = np.array([[[50, 0, 0, 128]]], dtype=np.uint8)  # premultiplied, ~50% alpha
    out = video.composite_over(video_frame, viz_rgba)
    # out = fg_premult + bg * (1 - alpha) = 50 + 100 * (1 - 128/255)
    expected = 50 + 100 * (1 - 128 / 255)
    assert out[0, 0, 0] == pytest.approx(expected, abs=1)
