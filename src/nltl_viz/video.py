from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


@dataclass
class VideoProbe:
    width: int
    height: int
    fps: float
    fps_rational: str
    duration_sec: float


def probe(video_path: Path) -> VideoProbe:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}):\n{result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    frac = Fraction(stream["r_frame_rate"])
    fps_rational = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"

    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(frac),
        fps_rational=fps_rational,
        duration_sec=float(data["format"]["duration"]),
    )


def extract_audio(video_path: Path, output_wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        str(output_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed (exit {result.returncode}):\n{result.stderr}")


def decode_frames(video_path: Path, width: int, height: int, n_frames: int) -> Iterator[np.ndarray]:
    """Reads the video's frames in native decode order (no resampling) as raw
    RGB24 — the caller is expected to have matched `n_frames` and the
    analysis fps to the video's own probed frame rate, so this just trusts
    that alignment rather than re-deriving it."""
    frame_bytes = width * height * 3
    cmd = ["ffmpeg", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for _ in range(n_frames):
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def composite_over(video_frame_rgb: np.ndarray, viz_rgba_premultiplied: np.ndarray) -> np.ndarray:
    """"Over" compositing with premultiplied source alpha:
    out = fg_premultiplied + bg * (1 - alpha)."""
    fg_rgb = viz_rgba_premultiplied[:, :, :3].astype(np.float32)
    alpha = viz_rgba_premultiplied[:, :, 3:4].astype(np.float32) / 255.0
    bg = video_frame_rgb.astype(np.float32)
    out = fg_rgb + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
