from pathlib import Path

from nltl_viz.encode import build_ffmpeg_cmd, build_overlay_ffmpeg_cmd


def test_build_ffmpeg_cmd_muxes_audio_with_aac():
    cmd = build_ffmpeg_cmd(Path("track.wav"), Path("out.mp4"), 1080, 1080, 30, 12.5, preview=False)
    assert "track.wav" in cmd
    assert "out.mp4" in cmd
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_build_overlay_ffmpeg_cmd_stream_copies_audio_from_source_video():
    cmd = build_overlay_ffmpeg_cmd(Path("clip.mp4"), Path("out.mp4"), 1920, 1080, "30000/1001", 12.5, preview=False)
    assert "clip.mp4" in cmd
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "1920x1080" in cmd
    assert "30000/1001" in cmd


def test_overlay_cmd_uses_preview_quality_flags():
    cmd = build_overlay_ffmpeg_cmd(Path("clip.mp4"), Path("out.mp4"), 1920, 1080, 30, 10.0, preview=True)
    assert "ultrafast" in cmd
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "10.000"
