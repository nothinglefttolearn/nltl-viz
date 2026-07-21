from __future__ import annotations

import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from nltl_viz import audio as audio_mod
from nltl_viz import config, encode, interactive, postprocess, render
from nltl_viz import preset as preset_mod
from nltl_viz import video as video_mod
from nltl_viz.audio import AudioAnalysis
from nltl_viz.preset import Preset
from nltl_viz.render import Motion, Shape

FPS = 30
SIZE = 1080
DEFAULT_COMMAND = "render"


class _DefaultCommandGroup(TyperGroup):
    """Route bare `nltl-viz [audio] [flags]` invocations to the `render`
    command, since Click otherwise consumes a leading positional value into
    the group callback's own argument before ever checking subcommand names —
    which would make an explicit `nltl-viz presets` unreachable."""

    def resolve_command(self, ctx, args):
        if not args or args[0] not in self.commands:
            args = [DEFAULT_COMMAND, *args]
        return super().resolve_command(ctx, args)


app = typer.Typer(
    add_completion=False,
    cls=_DefaultCommandGroup,
    context_settings={"ignore_unknown_options": True},
    help=(
        "Generate an audio-reactive NLTL face visualization from a music file, "
        "or — given a video file — overlay it with transparency onto that video, "
        "using the video's own audio track for analysis.\n\n"
        "Usage: nltl-viz [AUDIO|VIDEO] [--preset NAME] [--shape face|space] [--preview] "
        "[--output-dir DIR] [--config FILE] [--verbose]\n\n"
        "Motion (deform/rigid/pulse) comes from the chosen preset, not a separate flag — "
        "see `nltl-viz presets`.\n\n"
        "Run with no arguments for interactive mode (audio files only)."
    ),
)
console = Console()


def _frame_generator(
    analysis: AudioAnalysis, preset_obj: Preset, size: int, shape: Shape
) -> Iterable[np.ndarray]:
    motion = Motion(preset_obj.motion)
    vignette_mask = postprocess.build_vignette_mask(size, size, preset_obj.vignette_fraction)
    rng = np.random.default_rng()
    for i in range(analysis.n_frames):
        band_values = analysis.band_energy[i]
        brightness = float(analysis.flash_brightness[i])
        color = tuple(analysis.flash_color[i])
        scale_value = float(analysis.scale_envelope[i])
        frame = render.render_frame(
            band_values, brightness, color, preset_obj, size, shape, motion, scale_value
        )
        frame = postprocess.apply_vignette(frame, vignette_mask)
        frame = postprocess.apply_grain(frame, preset_obj.grain_strength, rng)
        yield frame


def _overlay_frame_generator(
    analysis: AudioAnalysis, preset_obj: Preset, width: int, height: int, shape: Shape
) -> Iterable[np.ndarray]:
    """Same per-frame render as `_frame_generator`, but on a transparent
    background sized to the source video's own resolution. Vignette is a
    pure multiply so it's inert at alpha=0 (corners stay untouched). Grain is
    additive, so — deliberately, unlike vignette — it isn't confined to the
    shape's alpha: through the `fg_premultiplied + bg * (1 - alpha)`
    compositing formula, grain noise added to the (zero, fully transparent)
    background leaks straight into the video underneath, giving the whole
    composited frame one continuous grain texture instead of an abrupt
    dropoff in texture right at the shape's silhouette."""
    motion = Motion(preset_obj.motion)
    size = min(width, height)
    vignette_mask = postprocess.build_vignette_mask(width, height, preset_obj.vignette_fraction)
    rng = np.random.default_rng()
    for i in range(analysis.n_frames):
        band_values = analysis.band_energy[i]
        brightness = float(analysis.flash_brightness[i])
        color = tuple(analysis.flash_color[i])
        scale_value = float(analysis.scale_envelope[i])
        frame = render.render_frame(
            band_values, brightness, color, preset_obj, size, shape, motion, scale_value,
            width=width, height=height, transparent_background=True,
        )
        rgb, alpha = frame[:, :, :3], frame[:, :, 3:4]
        rgb = postprocess.apply_vignette(rgb, vignette_mask)
        rgb = postprocess.apply_grain(rgb, preset_obj.grain_strength, rng)
        yield np.concatenate([rgb, alpha], axis=-1)


def run_render(
    audio_path: Path,
    preset_name: str,
    shape: Shape,
    output_dir: Optional[Path],
    config_path: Optional[Path],
    preview: bool,
    verbose: bool,
) -> None:
    resolved_preset = config.resolve_preset(preset_name, config_path)

    console.print()
    console.print(f"[bold]Preset[/bold]    {resolved_preset.name} — {resolved_preset.description}")
    console.print(f"[bold]Shape[/bold]     {shape.value}")
    console.print(f"[bold]Motion[/bold]    {resolved_preset.motion}")
    if preview:
        console.print("[bold]Preview[/bold]   10s low-quality render")

    with console.status("Analyzing audio..."):
        analysis = audio_mod.analyze(audio_path, resolved_preset, fps=FPS)

    out_dir = output_dir if output_dir is not None else audio_path.parent
    suffix = "viz_preview" if preview else "viz"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = out_dir / f"{audio_path.stem}_{suffix}_{timestamp}.mp4"

    total_frames = min(analysis.n_frames, FPS * 10) if preview else analysis.n_frames
    duration_sec = 10.0 if preview else analysis.duration_sec

    console.print(f"[bold]Output[/bold]    {output_path}")
    console.print()

    frames = _frame_generator(analysis, resolved_preset, SIZE, shape)
    cmd = encode.build_ffmpeg_cmd(audio_path, output_path, SIZE, SIZE, FPS, duration_sec, preview)
    encode.render_video(frames, cmd=cmd, total_frames=total_frames, preview=preview, verbose=verbose)
    console.print(f"[green]Done[/green] → {output_path}")


def run_overlay(
    video_path: Path,
    preset_name: str,
    shape: Shape,
    output_dir: Optional[Path],
    config_path: Optional[Path],
    preview: bool,
    verbose: bool,
) -> None:
    resolved_preset = config.resolve_preset(preset_name, config_path)
    probe = video_mod.probe(video_path)

    console.print()
    console.print(f"[bold]Preset[/bold]    {resolved_preset.name} — {resolved_preset.description}")
    console.print(f"[bold]Shape[/bold]     {shape.value}")
    console.print(f"[bold]Motion[/bold]    {resolved_preset.motion}")
    console.print(f"[bold]Overlay[/bold]   {probe.width}x{probe.height} @ {probe.fps_rational}fps")
    if preview:
        console.print("[bold]Preview[/bold]   10s low-quality render")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        with console.status("Extracting audio..."):
            video_mod.extract_audio(video_path, wav_path)
        with console.status("Analyzing audio..."):
            analysis = audio_mod.analyze(wav_path, resolved_preset, fps=probe.fps)

    out_dir = output_dir if output_dir is not None else video_path.parent
    suffix = "viz-overlay_preview" if preview else "viz-overlay"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = out_dir / f"{video_path.stem}_{suffix}_{timestamp}.mp4"

    total_frames = min(analysis.n_frames, math.ceil(probe.fps * 10)) if preview else analysis.n_frames
    duration_sec = 10.0 if preview else analysis.duration_sec

    console.print(f"[bold]Output[/bold]    {output_path}")
    console.print()

    viz_frames = _overlay_frame_generator(analysis, resolved_preset, probe.width, probe.height, shape)
    video_frames = video_mod.decode_frames(video_path, probe.width, probe.height, total_frames)
    composited = (
        video_mod.composite_over(video_frame, viz_frame)
        for video_frame, viz_frame in zip(video_frames, viz_frames)
    )

    cmd = encode.build_overlay_ffmpeg_cmd(
        video_path, output_path, probe.width, probe.height, probe.fps_rational, duration_sec, preview
    )
    encode.render_video(composited, cmd=cmd, total_frames=total_frames, preview=preview, verbose=verbose)
    console.print(f"[green]Done[/green] → {output_path}")


def _run_interactive() -> None:
    try:
        choices = interactive.run()
        run_render(
            choices.audio_path,
            choices.preset_name,
            Shape(choices.shape),
            None,
            None,
            False,
            False,
        )
    except Exception as exc:  # surface any failure as a clean CLI error, not a traceback
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"\nTo run this again without prompts:\n  {choices.headless_command()}")


@app.callback(invoke_without_command=True)
def default_callback(ctx: typer.Context) -> None:
    """Generate an audio-reactive NLTL face visualization from a music file.

    Run with no arguments for interactive mode."""
    # Click short-circuits truly-zero-arg invocations straight to the group
    # callback without ever reaching resolve_command, so the zero-args ->
    # interactive-mode path has to be handled here rather than in render_cmd.
    if ctx.invoked_subcommand is not None:
        return
    _run_interactive()


@app.command(DEFAULT_COMMAND, hidden=True)
def render_cmd(
    input_path: Optional[Path] = typer.Argument(
        None, help="Path to the input audio file, or a video file to overlay the visualization onto"
    ),
    preset: str = typer.Option("industrial", "--preset", "-p", help="Visual preset (also determines motion)"),
    shape: Shape = typer.Option(Shape.face, "--shape", help="Shape to visualize: face or space"),
    preview: bool = typer.Option(False, "--preview", help="Render a 10s low-quality preview"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML file with custom presets"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show raw ffmpeg output during render"),
) -> None:
    """Generate an audio-reactive NLTL face visualization from a music file,
    or overlay it with transparency onto a video file's own audio."""
    if input_path is None:
        _run_interactive()
        return

    try:
        if video_mod.is_video_file(input_path):
            run_overlay(input_path, preset, shape, output_dir, config_path, preview, verbose)
        else:
            run_render(input_path, preset, shape, output_dir, config_path, preview, verbose)
    except Exception as exc:  # surface any failure as a clean CLI error, not a traceback
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("presets")
def presets_cmd() -> None:
    """List available built-in presets."""
    table = Table(title="Built-in presets")
    table.add_column("Name", style="bold")
    table.add_column("Motion")
    table.add_column("Description")
    for name in preset_mod.names():
        p = preset_mod.get(name)
        table.add_row(name, p.motion, p.description)
    console.print(table)


if __name__ == "__main__":
    app()
