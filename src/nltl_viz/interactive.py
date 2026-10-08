from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import questionary

from nltl_viz import palette as palette_mod
from nltl_viz import preset as preset_mod

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


@dataclass
class Choices:
    audio_path: Path
    preset_name: str
    palette_name: str
    layout: str
    shape: str

    def headless_command(self) -> str:
        return (
            f"nltl-viz --preset {self.preset_name} --palette {self.palette_name} "
            f"--layout {self.layout} --shape {self.shape} {self.audio_path}"
        )


def scan_audio_files(cwd: Path) -> list[Path]:
    return sorted(p for p in cwd.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def run() -> Choices:
    cwd = Path.cwd()
    audio_files = scan_audio_files(cwd)
    if not audio_files:
        extensions = ", ".join(sorted(AUDIO_EXTENSIONS))
        raise RuntimeError(f"no audio files found in {cwd} (looked for {extensions})")

    audio_answer = questionary.select("Audio file", choices=[str(p) for p in audio_files]).ask()
    if audio_answer is None:
        raise RuntimeError("cancelled")

    preset_choices = [
        questionary.Choice(
            title=f"{name} — {preset_mod.get(name).description} [{preset_mod.get(name).motion}]", value=name
        )
        for name in preset_mod.names()
    ]
    preset_answer = questionary.select("Preset", choices=preset_choices).ask()
    if preset_answer is None:
        raise RuntimeError("cancelled")

    palette_choices = [
        questionary.Choice(title=f"{name} — {palette_mod.get(name).description}", value=name)
        for name in palette_mod.names()
    ]
    palette_answer = questionary.select("Palette", choices=palette_choices).ask()
    if palette_answer is None:
        raise RuntimeError("cancelled")

    layout_choices = [
        questionary.Choice(title="single — one shape centered in the frame", value="single"),
        questionary.Choice(title="grid — 2x2 grid of 4 identical shape copies", value="grid"),
    ]
    layout_answer = questionary.select("Layout", choices=layout_choices).ask()
    if layout_answer is None:
        raise RuntimeError("cancelled")

    shape_choices = [
        questionary.Choice(title="face — the NLTL face", value="face"),
        questionary.Choice(title="space — the NLTL space (the face's inverse)", value="space"),
    ]
    shape_answer = questionary.select("Shape", choices=shape_choices).ask()
    if shape_answer is None:
        raise RuntimeError("cancelled")

    return Choices(
        audio_path=Path(audio_answer),
        preset_name=preset_answer,
        palette_name=palette_answer,
        layout=layout_answer,
        shape=shape_answer,
    )
