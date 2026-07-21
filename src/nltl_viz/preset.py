from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Preset:
    name: str
    description: str = ""
    motion: str = "deform"  # one of Motion's values ("deform", "rigid", "pulse") — see render.Motion
    band_count: int = 32
    deform_amplitude: float = 0.0
    scale_amplitude: float = 0.0
    pulse_max_opacity: float = 0.85
    smoothing_attack: float = 0.03
    smoothing_release: float = 0.25
    onset_sensitivity: float = 1.0
    flash_decay_ms: float = 200.0
    flash_size: float = 0.18
    flash_intensity_scale: float = 1.0
    bass_color: str = "#6E5470"
    treble_color: str = "#2E8C8A"
    grain_strength: float = 14.0
    vignette_fraction: float = 0.45
    background_color: str = "#0A0A0C"
    outline_color: str = "#D8D8D2"


FIELD_NAMES = tuple(f.name for f in fields(Preset))

BUILTIN: dict[str, Preset] = {
    "industrial": Preset(
        name="industrial",
        description="Moderate deform, medium onset sensitivity",
        deform_amplitude=0.35,
        scale_amplitude=0.20,
        smoothing_attack=0.030,
        smoothing_release=0.250,
        onset_sensitivity=1.0,
        flash_decay_ms=200.0,
        flash_size=0.18,
        flash_intensity_scale=1.0,
        grain_strength=14.0,
        vignette_fraction=0.45,
    ),
    "subtle": Preset(
        name="subtle",
        description="Calmer, slower breathing, lighter grain/vignette",
        deform_amplitude=0.15,
        scale_amplitude=0.10,
        smoothing_attack=0.060,
        smoothing_release=0.450,
        onset_sensitivity=0.6,
        flash_decay_ms=150.0,
        flash_size=0.10,
        flash_intensity_scale=0.6,
        grain_strength=6.0,
        vignette_fraction=0.25,
    ),
    "3d-glasses": Preset(
        name="3d-glasses",
        description="Red bass color, cyan treble color, light background, dark outline",
        deform_amplitude=0.15,
        smoothing_attack=0.060,
        smoothing_release=0.450,
        onset_sensitivity=0.6,
        flash_decay_ms=150.0,
        flash_size=0.10,
        flash_intensity_scale=0.6,
        grain_strength=6.0,
        vignette_fraction=0.25,
        bass_color="#FF0000",
        treble_color="#00FFFF",
        background_color="#F2F0EF",
        outline_color="#0A0A0C",
    ),
    "aggressive": Preset(
        name="aggressive",
        description="Punchier deform and flash, faster attack, heavier grain/vignette",
        deform_amplitude=0.15,
        scale_amplitude=0.35,
        smoothing_attack=0.015,
        smoothing_release=0.150,
        onset_sensitivity=1.4,
        flash_decay_ms=220.0,
        flash_size=0.26,
        flash_intensity_scale=1.4,
        grain_strength=26.0,
        vignette_fraction=0.60,
    ),
}


def get(name: str) -> Preset | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return sorted(BUILTIN.keys())
