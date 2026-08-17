from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Preset:
    name: str
    description: str = ""
    motion: str = "deform"  # one of Motion's values ("deform", "rigid", "pulse", "split") — see render.Motion
    band_count: int = 32
    deform_amplitude: float = 0.0
    scale_amplitude: float = 0.0
    pulse_max_opacity: float = 0.85
    split_amplitude: float = 0.0
    smoothing_attack: float = 0.03
    smoothing_release: float = 0.25
    onset_sensitivity: float = 1.0
    flash_decay_ms: float = 200.0
    flash_size: float = 0.18
    flash_intensity_scale: float = 1.0
    grain_strength: float = 14.0
    vignette_fraction: float = 0.45


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
    "shutter": Preset(
        name="shutter",
        description="Horizontal split: top half slides left, bottom half slides right, snapping back to center with loudness",
        motion="split",
        split_amplitude=0.28,
        smoothing_attack=0.02,
        smoothing_release=0.18,
        grain_strength=14.0,
        vignette_fraction=0.45,
    ),
}


def get(name: str) -> Preset | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return sorted(BUILTIN.keys())
