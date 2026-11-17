from __future__ import annotations

from dataclasses import dataclass, fields

import nltl_values

# Brand colors come from nltl-values (pinned in pyproject.toml). Background
# colors (and black's off-white outline) are still defined locally here.
PRIMARY = nltl_values.color("primary")
SECONDARY = nltl_values.color("secondary")


@dataclass(frozen=True)
class Palette:
    name: str
    description: str = ""
    background_color: str = "#0A0A0C"
    outline_color: str = "#D8D8D2"
    flash_primary_color: str = PRIMARY
    # None => flash is monochrome (flash_primary_color only, no centroid lerp).
    # See audio.precompute_flash_signal.
    flash_secondary_color: str | None = None


FIELD_NAMES = tuple(f.name for f in fields(Palette))

BUILTIN: dict[str, Palette] = {
    "black": Palette(
        name="black",
        description="Dark background, off-white outline, flash lerps NLTL primary to secondary by spectral centroid",
        background_color="#0A0A0C",
        outline_color="#D8D8D2",
        flash_primary_color=PRIMARY,
        flash_secondary_color=SECONDARY,
    ),
    "violet": Palette(
        name="violet",
        description="Neutral off-white background, NLTL primary outline and flash",
        background_color="#F2F0EF",
        outline_color=PRIMARY,
        flash_primary_color=PRIMARY,
    ),
    "teal": Palette(
        name="teal",
        description="Neutral off-white background, NLTL secondary outline and flash",
        background_color="#F2F0EF",
        outline_color=SECONDARY,
        flash_primary_color=SECONDARY,
    ),
}


def get(name: str) -> Palette | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return sorted(BUILTIN.keys())
