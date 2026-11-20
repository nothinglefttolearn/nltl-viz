from __future__ import annotations

from dataclasses import dataclass, fields

import nltl_values

# NLTL colors come from nltl-values' `art` group only (pinned in
# pyproject.toml) — never `brand.*` or `web.*`. The one local color is
# black's cool-grey outline, which has no nltl-values equivalent.
PRIMARY = nltl_values.color("art.primary")
SECONDARY = nltl_values.color("art.secondary")
BG_DARK = nltl_values.color("art.bg-dark")
BG_LIGHT = nltl_values.color("art.bg-light")
CANVAS = nltl_values.color("art.canvas")
OUTLINE_GREY = "#D8D8D2"


@dataclass(frozen=True)
class Palette:
    name: str
    description: str = ""
    background_color: str = BG_DARK
    outline_color: str = OUTLINE_GREY
    flash_primary_color: str = PRIMARY
    # None => flash is monochrome (flash_primary_color only, no centroid lerp).
    # See audio.precompute_flash_signal.
    flash_secondary_color: str | None = None


FIELD_NAMES = tuple(f.name for f in fields(Palette))

BUILTIN: dict[str, Palette] = {
    "black": Palette(
        name="black",
        description="Dark background, off-white outline, flash lerps NLTL primary to secondary by spectral centroid",
        background_color=BG_DARK,
        outline_color=OUTLINE_GREY,
        flash_primary_color=PRIMARY,
        flash_secondary_color=SECONDARY,
    ),
    "grey": Palette(
        name="grey",
        description="Mid warm-grey background, canvas outline, flash lerps NLTL primary to secondary by spectral centroid",
        background_color=BG_LIGHT,
        outline_color=CANVAS,
        flash_primary_color=PRIMARY,
        flash_secondary_color=SECONDARY,
    ),
    "violet": Palette(
        name="violet",
        description="Warm canvas background, NLTL primary outline and flash",
        background_color=CANVAS,
        outline_color=PRIMARY,
        flash_primary_color=PRIMARY,
    ),
    "teal": Palette(
        name="teal",
        description="Warm canvas background, NLTL secondary outline and flash",
        background_color=CANVAS,
        outline_color=SECONDARY,
        flash_primary_color=SECONDARY,
    ),
}


def get(name: str) -> Palette | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return sorted(BUILTIN.keys())
