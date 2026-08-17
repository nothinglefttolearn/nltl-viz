from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Palette:
    name: str
    description: str = ""
    background_color: str = "#0A0A0C"
    outline_color: str = "#D8D8D2"
    flash_primary_color: str = "#6E5470"
    # None => flash is monochrome (flash_primary_color only, no centroid lerp).
    # See audio.precompute_flash_signal.
    flash_secondary_color: str | None = None


FIELD_NAMES = tuple(f.name for f in fields(Palette))

BUILTIN: dict[str, Palette] = {
    "black": Palette(
        name="black",
        description="Dark background, off-white outline, flash lerps NLTL violet to teal by spectral centroid",
        background_color="#0A0A0C",
        outline_color="#D8D8D2",
        flash_primary_color="#6E5470",
        flash_secondary_color="#2E8C8A",
    ),
    "violet": Palette(
        name="violet",
        description="Neutral off-white background, NLTL violet outline and flash",
        background_color="#F2F0EF",
        outline_color="#6E5470",
        flash_primary_color="#6E5470",
    ),
    "teal": Palette(
        name="teal",
        description="Neutral off-white background, NLTL teal outline and flash",
        background_color="#F2F0EF",
        outline_color="#2E8C8A",
        flash_primary_color="#2E8C8A",
    ),
}


def get(name: str) -> Palette | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return sorted(BUILTIN.keys())
