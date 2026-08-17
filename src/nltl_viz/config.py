from __future__ import annotations

from pathlib import Path

import yaml

from nltl_viz import palette as palette_mod
from nltl_viz import preset as preset_mod
from nltl_viz.palette import FIELD_NAMES as PALETTE_FIELD_NAMES
from nltl_viz.palette import Palette
from nltl_viz.preset import FIELD_NAMES as PRESET_FIELD_NAMES
from nltl_viz.preset import Preset
from nltl_viz.render import Motion


def _read_yaml(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ValueError(f"reading {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"parsing {path}: {exc}") from exc
    return raw or {}


def load(path: Path) -> list[Preset]:
    """Parse a YAML file's `presets:` list (motion-tuning presets) into Preset instances."""
    raw = _read_yaml(path)
    if "presets" not in raw:
        return []

    presets: list[Preset] = []
    for entry in raw["presets"]:
        unknown = set(entry) - set(PRESET_FIELD_NAMES)
        if unknown:
            valid = ", ".join(PRESET_FIELD_NAMES)
            raise ValueError(
                f"unknown preset field(s) {sorted(unknown)} in {path} — valid fields: {valid}"
            )
        if "motion" in entry:
            try:
                Motion(entry["motion"])
            except ValueError:
                valid_motions = ", ".join(m.value for m in Motion)
                raise ValueError(
                    f"preset {entry.get('name')!r} in {path} has invalid motion "
                    f"{entry['motion']!r} — valid: {valid_motions}"
                ) from None
        presets.append(Preset(**entry))
    return presets


def load_palettes(path: Path) -> list[Palette]:
    """Parse a YAML file's `palettes:` list (color schemes) into Palette instances."""
    raw = _read_yaml(path)
    if "palettes" not in raw:
        return []

    palettes: list[Palette] = []
    for entry in raw["palettes"]:
        unknown = set(entry) - set(PALETTE_FIELD_NAMES)
        if unknown:
            valid = ", ".join(PALETTE_FIELD_NAMES)
            raise ValueError(
                f"unknown palette field(s) {sorted(unknown)} in {path} — valid fields: {valid}"
            )
        palettes.append(Palette(**entry))
    return palettes


def resolve_preset(name: str, config_path: Path | None) -> Preset:
    """Look up `name` in the config file's presets first, then built-ins."""
    config_presets: list[Preset] = []
    if config_path is not None:
        config_presets = load(config_path)
        for p in config_presets:
            if p.name == name:
                return p

    builtin = preset_mod.get(name)
    if builtin is not None:
        return builtin

    available = sorted({p.name for p in config_presets} | set(preset_mod.names()))
    raise ValueError(
        f"preset {name!r} not found in config or built-ins — available: {', '.join(available)}"
    )


def resolve_palette(name: str, config_path: Path | None) -> Palette:
    """Look up `name` in the config file's palettes first, then built-ins."""
    config_palettes: list[Palette] = []
    if config_path is not None:
        config_palettes = load_palettes(config_path)
        for p in config_palettes:
            if p.name == name:
                return p

    builtin = palette_mod.get(name)
    if builtin is not None:
        return builtin

    available = sorted({p.name for p in config_palettes} | set(palette_mod.names()))
    raise ValueError(
        f"palette {name!r} not found in config or built-ins — available: {', '.join(available)}"
    )
