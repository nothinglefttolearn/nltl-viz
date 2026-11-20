import nltl_values
import pytest

from nltl_viz import config
from nltl_viz.palette import BUILTIN


def test_config_palette_overrides_builtin_of_same_name(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
palettes:
  - name: black
    description: custom override
    outline_color: "#FFFFFF"
"""
    )
    resolved = config.resolve_palette("black", yaml_path)
    assert resolved.description == "custom override"
    assert resolved.outline_color == "#FFFFFF"
    assert resolved.outline_color != BUILTIN["black"].outline_color


def test_name_missing_from_config_falls_back_to_builtin(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
palettes:
  - name: my-palette
    description: something else
"""
    )
    resolved = config.resolve_palette("teal", yaml_path)
    assert resolved.name == "teal"
    assert resolved.description == BUILTIN["teal"].description


def test_name_absent_from_both_lists_all_available(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
palettes:
  - name: my-palette
    description: something else
"""
    )
    with pytest.raises(ValueError) as exc_info:
        config.resolve_palette("nonexistent", yaml_path)
    message = str(exc_info.value)
    assert "my-palette" in message
    assert "black" in message
    assert "violet" in message
    assert "teal" in message


def test_unknown_yaml_field_raises_readable_error(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
palettes:
  - name: my-palette
    not_a_real_field: 123
"""
    )
    with pytest.raises(ValueError) as exc_info:
        config.load_palettes(yaml_path)
    message = str(exc_info.value)
    assert "not_a_real_field" in message
    assert "background_color" in message  # names a valid field


def test_monochrome_palette_has_no_secondary_color():
    assert BUILTIN["violet"].flash_secondary_color is None
    assert BUILTIN["teal"].flash_secondary_color is None


def test_violet_and_teal_use_nltl_values_primary_and_secondary():
    primary, secondary = nltl_values.color("art.primary"), nltl_values.color("art.secondary")
    assert BUILTIN["violet"].outline_color == BUILTIN["violet"].flash_primary_color == primary
    assert BUILTIN["teal"].outline_color == BUILTIN["teal"].flash_primary_color == secondary


def test_black_palette_keeps_duotone_flash():
    assert BUILTIN["black"].flash_primary_color == nltl_values.color("art.primary")
    assert BUILTIN["black"].flash_secondary_color == nltl_values.color("art.secondary")


def test_no_config_path_uses_builtin_directly():
    resolved = config.resolve_palette("violet", None)
    assert resolved.name == "violet"


def test_grey_palette_mirrors_black_on_bg_light():
    grey = BUILTIN["grey"]
    assert grey.background_color == nltl_values.color("art.bg-light")
    assert grey.outline_color == nltl_values.color("art.canvas")
    assert grey.flash_primary_color == nltl_values.color("art.primary")
    assert grey.flash_secondary_color == nltl_values.color("art.secondary")


def test_backgrounds_come_from_nltl_values_art_group():
    assert BUILTIN["black"].background_color == nltl_values.color("art.bg-dark")
    assert BUILTIN["violet"].background_color == nltl_values.color("art.canvas")
    assert BUILTIN["teal"].background_color == nltl_values.color("art.canvas")


def test_every_art_color_is_used_by_a_builtin_palette():
    used = {
        value
        for p in BUILTIN.values()
        for value in (
            p.background_color,
            p.outline_color,
            p.flash_primary_color,
            p.flash_secondary_color,
        )
    }
    art = {v for k, v in nltl_values.colors().items() if k.startswith("art.")}
    assert art <= used
