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


def test_black_palette_keeps_duotone_flash():
    assert BUILTIN["black"].flash_primary_color == "#6E5470"
    assert BUILTIN["black"].flash_secondary_color == "#2E8C8A"


def test_no_config_path_uses_builtin_directly():
    resolved = config.resolve_palette("violet", None)
    assert resolved.name == "violet"
