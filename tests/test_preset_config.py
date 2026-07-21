import pytest

from nltl_viz import config
from nltl_viz.preset import BUILTIN


def test_config_preset_overrides_builtin_of_same_name(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
presets:
  - name: industrial
    description: custom override
    deform_amplitude: 0.99
"""
    )
    resolved = config.resolve_preset("industrial", yaml_path)
    assert resolved.description == "custom override"
    assert resolved.deform_amplitude == 0.99
    assert resolved.deform_amplitude != BUILTIN["industrial"].deform_amplitude


def test_name_missing_from_config_falls_back_to_builtin(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
presets:
  - name: my-preset
    description: something else
"""
    )
    resolved = config.resolve_preset("subtle", yaml_path)
    assert resolved.name == "subtle"
    assert resolved.description == BUILTIN["subtle"].description


def test_name_absent_from_both_lists_all_available(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
presets:
  - name: my-preset
    description: something else
"""
    )
    with pytest.raises(ValueError) as exc_info:
        config.resolve_preset("nonexistent", yaml_path)
    message = str(exc_info.value)
    assert "my-preset" in message
    assert "industrial" in message
    assert "subtle" in message
    assert "aggressive" in message


def test_unknown_yaml_field_raises_readable_error(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
presets:
  - name: my-preset
    not_a_real_field: 123
"""
    )
    with pytest.raises(ValueError) as exc_info:
        config.load(yaml_path)
    message = str(exc_info.value)
    assert "not_a_real_field" in message
    assert "deform_amplitude" in message  # names a valid field


def test_no_config_path_uses_builtin_directly():
    resolved = config.resolve_preset("aggressive", None)
    assert resolved.name == "aggressive"
