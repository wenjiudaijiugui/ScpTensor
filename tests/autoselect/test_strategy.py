"""Tests for autoselect strategy presets."""

import pytest

from scptensor.autoselect.strategy import (
    STRATEGY_PRESETS,
    get_strategy_preset,
    list_strategy_presets,
)


def test_list_strategy_presets_order():
    """Supported strategies should be listed in canonical order."""
    assert list_strategy_presets() == ["speed", "balanced", "quality"]


def test_get_strategy_preset_success():
    """Lookup returns configured preset weights."""
    quality = get_strategy_preset("quality")
    balanced = get_strategy_preset("balanced")
    speed = get_strategy_preset("speed")

    assert quality.quality_weight == pytest.approx(1.0)
    assert quality.runtime_weight == pytest.approx(0.0)
    assert balanced.quality_weight == pytest.approx(0.85)
    assert balanced.runtime_weight == pytest.approx(0.15)
    assert speed.quality_weight == pytest.approx(0.65)
    assert speed.runtime_weight == pytest.approx(0.35)


def test_get_strategy_preset_case_insensitive():
    """Strategy names are normalized to lowercase canonical names."""
    preset = get_strategy_preset("  SPEED ")
    assert preset.name == "speed"


def test_get_strategy_preset_invalid():
    """Invalid strategy should raise actionable ValueError."""
    with pytest.raises(ValueError, match="selection_strategy must be one of"):
        get_strategy_preset("unknown")


def test_presets_registered_for_all_listed_names():
    """Every listed strategy should map to a preset entry."""
    for name in list_strategy_presets():
        assert name in STRATEGY_PRESETS
