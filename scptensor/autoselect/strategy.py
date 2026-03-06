"""Strategy presets for AutoSelect method ranking.

This module centralizes strategy definitions used to combine method quality
and runtime into a final selection score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyPreset:
    """Preset weights for selection scoring."""

    name: str
    quality_weight: float
    runtime_weight: float
    description: str = ""


_STRATEGY_ORDER = ("speed", "balanced", "quality")

STRATEGY_PRESETS: dict[str, StrategyPreset] = {
    "quality": StrategyPreset(
        name="quality",
        quality_weight=1.0,
        runtime_weight=0.0,
        description="Prioritize score quality only.",
    ),
    "balanced": StrategyPreset(
        name="balanced",
        quality_weight=0.85,
        runtime_weight=0.15,
        description="Prefer quality while mildly favoring faster methods.",
    ),
    "speed": StrategyPreset(
        name="speed",
        quality_weight=0.65,
        runtime_weight=0.35,
        description="Favor faster methods while retaining quality signal.",
    ),
}


def list_strategy_presets() -> list[str]:
    """Return supported strategy names in canonical display order."""
    return list(_STRATEGY_ORDER)


def get_strategy_preset(name: str) -> StrategyPreset:
    """Look up a strategy preset by name with actionable errors."""
    strategy_name = str(name).strip().lower()
    preset = STRATEGY_PRESETS.get(strategy_name)
    if preset is None:
        supported = ", ".join(f"'{strategy}'" for strategy in _STRATEGY_ORDER)
        raise ValueError(f"selection_strategy must be one of {{{supported}}}, got '{name}'")
    return preset


__all__ = [
    "StrategyPreset",
    "STRATEGY_PRESETS",
    "get_strategy_preset",
    "list_strategy_presets",
]
