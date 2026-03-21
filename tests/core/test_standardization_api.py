"""Tests for stable standardization namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.standardization as stable_standardization
from scptensor.standardization import zscore
from scptensor.standardization.zscore import zscore as zscore_core


def test_stable_standardization_namespace_all_freezes_package_surface() -> None:
    assert stable_standardization.__all__ == ["zscore"]


def test_zscore_is_reexported_from_stable_standardization_namespace() -> None:
    assert zscore is zscore_core


def test_zscore_is_reexported_from_top_level_package() -> None:
    assert scp.zscore is zscore_core
    assert "zscore" in scp.__all__
