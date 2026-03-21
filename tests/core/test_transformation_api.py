"""Tests for stable transformation namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.transformation as stable_transformation
from scptensor.transformation import log_transform
from scptensor.transformation.log_transform import log_transform as log_transform_core


def test_stable_transformation_namespace_all_freezes_package_surface() -> None:
    assert stable_transformation.__all__ == ["log_transform"]


def test_log_transform_is_reexported_from_stable_transformation_namespace() -> None:
    assert log_transform is log_transform_core


def test_log_transform_is_reexported_from_top_level_package() -> None:
    assert scp.log_transform is log_transform_core
    assert "log_transform" in scp.__all__
