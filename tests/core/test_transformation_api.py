"""Tests for stable transformation namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.transformation as stable_transformation
from scptensor.transformation import log_transform
from scptensor.transformation.log_transform import log_transform as log_transform_core

_INTERNAL_HELPERS = (
    "create_result_layer",
    "validate_assay_and_layer",
)


def test_stable_transformation_namespace_all_freezes_package_surface() -> None:
    assert stable_transformation.__all__ == ["log_transform"]


def test_log_transform_is_reexported_from_stable_transformation_namespace() -> None:
    assert log_transform is log_transform_core


def test_stable_transformation_namespace_hides_internal_helpers() -> None:
    for name in _INTERNAL_HELPERS:
        assert name not in stable_transformation.__all__
        assert not hasattr(stable_transformation, name)


def test_log_transform_is_not_reexported_from_top_level_package() -> None:
    assert "log_transform" not in scp.__all__
    assert not hasattr(scp, "log_transform")
    for name in _INTERNAL_HELPERS:
        assert name not in scp.__all__
        assert not hasattr(scp, name)
