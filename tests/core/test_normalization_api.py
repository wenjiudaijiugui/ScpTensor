"""Tests for stable normalization namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.normalization as stable_normalization
from scptensor.normalization import (
    norm_mean,
    norm_median,
    norm_none,
    norm_quantile,
    norm_trqn,
    normalize,
)
from scptensor.normalization.api import norm_none as norm_none_core
from scptensor.normalization.api import normalize as normalize_core
from scptensor.normalization.mean_normalization import norm_mean as norm_mean_core
from scptensor.normalization.median_normalization import norm_median as norm_median_core
from scptensor.normalization.quantile_normalization import norm_quantile as norm_quantile_core
from scptensor.normalization.trqn_normalization import norm_trqn as norm_trqn_core

_INTERNAL_HELPERS = (
    "apply_normalization",
    "create_result_layer",
    "ensure_dense",
    "get_layer_name",
    "log_operation",
    "validate_assay_and_layer",
)


def test_stable_normalization_namespace_all_exposes_only_user_facing_api() -> None:
    assert stable_normalization.__all__ == [
        "norm_none",
        "norm_mean",
        "norm_median",
        "norm_quantile",
        "norm_trqn",
        "normalize",
    ]


def test_stable_normalization_namespace_reexports_current_user_facing_implementations() -> None:
    assert norm_none is norm_none_core
    assert norm_mean is norm_mean_core
    assert norm_median is norm_median_core
    assert norm_quantile is norm_quantile_core
    assert norm_trqn is norm_trqn_core
    assert normalize is normalize_core


def test_stable_normalization_namespace_hides_internal_helpers() -> None:
    for name in _INTERNAL_HELPERS:
        assert name not in stable_normalization.__all__
        assert not hasattr(stable_normalization, name)


def test_top_level_package_reexports_only_user_facing_normalization_api() -> None:
    for name in (
        "norm_none",
        "norm_mean",
        "norm_median",
        "norm_quantile",
        "norm_trqn",
        "normalize",
    ):
        assert name not in scp.__all__
        assert not hasattr(scp, name)

    for name in _INTERNAL_HELPERS:
        assert name not in scp.__all__
        assert not hasattr(scp, name)
