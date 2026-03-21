"""Tests for stable normalization namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.normalization as stable_normalization
from scptensor.normalization import (
    apply_normalization,
    create_result_layer,
    ensure_dense,
    get_layer_name,
    log_operation,
    norm_mean,
    norm_median,
    norm_none,
    norm_quantile,
    norm_trqn,
    normalize,
    validate_assay_and_layer,
)
from scptensor.normalization.api import norm_none as norm_none_core
from scptensor.normalization.api import normalize as normalize_core
from scptensor.normalization.base import apply_normalization as apply_normalization_core
from scptensor.normalization.base import create_result_layer as create_result_layer_core
from scptensor.normalization.base import ensure_dense as ensure_dense_core
from scptensor.normalization.base import get_layer_name as get_layer_name_core
from scptensor.normalization.base import log_operation as log_operation_core
from scptensor.normalization.base import (
    validate_assay_and_layer as validate_assay_and_layer_core,
)
from scptensor.normalization.mean_normalization import norm_mean as norm_mean_core
from scptensor.normalization.median_normalization import norm_median as norm_median_core
from scptensor.normalization.quantile_normalization import norm_quantile as norm_quantile_core
from scptensor.normalization.trqn_normalization import norm_trqn as norm_trqn_core


def test_stable_normalization_namespace_all_freezes_package_surface() -> None:
    assert stable_normalization.__all__ == [
        "norm_none",
        "norm_mean",
        "norm_median",
        "norm_quantile",
        "norm_trqn",
        "normalize",
        "apply_normalization",
        "create_result_layer",
        "ensure_dense",
        "get_layer_name",
        "log_operation",
        "validate_assay_and_layer",
    ]


def test_stable_normalization_namespace_reexports_current_implementations() -> None:
    assert norm_none is norm_none_core
    assert norm_mean is norm_mean_core
    assert norm_median is norm_median_core
    assert norm_quantile is norm_quantile_core
    assert norm_trqn is norm_trqn_core
    assert normalize is normalize_core
    assert apply_normalization is apply_normalization_core
    assert create_result_layer is create_result_layer_core
    assert ensure_dense is ensure_dense_core
    assert get_layer_name is get_layer_name_core
    assert log_operation is log_operation_core
    assert validate_assay_and_layer is validate_assay_and_layer_core


def test_top_level_package_reexports_only_user_facing_normalization_api() -> None:
    assert scp.norm_none is norm_none_core
    assert scp.norm_mean is norm_mean_core
    assert scp.norm_median is norm_median_core
    assert scp.norm_quantile is norm_quantile_core
    assert scp.norm_trqn is norm_trqn_core
    assert scp.normalize is normalize_core

    for name in (
        "norm_none",
        "norm_mean",
        "norm_median",
        "norm_quantile",
        "norm_trqn",
        "normalize",
    ):
        assert name in scp.__all__

    for name in (
        "apply_normalization",
        "create_result_layer",
        "ensure_dense",
        "get_layer_name",
        "log_operation",
        "validate_assay_and_layer",
    ):
        assert name not in scp.__all__
