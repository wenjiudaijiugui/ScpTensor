"""Tests for stable impute namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.impute as stable_impute
from scptensor.impute import (
    impute,
    impute_bpca,
    impute_half_row_min,
    impute_iterative_svd,
    impute_knn,
    impute_lls,
    impute_mf,
    impute_minprob,
    impute_none,
    impute_qrilc,
    impute_row_mean,
    impute_row_median,
    impute_softimpute,
    impute_zero,
    infer_missing_mechanism,
    list_impute_methods,
    recommend_impute_method,
)
from scptensor.impute.base import impute as impute_core
from scptensor.impute.base import infer_missing_mechanism as infer_missing_mechanism_core
from scptensor.impute.base import list_impute_methods as list_impute_methods_core
from scptensor.impute.base import recommend_impute_method as recommend_impute_method_core
from scptensor.impute.baseline import impute_half_row_min as impute_half_row_min_core
from scptensor.impute.baseline import impute_none as impute_none_core
from scptensor.impute.baseline import impute_row_mean as impute_row_mean_core
from scptensor.impute.baseline import impute_row_median as impute_row_median_core
from scptensor.impute.baseline import impute_zero as impute_zero_core
from scptensor.impute.bpca import impute_bpca as impute_bpca_core
from scptensor.impute.knn import impute_knn as impute_knn_core
from scptensor.impute.lls import impute_lls as impute_lls_core
from scptensor.impute.minprob import impute_minprob as impute_minprob_core
from scptensor.impute.missforest import impute_mf as impute_mf_core
from scptensor.impute.qrilc import impute_qrilc as impute_qrilc_core
from scptensor.impute.svd import impute_iterative_svd as impute_iterative_svd_core
from scptensor.impute.svd import impute_softimpute as impute_softimpute_core


def test_stable_impute_namespace_all_freezes_package_surface() -> None:
    assert stable_impute.__all__ == [
        "impute_none",
        "impute_zero",
        "impute_row_mean",
        "impute_row_median",
        "impute_half_row_min",
        "impute_knn",
        "impute_bpca",
        "impute_mf",
        "impute_lls",
        "impute_iterative_svd",
        "impute_softimpute",
        "impute_qrilc",
        "impute_minprob",
        "impute",
        "list_impute_methods",
        "infer_missing_mechanism",
        "recommend_impute_method",
    ]


def test_stable_impute_namespace_reexports_stable_implementations() -> None:
    assert impute_none is impute_none_core
    assert impute_zero is impute_zero_core
    assert impute_row_mean is impute_row_mean_core
    assert impute_row_median is impute_row_median_core
    assert impute_half_row_min is impute_half_row_min_core
    assert impute_knn is impute_knn_core
    assert impute_bpca is impute_bpca_core
    assert impute_mf is impute_mf_core
    assert impute_lls is impute_lls_core
    assert impute_iterative_svd is impute_iterative_svd_core
    assert impute_softimpute is impute_softimpute_core
    assert impute_qrilc is impute_qrilc_core
    assert impute_minprob is impute_minprob_core
    assert impute is impute_core
    assert list_impute_methods is list_impute_methods_core
    assert infer_missing_mechanism is infer_missing_mechanism_core
    assert recommend_impute_method is recommend_impute_method_core


def test_only_individual_imputation_wrappers_are_reexported_from_top_level_package() -> None:
    assert scp.impute_none is impute_none_core
    assert scp.impute_zero is impute_zero_core
    assert scp.impute_row_mean is impute_row_mean_core
    assert scp.impute_row_median is impute_row_median_core
    assert scp.impute_half_row_min is impute_half_row_min_core
    assert scp.impute_knn is impute_knn_core
    assert scp.impute_lls is impute_lls_core
    assert scp.impute_iterative_svd is impute_iterative_svd_core
    assert scp.impute_softimpute is impute_softimpute_core
    assert scp.impute_bpca is impute_bpca_core
    assert scp.impute_mf is impute_mf_core
    assert scp.impute_qrilc is impute_qrilc_core
    assert scp.impute_minprob is impute_minprob_core

    for name in (
        "impute_none",
        "impute_zero",
        "impute_row_mean",
        "impute_row_median",
        "impute_half_row_min",
        "impute_knn",
        "impute_lls",
        "impute_iterative_svd",
        "impute_softimpute",
        "impute_bpca",
        "impute_mf",
        "impute_qrilc",
        "impute_minprob",
    ):
        assert name in scp.__all__

    for name in (
        "impute",
        "list_impute_methods",
        "infer_missing_mechanism",
        "recommend_impute_method",
    ):
        assert name not in scp.__all__
