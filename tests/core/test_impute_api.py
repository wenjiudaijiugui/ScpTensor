"""Tests for stable impute namespace exports."""

from __future__ import annotations

import numpy as np
import polars as pl

import scptensor as scp
import scptensor.impute as stable_impute
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
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
)
from scptensor.impute.base import impute as impute_core
from scptensor.impute.base import infer_missing_mechanism as infer_missing_mechanism_core
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


def _make_container(assay_key: str = "protein") -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2", "s3", "s4"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = np.array(
        [
            [1.0, np.nan, 3.0],
            [1.1, 2.0, 3.1],
            [10.0, 11.0, np.nan],
            [9.9, 11.2, 12.0],
        ],
        dtype=np.float64,
    )
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=None)})
    return ScpContainer(obs=obs, assays={assay_key: assay})


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


def test_stable_impute_namespace_does_not_reexport_selection_helpers() -> None:
    for name in (
        "list_impute_methods",
        "infer_missing_mechanism",
        "recommend_impute_method",
    ):
        assert name not in stable_impute.__all__
        assert not hasattr(stable_impute, name)


def test_imputation_api_is_not_reexported_from_top_level_package() -> None:
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
        assert name not in scp.__all__
        assert not hasattr(scp, name)

    for name in (
        "impute",
        "list_impute_methods",
        "infer_missing_mechanism",
        "recommend_impute_method",
    ):
        assert name not in scp.__all__

    # Importing `scptensor.impute` binds the submodule on the parent package
    # as `scp.impute`, but that is not a top-level function reexport.
    assert not callable(getattr(scp, "impute", None))


def test_impute_none_history_uses_resolved_container_assay_key() -> None:
    container = _make_container(assay_key="proteins")

    impute_none(container, assay_name="protein", source_layer="raw", new_layer_name="passthrough")

    assert "passthrough" in container.assays["proteins"].layers
    assert container.history[-1].params["assay"] == "proteins"


def test_impute_knn_history_uses_resolved_assay_name_for_aliases() -> None:
    container = _make_container(assay_key="protein")

    impute_knn(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="knn_alias",
        k=1,
    )

    assert "knn_alias" in container.assays["protein"].layers
    assert container.history[-1].params["assay"] == "protein"


def test_infer_missing_mechanism_resolves_assay_alias() -> None:
    container = _make_container(assay_key="proteins")

    mechanism, reason = infer_missing_mechanism_core(
        container,
        assay_name="protein",
        source_layer="raw",
    )

    assert mechanism in {"mcar", "mar", "mnar"}
    assert "missing_rate=" in reason
