"""Regression tests for PCA dimensionality reduction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction import reduce_pca


def _make_container(X: np.ndarray) -> ScpContainer:
    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(X.shape[0])]})
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(X.shape[1])]})
    assay = Assay(var=var, layers={"imputed": ScpMatrix(X=X)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_reduce_pca_arpack_rejects_full_rank_request() -> None:
    X = np.random.default_rng(0).normal(size=(50, 20))
    container = _make_container(X)

    with pytest.raises(ValueError, match="requires n_components < min"):
        reduce_pca(
            container,
            assay_name="proteins",
            base_layer="imputed",
            n_components=20,
            svd_solver="arpack",
        )


def test_reduce_pca_auto_handles_n_components_equal_min_dim() -> None:
    X = np.random.default_rng(1).normal(size=(50, 20))
    container = _make_container(X)

    result = reduce_pca(
        container,
        assay_name="proteins",
        base_layer="imputed",
        n_components=20,
        svd_solver="auto",
    )

    assert result.assays["pca"].layers["X"].X.shape == (50, 20)
    ratios = result.assays["pca"].var["explained_variance_ratio"].to_numpy()
    assert np.all(np.isfinite(ratios))


def test_reduce_pca_covariance_eigh_zero_variance_is_finite() -> None:
    X = np.ones((100, 5), dtype=np.float64)
    container = _make_container(X)

    result = reduce_pca(
        container,
        assay_name="proteins",
        base_layer="imputed",
        n_components=5,
        svd_solver="covariance_eigh",
    )

    scores = result.assays["pca"].layers["X"].X
    ratios = result.assays["pca"].var["explained_variance_ratio"].to_numpy()

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(ratios))
    assert np.allclose(ratios, 0.0)


def test_reduce_pca_invalid_solver_raises() -> None:
    X = np.random.default_rng(2).normal(size=(20, 10))
    container = _make_container(X)

    with pytest.raises(ValueError, match="Invalid svd_solver"):
        reduce_pca(
            container,
            assay_name="proteins",
            base_layer="imputed",
            n_components=5,
            svd_solver="invalid",  # type: ignore[arg-type]
        )


def test_reduce_pca_resolves_alias_without_creating_duplicate_source_assay() -> None:
    X = np.random.default_rng(3).normal(size=(20, 10))
    container = _make_container(X)

    result = reduce_pca(
        container,
        assay_name="protein",
        base_layer="imputed",
        n_components=5,
    )

    assert "protein" not in result.assays
    assert "proteins" in result.assays
    assert result.assays["proteins"] is not container.assays["proteins"]
    assert "pca_PC1_loading" in result.assays["proteins"].var.columns

    params = result.history[-1].params
    assert params["source_assay"] == "proteins"
    assert params["source_layer"] == "imputed"
    assert params["target_assay"] == "pca"


def test_reduce_pca_freezes_copy_and_source_mutation_contract() -> None:
    X = np.random.default_rng(4).normal(size=(20, 10))
    container = _make_container(X)

    result = reduce_pca(
        container,
        assay_name="proteins",
        base_layer="imputed",
        n_components=5,
    )

    assert result is not container
    assert result.obs is container.obs
    assert result.history is not container.history
    assert result.assays is not container.assays

    assert result.assays["proteins"] is not container.assays["proteins"]
    assert result.assays["proteins"].layers is not container.assays["proteins"].layers
    assert (
        result.assays["proteins"].layers["imputed"]
        is container.assays["proteins"].layers["imputed"]
    )
    assert result.assays["proteins"].var is not container.assays["proteins"].var
    assert "pca_PC1_loading" not in container.assays["proteins"].var.columns
    assert "pca_PC1_loading" in result.assays["proteins"].var.columns
