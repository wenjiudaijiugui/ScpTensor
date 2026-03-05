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
