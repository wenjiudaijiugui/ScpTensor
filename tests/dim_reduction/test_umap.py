"""Regression tests for UMAP dimensionality reduction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction import reduce_umap


def _make_container(n_samples: int = 30, n_features: int = 12) -> ScpContainer:
    X = np.random.default_rng(0).normal(size=(n_samples, n_features)).astype(np.float64)
    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})
    assay = Assay(var=var, layers={"imputed": ScpMatrix(X=X)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_reduce_umap_requires_n_neighbors_gt_one() -> None:
    container = _make_container()

    with pytest.raises(ValueError, match="n_neighbors must be > 1"):
        reduce_umap(
            container,
            assay_name="proteins",
            base_layer="imputed",
            n_neighbors=1,
        )


def test_reduce_umap_requires_n_neighbors_less_than_n_samples() -> None:
    container = _make_container(n_samples=10)

    with pytest.raises(ValueError, match="must be < n_samples"):
        reduce_umap(
            container,
            assay_name="proteins",
            base_layer="imputed",
            n_neighbors=10,
        )


def test_reduce_umap_accepts_min_dist_one() -> None:
    container = _make_container()

    result = reduce_umap(
        container,
        assay_name="proteins",
        base_layer="imputed",
        min_dist=1.0,
        n_neighbors=5,
        random_state=42,
    )

    assert result.assays["umap"].layers["X"].X.shape == (30, 2)


def test_reduce_umap_reuses_existing_assays_without_deepcopy() -> None:
    container = _make_container()

    result = reduce_umap(
        container,
        assay_name="proteins",
        base_layer="imputed",
        n_neighbors=5,
        random_state=42,
    )

    assert result.assays["proteins"] is container.assays["proteins"]


def test_reduce_umap_uses_unified_history_keys_with_resolved_assay() -> None:
    container = _make_container()

    result = reduce_umap(
        container,
        assay_name="protein",
        base_layer="imputed",
        n_neighbors=5,
        random_state=42,
    )

    params = result.history[-1].params
    assert params["source_assay"] == "proteins"
    assert params["source_layer"] == "imputed"
    assert params["target_assay"] == "umap"
    assert "assay_name" not in params
    assert "base_layer" not in params


def test_reduce_umap_freezes_copy_contract() -> None:
    container = _make_container()

    result = reduce_umap(
        container,
        assay_name="proteins",
        base_layer="imputed",
        n_neighbors=5,
        random_state=42,
    )

    assert result is not container
    assert result.obs is container.obs
    assert result.history is not container.history
    assert result.assays is not container.assays
    assert result.assays["proteins"] is container.assays["proteins"]
    assert "umap" not in container.assays
    assert "umap" in result.assays
