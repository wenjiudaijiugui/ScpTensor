"""Tests for t-SNE dimensionality reduction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction import reduce_tsne


def _make_container(
    *, n_samples: int = 30, with_nan: bool = False, use_sparse: bool = False
) -> ScpContainer:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 15)).astype(np.float64)

    if with_nan:
        X[0, 0] = np.nan

    X_layer: np.ndarray | sparse.csr_matrix
    if use_sparse:
        X_layer = sparse.csr_matrix(X)
    else:
        X_layer = X

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(15)]})

    assay = Assay(var=var, layers={"imputed": ScpMatrix(X=X_layer)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_reduce_tsne_basic() -> None:
    container = _make_container()

    result = reduce_tsne(
        container,
        assay_name="proteins",
        base_layer="imputed",
        perplexity=5.0,
        max_iter=250,
        random_state=42,
    )

    assert "tsne" in result.assays
    assert "tsne" not in container.assays

    tsne_assay = result.assays["tsne"]
    assert "X" in tsne_assay.layers
    assert tsne_assay.layers["X"].X.shape == (30, 2)
    assert tsne_assay.var["feature_id"].to_list() == ["TSNE_1", "TSNE_2"]
    assert result.history[-1].action == "reduce_tsne"


def test_reduce_tsne_custom_assay_name() -> None:
    container = _make_container()

    result = reduce_tsne(
        container,
        assay_name="proteins",
        base_layer="imputed",
        new_assay_name="embedding_tsne",
        perplexity=5.0,
        max_iter=250,
    )

    assert "embedding_tsne" in result.assays


def test_reduce_tsne_sparse_input() -> None:
    container = _make_container(use_sparse=True)

    result = reduce_tsne(
        container,
        assay_name="proteins",
        base_layer="imputed",
        perplexity=5.0,
        max_iter=250,
    )

    assert result.assays["tsne"].layers["X"].X.shape == (30, 2)
    assert result.assays["proteins"] is container.assays["proteins"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components must be positive"),
        ({"perplexity": 0.0}, "perplexity must be positive"),
        ({"early_exaggeration": 0.0}, "early_exaggeration must be positive"),
        ({"learning_rate": -1.0}, "learning_rate must be positive or 'auto'"),
        ({"max_iter": 100}, "max_iter must be >= 250"),
        ({"n_components": 4, "method": "barnes_hut"}, "n_components must be <= 3"),
        ({"init": "bad_init"}, "init must be one of"),
        ({"method": "bad_method"}, "method must be one of"),
    ],
)
def test_reduce_tsne_parameter_validation(kwargs: dict[str, int | float | str], match: str) -> None:
    container = _make_container()
    run_kwargs: dict[str, int | float | str] = {"perplexity": 5.0, "max_iter": 250}
    run_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        reduce_tsne(
            container,
            assay_name="proteins",
            base_layer="imputed",
            **run_kwargs,
        )


def test_reduce_tsne_init_pca_requires_valid_component_count() -> None:
    container = _make_container(n_samples=20)

    with pytest.raises(ValueError, match="init='pca' requires n_components <="):
        reduce_tsne(
            container,
            assay_name="proteins",
            base_layer="imputed",
            n_components=21,
            perplexity=5.0,
            max_iter=250,
            init="pca",
            method="exact",
        )


def test_reduce_tsne_perplexity_must_be_smaller_than_n_samples() -> None:
    container = _make_container(n_samples=20)

    with pytest.raises(ValueError, match="must be < n_samples"):
        reduce_tsne(
            container,
            assay_name="proteins",
            base_layer="imputed",
            perplexity=20.0,
            max_iter=250,
        )


def test_reduce_tsne_missing_assay_or_layer() -> None:
    container = _make_container()

    with pytest.raises(AssayNotFoundError):
        reduce_tsne(
            container,
            assay_name="missing_assay",
            base_layer="imputed",
            perplexity=5.0,
            max_iter=250,
        )

    with pytest.raises(LayerNotFoundError):
        reduce_tsne(
            container,
            assay_name="proteins",
            base_layer="missing_layer",
            perplexity=5.0,
            max_iter=250,
        )


def test_reduce_tsne_rejects_nan_input() -> None:
    container = _make_container(with_nan=True)

    with pytest.raises(ValidationError, match="NaN"):
        reduce_tsne(
            container,
            assay_name="proteins",
            base_layer="imputed",
            perplexity=5.0,
            max_iter=250,
        )


def test_reduce_tsne_uses_unified_history_keys_with_resolved_assay() -> None:
    container = _make_container()

    result = reduce_tsne(
        container,
        assay_name="protein",
        base_layer="imputed",
        perplexity=5.0,
        max_iter=250,
    )

    params = result.history[-1].params
    assert params["source_assay"] == "proteins"
    assert params["source_layer"] == "imputed"
    assert params["target_assay"] == "tsne"
    assert "assay_name" not in params
    assert "base_layer" not in params


def test_reduce_tsne_freezes_copy_contract() -> None:
    container = _make_container()

    result = reduce_tsne(
        container,
        assay_name="proteins",
        base_layer="imputed",
        perplexity=5.0,
        max_iter=250,
    )

    assert result is not container
    assert result.obs is container.obs
    assert result.history is not container.history
    assert result.assays is not container.assays
    assert result.assays["proteins"] is container.assays["proteins"]
    assert "tsne" not in container.assays
    assert "tsne" in result.assays
