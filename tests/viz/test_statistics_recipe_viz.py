"""Tests for statistics recipe visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.statistics import (
    correlation_matrix,
    dendrogram,
    plot_correlation_matrix,
    plot_dendrogram,
)


@pytest.fixture
def stats_container() -> ScpContainer:
    rng = np.random.default_rng(42)
    n_samples = 15
    n_features = 8

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:02d}" for i in range(n_samples)],
            "batch": np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5),
        }
    )
    var = pl.DataFrame({"_index": [f"P{i:02d}" for i in range(n_features)]})

    x = rng.normal(size=(n_samples, n_features))
    # Add separable group shifts to stabilize correlation structure
    x[:5] += 0.0
    x[5:10] += 1.0
    x[10:] += 2.0

    assay = Assay(var=var, layers={"norm": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_correlation_matrix_groupby_pearson_and_alias(stats_container: ScpContainer) -> None:
    plt.close("all")
    fig = correlation_matrix(
        stats_container,
        layer="norm",
        assay_name="proteins",
        groupby="batch",
        method="pearson",
        annot=True,
        show=False,
    )
    assert fig is not None
    assert len(fig.axes) >= 1

    fig_alias = plot_correlation_matrix(
        stats_container,
        layer="norm",
        assay_name="proteins",
        groupby="batch",
        show=False,
    )
    assert fig_alias is not None
    plt.close("all")


def test_correlation_matrix_spearman_samplewise(stats_container: ScpContainer) -> None:
    plt.close("all")
    fig = correlation_matrix(
        stats_container,
        layer="norm",
        assay_name="proteins",
        groupby=None,
        method="spearman",
        annot=False,
        show=False,
    )
    assert fig is not None
    plt.close("all")


def test_correlation_matrix_single_group_raises() -> None:
    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(5)], "batch": ["A"] * 5})
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(4)]})
    x = np.arange(20, dtype=float).reshape(5, 4)
    container = ScpContainer(
        obs=obs, assays={"proteins": Assay(var=var, layers={"norm": ScpMatrix(X=x)})}
    )

    with pytest.raises(ValueError, match="Need at least 2 groups for correlation matrix"):
        correlation_matrix(
            container, layer="norm", assay_name="proteins", groupby="batch", show=False
        )


def test_dendrogram_groupby_and_alias(stats_container: ScpContainer) -> None:
    plt.close("all")
    fig = dendrogram(
        stats_container,
        layer="norm",
        assay_name="proteins",
        groupby="batch",
        method="average",
        metric="euclidean",
        show=False,
    )
    assert fig is not None
    assert len(fig.axes) == 1
    assert fig.axes[0].legend_ is not None

    fig_alias = plot_dendrogram(
        stats_container,
        layer="norm",
        assay_name="proteins",
        show=False,
    )
    assert fig_alias is not None
    plt.close("all")


def test_dendrogram_ward_non_euclidean_raises(stats_container: ScpContainer) -> None:
    with pytest.raises(ValueError, match="Ward linkage only works with euclidean metric"):
        dendrogram(
            stats_container,
            layer="norm",
            assay_name="proteins",
            method="ward",
            metric="cityblock",
            show=False,
        )


def test_dendrogram_sample_limit_path() -> None:
    rng = np.random.default_rng(123)
    n_samples = 120
    n_features = 6

    obs = pl.DataFrame({"_index": [f"S{i:03d}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"P{i:02d}" for i in range(n_features)]})
    x = rng.normal(size=(n_samples, n_features))

    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"norm": ScpMatrix(X=x)})},
    )

    plt.close("all")
    fig = dendrogram(container, layer="norm", assay_name="proteins", show=False)
    assert fig is not None
    plt.close("all")
