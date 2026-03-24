"""Tests for advanced QC visualization recipes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.qc_advanced import (
    plot_cumulative_sensitivity,
    plot_cv_by_feature,
    plot_cv_comparison,
    plot_cv_distribution,
    plot_jaccard_heatmap,
    plot_missing_summary,
    plot_missing_type_heatmap,
    plot_sensitivity_summary,
)


@pytest.fixture
def qc_advanced_container() -> ScpContainer:
    rng = np.random.default_rng(42)
    n_samples = 12
    n_features = 10

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:02d}" for i in range(n_samples)],
            "batch": ["A"] * 6 + ["B"] * 6,
        },
    )
    var = pl.DataFrame({"_index": [f"P{i:02d}" for i in range(n_features)]})

    x = rng.uniform(10, 100, size=(n_samples, n_features))
    m = rng.choice([0, 1, 2, 5], size=(n_samples, n_features), p=[0.75, 0.1, 0.1, 0.05]).astype(
        np.int8,
    )
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


@pytest.fixture
def qc_container_mask_none(qc_advanced_container: ScpContainer) -> ScpContainer:
    container = qc_advanced_container.copy()
    x = container.assays["proteins"].layers["raw"].X.copy()
    container.assays["proteins"].layers["raw"] = ScpMatrix(X=x, M=None)
    return container


@pytest.fixture
def qc_container_all_missing(qc_advanced_container: ScpContainer) -> ScpContainer:
    container = qc_advanced_container.copy()
    x = container.assays["proteins"].layers["raw"].X.copy()
    m = np.ones_like(x, dtype=np.int8)
    container.assays["proteins"].layers["raw"] = ScpMatrix(X=x, M=m)
    return container


def test_plot_sensitivity_summary_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_sensitivity_summary(qc_advanced_container, group_by="batch")
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_sensitivity_summary_mask_none(qc_container_mask_none: ScpContainer) -> None:
    plt.close("all")
    ax = plot_sensitivity_summary(qc_container_mask_none, group_by="batch")
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_cumulative_sensitivity_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_cumulative_sensitivity(
        qc_advanced_container,
        group_by="batch",
        show_saturation=True,
        saturation_threshold=0.8,
    )
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_jaccard_heatmap_low_similarity_mode(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_jaccard_heatmap(
        qc_advanced_container,
        cluster=True,
        show_low_similarity_only=True,
        similarity_threshold=0.5,
    )
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_missing_type_heatmap_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_missing_type_heatmap(
        qc_advanced_container,
        cluster_samples=True,
        cluster_features=True,
        max_samples=12,
    )
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_missing_summary_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    fig = plot_missing_summary(qc_advanced_container, top_n_features=20, show_sample_labels=True)
    assert fig is not None
    assert len(fig.axes) == 4
    plt.close("all")


def test_plot_cv_distribution_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_cv_distribution(qc_advanced_container, group_by="batch", cv_threshold=0.4)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_cv_distribution_raises_when_no_valid(qc_container_all_missing: ScpContainer) -> None:
    with pytest.raises(ScpValueError, match="No valid features with enough detected values"):
        plot_cv_distribution(qc_container_all_missing)


def test_plot_cv_by_feature_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_cv_by_feature(qc_advanced_container, cv_threshold=0.4, use_log_scale=True)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_cv_by_feature_raises_when_no_valid(qc_container_all_missing: ScpContainer) -> None:
    with pytest.raises(ScpValueError, match="No valid features with enough detected values"):
        plot_cv_by_feature(qc_container_all_missing)


def test_plot_cv_comparison_basic(qc_advanced_container: ScpContainer) -> None:
    plt.close("all")
    ax = plot_cv_comparison(qc_advanced_container, batch_col="batch")
    assert isinstance(ax, Axes)
    assert hasattr(ax, "cv_comparison")
    result = ax.cv_comparison  # type: ignore[attr-defined]
    assert "within_batch" in result
    assert "between_batch" in result
    assert "overall" in result
    assert "batch_effect_ratio" in result
    plt.close("all")


def test_plot_cv_comparison_missing_batch_col_raises(qc_advanced_container: ScpContainer) -> None:
    with pytest.raises(ScpValueError, match="Batch column 'missing_batch' not found in obs"):
        plot_cv_comparison(qc_advanced_container, batch_col="missing_batch")


def test_plot_cv_comparison_raises_when_no_valid(qc_container_all_missing: ScpContainer) -> None:
    with pytest.raises(ScpValueError, match="No valid features with enough detected values"):
        plot_cv_comparison(qc_container_all_missing, batch_col="batch")
