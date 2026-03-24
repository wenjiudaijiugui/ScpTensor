"""Tests for workflow-oriented visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from scptensor.core import FilterCriteria, ScpMatrix
from scptensor.viz import (
    plot_data_overview,
    plot_embedding_panels,
    plot_missingness_reduction,
    plot_preprocessing_summary,
    plot_qc_filtering_summary,
    plot_recent_operations,
    plot_reduction_summary,
    plot_saved_artifact_sizes,
)


def test_plot_data_overview(sample_container):
    """Data overview should create 3-panel summary."""
    plt.close("all")
    axes = plot_data_overview(
        sample_container,
        assay_name="proteins",
        layer="raw",
        groupby="condition",
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_plot_qc_filtering_summary(sample_container):
    """QC filtering summary should compare before/after containers."""
    plt.close("all")
    after = sample_container.filter_samples(FilterCriteria.by_indices(np.arange(40)))
    after = after.filter_features("proteins", FilterCriteria.by_indices(np.arange(15)))

    axes = plot_qc_filtering_summary(
        sample_container,
        after,
        assay_name="proteins",
        layer="raw",
        min_features=5,
        max_missing_rate=0.8,
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_plot_preprocessing_summary(sample_container):
    """Preprocessing summary should compare multiple transformed layers."""
    plt.close("all")
    container = sample_container.copy()
    x_raw = container.assays["proteins"].layers["raw"].X
    x_log2 = np.log2(x_raw + 1.0)
    x_norm = x_log2 - np.nanmedian(x_log2, axis=1, keepdims=True)
    x_imputed = x_norm.copy()

    assay = container.assays["proteins"]
    assay.add_layer("log2", ScpMatrix(X=x_log2))
    assay.add_layer("norm", ScpMatrix(X=x_norm))
    assay.add_layer("imputed", ScpMatrix(X=x_imputed))

    axes = plot_preprocessing_summary(
        container,
        assay_name="proteins",
        raw_layer="raw",
        transformed_layers=("log2", "norm", "imputed"),
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_plot_missingness_reduction(sample_container):
    """Missingness reduction should create 2-bar summary."""
    plt.close("all")
    container = sample_container.copy()
    x_base = container.assays["proteins"].layers["raw"].X.copy()
    x_before = x_base.copy()
    x_before[0:3, 0:2] = np.nan
    x_after = np.nan_to_num(x_before, nan=float(np.nanmean(x_before)))

    assay = container.assays["proteins"]
    assay.add_layer("norm", ScpMatrix(X=x_before))
    assay.add_layer("imputed", ScpMatrix(X=x_after))

    ax = plot_missingness_reduction(
        container,
        assay_name="proteins",
        before_layer="norm",
        after_layer="imputed",
    )
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2
    plt.close("all")


def test_plot_reduction_summary(container_with_clusters):
    """Reduction summary should run even when explained variance is absent."""
    plt.close("all")
    axes = plot_reduction_summary(
        container_with_clusters,
        pca_assay_name="pca",
        cluster_col="kmeans_k3",
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    plt.close("all")


def test_plot_embedding_panels(container_with_clusters):
    """Embedding panels should render scatter from assay coordinates."""
    plt.close("all")
    axes = plot_embedding_panels(
        container_with_clusters,
        assay_names=("pca",),
        layer="X",
        color_by="kmeans_k3",
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 1
    assert len(axes[0].collections) > 0
    plt.close("all")


def test_plot_saved_artifact_sizes(tmp_path):
    """Artifact-size summary should render one bar per file."""
    plt.close("all")
    p1 = tmp_path / "result.h5"
    p2 = tmp_path / "result.npz"
    p1.write_bytes(b"a" * 1024)
    p2.write_bytes(b"b" * 2048)

    ax = plot_saved_artifact_sizes([p1, p2])
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2
    plt.close("all")


def test_plot_recent_operations_with_and_without_history(sample_container, container_with_clusters):
    """History summary should work for both empty and non-empty history."""
    plt.close("all")
    ax_empty = plot_recent_operations(sample_container)
    assert isinstance(ax_empty, Axes)
    assert len(ax_empty.texts) >= 1

    ax_hist = plot_recent_operations(container_with_clusters)
    assert isinstance(ax_hist, Axes)
    assert len(ax_hist.patches) >= 1
    plt.close("all")
