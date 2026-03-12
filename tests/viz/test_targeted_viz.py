"""Tests for targeted and canonical plot_* visualization APIs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from scptensor.aggregation import aggregate_to_protein
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.viz import (
    plot_aggregation_summary,
    plot_correlation_matrix,
    plot_integration_batch_summary,
    plot_normalization_summary,
    plot_qc_completeness,
    plot_qc_matrix_spy,
)


def test_plot_aggregation_summary() -> None:
    """Aggregation summary should render three targeted panels."""
    plt.close("all")
    rng = np.random.default_rng(42)
    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(8)]})
    var = pl.DataFrame(
        {
            "_index": [f"pep{i}" for i in range(12)],
            "PG.ProteinGroups": [
                "P1",
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P3",
                "P4",
                "P4",
                "P4",
                "P5",
                "P6",
            ],
        }
    )
    x = rng.uniform(10, 100, size=(8, 12))
    peptides = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    container = ScpContainer(obs=obs, assays={"peptides": peptides})
    container = aggregate_to_protein(
        container,
        source_assay="peptides",
        source_layer="raw",
        target_assay="proteins",
        method="sum",
    )

    axes = plot_aggregation_summary(container, source_assay="peptides", target_assay="proteins")
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_plot_normalization_summary(container_with_norm: ScpContainer) -> None:
    """Normalization summary should render median/CV/distribution panels."""
    plt.close("all")
    axes = plot_normalization_summary(
        container_with_norm,
        assay_name="proteins",
        before_layer="raw",
        after_layer="normalized",
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_plot_integration_batch_summary(container_with_batches: ScpContainer) -> None:
    """Integration summary should render before/after embedding + metric panel."""
    plt.close("all")
    before = container_with_batches
    after = container_with_batches.copy()
    x_raw = after.assays["proteins"].layers["raw"].X.copy()
    batches = after.obs["batch"].to_numpy()

    x_corrected = x_raw.copy()
    for batch in np.unique(batches):
        mask = batches == batch
        x_corrected[mask] = x_corrected[mask] - np.mean(x_corrected[mask], axis=0, keepdims=True)

    after.assays["proteins"].add_layer("integrated", ScpMatrix(X=x_corrected))

    axes = plot_integration_batch_summary(
        before,
        after,
        assay_name="proteins",
        before_layer="raw",
        after_layer="integrated",
        batch_key="batch",
    )
    assert isinstance(axes, np.ndarray)
    assert axes.size == 3
    plt.close("all")


def test_canonical_plot_names_are_callable(sample_container: ScpContainer) -> None:
    """Canonical plot_* names should be available and callable from top-level viz."""
    plt.close("all")
    ax1 = plot_qc_completeness(
        sample_container,
        assay_name="proteins",
        layer="raw",
        group_by="batch",
    )
    assert isinstance(ax1, Axes)

    ax2 = plot_qc_matrix_spy(sample_container, assay_name="proteins", layer="raw")
    assert isinstance(ax2, Axes)

    fig = plot_correlation_matrix(
        sample_container,
        assay_name="proteins",
        layer="raw",
        groupby="batch",
        show=False,
    )
    assert fig is not None
    plt.close("all")
