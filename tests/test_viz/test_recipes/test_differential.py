"""Tests for differential expression visualization recipes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.differential import (
    rank_genes_groups_dotplot,
    rank_genes_groups_stacked_violin,
    volcano,
)


@pytest.fixture
def de_container():
    """Create container with differential expression data."""
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(60)],
            "condition": np.repeat(["Control", "Treatment"], 30),
        }
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(20)], "protein": [f"P{i}" for i in range(20)]}
    )

    # Create data with some differential expression
    np.random.seed(42)
    X = np.random.rand(60, 20) * 10
    # Make some proteins higher in Treatment
    X[30:, :5] += 5  # Upregulated in treatment
    X[:30, 5:10] += 3  # Upregulated in control

    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay
    return container


def test_rank_genes_groups_dotplot_basic(de_container):
    """Test basic dot plot for ranked gene groups."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
    )
    assert ax is not None
    # Check title
    title = ax.get_title()
    assert "Treatment vs Control" in title


def test_rank_genes_groups_dotplot_auto_groups(de_container):
    """Test dot plot with auto-detected groups."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        n_genes=5,
        show=False,
    )
    assert ax is not None


def test_rank_genes_groups_dotplot_pval_values(de_container):
    """Test dot plot with p-value coloring."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        values_to_plot="pval",
        show=False,
    )
    assert ax is not None
    # Check xlabel
    xlabel = ax.get_xlabel()
    assert "-log10" in xlabel


def test_rank_genes_groups_dotplot_mean_values(de_container):
    """Test dot plot with mean expression coloring."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        values_to_plot="mean",
        show=False,
    )
    assert ax is not None
    xlabel = ax.get_xlabel()
    assert "Mean" in xlabel


def test_rank_genes_groups_dotplot_custom_cmap(de_container):
    """Test dot plot with custom colormap."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        cmap="plasma",
        show=False,
    )
    assert ax is not None


def test_rank_genes_groups_dotplot_invalid_layer(de_container):
    """Test dot plot with invalid layer."""
    with pytest.raises((VisualizationError, Exception)):
        rank_genes_groups_dotplot(
            de_container,
            layer="nonexistent",
            groupby="condition",
            show=False,
        )


def test_rank_genes_groups_dotplot_invalid_groupby(de_container):
    """Test dot plot with invalid groupby column."""
    with pytest.raises(VisualizationError, match="not found"):
        rank_genes_groups_dotplot(
            de_container,
            layer="normalized",
            groupby="invalid_column",
            show=False,
        )


def test_rank_genes_groups_dotplot_invalid_group(de_container):
    """Test dot plot with invalid group name."""
    with pytest.raises(ValueError, match="not found"):
        rank_genes_groups_dotplot(
            de_container,
            layer="normalized",
            groupby="condition",
            group1="NonExistent",
            show=False,
        )


def test_rank_genes_groups_stacked_violin_basic(de_container):
    """Test basic stacked violin plot for ranked gene groups."""
    ax = rank_genes_groups_stacked_violin(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
    )
    assert ax is not None
    # Check title
    title = ax.get_title()
    assert "Treatment vs Control" in title


def test_rank_genes_groups_stacked_violin_auto_groups(de_container):
    """Test stacked violin plot with auto-detected groups."""
    ax = rank_genes_groups_stacked_violin(
        de_container,
        layer="normalized",
        groupby="condition",
        n_genes=5,
        show=False,
    )
    assert ax is not None


def test_rank_genes_groups_stacked_violin_custom_cmap(de_container):
    """Test stacked violin plot with custom colormap."""
    ax = rank_genes_groups_stacked_violin(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        cmap="plasma",
        show=False,
    )
    assert ax is not None


def test_rank_genes_groups_stacked_violin_invalid_layer(de_container):
    """Test stacked violin plot with invalid layer."""
    with pytest.raises((VisualizationError, Exception)):
        rank_genes_groups_stacked_violin(
            de_container,
            layer="nonexistent",
            groupby="condition",
            show=False,
        )


def test_rank_genes_groups_stacked_violin_invalid_groupby(de_container):
    """Test stacked violin plot with invalid groupby column."""
    with pytest.raises(VisualizationError, match="not found"):
        rank_genes_groups_stacked_violin(
            de_container,
            layer="normalized",
            groupby="invalid_column",
            show=False,
        )


def test_volcano_basic(de_container):
    """Test basic volcano plot."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        show=False,
    )
    assert fig is not None
    # Check number of axes (multi-panel layout)
    assert len(fig.axes) == 6


def test_volcano_auto_groups(de_container):
    """Test volcano plot with auto-detected groups."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        show=False,
    )
    assert fig is not None


def test_volcano_custom_thresholds(de_container):
    """Test volcano plot with custom significance thresholds."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        pval_threshold=0.01,
        logfc_threshold=0.5,
        show=False,
    )
    assert fig is not None


def test_volcano_custom_colors(de_container):
    """Test volcano plot with custom colors."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        colors=("lightgray", "green", "purple"),
        show=False,
    )
    assert fig is not None


def test_volcano_no_labels(de_container):
    """Test volcano plot without gene labels."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        show_labels=False,
        show=False,
    )
    assert fig is not None


def test_volcano_custom_n_labels(de_container):
    """Test volcano plot with custom number of labels."""
    fig = volcano(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_labels=10,
        show=False,
    )
    assert fig is not None


def test_volcano_invalid_layer(de_container):
    """Test volcano plot with invalid layer."""
    with pytest.raises((VisualizationError, Exception)):
        volcano(
            de_container,
            layer="nonexistent",
            groupby="condition",
            show=False,
        )


def test_volcano_invalid_groupby(de_container):
    """Test volcano plot with invalid groupby column."""
    with pytest.raises(VisualizationError, match="not found"):
        volcano(
            de_container,
            layer="normalized",
            groupby="invalid_column",
            show=False,
        )


def test_volcano_invalid_group(de_container):
    """Test volcano plot with invalid group name."""
    with pytest.raises(ValueError, match="not found"):
        volcano(
            de_container,
            layer="normalized",
            groupby="condition",
            group1="NonExistent",
            show=False,
        )


def test_rank_genes_groups_dotplot_with_ax(de_container):
    """Test dot plot with provided axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
        ax=ax,
    )
    assert result_ax is ax


def test_rank_genes_groups_stacked_violin_with_ax(de_container):
    """Test stacked violin plot with provided axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax = rank_genes_groups_stacked_violin(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
        ax=ax,
    )
    assert result_ax is ax


def test_rank_genes_groups_dotplot_dendrogram_reserved(de_container):
    """Test that dendrogram parameter is accepted but not implemented."""
    ax = rank_genes_groups_dotplot(
        de_container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        dendrogram=True,  # Should be accepted but not implemented
        show=False,
    )
    assert ax is not None


def test_single_group_error(de_container):
    """Test error when only one group is available."""
    # Create container with single group
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(30)],
            "condition": ["Control"] * 30,
        }
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(10)], "protein": [f"P{i}" for i in range(10)]}
    )
    X = np.random.rand(30, 10) * 10
    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    # Should raise error when needing to auto-detect groups
    with pytest.raises(ValueError, match="at least 2 groups"):
        rank_genes_groups_dotplot(
            container,
            layer="normalized",
            groupby="condition",
            show=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
