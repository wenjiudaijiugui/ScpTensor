"""Tests for feature visualization recipes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.feature import dotplot


@pytest.fixture
def feature_container():
    """Create container with feature data."""
    obs = pl.DataFrame(
        {"_index": [f"S{i}" for i in range(60)], "cluster": np.repeat(["A", "B", "C"], 20)}
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(10)], "protein": [f"P{i}" for i in range(10)]}
    )
    X = np.random.rand(60, 10) * 10
    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay
    return container


def test_dotplot_basic(feature_container):
    """Test basic dot plot."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None


def test_dotplot_with_features(feature_container):
    """Test dot plot with multiple features."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1", "P2", "P3", "P4"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None
    # Check x-axis labels
    x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert set(x_labels) == {"P0", "P1", "P2", "P3", "P4"}


def test_dotplot_no_log(feature_container):
    """Test dot plot without log transform."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        log=False,
        show=False,
    )
    assert ax is not None


def test_dotplot_custom_cmap(feature_container):
    """Test dot plot with custom colormap."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        cmap="plasma",
        show=False,
    )
    assert ax is not None


def test_dotplot_custom_dot_size(feature_container):
    """Test dot plot with custom dot size."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        dot_size=10.0,
        show=False,
    )
    assert ax is not None


def test_dotplot_no_standard_scale(feature_container):
    """Test dot plot without standard scaling."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        standard_scale=None,
        show=False,
    )
    assert ax is not None


def test_dotplot_obs_standard_scale(feature_container):
    """Test dot plot with obs standard scaling."""
    ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        standard_scale="obs",
        show=False,
    )
    assert ax is not None


def test_dotplot_invalid_assay(feature_container):
    """Test dot plot with invalid assay."""
    with pytest.raises((VisualizationError, Exception)):
        dotplot(
            feature_container,
            layer="nonexistent",
            var_names=["P0"],
            groupby="cluster",
            show=False,
        )


def test_dotplot_invalid_feature(feature_container):
    """Test dot plot with invalid feature name."""
    with pytest.raises(VisualizationError, match="not found"):
        dotplot(
            feature_container,
            layer="normalized",
            var_names=["INVALID_PROTEIN"],
            groupby="cluster",
            show=False,
        )


def test_dotplot_invalid_groupby(feature_container):
    """Test dot plot with invalid groupby column."""
    with pytest.raises(VisualizationError, match="not found"):
        dotplot(
            feature_container,
            layer="normalized",
            var_names=["P0"],
            groupby="invalid_column",
            show=False,
        )


def test_dotplot_with_ax(feature_container):
    """Test dot plot with provided axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax = dotplot(
        feature_container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        show=False,
        ax=ax,
    )
    assert result_ax is ax


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
