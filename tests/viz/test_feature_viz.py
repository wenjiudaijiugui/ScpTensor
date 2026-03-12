"""Tests for feature visualization recipe."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.feature import dotplot, plot_feature_dotplot


def test_dotplot_basic(sample_container: ScpContainer) -> None:
    """dotplot should render one marker per (group, feature) cell."""
    plt.close("all")
    ax = dotplot(
        sample_container,
        layer="raw",
        var_names=["P0", "P1"],
        groupby="condition",
        assay_name="proteins",
        show=False,
    )
    assert isinstance(ax, Axes)
    # 2 groups x 2 features -> 4 scatter artists
    assert len(ax.collections) == 4
    plt.close("all")


def test_dotplot_alias_callable(sample_container: ScpContainer) -> None:
    """Canonical plot_* alias should delegate to dotplot."""
    plt.close("all")
    ax = plot_feature_dotplot(
        sample_container,
        layer="raw",
        var_names=["P0"],
        groupby="condition",
        assay_name="proteins",
        show=False,
    )
    assert isinstance(ax, Axes)
    plt.close("all")


def test_dotplot_preserves_group_order() -> None:
    """Group order should follow first appearance in obs, not lexical sort."""
    plt.close("all")
    obs = pl.DataFrame({"_index": ["S1", "S2", "S3", "S4"], "grp": ["B", "A", "B", "A"]})
    var = pl.DataFrame({"_index": ["P1"]})
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x)})},
    )

    ax = dotplot(
        container,
        layer="raw",
        var_names=["P1"],
        groupby="grp",
        assay_name="proteins",
        show=False,
    )
    y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert y_labels == ["B", "A"]
    plt.close("all")


def test_dotplot_sparse_input_supported() -> None:
    """dotplot should accept sparse expression layer."""
    plt.close("all")
    obs = pl.DataFrame({"_index": ["S1", "S2", "S3"], "grp": ["G1", "G1", "G2"]})
    var = pl.DataFrame({"_index": ["P1", "P2"]})
    x_sparse = sparse.csr_matrix(np.array([[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]]))
    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x_sparse)})},
    )

    ax = dotplot(
        container,
        layer="raw",
        var_names=["P1", "P2"],
        groupby="grp",
        assay_name="proteins",
        show=False,
    )
    assert isinstance(ax, Axes)
    plt.close("all")


def test_dotplot_invalid_runtime_params_raise(sample_container: ScpContainer) -> None:
    """dotplot should reject empty var_names and invalid standard_scale."""
    with pytest.raises(VisualizationError, match="at least one feature"):
        dotplot(
            sample_container,
            layer="raw",
            var_names=[],
            groupby="condition",
            assay_name="proteins",
            show=False,
        )

    with pytest.raises(VisualizationError, match="standard_scale"):
        dotplot(
            sample_container,
            layer="raw",
            var_names=["P0"],
            groupby="condition",
            assay_name="proteins",
            standard_scale="invalid",  # type: ignore[arg-type]
            show=False,
        )


def test_dotplot_dendrogram_not_supported(sample_container: ScpContainer) -> None:
    """dendrogram flag should fail fast with explicit unsupported error."""
    with pytest.raises(VisualizationError, match="dendrogram=True is not supported"):
        dotplot(
            sample_container,
            layer="raw",
            var_names=["P0"],
            groupby="condition",
            assay_name="proteins",
            dendrogram=True,
            show=False,
        )


def test_dotplot_standard_scale_none_uses_data_range() -> None:
    """Without scaling, colorbar should reflect expression range instead of fixed [0, 1]."""
    plt.close("all")
    obs = pl.DataFrame({"_index": ["S1", "S2", "S3", "S4"], "grp": ["A", "A", "B", "B"]})
    var = pl.DataFrame({"_index": ["P1"]})
    x = np.array([[10.0], [12.0], [20.0], [22.0]])
    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x)})},
    )

    ax = dotplot(
        container,
        layer="raw",
        var_names=["P1"],
        groupby="grp",
        assay_name="proteins",
        standard_scale=None,
        log=False,
        show=False,
    )
    fig = ax.figure
    assert fig is not None
    assert len(fig.axes) >= 2  # main plot + colorbar
    cbar_ylim = fig.axes[-1].get_ylim()
    assert not np.allclose(cbar_ylim, (0.0, 1.0))
    plt.close("all")
