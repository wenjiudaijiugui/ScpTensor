"""Tests for matrix-style visualization recipes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from scptensor.core.exceptions import VisualizationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.matrix import (
    heatmap,
    matrixplot,
    plot_matrix_heatmap,
    plot_matrixplot,
    plot_tracksplot,
    tracksplot,
)


@pytest.fixture
def matrix_container() -> ScpContainer:
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(6)],
            "group": ["G1", "G1", "G1", "G2", "G2", "G2"],
        }
    )
    var = pl.DataFrame({"_index": ["P0", "P1", "P2", "P3"]})

    # Group G1 means for P1/P3: 1 / 3, group G2 means: 11 / 13
    x = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [10.0, 11.0, 12.0, 13.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )
    assay = Assay(var=var, layers={"norm": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_matrixplot_standard_scale_var(matrix_container: ScpContainer) -> None:
    plt.close("all")
    ax = matrixplot(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        standard_scale="var",
        show=False,
    )
    assert isinstance(ax, Axes)
    data = np.asarray(ax.images[0].get_array())
    assert data.shape == (2, 2)
    assert np.allclose(data.min(axis=0), 0.0)
    assert np.allclose(data.max(axis=0), 1.0)
    plt.close("all")


def test_matrixplot_standard_scale_obs(matrix_container: ScpContainer) -> None:
    plt.close("all")
    ax = matrixplot(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        standard_scale="obs",
        show=False,
    )
    data = np.asarray(ax.images[0].get_array())
    assert data.shape == (2, 2)
    assert np.allclose(data.min(axis=1), 0.0)
    assert np.allclose(data.max(axis=1), 1.0)
    plt.close("all")


def test_heatmap_swap_axes_changes_shape(matrix_container: ScpContainer) -> None:
    plt.close("all")
    ax_normal = heatmap(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        log=False,
        swap_axes=False,
        show=False,
    )
    normal_data = np.asarray(ax_normal.images[0].get_array())
    assert normal_data.shape == (6, 2)

    ax_swapped = heatmap(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        log=False,
        swap_axes=True,
        show=False,
    )
    swapped_data = np.asarray(ax_swapped.images[0].get_array())
    assert swapped_data.shape == (2, 6)
    plt.close("all")


def test_tracksplot_bar_count_matches_groups_times_features(
    matrix_container: ScpContainer,
) -> None:
    plt.close("all")
    ax = tracksplot(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        show=False,
    )
    # 2 groups x 2 features -> 4 bars
    assert len(ax.patches) == 4
    plt.close("all")


def test_matrix_alias_functions_are_callable(matrix_container: ScpContainer) -> None:
    plt.close("all")
    ax1 = plot_matrixplot(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        show=False,
    )
    ax2 = plot_matrix_heatmap(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        log=False,
        show=False,
    )
    ax3 = plot_tracksplot(
        matrix_container,
        layer="norm",
        var_names=["P1", "P3"],
        groupby="group",
        assay_name="proteins",
        show=False,
    )
    assert isinstance(ax1, Axes)
    assert isinstance(ax2, Axes)
    assert isinstance(ax3, Axes)
    plt.close("all")


def test_matrixplot_invalid_groupby_raises(matrix_container: ScpContainer) -> None:
    with pytest.raises(VisualizationError, match="Column 'missing_group' not found in obs"):
        matrixplot(
            matrix_container,
            layer="norm",
            var_names=["P1", "P3"],
            groupby="missing_group",
            assay_name="proteins",
            show=False,
        )
