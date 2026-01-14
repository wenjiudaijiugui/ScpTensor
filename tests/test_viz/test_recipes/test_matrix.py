"""Tests for matrix visualization recipes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.matrix import heatmap, matrixplot, tracksplot


@pytest.fixture
def matrix_container():
    """Create container with matrix data."""
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


class TestMatrixplot:
    """Tests for matrixplot function."""

    def test_matrixplot_basic(self, matrix_container):
        """Test basic matrix plot."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            show=False,
        )
        assert ax is not None

    def test_matrixplot_no_dendrogram(self, matrix_container):
        """Test matrix plot without dendrogram."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            dendrogram=False,
            show=False,
        )
        assert ax is not None

    def test_matrixplot_with_dendrogram(self, matrix_container):
        """Test matrix plot with dendrogram."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            dendrogram=True,
            show=False,
        )
        assert ax is not None

    def test_matrixplot_custom_cmap(self, matrix_container):
        """Test matrix plot with custom colormap."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            cmap="plasma",
            show=False,
        )
        assert ax is not None

    def test_matrixplot_var_standard_scale(self, matrix_container):
        """Test matrix plot with var standard scaling."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            standard_scale="var",
            show=False,
        )
        assert ax is not None

    def test_matrixplot_obs_standard_scale(self, matrix_container):
        """Test matrix plot with obs standard scaling."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            standard_scale="obs",
            show=False,
        )
        assert ax is not None

    def test_matrixplot_no_standard_scale(self, matrix_container):
        """Test matrix plot without standard scaling."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            standard_scale=None,
            show=False,
        )
        assert ax is not None

    def test_matrixplot_custom_colorbar_title(self, matrix_container):
        """Test matrix plot with custom colorbar title."""
        ax = matrixplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            colorbar_title="Custom expression",
            show=False,
        )
        assert ax is not None

    def test_matrixplot_invalid_layer(self, matrix_container):
        """Test matrix plot with invalid layer."""
        with pytest.raises((VisualizationError, Exception)):
            matrixplot(
                matrix_container,
                layer="nonexistent",
                var_names=["P0"],
                groupby="cluster",
                show=False,
            )

    def test_matrixplot_invalid_feature(self, matrix_container):
        """Test matrix plot with invalid feature name."""
        with pytest.raises(VisualizationError, match="not found"):
            matrixplot(
                matrix_container,
                layer="normalized",
                var_names=["INVALID_PROTEIN"],
                groupby="cluster",
                show=False,
            )

    def test_matrixplot_invalid_groupby(self, matrix_container):
        """Test matrix plot with invalid groupby column."""
        with pytest.raises(VisualizationError, match="not found"):
            matrixplot(
                matrix_container,
                layer="normalized",
                var_names=["P0"],
                groupby="invalid_column",
                show=False,
            )


class TestHeatmap:
    """Tests for heatmap function."""

    def test_heatmap_basic(self, matrix_container):
        """Test basic heatmap."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            show=False,
        )
        assert ax is not None

    def test_heatmap_with_log(self, matrix_container):
        """Test heatmap with log transform."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            log=True,
            show=False,
        )
        assert ax is not None

    def test_heatmap_without_log(self, matrix_container):
        """Test heatmap without log transform."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            log=False,
            show=False,
        )
        assert ax is not None

    def test_heatmap_custom_cmap(self, matrix_container):
        """Test heatmap with custom colormap."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            cmap="plasma",
            show=False,
        )
        assert ax is not None

    def test_heatmap_swap_axes(self, matrix_container):
        """Test heatmap with swapped axes."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            swap_axes=True,
            show=False,
        )
        assert ax is not None

    def test_heatmap_with_dendrogram(self, matrix_container):
        """Test heatmap with dendrogram."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            dendrogram=True,
            show=False,
        )
        assert ax is not None

    def test_heatmap_show_true(self, matrix_container):
        """Test heatmap with show=True."""
        ax = heatmap(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            show=True,
        )
        assert ax is not None

    def test_heatmap_invalid_layer(self, matrix_container):
        """Test heatmap with invalid layer."""
        with pytest.raises((VisualizationError, Exception)):
            heatmap(
                matrix_container,
                layer="nonexistent",
                var_names=["P0"],
                groupby="cluster",
                show=False,
            )

    def test_heatmap_invalid_feature(self, matrix_container):
        """Test heatmap with invalid feature name."""
        with pytest.raises(VisualizationError, match="not found"):
            heatmap(
                matrix_container,
                layer="normalized",
                var_names=["INVALID_PROTEIN"],
                groupby="cluster",
                show=False,
            )

    def test_heatmap_invalid_groupby(self, matrix_container):
        """Test heatmap with invalid groupby column."""
        with pytest.raises(VisualizationError, match="not found"):
            heatmap(
                matrix_container,
                layer="normalized",
                var_names=["P0"],
                groupby="invalid_column",
                show=False,
            )


class TestTracksplot:
    """Tests for tracksplot function."""

    def test_tracksplot_basic(self, matrix_container):
        """Test basic tracks plot."""
        ax = tracksplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            show=False,
        )
        assert ax is not None

    def test_tracksplot_with_dendrogram(self, matrix_container):
        """Test tracks plot with dendrogram."""
        ax = tracksplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1"],
            groupby="cluster",
            dendrogram=True,
            show=False,
        )
        assert ax is not None

    def test_tracksplot_no_dendrogram(self, matrix_container):
        """Test tracks plot without dendrogram."""
        ax = tracksplot(
            matrix_container,
            layer="normalized",
            var_names=["P0", "P1", "P2"],
            groupby="cluster",
            dendrogram=False,
            show=False,
        )
        assert ax is not None

    def test_tracksplot_invalid_layer(self, matrix_container):
        """Test tracks plot with invalid layer."""
        with pytest.raises((VisualizationError, Exception)):
            tracksplot(
                matrix_container,
                layer="nonexistent",
                var_names=["P0"],
                groupby="cluster",
                show=False,
            )

    def test_tracksplot_invalid_feature(self, matrix_container):
        """Test tracks plot with invalid feature name."""
        with pytest.raises(VisualizationError, match="not found"):
            tracksplot(
                matrix_container,
                layer="normalized",
                var_names=["INVALID_PROTEIN"],
                groupby="cluster",
                show=False,
            )

    def test_tracksplot_invalid_groupby(self, matrix_container):
        """Test tracks plot with invalid groupby column."""
        with pytest.raises(VisualizationError, match="not found"):
            tracksplot(
                matrix_container,
                layer="normalized",
                var_names=["P0"],
                groupby="invalid_column",
                show=False,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
