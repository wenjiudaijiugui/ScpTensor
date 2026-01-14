"""Tests for statistics visualization recipes."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import LayerNotFoundError, VisualizationError
from scptensor.viz.recipes.statistics import correlation_matrix, dendrogram


@pytest.fixture
def test_container():
    """Create a test container."""
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(60)],
            "cluster": np.repeat(["A", "B", "C"], 20),
            "condition": np.repeat(["Control", "Treatment"], 30),
        }
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {
            "_index": [f"P{i}" for i in range(15)],
            "protein": [f"P{i}" for i in range(15)],
        }
    )

    # Create data with group differences
    np.random.seed(42)
    X = np.random.rand(60, 15) * 10
    # Add some structure for better clustering
    X[:20, :5] += 5  # Cluster A
    X[20:40, 5:10] += 5  # Cluster B
    X[40:, 10:] += 5  # Cluster C

    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    return container


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    def test_basic_correlation(self, test_container):
        """Test basic correlation matrix visualization."""
        fig = correlation_matrix(test_container, layer="normalized", show=False)
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 1

    def test_correlation_with_groupby(self, test_container):
        """Test correlation matrix with grouping."""
        fig = correlation_matrix(test_container, layer="normalized", groupby="cluster", show=False)
        assert fig is not None

    def test_correlation_spearman(self, test_container):
        """Test correlation matrix with spearman method."""
        fig = correlation_matrix(test_container, layer="normalized", method="spearman", show=False)
        assert fig is not None

    def test_correlation_with_groupby_spearman(self, test_container):
        """Test correlation matrix with groupby and spearman."""
        fig = correlation_matrix(
            test_container,
            layer="normalized",
            groupby="cluster",
            method="spearman",
            show=False,
        )
        assert fig is not None

    def test_correlation_no_annot(self, test_container):
        """Test correlation matrix without annotations."""
        fig = correlation_matrix(test_container, layer="normalized", annot=False, show=False)
        assert fig is not None

    def test_correlation_custom_cmap(self, test_container):
        """Test correlation matrix with custom colormap."""
        fig = correlation_matrix(test_container, layer="normalized", cmap="coolwarm", show=False)
        assert fig is not None

    def test_correlation_invalid_layer(self, test_container):
        """Test correlation matrix with invalid layer."""
        with pytest.raises((VisualizationError, LayerNotFoundError)):
            correlation_matrix(test_container, layer="nonexistent", show=False)

    def test_correlation_invalid_groupby(self, test_container):
        """Test correlation matrix with invalid groupby."""
        with pytest.raises(VisualizationError):
            correlation_matrix(
                test_container, layer="normalized", groupby="invalid_column", show=False
            )


class TestDendrogram:
    """Tests for dendrogram function."""

    def test_basic_dendrogram(self, test_container):
        """Test basic dendrogram visualization."""
        fig = dendrogram(test_container, layer="normalized", show=False)
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 1

    def test_dendrogram_with_groupby(self, test_container):
        """Test dendrogram with grouping."""
        fig = dendrogram(test_container, layer="normalized", groupby="cluster", show=False)
        assert fig is not None

    def test_dendrogram_single_method(self, test_container):
        """Test dendrogram with single linkage."""
        fig = dendrogram(test_container, layer="normalized", method="single", show=False)
        assert fig is not None

    def test_dendrogram_complete_method(self, test_container):
        """Test dendrogram with complete linkage."""
        fig = dendrogram(test_container, layer="normalized", method="complete", show=False)
        assert fig is not None

    def test_dendrogram_ward_method(self, test_container):
        """Test dendrogram with ward linkage."""
        fig = dendrogram(test_container, layer="normalized", method="ward", show=False)
        assert fig is not None

    def test_dendrogram_custom_metric(self, test_container):
        """Test dendrogram with custom distance metric."""
        fig = dendrogram(test_container, layer="normalized", metric="correlation", show=False)
        assert fig is not None

    def test_dendrogram_invalid_layer(self, test_container):
        """Test dendrogram with invalid layer."""
        with pytest.raises((VisualizationError, LayerNotFoundError)):
            dendrogram(test_container, layer="nonexistent", show=False)

    def test_dendrogram_invalid_groupby(self, test_container):
        """Test dendrogram with invalid groupby."""
        with pytest.raises(VisualizationError):
            dendrogram(test_container, layer="normalized", groupby="invalid_column", show=False)

    def test_dendrogram_ward_with_non_euclidean(self, test_container):
        """Test that ward linkage with non-euclidean metric raises error."""
        with pytest.raises(ValueError):
            dendrogram(test_container, layer="normalized", method="ward", metric="correlation")
