"""Tests for clustering visualization recipes.

This module provides comprehensive tests for clustering visualization functions,
including optimization across k values and quality assessment.
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.clustering import (
    plot_clustering_optimization,
    plot_clustering_quality,
)


@pytest.fixture
def container_with_pca() -> ScpContainer:
    """Create container with PCA data for clustering.

    Returns
    -------
    ScpContainer
        Container with PCA assay and synthetic data.
    """
    np.random.seed(42)

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(100)],
            "batch": np.repeat(["A", "B"], 50),
        }
    )

    var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})

    container = ScpContainer(obs=obs)

    # Create synthetic PCA data with 3 clusters
    X = np.random.randn(100, 10) * 0.5
    X[:33, :] += 3  # Cluster 0
    X[33:66, :] -= 3  # Cluster 1
    # Cluster 2 stays around 0

    assay = Assay(var=var)
    assay.add_layer("X", ScpMatrix(X=X))
    container.add_assay("pca", assay)

    return container


@pytest.fixture
def container_with_clusters(container_with_pca: ScpContainer) -> ScpContainer:
    """Create container with pre-computed clustering results.

    Parameters
    ----------
    container_with_pca : ScpContainer
        Container with PCA data.

    Returns
    -------
    ScpContainer
        Container with cluster labels in obs.
    """
    # Add cluster labels manually for testing
    labels = ["0"] * 33 + ["1"] * 33 + ["2"] * 34
    new_obs = container_with_pca.obs.with_columns(pl.Series("kmeans_k3", labels).cast(pl.String))

    return ScpContainer(
        obs=new_obs, assays=container_with_pca.assays, history=list(container_with_pca.history)
    )


@pytest.fixture
def container_with_embedding(container_with_clusters: ScpContainer) -> ScpContainer:
    """Create container with PCA coordinates for embedding visualization.

    Parameters
    ----------
    container_with_clusters : ScpContainer
        Container with cluster labels.

    Returns
    -------
    ScpContainer
        Container with PCA coordinate columns.
    """
    # Add PCA coordinates for embedding visualization
    X = container_with_clusters.assays["pca"].layers["X"].X

    if sparse.issparse(X):
        X = X.toarray()

    new_obs = container_with_clusters.obs.with_columns(
        [
            pl.Series("pca_1", X[:, 0]),
            pl.Series("pca_2", X[:, 1]),
        ]
    )

    return ScpContainer(obs=new_obs, assays=container_with_clusters.assays, history=[])


class TestClusteringOptimization:
    """Test suite for plot_clustering_optimization function."""

    def test_plot_clustering_optimization_basic(self, container_with_pca: ScpContainer) -> None:
        """Test basic clustering optimization plot."""
        fig = plot_clustering_optimization(
            container_with_pca,
            assay_name="pca",
            layer="X",
            k_range=(2, 6),
            show=False,
        )

        assert fig is not None
        # Should have 2 panels (elbow + silhouette)
        assert len(fig.axes) == 2

    @pytest.mark.parametrize("k_range", [(2, 6), (3, 8), (2, 10)])
    def test_plot_clustering_optimization_ranges(
        self, container_with_pca: ScpContainer, k_range: tuple[int, int]
    ) -> None:
        """Test optimization with different k ranges."""
        fig = plot_clustering_optimization(
            container_with_pca,
            assay_name="pca",
            layer="X",
            k_range=k_range,
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2

    def test_plot_clustering_optimization_small_k_range(self) -> None:
        """Test with minimal k range."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(30)]})
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(5)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(30, 5)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_optimization(
            container,
            assay_name="pca",
            layer="X",
            k_range=(2, 4),
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2

    def test_plot_clustering_optimization_large_k_range(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test with larger k range."""
        fig = plot_clustering_optimization(
            container_with_pca,
            assay_name="pca",
            layer="X",
            k_range=(2, 15),
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2

    def test_plot_clustering_optimization_sparse_matrix(self) -> None:
        """Test with sparse matrix data."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(50)]})
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})
        container = ScpContainer(obs=obs)

        # Create sparse data
        X_dense = np.random.randn(50, 10)
        X_sparse = sparse.csr_matrix(X_dense)

        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X_sparse))
        container.add_assay("pca", assay)

        fig = plot_clustering_optimization(
            container,
            assay_name="pca",
            layer="X",
            k_range=(2, 5),
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2


class TestClusteringQuality:
    """Test suite for plot_clustering_quality function."""

    def test_plot_clustering_quality_basic(self, container_with_clusters: ScpContainer) -> None:
        """Test basic clustering quality assessment."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            assay_name="pca",
            layer="X",
            show_embedding=False,
            show=False,
        )

        assert fig is not None
        # Should have 2 panels (silhouette + cluster sizes)
        assert len(fig.axes) >= 2

    def test_plot_clustering_quality_with_embedding(
        self, container_with_embedding: ScpContainer
    ) -> None:
        """Test clustering quality with embedding visualization."""
        fig = plot_clustering_quality(
            container_with_embedding,
            cluster_key="kmeans_k3",
            assay_name="pca",
            layer="X",
            show_embedding=True,
            show=False,
        )

        assert fig is not None
        # Should have 4 panels (silhouette, sizes, embedding, metrics)
        assert len(fig.axes) >= 3

    def test_plot_clustering_quality_without_embedding(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test clustering quality without embedding visualization."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            assay_name="pca",
            layer="X",
            show_embedding=False,
            show=False,
        )

        assert fig is not None
        # Should have 2 panels
        assert len(fig.axes) >= 2

    def test_plot_clustering_quality_different_assay(self) -> None:
        """Test with different assay name."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(50)],
                "clusters": ["0"] * 25 + ["1"] * 25,
            }
        )
        var = pl.DataFrame({"_index": [f"F{i}" for i in range(8)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(50, 8)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("features", assay)

        fig = plot_clustering_quality(
            container,
            cluster_key="clusters",
            assay_name="features",
            layer="X",
            show_embedding=False,
            show=False,
        )

        assert fig is not None

    def test_plot_clustering_quality_sparse_matrix(self) -> None:
        """Test clustering quality with sparse matrix."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(50)],
                "clusters": ["0"] * 25 + ["1"] * 25,
            }
        )
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})
        container = ScpContainer(obs=obs)

        # Create sparse data
        X_dense = np.random.randn(50, 10)
        X_sparse = sparse.csr_matrix(X_dense)

        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X_sparse))
        container.add_assay("pca", assay)

        fig = plot_clustering_quality(
            container,
            cluster_key="clusters",
            assay_name="pca",
            layer="X",
            show_embedding=False,
            show=False,
        )

        assert fig is not None


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_plot_clustering_optimization_single_cluster_perfect(self) -> None:
        """Test with data that forms one clear cluster."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(30)]})
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(5)]})
        container = ScpContainer(obs=obs)

        # Very tight single cluster
        X = np.random.randn(30, 5) * 0.1
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_optimization(
            container,
            k_range=(2, 5),
            show=False,
        )

        assert fig is not None

    def test_plot_clustering_quality_very_small_dataset(self) -> None:
        """Test with minimal dataset (10 samples)."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(10)],
                "clusters": ["0"] * 5 + ["1"] * 5,
            }
        )
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(3)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(10, 3)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_quality(
            container,
            cluster_key="clusters",
            show_embedding=False,
            show=False,
        )

        assert fig is not None

    def test_plot_clustering_quality_imbalanced_clusters(self) -> None:
        """Test with highly imbalanced cluster sizes."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(100)],
                "clusters": ["0"] * 80 + ["1"] * 15 + ["2"] * 5,
            }
        )
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(100, 10)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_quality(
            container,
            cluster_key="clusters",
            show_embedding=False,
            show=False,
        )

        assert fig is not None

    def test_plot_clustering_optimization_minimum_k(self) -> None:
        """Test with minimum valid k range (2, 3)."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(20)]})
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(5)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(20, 5)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_optimization(
            container,
            k_range=(2, 3),
            show=False,
        )

        assert fig is not None

    def test_plot_clustering_with_umap_embedding(self) -> None:
        """Test with UMAP coordinates instead of PCA."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(50)],
                "clusters": ["0"] * 25 + ["1"] * 25,
                "umap_1": np.random.randn(50),
                "umap_2": np.random.randn(50),
            }
        )
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})
        container = ScpContainer(obs=obs)

        X = np.random.randn(50, 10)
        assay = Assay(var=var)
        assay.add_layer("X", ScpMatrix(X=X))
        container.add_assay("pca", assay)

        fig = plot_clustering_quality(
            container,
            cluster_key="clusters",
            show_embedding=True,
            show=False,
        )

        assert fig is not None
        # Should show embedding
        assert len(fig.axes) >= 3


class TestErrorHandling:
    """Test suite for error handling and validation."""

    def test_plot_clustering_quality_missing_cluster_key(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test error handling for missing cluster column."""
        with pytest.raises(VisualizationError, match="not found"):
            plot_clustering_quality(
                container_with_pca,
                cluster_key="nonexistent_clusters",
                show=False,
            )

    def test_plot_clustering_optimization_missing_assay(self) -> None:
        """Test error handling for missing assay."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(20)]})
        container = ScpContainer(obs=obs)

        with pytest.raises(VisualizationError, match="not found"):
            plot_clustering_optimization(
                container,
                assay_name="nonexistent_assay",
                show=False,
            )

    def test_plot_clustering_optimization_missing_layer(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test error handling for missing layer."""
        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises((VisualizationError, LayerNotFoundError)):
            plot_clustering_optimization(
                container_with_pca,
                assay_name="pca",
                layer="nonexistent_layer",
                show=False,
            )

    def test_plot_clustering_quality_missing_layer(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test error handling for missing layer in quality plot."""
        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises((VisualizationError, LayerNotFoundError)):
            plot_clustering_quality(
                container_with_clusters,
                cluster_key="kmeans_k3",
                assay_name="pca",
                layer="nonexistent_layer",
                show=False,
            )

    def test_plot_clustering_optimization_invalid_k_range_min(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test error handling for invalid k range (min < 2)."""
        with pytest.raises(VisualizationError, match="must be >= 2"):
            plot_clustering_optimization(
                container_with_pca,
                k_range=(1, 5),
                show=False,
            )

    def test_plot_clustering_optimization_invalid_k_range_max(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test error handling for invalid k range (max <= min)."""
        with pytest.raises(VisualizationError, match="must be >"):
            plot_clustering_optimization(
                container_with_pca,
                k_range=(5, 3),
                show=False,
            )


class TestVisualizationOutput:
    """Test suite for visualization output properties."""

    def test_plot_clustering_optimization_return_type(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test that optimization returns correct type."""
        fig = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            show=False,
        )

        # Check it's a matplotlib Figure
        assert hasattr(fig, "axes")
        assert hasattr(fig, "subplots")
        assert hasattr(fig, "savefig")

    def test_plot_clustering_quality_return_type(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test that quality assessment returns correct type."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        # Check it's a matplotlib Figure
        assert hasattr(fig, "axes")
        assert hasattr(fig, "subplots")
        assert hasattr(fig, "savefig")

    def test_plot_clustering_optimization_panels_have_titles(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test that all optimization panels have titles."""
        fig = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            show=False,
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        # Should have titles for elbow and silhouette plots
        assert len(titles) >= 2
        assert all(title != "" for title in titles)

    def test_plot_clustering_quality_panels_have_titles(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test that all quality panels have titles."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        # Should have at least 2 titles
        assert len(titles) >= 2
        assert all(title != "" for title in titles)

    def test_plot_clustering_optimization_show_false(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test that show=False prevents display."""
        fig = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            show=False,
        )

        assert fig is not None
        # Figure should exist but not be shown

    def test_plot_clustering_quality_show_false(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test that show=False prevents display for quality plot."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        assert fig is not None
        # Figure should exist but not be shown

    def test_plot_clustering_optimization_has_labels(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test that optimization plots have proper axis labels."""
        fig = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            show=False,
        )

        for ax in fig.axes:
            # At minimum, panels should have labels
            assert ax.get_xlabel() != "" or ax.get_ylabel() != ""

    def test_plot_clustering_quality_has_labels(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test that quality plots have proper axis labels."""
        fig = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        for ax in fig.axes:
            # At minimum, panels should have labels
            assert ax.get_xlabel() != "" or ax.get_ylabel() != ""


class TestMultipleCalls:
    """Test suite for consistency across multiple calls."""

    def test_plot_clustering_optimization_consistent_results(
        self, container_with_pca: ScpContainer
    ) -> None:
        """Test that multiple calls produce consistent results."""
        fig1 = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            random_state=42,
            show=False,
        )

        fig2 = plot_clustering_optimization(
            container_with_pca,
            k_range=(2, 5),
            random_state=42,
            show=False,
        )

        assert fig1 is not None
        assert fig2 is not None
        # Same number of axes
        assert len(fig1.axes) == len(fig2.axes)

    def test_plot_clustering_quality_consistent_results(
        self, container_with_clusters: ScpContainer
    ) -> None:
        """Test that multiple quality calls produce consistent results."""
        fig1 = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        fig2 = plot_clustering_quality(
            container_with_clusters,
            cluster_key="kmeans_k3",
            show_embedding=False,
            show=False,
        )

        assert fig1 is not None
        assert fig2 is not None
        # Same number of axes
        assert len(fig1.axes) == len(fig2.axes)
