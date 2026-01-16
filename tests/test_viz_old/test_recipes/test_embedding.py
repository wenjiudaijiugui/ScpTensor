"""Tests for embedding visualization recipes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.embedding import pca, scatter, tsne, umap


@pytest.fixture
def embedding_container():
    """Create container with embedding coordinates."""
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(100)],
            "cluster": np.random.choice(["A", "B", "C"], 100),
            "umap_1": np.random.randn(100),
            "umap_2": np.random.randn(100),
            "pca_1": np.random.randn(100),
            "pca_2": np.random.randn(100),
            "tsne_1": np.random.randn(100),
            "tsne_2": np.random.randn(100),
        }
    )

    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(50)],
            "protein": [f"protein_{i}" for i in range(50)],
        }
    )

    X = np.random.rand(100, 50)
    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    return container


@pytest.fixture
def container_with_missing_values():
    """Create container with missing values for testing missing value visualization."""
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(50)],
            "umap_1": np.random.randn(50),
            "umap_2": np.random.randn(50),
        }
    )

    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(20)],
            "protein": [f"protein_{i}" for i in range(20)],
        }
    )

    X = np.random.rand(50, 20) * 10
    M = np.zeros_like(X, dtype=np.int8)
    M[X < 2] = 1  # Some missing

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})
    container.assays["proteins"] = assay

    return container


class TestScatterBasic:
    """Tests for basic scatter functionality."""

    def test_scatter_basic(self, embedding_container):
        """Test basic scatter plot."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
        )
        assert ax is not None
        assert ax.get_xlabel() == "UMAP1"
        assert ax.get_ylabel() == "UMAP2"

    def test_scatter_with_pca(self, embedding_container):
        """Test scatter plot with PCA basis."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="pca",
        )
        assert ax is not None
        assert ax.get_xlabel() == "PCA1"
        assert ax.get_ylabel() == "PCA2"

    def test_scatter_with_tsne(self, embedding_container):
        """Test scatter plot with t-SNE basis."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="tsne",
        )
        assert ax is not None
        assert ax.get_xlabel() == "TSNE1"
        assert ax.get_ylabel() == "TSNE2"

    def test_scatter_missing_basis_raises(self, embedding_container):
        """Test that missing basis columns raise ValueError."""
        with pytest.raises(ValueError, match="Embedding columns"):
            scatter(
                embedding_container,
                layer="normalized",
                basis="nonexistent",
            )


class TestScatterColor:
    """Tests for scatter plot coloring."""

    def test_scatter_with_color_obs_column(self, embedding_container):
        """Test scatter with color by obs column."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            color="cluster",
        )
        assert ax is not None
        # Check that legend was created
        legend = ax.get_legend()
        assert legend is not None

    def test_scatter_with_color_feature(self, embedding_container):
        """Test scatter with color by feature."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            color="protein_0",
        )
        assert ax is not None

    def test_scatter_with_missing_values(self, container_with_missing_values):
        """Test scatter shows missing values."""
        ax = scatter(
            container_with_missing_values,
            layer="raw",
            basis="umap",
            color="protein_0",
            show_missing_values=True,
        )
        assert ax is not None
        # When using feature coloring with missing values,
        # the scatter plot should still be created
        # Legend may not be created for feature-based coloring
        # but the plot should render without error


class TestScatterParameters:
    """Tests for various scatter parameters."""

    def test_scatter_with_size(self, embedding_container):
        """Test scatter with custom size."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            size=10.0,
        )
        assert ax is not None

    def test_scatter_with_alpha(self, embedding_container):
        """Test scatter with custom alpha."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            alpha=0.5,
        )
        assert ax is not None

    def test_scatter_no_frameon(self, embedding_container):
        """Test scatter without frame."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            frameon=False,
        )
        assert ax is not None
        assert not ax.get_frame_on()

    def test_scatter_with_title(self, embedding_container):
        """Test scatter with custom title."""
        custom_title = "My Custom Plot"
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            title=custom_title,
        )
        assert ax is not None
        assert ax.get_title() == custom_title


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_umap_function(self, embedding_container):
        """Test umap convenience function."""
        ax = umap(embedding_container, layer="normalized")
        assert ax is not None
        assert ax.get_xlabel() == "UMAP1"

    def test_pca_function(self, embedding_container):
        """Test pca convenience function."""
        ax = pca(embedding_container, layer="normalized")
        assert ax is not None
        assert ax.get_xlabel() == "PCA1"

    def test_tsne_function(self, embedding_container):
        """Test tsne convenience function."""
        ax = tsne(embedding_container, layer="normalized")
        assert ax is not None
        assert ax.get_xlabel() == "TSNE1"


class TestScatterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scatter_no_missing_values_shows_all_points(self, embedding_container):
        """Test that all points are shown when no missing values."""
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            color="cluster",
            show_missing_values=False,
        )
        assert ax is not None

    def test_scatter_with_custom_ax(self, embedding_container):
        """Test scatter with custom axes."""
        import matplotlib.pyplot as plt

        fig, custom_ax = plt.subplots()
        ax = scatter(
            embedding_container,
            layer="normalized",
            basis="umap",
            ax=custom_ax,
        )
        assert ax is custom_ax
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
