"""Tests for QC visualization recipes."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import LayerNotFoundError, VisualizationError
from scptensor.dim_reduction import pca as compute_pca
from scptensor.viz.recipes.qc import (
    missing_value_patterns,
    pca_overview,
    qc_completeness,
    qc_matrix_spy,
)


@pytest.fixture
def test_container():
    """Create a test container with PCA results."""
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(100)],
            "cluster": np.random.choice(["A", "B", "C"], 100),
            "batch": np.random.choice(["Batch1", "Batch2"], 100),
        }
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {
            "_index": [f"P{i}" for i in range(20)],
            "protein": [f"P{i}" for i in range(20)],
        }
    )

    # Create data with some missing values
    np.random.seed(42)
    X = np.random.rand(100, 20) * 10
    X[X < 2] = 0  # Simulate missing values

    M = np.zeros_like(X, dtype=np.int8)
    M[X == 0] = 1  # MBR

    assay = Assay(var=var, layers={"imputed": ScpMatrix(X=X, M=M)})
    container.assays["proteins"] = assay

    # Run PCA
    container = compute_pca(container, "proteins", "imputed", n_components=5)

    return container


class TestPcaOverview:
    """Tests for pca_overview function."""

    def test_basic_pca_overview(self, test_container):
        """Test basic PCA overview visualization."""
        fig = pca_overview(test_container, layer="scores", show=False)
        assert fig is not None
        assert hasattr(fig, "axes")
        # Figure has 4 plots + colorbar = 5 axes
        assert len(fig.axes) >= 4

    def test_pca_overview_with_color(self, test_container):
        """Test PCA overview with color grouping."""
        fig = pca_overview(test_container, layer="scores", color="cluster", show=False)
        assert fig is not None

    def test_pca_overview_custom_n_pcs(self, test_container):
        """Test PCA overview with custom n_pcs."""
        fig = pca_overview(test_container, layer="scores", n_pcs=5, show=False)
        assert fig is not None

    def test_pca_overview_invalid_layer(self, test_container):
        """Test PCA overview with invalid layer."""
        with pytest.raises((VisualizationError, LayerNotFoundError)):
            pca_overview(test_container, layer="nonexistent", show=False)

    def test_pca_overview_invalid_color(self, test_container):
        """Test PCA overview with invalid color column."""
        with pytest.raises(VisualizationError):
            pca_overview(test_container, layer="scores", color="invalid_column", show=False)


class TestMissingValuePatterns:
    """Tests for missing_value_patterns function."""

    def test_basic_missing_patterns(self, test_container):
        """Test basic missing value patterns visualization."""
        fig = missing_value_patterns(test_container, layer="imputed", show=False)
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 4  # 2x2 grid

    def test_missing_patterns_with_groupby(self, test_container):
        """Test missing value patterns with grouping."""
        fig = missing_value_patterns(test_container, layer="imputed", groupby="cluster", show=False)
        assert fig is not None

    def test_missing_patterns_invalid_layer(self, test_container):
        """Test missing value patterns with invalid layer."""
        with pytest.raises((VisualizationError, LayerNotFoundError)):
            missing_value_patterns(test_container, layer="nonexistent", show=False)

    def test_missing_patterns_invalid_groupby(self, test_container):
        """Test missing value patterns with invalid groupby."""
        with pytest.raises(VisualizationError):
            missing_value_patterns(
                test_container, layer="imputed", groupby="invalid_column", show=False
            )


class TestExistingQcFunctions:
    """Tests for existing QC functions to ensure they still work."""

    def test_qc_completeness(self, test_container):
        """Test qc_completeness function."""
        ax = qc_completeness(
            test_container,
            assay_name="proteins",
            layer="imputed",
            group_by="cluster",
        )
        assert ax is not None

    def test_qc_matrix_spy(self, test_container):
        """Test qc_matrix_spy function."""
        ax = qc_matrix_spy(test_container, assay_name="proteins", layer="imputed")
        assert ax is not None
