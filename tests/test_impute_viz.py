"""Tests for imputation visualization recipes.

Tests cover:
- scptensor.viz.recipes.impute: All 4 imputation visualization functions
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.viz.recipes.impute import (
    plot_imputation_comparison,
    plot_imputation_metrics,
    plot_imputation_scatter,
    plot_missing_pattern,
)

# ============================================================================
# Fixtures for imputation visualization tests
# ============================================================================


@pytest.fixture
def impute_viz_container():
    """Create a container with imputed layers for visualization tests."""
    np.random.seed(42)

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(1, 21)],
            "batch": ["A"] * 10 + ["B"] * 10,
            "group": ["control"] * 10 + ["treated"] * 10,
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"P{i}" for i in range(1, 51)],
            "protein_name": [f"Protein{i}" for i in range(1, 51)],
        }
    )

    # Create base data with some missing values
    X = np.random.rand(20, 50) * 10 + 1
    M = np.zeros((20, 50), dtype=np.int8)

    # Add missing values (20% missing)
    missing_indices = np.random.choice(20 * 50, size=200, replace=False)
    for idx in missing_indices:
        i, j = idx // 50, idx % 50
        M[i, j] = MaskCode.LOD
        X[i, j] = np.nan

    # Replace NaN with placeholder
    X[np.isnan(X)] = 0

    # Create imputed versions
    X_knn = X.copy()
    X_qrilc = X.copy()
    X_bpca = X.copy()

    for idx in missing_indices:
        i, j = idx // 50, idx % 50
        # Simulate imputed values
        X_knn[i, j] = np.random.rand() * 5 + 2
        X_qrilc[i, j] = np.random.rand() * 3 + 1
        X_bpca[i, j] = np.random.rand() * 4 + 1.5

    # Create masks for imputed layers
    M_imp = M.copy()
    for idx in missing_indices:
        i, j = idx // 50, idx % 50
        M_imp[i, j] = MaskCode.IMPUTED

    assay = Assay(
        var=var,
        layers={
            "raw": ScpMatrix(X=X, M=M),
            "knn_": ScpMatrix(X=X_knn, M=M_imp),
            "qrilc_": ScpMatrix(X=X_qrilc, M=M_imp),
            "bpca_": ScpMatrix(X=X_bpca, M=M_imp),
        },
    )

    return ScpContainer(obs=obs, assays={"proteins": assay})


@pytest.fixture
def true_container():
    """Create a container with true (complete) values for comparison."""
    np.random.seed(42)

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(1, 21)]})

    var = pl.DataFrame({"_index": [f"P{i}" for i in range(1, 51)]})

    # True values without missing
    X_true = np.random.rand(20, 50) * 10 + 1

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X_true, M=None)})

    return ScpContainer(obs=obs, assays={"proteins": assay})


@pytest.fixture
def minimal_impute_container():
    """Create minimal container for edge case tests."""
    obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
    var = pl.DataFrame({"_index": ["P1", "P2"]})

    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    M = np.array([[0, 0], [0, 1], [0, 0]], dtype=np.int8)
    X[np.isnan(X)] = 0

    X_imp = X.copy()
    X_imp[1, 1] = 2.5
    M_imp = M.copy()
    M_imp[1, 1] = MaskCode.IMPUTED

    assay = Assay(
        var=var,
        layers={
            "raw": ScpMatrix(X=X, M=M),
            "knn": ScpMatrix(X=X_imp, M=M_imp),
        },
    )

    return ScpContainer(obs=obs, assays={"proteins": assay})


# ============================================================================
# Tests for plot_imputation_comparison
# ============================================================================


class TestPlotImputationComparison:
    """Tests for plot_imputation_comparison function."""

    def test_returns_axes(self, impute_viz_container):
        """plot_imputation_comparison should return matplotlib Axes."""
        plt.close("all")
        ax = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=["knn_", "qrilc_"],
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_basic_comparison(self, impute_viz_container):
        """Test basic imputation comparison plot."""
        plt.close("all")
        ax = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=["knn_", "qrilc_"],
        )

        # Check that bars were created
        assert len(ax.containers) > 0

        # Check that results are attached
        assert hasattr(ax, "imputation_results")
        assert "knn_" in ax.imputation_results
        plt.close("all")

    def test_auto_detect_methods(self, impute_viz_container):
        """Test auto-detection of imputed layers."""
        plt.close("all")
        ax = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=None,  # Auto-detect
        )

        # Should detect all imputed layers
        assert hasattr(ax, "imputation_results")
        assert len(ax.imputation_results) >= 2
        plt.close("all")

    def test_custom_metrics(self, impute_viz_container):
        """Test with custom metrics selection."""
        plt.close("all")
        ax = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=["knn_"],
            metrics=["nrmse"],
        )

        assert hasattr(ax, "imputation_results")
        assert "nrmse" in ax.imputation_results["knn_"]
        plt.close("all")

    def test_invalid_assay_raises(self, impute_viz_container):
        """Test with invalid assay name."""
        from scptensor.core.exceptions import AssayNotFoundError

        with pytest.raises(AssayNotFoundError):
            plot_imputation_comparison(
                impute_viz_container,
                "invalid_assay",
                "raw",
            )

    def test_invalid_layer_raises(self, impute_viz_container):
        """Test with invalid layer name."""
        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises(LayerNotFoundError):
            plot_imputation_comparison(
                impute_viz_container,
                "proteins",
                "invalid_layer",
            )

    def test_no_imputed_layers_raises(self, impute_viz_container):
        """Test with no imputed layers available."""
        # Create container without imputed layers
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.array([[1.0]])
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(ValueError, match="No imputed layers"):
            plot_imputation_comparison(
                container,
                "proteins",
                "raw",
                methods=None,
            )


# ============================================================================
# Tests for plot_imputation_scatter
# ============================================================================


class TestPlotImputationScatter:
    """Tests for plot_imputation_scatter function."""

    def test_returns_axes(self, true_container, impute_viz_container):
        """plot_imputation_scatter should return matplotlib Axes."""
        plt.close("all")
        ax = plot_imputation_scatter(
            true_container,
            impute_viz_container,
            "proteins",
            "raw",
            "knn_",
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_basic_scatter(self, true_container, impute_viz_container):
        """Test basic scatter plot creation."""
        plt.close("all")
        ax = plot_imputation_scatter(
            true_container,
            impute_viz_container,
            "proteins",
            "raw",
            "knn_",
        )

        # Check that scatter points were created
        assert len(ax.collections) >= 2  # At least observed and imputed

        # Check that statistics are attached
        assert hasattr(ax, "scatter_stats")
        assert "pcc_all" in ax.scatter_stats
        assert "nrmse" in ax.scatter_stats
        plt.close("all")

    def test_identity_line_present(self, true_container, impute_viz_container):
        """Test that identity line is plotted."""
        plt.close("all")
        ax = plot_imputation_scatter(
            true_container,
            impute_viz_container,
            "proteins",
            "raw",
            "knn_",
        )

        # Check for identity line (line plot)
        lines = ax.get_lines()
        assert len(lines) > 0
        plt.close("all")

    def test_legend_present(self, true_container, impute_viz_container):
        """Test that legend is present."""
        plt.close("all")
        ax = plot_imputation_scatter(
            true_container,
            impute_viz_container,
            "proteins",
            "raw",
            "knn_",
        )

        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")

    def test_invalid_assay_raises(self, true_container, impute_viz_container):
        """Test with invalid assay name."""
        from scptensor.core.exceptions import AssayNotFoundError

        with pytest.raises(AssayNotFoundError):
            plot_imputation_scatter(
                true_container,
                impute_viz_container,
                "invalid_assay",
                "raw",
                "knn_",
            )

    def test_invalid_layer_raises(self, true_container, impute_viz_container):
        """Test with invalid layer name."""
        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises(LayerNotFoundError):
            plot_imputation_scatter(
                true_container,
                impute_viz_container,
                "proteins",
                "invalid_layer",
                "knn_",
            )


# ============================================================================
# Tests for plot_imputation_metrics
# ============================================================================


class TestPlotImputationMetrics:
    """Tests for plot_imputation_metrics function."""

    def test_returns_axes(self):
        """plot_imputation_metrics should return matplotlib Axes."""
        plt.close("all")
        metrics = {
            "KNN": {"nrmse": 0.15, "pcc": 0.92},
            "QRILC": {"nrmse": 0.18, "pcc": 0.89},
        }
        ax = plot_imputation_metrics(metrics)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_basic_metrics_plot(self):
        """Test basic metrics bar chart."""
        plt.close("all")
        metrics = {
            "KNN": {"nrmse": 0.15, "pcc": 0.92},
            "QRILC": {"nrmse": 0.18, "pcc": 0.89},
            "BPCA": {"nrmse": 0.12, "pcc": 0.95},
        }
        ax = plot_imputation_metrics(metrics)

        # Check that bars were created
        assert len(ax.containers) > 0

        # Check labels
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "KNN" in xtick_labels
        assert "QRILC" in xtick_labels
        plt.close("all")

    def test_custom_metric_names(self):
        """Test with custom metric display names."""
        plt.close("all")
        metrics = {
            "KNN": {"nrmse": 0.15},
            "BPCA": {"nrmse": 0.12},
        }
        ax = plot_imputation_metrics(
            metrics,
            metric_names=["RMSE"],  # Maps to 'nrmse' key
        )

        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")

    def test_single_method(self):
        """Test with single imputation method."""
        plt.close("all")
        metrics = {"KNN": {"nrmse": 0.15, "pcc": 0.92}}
        ax = plot_imputation_metrics(metrics)

        assert len(ax.containers) > 0
        plt.close("all")

    def test_empty_metrics_works(self):
        """Test with empty metrics dictionary creates empty plot."""
        plt.close("all")
        ax = plot_imputation_metrics({})
        assert ax is not None
        plt.close("all")


# ============================================================================
# Tests for plot_missing_pattern
# ============================================================================


class TestPlotMissingPattern:
    """Tests for plot_missing_pattern function."""

    def test_returns_axes(self, impute_viz_container):
        """plot_missing_pattern should return matplotlib Axes."""
        plt.close("all")
        ax = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_basic_pattern_plot(self, impute_viz_container):
        """Test basic missing pattern heatmap."""
        plt.close("all")
        ax = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
        )

        # Check that image was created
        assert len(ax.images) > 0

        # Check that statistics are attached
        assert hasattr(ax, "pattern_stats")
        assert "missing_rate" in ax.pattern_stats
        plt.close("all")

    def test_custom_limits(self, impute_viz_container):
        """Test with custom max_features and max_samples."""
        plt.close("all")
        ax = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
            max_features=20,
            max_samples=10,
        )

        assert hasattr(ax, "pattern_stats")
        plt.close("all")

    def test_no_missing_data(self):
        """Test with container without missing data."""
        plt.close("all")
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        ax = plot_missing_pattern(container, "proteins", "raw")

        # Should show no missing values
        assert ax.pattern_stats["missing_rate"] == 0.0
        plt.close("all")

    def test_colorbar_present(self, impute_viz_container):
        """Test that colorbar is present."""
        plt.close("all")
        ax = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
        )

        # Check for colorbar images
        assert len(ax.images) > 0
        plt.close("all")

    def test_invalid_assay_raises(self, impute_viz_container):
        """Test with invalid assay name."""
        from scptensor.core.exceptions import AssayNotFoundError

        plt.close("all")
        with pytest.raises(AssayNotFoundError):
            plot_missing_pattern(
                impute_viz_container,
                "invalid_assay",
                "raw",
            )

    def test_invalid_layer_raises(self, impute_viz_container):
        """Test with invalid layer name."""
        from scptensor.core.exceptions import LayerNotFoundError

        plt.close("all")
        with pytest.raises(LayerNotFoundError):
            plot_missing_pattern(
                impute_viz_container,
                "proteins",
                "invalid_layer",
            )


# ============================================================================
# Integration tests
# ============================================================================


class TestImputationVizIntegration:
    """Integration tests for imputation visualization."""

    def test_full_workflow(self, impute_viz_container):
        """Test complete imputation visualization workflow."""
        plt.close("all")

        # 1. Compare methods
        ax1 = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=["knn_", "qrilc_"],
        )
        assert ax1 is not None

        # 2. Show missing pattern
        ax2 = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
        )
        assert ax2 is not None

        # 3. Plot metrics from results
        results = ax1.imputation_results
        ax3 = plot_imputation_metrics(results)
        assert ax3 is not None

        plt.close("all")

    def test_different_figsize(self, impute_viz_container):
        """Test with custom figure sizes."""
        plt.close("all")

        ax1 = plot_imputation_comparison(
            impute_viz_container,
            "proteins",
            "raw",
            methods=["knn_"],
            figsize=(8, 4),
        )
        assert ax1.figure.get_size_inches()[0] == 8

        ax2 = plot_missing_pattern(
            impute_viz_container,
            "proteins",
            "raw",
            figsize=(6, 6),
        )
        assert ax2.figure.get_size_inches()[0] == 6

        plt.close("all")


# ============================================================================
# Edge case tests
# ============================================================================


class TestImputationVizEdgeCases:
    """Edge case tests for imputation visualization."""

    def test_minimal_data_comparison(self, minimal_impute_container):
        """Test comparison with minimal data."""
        plt.close("all")
        ax = plot_imputation_comparison(
            minimal_impute_container,
            "proteins",
            "raw",
            methods=["knn"],
        )

        assert ax is not None
        plt.close("all")

    def test_minimal_data_pattern(self, minimal_impute_container):
        """Test pattern plot with minimal data."""
        plt.close("all")
        ax = plot_missing_pattern(
            minimal_impute_container,
            "proteins",
            "raw",
        )

        assert ax is not None
        plt.close("all")

    def test_single_sample(self):
        """Test with single sample."""
        plt.close("all")
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
        X = np.array([[1.0, np.nan, 3.0]])
        M = np.array([[0, 1, 0]], dtype=np.int8)
        X[np.isnan(X)] = 0

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        ax = plot_missing_pattern(container, "proteins", "raw")
        assert ax is not None
        plt.close("all")

    def test_single_feature(self):
        """Test with single feature."""
        plt.close("all")
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.array([[1.0], [np.nan], [3.0]])
        M = np.array([[0], [1], [0]], dtype=np.int8)
        X[np.isnan(X)] = 0

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        ax = plot_missing_pattern(container, "proteins", "raw")
        assert ax is not None
        plt.close("all")
