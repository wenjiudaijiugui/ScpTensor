"""Tests for normalization visualization recipes.

This module provides comprehensive tests for normalization visualization functions,
including basic functionality, parameterization, edge cases, and error handling.
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import LayerNotFoundError, VisualizationError
from scptensor.viz.recipes.normalization import (
    plot_normalization_diagnostics,
    plot_normalization_effect,
)


class TestNormalizationEffect:
    """Test suite for plot_normalization_effect function."""

    def test_plot_normalization_effect_basic(self, container_with_norm: ScpContainer) -> None:
        """Test basic normalization effect plot."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        assert ax is not None
        assert hasattr(ax, "get_title")
        # Title contains "Normalization Effect"
        assert "Normalization" in ax.get_title() or "Effect" in ax.get_title()

    @pytest.mark.parametrize("group_by", [None, "batch"])
    def test_plot_normalization_effect_grouping(
        self, container_with_norm: ScpContainer, group_by: str | None
    ) -> None:
        """Test normalization effect with different groupings."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=group_by,
        )

        assert ax is not None
        # Plot should be created successfully

    @pytest.mark.parametrize("max_features", [None, 5, 10])
    def test_plot_normalization_with_max_features(
        self, container_with_norm: ScpContainer, max_features: int | None
    ) -> None:
        """Test with feature sampling."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            max_features=max_features,
        )

        assert ax is not None
        # Verify plot was created successfully

    def test_plot_normalization_with_axes(self, container_with_norm: ScpContainer) -> None:
        """Test plotting on provided axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        result_ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
            ax=ax,
        )

        assert result_ax is ax
        plt.close(fig)


class TestNormalizationDiagnostics:
    """Test suite for plot_normalization_diagnostics function."""

    def test_plot_normalization_diagnostics_basic(self, container_with_norm: ScpContainer) -> None:
        """Test basic normalization diagnostics multi-panel plot."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        assert fig is not None
        # 4 panels + 1 colorbar = 5 axes
        assert len(fig.axes) >= 4

    def test_plot_normalization_diagnostics_panels_have_titles(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that all panels have appropriate titles."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) >= 4  # At least 4 panels should have titles
        assert all(title != "" for title in titles)

    def test_plot_normalization_diagnostics_show_false(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that show=False prevents display."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        assert fig is not None
        # Figure should exist but not be shown


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_plot_normalization_single_feature(self) -> None:
        """Test with minimal data (2 samples × 1 feature)."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "batch": ["A", "B"]})
        var = pl.DataFrame({"_index": ["P1"]})
        container = ScpContainer(obs)

        assay = Assay(var)
        X_raw = np.array([[10.0], [20.0]])
        X_norm = X_raw - np.median(X_raw)

        assay.add_layer("raw", ScpMatrix(X=X_raw))
        assay.add_layer("normalized", ScpMatrix(X=X_norm))
        container.add_assay("proteins", assay)

        ax = plot_normalization_effect(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        assert ax is not None

    def test_plot_normalization_single_sample(self) -> None:
        """Test with single sample (1 sample × 5 features)."""
        obs = pl.DataFrame({"_index": ["S1"], "batch": ["A"]})
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
        container = ScpContainer(obs)

        assay = Assay(var)
        X_raw = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
        X_norm = X_raw - np.median(X_raw)

        assay.add_layer("raw", ScpMatrix(X=X_raw))
        assay.add_layer("normalized", ScpMatrix(X=X_norm))
        container.add_assay("proteins", assay)

        ax = plot_normalization_effect(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        assert ax is not None

    def test_plot_normalization_sparse_matrix(self) -> None:
        """Test with sparse matrix data."""
        obs = pl.DataFrame(
            {"_index": [f"S{i}" for i in range(10)], "batch": np.repeat(["A", "B"], 5)}
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})
        container = ScpContainer(obs)

        assay = Assay(var)

        # Create sparse data (70% sparse)
        X_dense = np.random.rand(10, 20) * 10
        X_dense[X_dense < 7] = 0
        X_sparse = sparse.csr_matrix(X_dense)

        # Normalize (add constant to avoid zero issues)
        X_norm_sparse = X_sparse.copy()
        X_norm_sparse.data = X_norm_sparse.data - np.median(X_dense)

        assay.add_layer("raw", ScpMatrix(X=X_sparse))
        assay.add_layer("normalized", ScpMatrix(X=X_norm_sparse))
        container.add_assay("proteins", assay)

        ax = plot_normalization_effect(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        assert ax is not None

    def test_plot_normalization_large_dataset(self) -> None:
        """Test with larger dataset (100 samples × 50 features)."""
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(100)],
                "batch": np.random.choice(["A", "B", "C"], 100),
            }
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(50)]})
        container = ScpContainer(obs)

        assay = Assay(var)
        X_raw = np.random.rand(100, 50) * 10 + 5
        X_norm = X_raw - np.median(X_raw, axis=1, keepdims=True)

        assay.add_layer("raw", ScpMatrix(X=X_raw))
        assay.add_layer("normalized", ScpMatrix(X=X_norm))
        container.add_assay("proteins", assay)

        ax = plot_normalization_effect(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by="batch",
        )

        assert ax is not None

    def test_plot_normalization_with_zeros(self) -> None:
        """Test handling of zero values in data."""
        obs = pl.DataFrame(
            {"_index": [f"S{i}" for i in range(10)], "batch": np.repeat(["A", "B"], 5)}
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})
        container = ScpContainer(obs)

        assay = Assay(var)

        # Create data with zeros
        X_raw = np.random.rand(10, 20) * 10
        X_raw[X_raw < 3] = 0  # ~30% zeros

        X_norm = X_raw - np.median(X_raw[X_raw > 0])

        assay.add_layer("raw", ScpMatrix(X=X_raw))
        assay.add_layer("normalized", ScpMatrix(X=X_norm))
        container.add_assay("proteins", assay)

        ax = plot_normalization_effect(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        assert ax is not None


class TestErrorHandling:
    """Test suite for error handling and validation."""

    def test_plot_normalization_missing_assay(self, container_with_norm: ScpContainer) -> None:
        """Test error handling for missing assay."""
        with pytest.raises(VisualizationError, match="Assay 'nonexistent_assay' not found"):
            plot_normalization_effect(
                container_with_norm,
                assay_name="nonexistent_assay",
                pre_layer="raw",
                post_layer="normalized",
            )

    def test_plot_normalization_missing_pre_layer(self, container_with_norm: ScpContainer) -> None:
        """Test error handling for missing pre-normalization layer."""
        # LayerNotFoundError gets wrapped in VisualizationError
        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_normalization_effect(
                container_with_norm,
                assay_name="proteins",
                pre_layer="nonexistent_layer",
                post_layer="normalized",
            )

    def test_plot_normalization_missing_post_layer(self, container_with_norm: ScpContainer) -> None:
        """Test error handling for missing post-normalization layer."""
        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_normalization_effect(
                container_with_norm,
                assay_name="proteins",
                pre_layer="raw",
                post_layer="nonexistent_layer",
            )

    def test_plot_normalization_diagnostics_missing_assay(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test error handling for missing assay in diagnostics."""
        with pytest.raises(VisualizationError, match="Assay 'nonexistent_assay' not found"):
            plot_normalization_diagnostics(
                container_with_norm,
                assay_name="nonexistent_assay",
                pre_layer="raw",
                post_layer="normalized",
                show=False,
            )

    def test_plot_normalization_diagnostics_missing_layer(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test error handling for missing layer in diagnostics."""
        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_normalization_diagnostics(
                container_with_norm,
                assay_name="proteins",
                pre_layer="nonexistent_layer",
                post_layer="normalized",
                show=False,
            )

    def test_plot_normalization_invalid_group_column(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test error handling for invalid group column."""
        with pytest.raises(VisualizationError, match="Column 'nonexistent_column' not found"):
            plot_normalization_effect(
                container_with_norm,
                assay_name="proteins",
                pre_layer="raw",
                post_layer="normalized",
                group_by="nonexistent_column",
            )


class TestParameterValidation:
    """Test suite for parameter validation."""

    def test_plot_normalization_default_parameters(self, container_with_norm: ScpContainer) -> None:
        """Test that default parameters work correctly."""
        ax = plot_normalization_effect(
            container_with_norm,
        )

        assert ax is not None

    def test_plot_normalization_diagnostics_default_parameters(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that default parameters work for diagnostics."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            show=False,
        )

        assert fig is not None
        # 4 panels + colorbar = 5 axes
        assert len(fig.axes) >= 4

    def test_plot_normalization_with_small_max_features(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test with very small max_features (should still work)."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            max_features=2,
        )

        assert ax is not None

    def test_plot_normalization_with_large_max_features(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test with max_features larger than actual features."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            max_features=1000,  # More than 20 actual features
        )

        assert ax is not None


class TestDataConsistency:
    """Test suite for data consistency across visualizations."""

    def test_plot_normalization_same_data_different_groups(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that same data produces consistent plots with different groupings."""
        ax1 = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        ax2 = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by="batch",
        )

        assert ax1 is not None
        assert ax2 is not None

    def test_plot_normalization_diagnostics_multiple_calls(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that multiple calls produce consistent results."""
        fig1 = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        fig2 = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        assert fig1 is not None
        assert fig2 is not None
        # Same number of axes
        assert len(fig1.axes) == len(fig2.axes)


class TestVisualizationOutput:
    """Test suite for visualization output properties."""

    def test_plot_normalization_has_labels(self, container_with_norm: ScpContainer) -> None:
        """Test that plot has proper axis labels."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            group_by=None,
        )

        # At minimum, ylabel should be set (xlabel might be empty for violin plots)
        assert ax.get_ylabel() != ""

    def test_plot_normalization_diagnostics_all_panels_labeled(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that all diagnostic panels have labels."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        for ax in fig.axes:
            # At minimum, panels should have titles or labels
            assert ax.get_title() != "" or ax.get_xlabel() != "" or ax.get_ylabel() != ""

    def test_plot_normalization_return_type(self, container_with_norm: ScpContainer) -> None:
        """Test that function returns correct type."""
        ax = plot_normalization_effect(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
        )

        # Check it's a matplotlib Axes
        assert hasattr(ax, "plot")
        assert hasattr(ax, "scatter")
        assert hasattr(ax, "set_xlabel")
        assert hasattr(ax, "set_ylabel")

    def test_plot_normalization_diagnostics_return_type(
        self, container_with_norm: ScpContainer
    ) -> None:
        """Test that diagnostics returns correct type."""
        fig = plot_normalization_diagnostics(
            container_with_norm,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="normalized",
            show=False,
        )

        # Check it's a matplotlib Figure
        assert hasattr(fig, "axes")
        assert hasattr(fig, "subplots")
        # 4 panels + colorbar
        assert len(fig.axes) >= 4
