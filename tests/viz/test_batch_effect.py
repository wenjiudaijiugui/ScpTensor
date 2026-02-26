"""Tests for batch effect visualization recipes.

This module provides comprehensive tests for batch effect visualization functions,
including basic functionality, parameterization, edge cases, and error handling.
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import VisualizationError
from scptensor.viz.recipes.batch_effect import (
    plot_batch_correction_comparison,
    plot_batch_effect,
)


class TestPlotBatchEffect:
    """Test suite for plot_batch_effect function."""

    def test_plot_batch_effect_basic(self, container_with_batches: ScpContainer) -> None:
        """Test basic batch effect plot."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        # 4 panels
        assert len(fig.axes) == 4

    def test_plot_batch_effect_with_assay_name(self, container_with_batches: ScpContainer) -> None:
        """Test with explicit assay name."""
        fig = plot_batch_effect(
            container_with_batches,
            assay_name="proteins",
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_with_layer(self, container_with_batches: ScpContainer) -> None:
        """Test with explicit layer name."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            layer="raw",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_pca_method(self, container_with_batches: ScpContainer) -> None:
        """Test with PCA method."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            method="pca",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_with_color_by(self, container_with_batches: ScpContainer) -> None:
        """Test with color_by parameter."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            color_by="condition",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_panels_have_titles(
        self, container_with_batches: ScpContainer
    ) -> None:
        """Test that all panels have appropriate titles."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) == 4  # All 4 panels should have titles
        assert all(title != "" for title in titles)

    def test_plot_batch_effect_show_false(self, container_with_batches: ScpContainer) -> None:
        """Test that show=False prevents display."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        # Figure should exist but not be shown


class TestPlotBatchCorrectionComparison:
    """Test suite for plot_batch_correction_comparison function."""

    def test_plot_batch_correction_comparison_basic(self) -> None:
        """Test basic batch correction comparison."""
        # Create container with pre/post correction layers
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(60)],
                "batch": np.repeat(["batch1", "batch2", "batch3"], 20),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(15)]})

        # Simulate batch effects
        X_pre = np.random.rand(60, 15) * 5
        X_pre[0:20] += 5  # batch1 offset
        X_pre[20:40] += 10  # batch2 offset

        # Simulate corrected data (remove offsets)
        X_post = X_pre.copy()
        for start_idx in [0, 20, 40]:
            end_idx = start_idx + 20
            X_post[start_idx:end_idx] -= np.mean(X_pre[start_idx:end_idx], axis=0)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        # Note: Creates 8 axes (4 panels with subplots in some panels)
        assert len(fig.axes) >= 4

    def test_plot_batch_correction_comparison_with_assay_name(self) -> None:
        """Test with explicit assay name."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(40)],
                "batch": np.repeat(["A", "B"], 20),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(10)]})

        X_pre = np.random.rand(40, 10) * 10
        X_pre[0:20] += 5

        X_post = X_pre.copy()
        X_post[0:20] -= np.mean(X_pre[0:20], axis=0)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            assay_name="proteins",
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        # Note: Creates 8 axes (4 panels with subplots in some panels)
        assert len(fig.axes) >= 4

    def test_plot_batch_correction_comparison_panels_have_titles(self) -> None:
        """Test that all panels have appropriate titles."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(40)],
                "batch": np.repeat(["A", "B"], 20),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(10)]})

        X_pre = np.random.rand(40, 10) * 10
        X_pre[0:20] += 5
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) >= 4  # At least 4 panels should have titles

    def test_plot_batch_correction_comparison_show_false(self) -> None:
        """Test that show=False prevents display."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(30)],
                "batch": np.repeat(["A", "B"], 15),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(8)]})

        X_pre = np.random.rand(30, 8) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        assert fig is not None


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_plot_batch_effect_single_batch(self) -> None:
        """Test with only one batch."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.array(["batch1"] * 20),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(10)]})

        X = np.random.rand(20, 10) * 10
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_effect(
            container,
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_many_batches(self) -> None:
        """Test with many batches (5+)."""
        np.random.seed(42)

        n_samples = 96  # Must be divisible by 6
        n_features = 20
        n_batches = 6

        batch_labels = []
        for i in range(n_batches):
            batch_labels.extend([f"batch{i}"] * (n_samples // n_batches))

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(n_samples)],
                "batch": np.array(batch_labels),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})

        X = np.random.rand(n_samples, n_features) * 10
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_effect(
            container,
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4

    def test_plot_batch_effect_sparse_matrix(self) -> None:
        """Test with sparse matrix data."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(30)],
                "batch": np.repeat(["A", "B"], 15),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})

        # Create sparse data (70% sparse)
        X_dense = np.random.rand(30, 20) * 10
        X_dense[X_dense < 7] = 0
        X_sparse = sparse.csr_matrix(X_dense)

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X_sparse)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_effect(
            container,
            batch_key="batch",
            show=False,
        )

        assert fig is not None

    def test_plot_batch_correction_comparison_single_batch(self) -> None:
        """Test comparison with single batch."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.array(["batch1"] * 20),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(10)]})

        X_pre = np.random.rand(20, 10) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        assert fig is not None

    def test_plot_batch_effect_two_batches_only(self) -> None:
        """Test with minimal case of 2 batches."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.repeat(["A", "B"], 10),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})

        X = np.random.rand(20, 5) * 10
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_effect(
            container,
            batch_key="batch",
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4


class TestErrorHandling:
    """Test suite for error handling and validation."""

    def test_plot_batch_effect_missing_batch_key(
        self, container_with_batches: ScpContainer
    ) -> None:
        """Test error handling for missing batch column."""
        with pytest.raises(VisualizationError, match="Column.*not found"):
            plot_batch_effect(
                container_with_batches,
                batch_key="nonexistent_batch",
                show=False,
            )

    def test_plot_batch_effect_missing_assay(self, container_with_batches: ScpContainer) -> None:
        """Test error handling for missing assay."""
        with pytest.raises(VisualizationError, match="Assay.*not found"):
            plot_batch_effect(
                container_with_batches,
                assay_name="nonexistent_assay",
                batch_key="batch",
                show=False,
            )

    def test_plot_batch_effect_missing_layer(self, container_with_batches: ScpContainer) -> None:
        """Test error handling for missing layer."""
        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_batch_effect(
                container_with_batches,
                batch_key="batch",
                layer="nonexistent_layer",
                show=False,
            )

    def test_plot_batch_effect_invalid_method(self, container_with_batches: ScpContainer) -> None:
        """Test error handling for invalid method."""
        with pytest.raises((ValueError, VisualizationError), match="method"):
            plot_batch_effect(
                container_with_batches,
                batch_key="batch",
                method="invalid_method",
                show=False,
            )

    def test_plot_batch_effect_missing_color_by(self, container_with_batches: ScpContainer) -> None:
        """Test error handling for missing color_by column."""
        with pytest.raises(VisualizationError, match="Column.*not found"):
            plot_batch_effect(
                container_with_batches,
                batch_key="batch",
                color_by="nonexistent_column",
                show=False,
            )

    def test_plot_batch_correction_comparison_missing_pre_layer(self) -> None:
        """Test error handling for missing pre-correction layer."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.repeat(["A", "B"], 10),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})

        X = np.random.rand(20, 5) * 10
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_batch_correction_comparison(
                container,
                pre_layer="nonexistent_pre",
                post_layer="raw",
                batch_key="batch",
                show=False,
            )

    def test_plot_batch_correction_comparison_missing_post_layer(self) -> None:
        """Test error handling for missing post-correction layer."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.repeat(["A", "B"], 10),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})

        X = np.random.rand(20, 5) * 10
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        from scptensor.core.exceptions import LayerNotFoundError

        with pytest.raises((LayerNotFoundError, VisualizationError)):
            plot_batch_correction_comparison(
                container,
                pre_layer="raw",
                post_layer="nonexistent_post",
                batch_key="batch",
                show=False,
            )

    def test_plot_batch_correction_comparison_missing_batch_key(self) -> None:
        """Test error handling for missing batch column in comparison."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})

        X_pre = np.random.rand(20, 5) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(VisualizationError, match="Column.*not found"):
            plot_batch_correction_comparison(
                container,
                pre_layer="raw",
                post_layer="corrected",
                batch_key="nonexistent_batch",
                show=False,
            )

    def test_plot_batch_correction_comparison_missing_assay(self) -> None:
        """Test error handling for missing assay in comparison."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(20)],
                "batch": np.repeat(["A", "B"], 10),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})

        X_pre = np.random.rand(20, 5) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(VisualizationError, match="Assay.*not found"):
            plot_batch_correction_comparison(
                container,
                assay_name="nonexistent_assay",
                pre_layer="raw",
                post_layer="corrected",
                batch_key="batch",
                show=False,
            )


class TestVisualizationOutput:
    """Test suite for visualization output properties."""

    def test_plot_batch_effect_return_type(self, container_with_batches: ScpContainer) -> None:
        """Test that function returns correct type."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        # Check it's a matplotlib Figure
        assert hasattr(fig, "axes")
        assert hasattr(fig, "subplots")
        assert len(fig.axes) == 4

    def test_plot_batch_correction_comparison_return_type(self) -> None:
        """Test that comparison returns correct type."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(30)],
                "batch": np.repeat(["A", "B"], 15),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(8)]})

        X_pre = np.random.rand(30, 8) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        # Check it's a matplotlib Figure
        assert hasattr(fig, "axes")
        assert hasattr(fig, "subplots")
        # Note: Creates 8 axes (4 panels with subplots in some panels)
        assert len(fig.axes) >= 4

    def test_plot_batch_effect_axes_have_labels(self, container_with_batches: ScpContainer) -> None:
        """Test that axes have proper labels."""
        fig = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        # At least some axes should have labels
        labeled_axes = [ax for ax in fig.axes if ax.get_xlabel() != "" or ax.get_ylabel() != ""]
        assert len(labeled_axes) > 0

    def test_plot_batch_correction_comparison_axes_have_labels(self) -> None:
        """Test that comparison axes have proper labels."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(30)],
                "batch": np.repeat(["A", "B"], 15),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(8)]})

        X_pre = np.random.rand(30, 8) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        # At least some axes should have labels
        labeled_axes = [ax for ax in fig.axes if ax.get_xlabel() != "" or ax.get_ylabel() != ""]
        assert len(labeled_axes) > 0


class TestMultipleCalls:
    """Test suite for consistency across multiple calls."""

    def test_plot_batch_effect_multiple_calls_consistent(
        self, container_with_batches: ScpContainer
    ) -> None:
        """Test that multiple calls produce consistent results."""
        fig1 = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        fig2 = plot_batch_effect(
            container_with_batches,
            batch_key="batch",
            show=False,
        )

        assert fig1 is not None
        assert fig2 is not None
        # Same number of axes
        assert len(fig1.axes) == len(fig2.axes)

    def test_plot_batch_correction_comparison_multiple_calls_consistent(self) -> None:
        """Test that multiple comparison calls are consistent."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(30)],
                "batch": np.repeat(["A", "B"], 15),
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(8)]})

        X_pre = np.random.rand(30, 8) * 10
        X_post = X_pre - np.mean(X_pre, axis=0, keepdims=True)

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_pre))
        assay.add_layer("corrected", ScpMatrix(X=X_post))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        fig1 = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        fig2 = plot_batch_correction_comparison(
            container,
            pre_layer="raw",
            post_layer="corrected",
            batch_key="batch",
            show=False,
        )

        assert fig1 is not None
        assert fig2 is not None
        # Same number of axes
        assert len(fig1.axes) == len(fig2.axes)
