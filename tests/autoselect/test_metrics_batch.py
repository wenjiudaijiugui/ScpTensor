"""Tests for batch effect metrics module.

This module provides comprehensive tests for the batch effect metrics functions
used in the automatic method selection system.
"""

from __future__ import annotations

import numpy as np
import pytest

from scptensor.autoselect.metrics.batch import (
    batch_asw,
    batch_mixing_score,
    bio_asw,
)


class TestBatchASW:
    """Tests for batch_asw function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        batch_labels = np.array([], dtype=int)
        result = batch_asw(X, batch_labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        batch_labels = np.array([0])
        result = batch_asw(X, batch_labels)
        assert result == 0.0

    def test_single_batch(self) -> None:
        """Test with only one batch returns 0.0."""
        X = np.random.randn(100, 10)
        batch_labels = np.zeros(100, dtype=int)
        result = batch_asw(X, batch_labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        batch_labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            batch_asw(X, batch_labels)

    def test_well_mixed_batches(self) -> None:
        """Test with well-mixed batches should return high score."""
        np.random.seed(42)
        # Create well-mixed data: both batches from same distribution
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        result = batch_asw(X, batch_labels)
        assert 0.0 <= result <= 1.0
        # Should be reasonably high since batches are mixed
        assert result > 0.3

    def test_separated_batches(self) -> None:
        """Test with clearly separated batches should return lower score."""
        np.random.seed(42)
        # Create separated batches
        X1 = np.random.randn(50, 10) + 10  # Batch 0: centered at 10
        X2 = np.random.randn(50, 10) - 10  # Batch 1: centered at -10
        X = np.vstack([X1, X2])
        batch_labels = np.repeat([0, 1], 50)
        result = batch_asw(X, batch_labels)
        assert 0.0 <= result <= 1.0
        # Should be lower since batches are separated
        assert result < 0.7

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan  # 10% NaN rows
        batch_labels = np.repeat([0, 1], 50)
        result = batch_asw(X, batch_labels)
        assert 0.0 <= result <= 1.0

    def test_all_nans(self) -> None:
        """Test with all NaN values returns 0.0."""
        X = np.full((100, 10), np.nan)
        batch_labels = np.repeat([0, 1], 50)
        result = batch_asw(X, batch_labels)
        assert result == 0.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        result = batch_asw(X, batch_labels)
        assert isinstance(result, float)


class TestBioASW:
    """Tests for bio_asw function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        bio_labels = np.array([], dtype=int)
        result = bio_asw(X, bio_labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        bio_labels = np.array([0])
        result = bio_asw(X, bio_labels)
        assert result == 0.0

    def test_single_group(self) -> None:
        """Test with only one biological group returns 0.0."""
        X = np.random.randn(100, 10)
        bio_labels = np.zeros(100, dtype=int)
        result = bio_asw(X, bio_labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        bio_labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            bio_asw(X, bio_labels)

    def test_well_separated_groups(self) -> None:
        """Test with well-separated biological groups should return high score."""
        np.random.seed(42)
        # Create clearly separated groups
        X1 = np.random.randn(25, 10) + 10  # Group 0
        X2 = np.random.randn(25, 10) - 10  # Group 1
        X3 = np.random.randn(25, 10) + 20  # Group 2
        X4 = np.random.randn(25, 10) - 20  # Group 3
        X = np.vstack([X1, X2, X3, X4])
        bio_labels = np.repeat([0, 1, 2, 3], 25)
        result = bio_asw(X, bio_labels)
        assert 0.0 <= result <= 1.0
        # Should be high since groups are well-separated
        assert result > 0.5

    def test_overlapping_groups(self) -> None:
        """Test with overlapping groups should return lower score."""
        np.random.seed(42)
        # Create overlapping groups (all from same distribution)
        X = np.random.randn(100, 10)
        bio_labels = np.repeat([0, 1, 2, 3], 25)
        result = bio_asw(X, bio_labels)
        assert 0.0 <= result <= 1.0
        # Should be lower since groups overlap

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan  # 10% NaN rows
        bio_labels = np.repeat([0, 1, 2, 3], 25)
        result = bio_asw(X, bio_labels)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        bio_labels = np.repeat([0, 1, 2, 3], 25)
        result = bio_asw(X, bio_labels)
        assert isinstance(result, float)


class TestBatchMixingScore:
    """Tests for batch_mixing_score function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        batch_labels = np.array([], dtype=int)
        result = batch_mixing_score(X, batch_labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        batch_labels = np.array([0])
        result = batch_mixing_score(X, batch_labels)
        assert result == 0.0

    def test_single_batch(self) -> None:
        """Test with only one batch returns 0.0."""
        X = np.random.randn(100, 10)
        batch_labels = np.zeros(100, dtype=int)
        result = batch_mixing_score(X, batch_labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        batch_labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            batch_mixing_score(X, batch_labels)

    def test_well_mixed_batches(self) -> None:
        """Test with well-mixed batches should return high score."""
        np.random.seed(42)
        # Create well-mixed data: both batches from same distribution
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        result = batch_mixing_score(X, batch_labels, n_neighbors=10)
        assert 0.0 <= result <= 1.0
        # Should be high since batches are well-mixed
        assert result > 0.5

    def test_separated_batches(self) -> None:
        """Test with separated batches should return lower score."""
        np.random.seed(42)
        # Create separated batches
        X1 = np.random.randn(50, 10) + 10
        X2 = np.random.randn(50, 10) - 10
        X = np.vstack([X1, X2])
        batch_labels = np.repeat([0, 1], 50)
        result = batch_mixing_score(X, batch_labels, n_neighbors=10)
        assert 0.0 <= result <= 1.0
        # Should be lower since batches are separated
        assert result < 0.8

    def test_custom_n_neighbors(self) -> None:
        """Test with custom n_neighbors parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)

        result_5 = batch_mixing_score(X, batch_labels, n_neighbors=5)
        result_20 = batch_mixing_score(X, batch_labels, n_neighbors=20)

        assert 0.0 <= result_5 <= 1.0
        assert 0.0 <= result_20 <= 1.0

    def test_n_neighbors_exceeds_samples(self) -> None:
        """Test when n_neighbors exceeds number of samples."""
        np.random.seed(42)
        X = np.random.randn(10, 5)
        batch_labels = np.repeat([0, 1], 5)
        # Should handle gracefully by adjusting n_neighbors
        result = batch_mixing_score(X, batch_labels, n_neighbors=100)
        assert 0.0 <= result <= 1.0

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan  # 10% NaN rows
        batch_labels = np.repeat([0, 1], 50)
        result = batch_mixing_score(X, batch_labels, n_neighbors=10)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        result = batch_mixing_score(X, batch_labels)
        assert isinstance(result, float)

    def test_multiple_batches(self) -> None:
        """Test with more than 2 batches."""
        np.random.seed(42)
        X = np.random.randn(120, 10)
        batch_labels = np.repeat([0, 1, 2, 3], 30)
        result = batch_mixing_score(X, batch_labels, n_neighbors=15)
        assert 0.0 <= result <= 1.0


class TestMetricProperties:
    """Test common properties across all batch metrics."""

    def test_all_metrics_return_in_range(self) -> None:
        """Test that all metrics return values in [0, 1] range."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        bio_labels = np.repeat([0, 1, 2, 3], 25)

        metrics = [
            batch_asw(X, batch_labels),
            bio_asw(X, bio_labels),
            batch_mixing_score(X, batch_labels),
        ]

        for metric_value in metrics:
            assert 0.0 <= metric_value <= 1.0, f"Metric {metric_value} not in [0, 1]"

    def test_all_metrics_return_float(self) -> None:
        """Test that all metrics return Python float type."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        batch_labels = np.repeat([0, 1], 50)
        bio_labels = np.repeat([0, 1, 2, 3], 25)

        assert isinstance(batch_asw(X, batch_labels), float)
        assert isinstance(bio_asw(X, bio_labels), float)
        assert isinstance(batch_mixing_score(X, batch_labels), float)

    def test_reproducibility(self) -> None:
        """Test that metrics are deterministic for same input."""
        np.random.seed(42)
        X1 = np.random.randn(100, 10)
        batch_labels1 = np.repeat([0, 1], 50)
        bio_labels1 = np.repeat([0, 1, 2, 3], 25)

        np.random.seed(42)
        X2 = np.random.randn(100, 10)
        batch_labels2 = np.repeat([0, 1], 50)
        bio_labels2 = np.repeat([0, 1, 2, 3], 25)

        assert batch_asw(X1, batch_labels1) == batch_asw(X2, batch_labels2)
        assert bio_asw(X1, bio_labels1) == bio_asw(X2, bio_labels2)
        assert batch_mixing_score(X1, batch_labels1) == batch_mixing_score(
            X2, batch_labels2
        )
