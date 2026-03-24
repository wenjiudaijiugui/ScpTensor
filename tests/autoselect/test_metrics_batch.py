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
    ilisi_score,
    kbet_score,
    lisi_approx_score,
)
from scptensor.core._batch_metrics_kernel import compute_self_excluded_knn


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

    def test_string_labels_are_supported(self) -> None:
        """String-valued batch labels should not require caller-side encoding."""
        np.random.seed(42)
        X = np.random.randn(60, 8)
        batch_labels = np.array(["B1"] * 30 + ["B2"] * 30)

        assert 0.0 <= batch_asw(X, batch_labels) <= 1.0
        assert 0.0 <= batch_mixing_score(X, batch_labels, n_neighbors=10) <= 1.0


class TestStandardizedBatchMetrics:
    """Tests for standardized kBET / iLISI scores."""

    @staticmethod
    def _build_mixed_and_segregated() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mixed_x = np.array([[i // 2, 0.0] for i in range(12)], dtype=float)
        segregated_x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
                [0.5, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
                [10.2, 0.0],
                [10.3, 0.0],
                [10.4, 0.0],
                [10.5, 0.0],
            ],
        )
        labels = np.array(["A", "B"] * 6)
        segregated_labels = np.array(["A"] * 6 + ["B"] * 6)
        return mixed_x, segregated_x, labels, segregated_labels

    def test_kbet_distinguishes_mixed_from_segregated(self) -> None:
        mixed_x, segregated_x, labels, segregated_labels = self._build_mixed_and_segregated()

        mixed_score = kbet_score(mixed_x, labels, n_neighbors=5, alpha=0.05)
        segregated_score = kbet_score(
            segregated_x,
            segregated_labels,
            n_neighbors=5,
            alpha=0.05,
        )

        assert 0.0 <= mixed_score <= 1.0
        assert 0.0 <= segregated_score <= 1.0
        assert mixed_score > segregated_score
        assert mixed_score > 0.8
        assert segregated_score < 0.2

    def test_ilisi_distinguishes_mixed_from_segregated(self) -> None:
        mixed_x, segregated_x, labels, segregated_labels = self._build_mixed_and_segregated()

        mixed_scaled = ilisi_score(mixed_x, labels, n_neighbors=5, perplexity=4.0)
        mixed_raw = ilisi_score(mixed_x, labels, n_neighbors=5, perplexity=4.0, scale=False)
        segregated_scaled = ilisi_score(
            segregated_x,
            segregated_labels,
            n_neighbors=5,
            perplexity=4.0,
        )
        segregated_raw = ilisi_score(
            segregated_x,
            segregated_labels,
            n_neighbors=5,
            perplexity=4.0,
            scale=False,
        )

        assert 0.0 <= mixed_scaled <= 1.0
        assert 0.0 <= segregated_scaled <= 1.0
        assert 1.0 <= mixed_raw <= 2.0
        assert 1.0 <= segregated_raw <= 2.0
        assert mixed_scaled == pytest.approx(mixed_raw - 1.0)
        assert segregated_scaled == pytest.approx(segregated_raw - 1.0)
        assert mixed_scaled > segregated_scaled
        assert mixed_scaled > 0.55
        assert segregated_scaled < 0.25

    def test_standardized_metrics_validate_inputs(self) -> None:
        X = np.random.randn(20, 5)
        batch_labels = np.repeat(["A", "B"], 10)

        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            kbet_score(X, batch_labels, n_neighbors=0)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            kbet_score(X, batch_labels, alpha=1.0)

        with pytest.raises(ValueError, match="perplexity must be positive"):
            ilisi_score(X, batch_labels, perplexity=0.0)


class TestApproximateLISI:
    """Tests for the shared historical approximate LISI metric."""

    def test_lisi_approx_empty_array(self) -> None:
        X = np.array([]).reshape(0, 0)
        batch_labels = np.array([], dtype=int)
        assert lisi_approx_score(X, batch_labels) == 0.0

    def test_lisi_approx_single_batch_fails_closed(self) -> None:
        X = np.random.randn(20, 5)
        batch_labels = np.zeros(20, dtype=int)
        assert lisi_approx_score(X, batch_labels, n_neighbors=5) == 0.0

    def test_lisi_approx_distinguishes_mixed_from_segregated(self) -> None:
        mixed_x = np.array([[i // 2, 0.0] for i in range(12)], dtype=float)
        segregated_x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
                [0.5, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
                [10.2, 0.0],
                [10.3, 0.0],
                [10.4, 0.0],
                [10.5, 0.0],
            ],
        )
        mixed_labels = np.array(["A", "B"] * 6)
        segregated_labels = np.array(["A"] * 6 + ["B"] * 6)

        mixed_score = lisi_approx_score(mixed_x, mixed_labels, n_neighbors=5)
        segregated_score = lisi_approx_score(
            segregated_x,
            segregated_labels,
            n_neighbors=5,
        )

        assert 1.0 <= mixed_score <= 2.0
        assert 1.0 <= segregated_score <= 2.0
        assert mixed_score > segregated_score


def test_shared_knn_kernel_excludes_self_indices() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )

    _, neighbor_indices = compute_self_excluded_knn(X, n_neighbors=2)

    assert neighbor_indices.shape == (4, 2)
    for row_idx, neighbors in enumerate(neighbor_indices):
        assert row_idx not in set(neighbors.tolist())


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
            kbet_score(X, batch_labels),
            ilisi_score(X, batch_labels),
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
        assert isinstance(kbet_score(X, batch_labels), float)
        assert isinstance(ilisi_score(X, batch_labels), float)

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
        assert batch_mixing_score(X1, batch_labels1) == batch_mixing_score(X2, batch_labels2)
        assert kbet_score(X1, batch_labels1) == kbet_score(X2, batch_labels2)
        assert ilisi_score(X1, batch_labels1) == ilisi_score(X2, batch_labels2)
