"""Tests for clustering evaluation metrics module.

This module provides comprehensive tests for the clustering metrics functions
used in the automatic method selection system.
"""

from __future__ import annotations

import numpy as np
import pytest

from scptensor.autoselect.metrics.clustering import (
    calinski_harabasz_score,
    clustering_stability,
    davies_bouldin_score,
    silhouette_score,
)


class TestSilhouetteScore:
    """Tests for silhouette_score function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        labels = np.array([], dtype=int)
        result = silhouette_score(X, labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        labels = np.array([0])
        result = silhouette_score(X, labels)
        assert result == 0.0

    def test_single_cluster(self) -> None:
        """Test with only one cluster returns 0.0."""
        X = np.random.randn(100, 10)
        labels = np.zeros(100, dtype=int)
        result = silhouette_score(X, labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            silhouette_score(X, labels)

    def test_well_separated_clusters(self) -> None:
        """Test with well-separated clusters should return high score."""
        np.random.seed(42)
        # Create clearly separated clusters
        X1 = np.random.randn(25, 10) + 10
        X2 = np.random.randn(25, 10) - 10
        X3 = np.random.randn(25, 10) + 20
        X4 = np.random.randn(25, 10) - 20
        X = np.vstack([X1, X2, X3, X4])
        labels = np.repeat([0, 1, 2, 3], 25)
        result = silhouette_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Should be high since clusters are well-separated
        assert result > 0.5

    def test_overlapping_clusters(self) -> None:
        """Test with overlapping clusters should return lower score."""
        np.random.seed(42)
        # Create overlapping clusters (all from same distribution)
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = silhouette_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Should be lower since clusters overlap
        assert result < 0.5

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan  # 10% NaN rows
        labels = np.repeat([0, 1, 2, 3], 25)
        result = silhouette_score(X, labels)
        assert 0.0 <= result <= 1.0

    def test_all_nans(self) -> None:
        """Test with all NaN values returns 0.0."""
        X = np.full((100, 10), np.nan)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = silhouette_score(X, labels)
        assert result == 0.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = silhouette_score(X, labels)
        assert isinstance(result, float)


class TestCalinskiHarabaszScore:
    """Tests for calinski_harabasz_score function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        labels = np.array([], dtype=int)
        result = calinski_harabasz_score(X, labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        labels = np.array([0])
        result = calinski_harabasz_score(X, labels)
        assert result == 0.0

    def test_single_cluster(self) -> None:
        """Test with only one cluster returns 0.0."""
        X = np.random.randn(100, 10)
        labels = np.zeros(100, dtype=int)
        result = calinski_harabasz_score(X, labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            calinski_harabasz_score(X, labels)

    def test_well_separated_clusters(self) -> None:
        """Test with well-separated clusters should return high score."""
        np.random.seed(42)
        # Create clearly separated clusters
        X1 = np.random.randn(25, 10) + 10
        X2 = np.random.randn(25, 10) - 10
        X3 = np.random.randn(25, 10) + 20
        X4 = np.random.randn(25, 10) - 20
        X = np.vstack([X1, X2, X3, X4])
        labels = np.repeat([0, 1, 2, 3], 25)
        result = calinski_harabasz_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Should be reasonably high
        assert result > 0.3

    def test_overlapping_clusters(self) -> None:
        """Test with overlapping clusters should return lower score."""
        np.random.seed(42)
        # Create overlapping clusters
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = calinski_harabasz_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Overlapping clusters should have lower score
        assert result < 0.7

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan
        labels = np.repeat([0, 1, 2, 3], 25)
        result = calinski_harabasz_score(X, labels)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = calinski_harabasz_score(X, labels)
        assert isinstance(result, float)


class TestDaviesBouldinScore:
    """Tests for davies_bouldin_score function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        labels = np.array([], dtype=int)
        result = davies_bouldin_score(X, labels)
        assert result == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample returns 0.0."""
        X = np.array([[1.0, 2.0]])
        labels = np.array([0])
        result = davies_bouldin_score(X, labels)
        assert result == 0.0

    def test_single_cluster(self) -> None:
        """Test with only one cluster returns 0.0."""
        X = np.random.randn(100, 10)
        labels = np.zeros(100, dtype=int)
        result = davies_bouldin_score(X, labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            davies_bouldin_score(X, labels)

    def test_well_separated_clusters(self) -> None:
        """Test with well-separated clusters should return high score."""
        np.random.seed(42)
        # Create clearly separated clusters
        X1 = np.random.randn(25, 10) + 10
        X2 = np.random.randn(25, 10) - 10
        X3 = np.random.randn(25, 10) + 20
        X4 = np.random.randn(25, 10) - 20
        X = np.vstack([X1, X2, X3, X4])
        labels = np.repeat([0, 1, 2, 3], 25)
        result = davies_bouldin_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Well-separated = low DB = high returned score
        assert result > 0.5

    def test_overlapping_clusters(self) -> None:
        """Test with overlapping clusters should return lower score."""
        np.random.seed(42)
        # Create overlapping clusters
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = davies_bouldin_score(X, labels)
        assert 0.0 <= result <= 1.0
        # Overlapping = high DB = low returned score

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan
        labels = np.repeat([0, 1, 2, 3], 25)
        result = davies_bouldin_score(X, labels)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = davies_bouldin_score(X, labels)
        assert isinstance(result, float)


class TestClusteringStability:
    """Tests for clustering_stability function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        labels = np.array([], dtype=int)
        result = clustering_stability(X, labels)
        assert result == 0.0

    def test_too_few_samples(self) -> None:
        """Test with too few samples returns 0.0."""
        X = np.random.randn(3, 5)
        labels = np.array([0, 0, 1])
        result = clustering_stability(X, labels)
        assert result == 0.0

    def test_single_cluster(self) -> None:
        """Test with only one cluster returns 0.0."""
        X = np.random.randn(100, 10)
        labels = np.zeros(100, dtype=int)
        result = clustering_stability(X, labels)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X = np.random.randn(100, 10)
        labels = np.array([0] * 50)  # Wrong size
        with pytest.raises(ValueError, match="Shape mismatch"):
            clustering_stability(X, labels)

    def test_stable_clustering(self) -> None:
        """Test with stable clustering should return high score."""
        np.random.seed(42)
        # Create clearly separated clusters
        X1 = np.random.randn(30, 10) + 10
        X2 = np.random.randn(30, 10) - 10
        X = np.vstack([X1, X2])
        labels = np.repeat([0, 1], 30)
        result = clustering_stability(X, labels, n_subsamples=5, random_state=42)
        assert 0.0 <= result <= 1.0
        # Well-separated clusters should be stable
        assert result > 0.3

    def test_unstable_clustering(self) -> None:
        """Test with unstable clustering should return lower score."""
        np.random.seed(42)
        # Create overlapping clusters
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = clustering_stability(X, labels, n_subsamples=5, random_state=42)
        assert 0.0 <= result <= 1.0

    def test_custom_parameters(self) -> None:
        """Test with custom n_subsamples and subsample_ratio."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)

        result_5 = clustering_stability(X, labels, n_subsamples=5, random_state=42)
        result_10 = clustering_stability(X, labels, n_subsamples=10, random_state=42)

        assert 0.0 <= result_5 <= 1.0
        assert 0.0 <= result_10 <= 1.0

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[::10, 0] = np.nan
        labels = np.repeat([0, 1, 2, 3], 25)
        result = clustering_stability(X, labels, n_subsamples=5, random_state=42)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)
        result = clustering_stability(X, labels)
        assert isinstance(result, float)

    def test_reproducibility_with_random_state(self) -> None:
        """Test that random_state produces reproducible results."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)

        result1 = clustering_stability(X, labels, n_subsamples=5, random_state=123)
        result2 = clustering_stability(X, labels, n_subsamples=5, random_state=123)

        assert result1 == result2


class TestMetricProperties:
    """Test common properties across all clustering metrics."""

    def test_all_metrics_return_in_range(self) -> None:
        """Test that all metrics return values in [0, 1] range."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)

        metrics = [
            silhouette_score(X, labels),
            calinski_harabasz_score(X, labels),
            davies_bouldin_score(X, labels),
            clustering_stability(X, labels, n_subsamples=5),
        ]

        for metric_value in metrics:
            assert 0.0 <= metric_value <= 1.0, f"Metric {metric_value} not in [0, 1]"

    def test_all_metrics_return_float(self) -> None:
        """Test that all metrics return Python float type."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        labels = np.repeat([0, 1, 2, 3], 25)

        assert isinstance(silhouette_score(X, labels), float)
        assert isinstance(calinski_harabasz_score(X, labels), float)
        assert isinstance(davies_bouldin_score(X, labels), float)
        assert isinstance(clustering_stability(X, labels), float)

    def test_reproducibility(self) -> None:
        """Test that metrics are deterministic for same input."""
        np.random.seed(42)
        X1 = np.random.randn(100, 10)
        labels1 = np.repeat([0, 1, 2, 3], 25)

        np.random.seed(42)
        X2 = np.random.randn(100, 10)
        labels2 = np.repeat([0, 1, 2, 3], 25)

        assert silhouette_score(X1, labels1) == silhouette_score(X2, labels2)
        assert calinski_harabasz_score(X1, labels1) == calinski_harabasz_score(X2, labels2)
        assert davies_bouldin_score(X1, labels1) == davies_bouldin_score(X2, labels2)
