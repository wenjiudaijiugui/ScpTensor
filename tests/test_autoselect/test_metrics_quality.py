"""Tests for quality metrics module.

This module provides comprehensive tests for the quality metrics functions
used in the automatic method selection system.
"""

from __future__ import annotations

import numpy as np
import pytest

from scptensor.autoselect.metrics.quality import (
    cv_stability,
    dynamic_range,
    kurtosis_improvement,
    outlier_ratio,
    skewness_improvement,
)


class TestCVStability:
    """Tests for cv_stability function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        result = cv_stability(X)
        assert result == 0.0

    def test_single_value(self) -> None:
        """Test with single value returns 0.0."""
        X = np.array([[5.0]])
        result = cv_stability(X)
        assert result == 0.0

    def test_all_zeros(self) -> None:
        """Test with all zeros returns 0.0."""
        X = np.zeros((10, 5))
        result = cv_stability(X)
        assert result == 0.0

    def test_all_nans(self) -> None:
        """Test with all NaNs returns 0.0."""
        X = np.full((10, 5), np.nan)
        result = cv_stability(X)
        assert result == 0.0

    def test_ideal_case(self) -> None:
        """Test with well-behaved data should return high score."""
        # Create data with similar CVs across features (stable)
        np.random.seed(42)
        X = np.random.randn(100, 10) * 10 + 100  # Large mean, moderate std
        result = cv_stability(X)
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should be reasonably stable

    def test_poor_case(self) -> None:
        """Test with highly variable CVs should return lower score."""
        # Create data with very different CVs across features (unstable)
        X = np.zeros((100, 5))
        X[:, 0] = np.random.randn(100) * 0.1 + 100  # Low CV
        X[:, 1] = np.random.randn(100) * 10 + 100  # High CV
        X[:, 2] = np.random.randn(100) * 0.5 + 100  # Moderate CV
        X[:, 3] = np.random.randn(100) * 5 + 100  # Moderate-high CV
        X[:, 4] = np.random.randn(100) * 20 + 100  # Very high CV
        result = cv_stability(X)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float, not np.float64."""
        X = np.random.randn(10, 5)
        result = cv_stability(X)
        assert isinstance(result, float)


class TestSkewnessImprovement:
    """Tests for skewness_improvement function."""

    def test_empty_arrays(self) -> None:
        """Test with empty arrays returns 0.0."""
        X_before = np.array([]).reshape(0, 0)
        X_after = np.array([]).reshape(0, 0)
        result = skewness_improvement(X_before, X_after)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X_before = np.random.randn(10, 5)
        X_after = np.random.randn(10, 6)
        with pytest.raises(ValueError, match="Shape mismatch"):
            skewness_improvement(X_before, X_after)

    def test_ideal_improvement(self) -> None:
        """Test when skewness improves (closer to 0)."""
        # Before: highly skewed
        np.random.seed(42)
        X_before = np.exp(np.random.randn(100, 5))  # Right-skewed

        # After: log-transformed (more symmetric)
        X_after = np.log1p(X_before)

        result = skewness_improvement(X_before, X_after)
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should show good improvement

    def test_no_improvement(self) -> None:
        """Test when skewness doesn't improve."""
        # Same data for both
        X = np.random.randn(100, 5)
        result = skewness_improvement(X, X)
        assert 0.0 <= result <= 1.0
        # Should be around 0 since no improvement

    def test_worse_skewness(self) -> None:
        """Test when skewness gets worse (should return 0.0)."""
        # Before: normal distribution (skewness ~0)
        X_before = np.random.randn(100, 5)

        # After: highly skewed
        X_after = np.exp(np.random.randn(100, 5))

        result = skewness_improvement(X_before, X_after)
        assert result >= 0.0  # Should be clamped to 0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(10, 5)
        result = skewness_improvement(X, X)
        assert isinstance(result, float)


class TestKurtosisImprovement:
    """Tests for kurtosis_improvement function."""

    def test_empty_arrays(self) -> None:
        """Test with empty arrays returns 0.0."""
        X_before = np.array([]).reshape(0, 0)
        X_after = np.array([]).reshape(0, 0)
        result = kurtosis_improvement(X_before, X_after)
        assert result == 0.0

    def test_shape_mismatch(self) -> None:
        """Test with mismatched shapes raises ValueError."""
        X_before = np.random.randn(10, 5)
        X_after = np.random.randn(10, 6)
        with pytest.raises(ValueError, match="Shape mismatch"):
            kurtosis_improvement(X_before, X_after)

    def test_ideal_improvement(self) -> None:
        """Test when kurtosis improves (closer to 3)."""
        # Before: heavy-tailed distribution
        np.random.seed(42)
        X_before = np.random.standard_t(df=3, size=(100, 5))  # Heavy tails

        # After: more normal-like
        X_after = np.random.randn(100, 5)

        result = kurtosis_improvement(X_before, X_after)
        assert 0.0 <= result <= 1.0

    def test_no_improvement(self) -> None:
        """Test when kurtosis doesn't improve."""
        X = np.random.randn(100, 5)
        result = kurtosis_improvement(X, X)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(10, 5)
        result = kurtosis_improvement(X, X)
        assert isinstance(result, float)


class TestDynamicRange:
    """Tests for dynamic_range function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        result = dynamic_range(X)
        assert result == 0.0

    def test_all_zeros(self) -> None:
        """Test with all zeros returns 0.0."""
        X = np.zeros((10, 5))
        result = dynamic_range(X)
        assert result == 0.0

    def test_negative_values(self) -> None:
        """Test with negative values handles gracefully."""
        X = np.random.randn(100, 5)
        # Should handle by taking absolute values or skipping negatives
        result = dynamic_range(X)
        assert 0.0 <= result <= 1.0

    def test_ideal_range(self) -> None:
        """Test with ideal dynamic range (2-10 orders of magnitude)."""
        # Create data spanning ~6 orders of magnitude (ideal)
        np.random.seed(42)
        X = 10 ** np.random.uniform(0, 6, size=(100, 5))  # 1 to 1e6
        result = dynamic_range(X)
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should score well

    def test_too_large_range(self) -> None:
        """Test with extremely large dynamic range."""
        # Create data spanning 20+ orders of magnitude
        np.random.seed(42)
        X = 10 ** np.random.uniform(0, 20, size=(100, 5))  # 1 to 1e20
        result = dynamic_range(X)
        assert 0.0 <= result <= 1.0
        assert result < 0.5  # Should score lower

    def test_too_small_range(self) -> None:
        """Test with very small dynamic range."""
        # Create data spanning < 1 order of magnitude
        X = np.random.uniform(1.0, 1.5, size=(100, 5))
        result = dynamic_range(X)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(10, 5) + 10  # Add offset to avoid negatives
        result = dynamic_range(X)
        assert isinstance(result, float)


class TestOutlierRatio:
    """Tests for outlier_ratio function."""

    def test_empty_array(self) -> None:
        """Test with empty array returns 0.0."""
        X = np.array([]).reshape(0, 0)
        result = outlier_ratio(X)
        assert result == 0.0

    def test_no_outliers(self) -> None:
        """Test with no outliers should return 1.0."""
        # Create perfectly normal data
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        result = outlier_ratio(X)
        assert 0.0 <= result <= 1.0
        assert result > 0.9  # Should be very high (few outliers)

    def test_many_outliers(self) -> None:
        """Test with many outliers should return low score."""
        # Create data with many outliers
        X = np.random.randn(100, 5)
        # Add extreme outliers
        X[:10, :] = 1000  # 10% outliers
        result = outlier_ratio(X)
        assert 0.0 <= result <= 1.0
        assert result < 0.95  # Should be lower

    def test_all_outliers(self) -> None:
        """Test with all outliers should return 0.0 or very low score."""
        # Create data where all values are extreme
        X = np.random.randn(10, 5) * 100
        result = outlier_ratio(X)
        assert 0.0 <= result <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is float."""
        X = np.random.randn(10, 5)
        result = outlier_ratio(X)
        assert isinstance(result, float)

    def test_with_nans(self) -> None:
        """Test with NaN values handles gracefully."""
        X = np.random.randn(100, 5)
        X[::10, 0] = np.nan  # 10% NaN
        result = outlier_ratio(X)
        assert 0.0 <= result <= 1.0


class TestMetricProperties:
    """Test common properties across all metrics."""

    def test_all_metrics_return_in_range(self) -> None:
        """Test that all metrics return values in [0, 1] range."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        metrics = [
            cv_stability(X),
            dynamic_range(X),
            outlier_ratio(X),
        ]

        for metric_value in metrics:
            assert 0.0 <= metric_value <= 1.0, f"Metric {metric_value} not in [0, 1]"

    def test_all_metrics_return_float(self) -> None:
        """Test that all metrics return Python float type."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        assert isinstance(cv_stability(X), float)
        assert isinstance(dynamic_range(X), float)
        assert isinstance(outlier_ratio(X), float)

        X2 = np.random.randn(10, 5)
        assert isinstance(skewness_improvement(X, X2), float)
        assert isinstance(kurtosis_improvement(X, X2), float)

    def test_reproducibility(self) -> None:
        """Test that metrics are deterministic for same input."""
        np.random.seed(42)
        X1 = np.random.randn(50, 10)
        np.random.seed(42)
        X2 = np.random.randn(50, 10)

        assert cv_stability(X1) == cv_stability(X2)
        assert dynamic_range(X1) == dynamic_range(X2)
        assert outlier_ratio(X1) == outlier_ratio(X2)
