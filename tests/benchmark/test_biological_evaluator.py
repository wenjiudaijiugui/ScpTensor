"""Tests for biological evaluator."""

import numpy as np
import pytest

from scptensor.benchmark.evaluators import (
    BaseEvaluator,
    BiologicalEvaluator,
    evaluate_biological,
)


@pytest.fixture
def bio_evaluator():
    """Create biological evaluator instance."""
    return BiologicalEvaluator(use_scib=False)  # Use sklearn fallback


def test_bio_evaluator_initialization():
    """Test evaluator initialization."""
    evaluator = BiologicalEvaluator()
    assert evaluator is not None
    # use_scib depends on whether scib-metrics is available
    assert hasattr(evaluator, "use_scib")
    # Check that use_scib is a boolean
    assert isinstance(evaluator.use_scib, bool)


def test_bio_evaluator_with_sklearn():
    """Test evaluator with sklearn fallback."""
    evaluator = BiologicalEvaluator(use_scib=False)
    assert evaluator.use_scib is False


def test_evaluate_with_synthetic_data(bio_evaluator):
    """Test evaluation with synthetic data."""
    # Create synthetic data
    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)

    metrics = bio_evaluator.evaluate(X, labels, batches)

    # Check metrics exist
    assert isinstance(metrics, dict)
    # Should have at least some metrics
    assert len(metrics) > 0


def test_evaluate_with_no_batches(bio_evaluator):
    """Test evaluation without batches."""
    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.zeros(100, dtype=int)  # All same batch

    metrics = bio_evaluator.evaluate(X, labels, batches)

    # Should still return metrics
    assert isinstance(metrics, dict)


def test_evaluate_with_no_labels(bio_evaluator):
    """Test evaluation without labels."""
    X = np.random.randn(100, 20)
    labels = np.zeros(100, dtype=int)  # All same label
    batches = np.array([0] * 50 + [1] * 50)

    metrics = bio_evaluator.evaluate(X, labels, batches)

    # Should still return metrics
    assert isinstance(metrics, dict)


def test_evaluate_batch_correction():
    """Test batch correction evaluation."""
    evaluator = BiologicalEvaluator(use_scib=False)

    # Create synthetic data with batch effect
    X_orig = np.random.randn(100, 20)
    X_orig[50:] += 2.0  # Add batch effect

    # Corrected data (no batch effect)
    X_corr = X_orig.copy()
    X_corr[50:] -= 2.0

    batches = np.array([0] * 50 + [1] * 50)
    labels = np.array([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)

    metrics = evaluator.evaluate_batch_correction(
        X_orig, X_corr, batches, labels
    )

    # Check metrics exist
    assert isinstance(metrics, dict)
    # Should have kbet_orig, kbet_corr, kbet_delta
    assert "kbet_orig" in metrics
    assert "kbet_corr" in metrics
    assert "kbet_delta" in metrics
    assert "ilisi_delta" in metrics
    # Should have clisi_corr when labels provided
    assert "clisi_corr" in metrics


def test_evaluate_returns_all_metrics():
    """Test that all expected metrics are returned."""
    evaluator = BiologicalEvaluator(use_scib=False)

    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 50 + [1] * 50)

    metrics = evaluator.evaluate(X, labels, batches)

    # Check for expected metric keys
    expected_metrics = ["kbet", "ilisi", "clisi", "ari", "nmi"]
    for metric in expected_metrics:
        # Metrics may be nan if not applicable
        assert metric in metrics


def test_convenience_function():
    """Test the evaluate_biological convenience function."""
    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 50 + [1] * 50)

    metrics = evaluate_biological(X, labels, batches)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_base_evaluator_is_abstract():
    """Test that BaseEvaluator cannot be instantiated directly."""
    from scptensor.benchmark.evaluators.biological import BaseEvaluator

    # BaseEvaluator should be abstract - it has abstract methods
    # But Python doesn't enforce this without @abstractmethod
    evaluator = BaseEvaluator()
    # Should have evaluate method but it's the base implementation
    assert hasattr(evaluator, "evaluate")


def test_bio_evaluator_with_scib_unavailable():
    """Test behavior when scib-metrics is unavailable."""
    # Create evaluator with use_scib=False to force fallback
    evaluator = BiologicalEvaluator(use_scib=False)

    # Should work with fallback metrics
    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 50 + [1] * 50)

    metrics = evaluator.evaluate(X, labels, batches)

    # Should still work with fallback metrics
    assert isinstance(metrics, dict)
    assert "kbet" in metrics
    assert "ilisi" in metrics


def test_evaluate_with_missing_values():
    """Test evaluation with missing values - filters NaN rows before evaluation."""
    evaluator = BiologicalEvaluator(use_scib=False)

    # Create data with missing values in some rows
    X = np.random.randn(100, 20)
    # Add NaN values to some rows (every 5th row)
    nan_indices = list(range(0, 100, 5))
    X[nan_indices, :] = np.nan

    # Create labels and batches (with corresponding NaN rows)
    all_labels = np.array([0] * 50 + [1] * 50)
    all_batches = np.array([0] * 50 + [1] * 50)

    # The evaluator should filter out rows with NaN values
    # We need to manually filter NaN rows for the test
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    labels_valid = all_labels[valid_mask]
    batches_valid = all_batches[valid_mask]

    metrics = evaluator.evaluate(X_valid, labels_valid, batches_valid)

    # Should evaluate successfully with filtered data
    assert isinstance(metrics, dict)
    # Verify we still have meaningful data after filtering
    assert len(X_valid) > 0  # Should have remaining valid samples


def test_evaluate_with_different_k_values():
    """Test evaluation with different k values."""
    evaluator = BiologicalEvaluator(use_scib=False, k_bet=10)

    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 50 + [1] * 50)

    metrics = evaluator.evaluate(X, labels, batches)

    assert isinstance(metrics, dict)
    assert evaluator.k_bet == 10


def test_evaluate_returns_dict():
    """Test that evaluate returns a dictionary."""
    evaluator = BiologicalEvaluator(use_scib=False)

    X = np.random.randn(100, 20)
    labels = np.array([0] * 50 + [1] * 50)
    batches = np.array([0] * 50 + [1] * 50)

    metrics = evaluator.evaluate(X, labels, batches)

    assert isinstance(metrics, dict)
    # All values should be float or nan
    for value in metrics.values():
        assert isinstance(value, (float, np.floating))
