"""Tests for accuracy evaluator."""

import numpy as np
import pytest

from scptensor.benchmark.evaluators import (
    AccuracyEvaluator,
    AccuracyResult,
    evaluate_accuracy,
    evaluate_classification_accuracy,
    evaluate_regression_accuracy,
)


@pytest.fixture
def accuracy_evaluator():
    """Create accuracy evaluator instance."""
    return AccuracyEvaluator()


@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = y_true + np.random.randn(10) * 0.1
    return y_true, y_pred


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 2, 2, 0])
    return y_true, y_pred


# ===========================================================================
# Initialization Tests
# ===========================================================================


def test_accuracy_evaluator_initialization():
    """Test evaluator initialization."""
    evaluator = AccuracyEvaluator()
    assert evaluator is not None
    assert evaluator.mask_sensitive is False
    assert evaluator.task_type == "auto"


def test_accuracy_evaluator_with_options():
    """Test evaluator initialization with options."""
    evaluator = AccuracyEvaluator(mask_sensitive=True, task_type="regression")
    assert evaluator.mask_sensitive is True
    assert evaluator.task_type == "regression"


# ===========================================================================
# Regression Tests
# ===========================================================================


def test_evaluate_regression_basic(accuracy_evaluator, sample_regression_data):
    """Test basic regression evaluation."""
    y_true, y_pred = sample_regression_data

    metrics = accuracy_evaluator.evaluate(y_true, y_pred, task_type="regression")

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "correlation" in metrics
    assert "spearman_correlation" in metrics

    # Check reasonable values
    assert metrics["mae"] >= 0
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert not np.isnan(metrics["mae"])
    assert not np.isnan(metrics["mse"])


def test_evaluate_regression_perfect_prediction():
    """Test regression with perfect prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)
    assert metrics["correlation"] == pytest.approx(1.0, abs=1e-10)


def test_evaluate_regression_with_correlation():
    """Test that correlation metrics work correctly."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect linear relationship

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # Perfect correlation despite different scale
    assert metrics["correlation"] == pytest.approx(1.0, abs=1e-10)
    assert metrics["spearman_correlation"] == pytest.approx(1.0, abs=1e-10)


def test_evaluate_regression_method(accuracy_evaluator):
    """Test the evaluate_regression method directly."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = accuracy_evaluator.evaluate_regression(y_true, y_pred)

    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["r2"] > 0.9  # Should be high for close predictions


def test_evaluate_regression_with_nan():
    """Test regression evaluation with NaN values."""
    y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 3.0, 4.1, np.nan])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # Should handle NaN by excluding those values
    assert not np.isnan(metrics["mae"])
    assert metrics["mae"] > 0


# ===========================================================================
# Classification Tests
# ===========================================================================


def test_evaluate_classification_basic(accuracy_evaluator, sample_classification_data):
    """Test basic classification evaluation."""
    y_true, y_pred = sample_classification_data

    metrics = accuracy_evaluator.evaluate(y_true, y_pred, task_type="classification")

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    # Check reasonable values
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_evaluate_classification_perfect():
    """Test classification with perfect prediction."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="classification")

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)


def test_evaluate_classification_binary():
    """Test binary classification."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="classification")

    assert metrics["accuracy"] == pytest.approx(5 / 6, abs=1e-10)
    assert not np.isnan(metrics["f1_score"])


def test_evaluate_classification_multiclass():
    """Test multiclass classification."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="classification")

    assert not np.isnan(metrics["accuracy"])
    assert not np.isnan(metrics["f1_score"])


def test_evaluate_classification_average_parameter():
    """Test classification with different averaging methods."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])

    evaluator = AccuracyEvaluator()

    # Test weighted average
    metrics_weighted = evaluator.evaluate(
        y_true, y_pred, task_type="classification", average="weighted"
    )
    assert "f1_score" in metrics_weighted

    # Test macro average
    metrics_macro = evaluator.evaluate(
        y_true, y_pred, task_type="classification", average="macro"
    )
    assert "f1_score" in metrics_macro

    # Test micro average
    metrics_micro = evaluator.evaluate(
        y_true, y_pred, task_type="classification", average="micro"
    )
    assert "f1_score" in metrics_micro


# ===========================================================================
# Auto Detection Tests
# ===========================================================================


def test_auto_detect_classification():
    """Test automatic detection of classification task."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="auto")

    # Should detect as classification and have classification metrics
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_auto_detect_regression():
    """Test automatic detection of regression task."""
    y_true = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    y_pred = np.array([1.4, 2.8, 3.1, 4.9, 5.0])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="auto")

    # Should detect as regression and have regression metrics
    assert "mae" in metrics
    assert "r2" in metrics


# ===========================================================================
# Mask Handling Tests
# ===========================================================================


def test_evaluate_with_mask():
    """Test evaluation with explicit mask."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 3.1, 4.2, 5.1])
    mask = np.array([0, 0, 1, 0, 0])  # Third value is masked

    evaluator = AccuracyEvaluator(mask_sensitive=True)
    metrics = evaluator.evaluate_with_mask(y_true, y_pred, mask)

    # Should exclude masked value from computation
    assert "mae" in metrics
    assert not np.isnan(metrics["mae"])


def test_mask_sensitive_parameter():
    """Test mask_sensitive parameter."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([0, 0, 2, 0, 0])  # LOD masked value

    evaluator_sensitive = AccuracyEvaluator(mask_sensitive=True)
    evaluator_not_sensitive = AccuracyEvaluator(mask_sensitive=False)

    # With mask sensitivity, different results possible
    metrics_sensitive = evaluator_sensitive.evaluate(
        y_true, y_pred, mask=mask, task_type="regression"
    )
    metrics_not_sensitive = evaluator_not_sensitive.evaluate(
        y_true, y_pred, mask=mask, task_type="regression"
    )

    assert "mae" in metrics_sensitive
    assert "mae" in metrics_not_sensitive


# ===========================================================================
# Detailed Result Tests
# ===========================================================================


def test_get_detailed_result(accuracy_evaluator):
    """Test getting detailed result."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    result = accuracy_evaluator.get_detailed_result(y_true, y_pred, task_type="regression")

    assert isinstance(result, AccuracyResult)
    assert result.task_type == "regression"
    assert result.n_samples == 5
    assert result.n_valid == 5
    assert isinstance(result.metrics, dict)
    assert "mae" in result.metrics


def test_accuracy_result_to_dict():
    """Test AccuracyResult to_dict method."""
    result = AccuracyResult(
        task_type="regression",
        metrics={"mae": 0.1, "r2": 0.95},
        n_samples=100,
        n_valid=95,
    )

    result_dict = result.to_dict()

    assert result_dict["task_type"] == "regression"
    assert result_dict["n_samples"] == 100
    assert result_dict["n_valid"] == 95
    assert result_dict["mae"] == 0.1
    assert result_dict["r2"] == 0.95


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================


def test_empty_arrays():
    """Test evaluation with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)

    # Should return empty result with NaN values
    assert isinstance(metrics, dict)
    assert np.isnan(metrics.get("mae", np.nan))


def test_shape_mismatch():
    """Test that shape mismatch raises an error."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])  # Different shape

    evaluator = AccuracyEvaluator()
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator.evaluate(y_true, y_pred)


def test_multidimensional_input():
    """Test evaluation with multidimensional input."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # Should flatten and compute metrics
    assert "mae" in metrics
    assert not np.isnan(metrics["mae"])


def test_all_nan_values():
    """Test evaluation with all NaN values."""
    y_true = np.array([np.nan, np.nan, np.nan])
    y_pred = np.array([np.nan, np.nan, np.nan])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)

    # Should handle gracefully
    assert isinstance(metrics, dict)


# ===========================================================================
# Convenience Functions Tests
# ===========================================================================


def test_evaluate_accuracy_convenience():
    """Test the evaluate_accuracy convenience function."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = evaluate_accuracy(y_true, y_pred, task_type="regression")

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "r2" in metrics


def test_evaluate_regression_accuracy_convenience():
    """Test the evaluate_regression_accuracy convenience function."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = evaluate_regression_accuracy(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "correlation" in metrics


def test_evaluate_classification_accuracy_convenience():
    """Test the evaluate_classification_accuracy convenience function."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = evaluate_classification_accuracy(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


def test_evaluate_classification_accuracy_with_average():
    """Test classification convenience with different averaging."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics_macro = evaluate_classification_accuracy(y_true, y_pred, average="macro")
    metrics_weighted = evaluate_classification_accuracy(y_true, y_pred, average="weighted")

    assert "f1_score" in metrics_macro
    assert "f1_score" in metrics_weighted


# ===========================================================================
# Metric Calculation Validation
# ===========================================================================


def test_mae_calculation():
    """Test MAE calculation is correct."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # MAE = (|1-2| + |2-3| + |3-4|) / 3 = 3/3 = 1
    assert metrics["mae"] == pytest.approx(1.0)


def test_mse_calculation():
    """Test MSE calculation is correct."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 3/3 = 1
    assert metrics["mse"] == pytest.approx(1.0)


def test_rmse_calculation():
    """Test RMSE is sqrt of MSE."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")

    # RMSE should be sqrt(MSE)
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_accuracy_calculation():
    """Test accuracy calculation is correct."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])  # 5 out of 6 correct

    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, task_type="classification")

    assert metrics["accuracy"] == pytest.approx(5.0 / 6.0)


# ===========================================================================
# Integration with BaseEvaluator
# ===========================================================================


def test_inherits_from_base_evaluator():
    """Test that AccuracyEvaluator is a BaseEvaluator."""
    from scptensor.benchmark.evaluators.accuracy import BaseEvaluator

    evaluator = AccuracyEvaluator()
    assert isinstance(evaluator, BaseEvaluator)
    assert hasattr(evaluator, "evaluate")


# ===========================================================================
# Slot Usage
# ===========================================================================


def test_uses_slots():
    """Test that AccuracyEvaluator uses __slots__."""
    # Should have __slots__ defined
    assert hasattr(AccuracyEvaluator, "__slots__")

    # Verify the expected slot attributes
    assert "mask_sensitive" in AccuracyEvaluator.__slots__
    assert "task_type" in AccuracyEvaluator.__slots__

    # Verify __dict__ is not present (slots classes typically don't have __dict__)
    # unless __dict__ is explicitly in __slots__
    assert not hasattr(AccuracyEvaluator, "__dict__") or "__dict__" not in AccuracyEvaluator.__slots__
