"""Accuracy evaluator for algorithm quality assessment.

This module provides metrics for evaluating the accuracy of analysis methods,
including regression metrics (MAE, MSE, RMSE, R2, correlation) and classification
metrics (accuracy, precision, recall, F1 score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

if TYPE_CHECKING:
    pass

# Import BaseEvaluator from biological module
from .biological import BaseEvaluator

# =============================================================================
# Type Aliases
# =============================================================================

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass(slots=True)
class AccuracyResult:
    """Result of accuracy evaluation.

    Attributes
    ----------
    task_type : str
        Type of task: "regression" or "classification".
    metrics : dict[str, float]
        Computed metric values.
    n_samples : int
        Number of samples evaluated.
    n_valid : int
        Number of valid (non-NaN) samples.
    """

    task_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_valid: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        """Convert result to a dictionary.

        Returns
        -------
        dict[str, float | str | int]
            Dictionary representation of the result.
        """
        result: dict[str, float | str | int] = {
            "task_type": self.task_type,
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
        }
        result.update(self.metrics)
        return result


# =============================================================================
# Accuracy Evaluator
# =============================================================================


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator for accuracy metrics of algorithm results.

    This evaluator computes accuracy metrics comparing predicted values
    against ground truth values. It supports both regression and
    classification tasks.

    Regression Metrics
    ------------------
    - mae: Mean Absolute Error. Lower is better.
    - mse: Mean Squared Error. Lower is better.
    - rmse: Root Mean Squared Error. Lower is better.
    - r2: R-squared (coefficient of determination). Higher is better.
    - correlation: Pearson correlation coefficient. Higher is better.
    - spearman_correlation: Spearman rank correlation. Higher is better.

    Classification Metrics
    ----------------------
    - accuracy: Classification accuracy. Higher is better.
    - precision: Precision score (weighted average). Higher is better.
    - recall: Recall score (weighted average). Higher is better.
    - f1_score: F1 score (weighted average). Higher is better.

    Parameters
    ----------
    mask_sensitive : bool, default=False
        Whether to handle masked values (indicated by NaN or specific codes).
        When True, masked values are excluded from computation.
    task_type : {"auto", "regression", "classification"}, default="auto"
        Type of evaluation task. "auto" detects based on data characteristics.

    Attributes
    ----------
    mask_sensitive : bool
        Whether masked value handling is enabled.
    task_type : str
        The configured task type.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import AccuracyEvaluator
    >>>
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    >>>
    >>> evaluator = AccuracyEvaluator()
    >>> metrics = evaluator.evaluate(y_true, y_pred, task_type="regression")
    >>> print(f"MAE: {metrics['mae']:.3f}")
    >>> print(f"R2: {metrics['r2']:.3f}")
    """

    __slots__ = ("mask_sensitive", "task_type")

    # Mask codes for ScpTensor data structures
    _VALID_MASK = 0
    _MBR_MASK = 1
    _LOD_MASK = 2
    _FILTERED_MASK = 3
    _IMPUTED_MASK = 5

    def __init__(
        self,
        mask_sensitive: bool = False,
        task_type: str = "auto",
    ) -> None:
        """Initialize the accuracy evaluator.

        Parameters
        ----------
        mask_sensitive : bool, default=False
            Whether to handle masked values.
        task_type : {"auto", "regression", "classification"}, default="auto"
            Type of evaluation task.
        """
        self.mask_sensitive = mask_sensitive
        self.task_type = task_type

    def evaluate(
        self,
        y_true: ArrayFloat | ArrayInt,
        y_pred: ArrayFloat | ArrayInt,
        task_type: str = "auto",
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate accuracy metrics on the given predictions.

        Parameters
        ----------
        y_true : ArrayFloat | ArrayInt
            Ground truth values of shape (n_samples,) or (n_samples, n_features).
        y_pred : ArrayFloat | ArrayInt
            Predicted values, same shape as y_true.
        task_type : {"auto", "regression", "classification"}, default="auto"
            Type of evaluation. "auto" detects based on data.
        **kwargs
            Additional parameters including:
            - mask: Optional mask array of same shape as data.
            - average: Averaging method for classification (default="weighted").

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.

        Raises
        ------
        ValueError
            If y_true and y_pred have incompatible shapes.
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # Validate shapes
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true={y_true_arr.shape}, "
                f"y_pred={y_pred_arr.shape}"
            )

        # Handle mask if provided
        mask = kwargs.get("mask")
        if mask is not None and self.mask_sensitive:
            mask_arr = np.asarray(mask)
            valid_mask = self._get_valid_mask(mask_arr)
            y_true_arr = y_true_arr[valid_mask]
            y_pred_arr = y_pred_arr[valid_mask]

        # Flatten if needed
        y_true_flat = y_true_arr.ravel()
        y_pred_flat = y_pred_arr.ravel()

        # Handle NaN values
        valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_flat = y_true_flat[valid_idx]
        y_pred_flat = y_pred_flat[valid_idx]

        if len(y_true_flat) == 0:
            return self._empty_result()

        # Detect task type if auto
        if task_type == "auto":
            task_type = self._detect_task_type(
                y_true_flat, y_pred_flat, kwargs.get("task_type", self.task_type)
            )

        # Route to appropriate evaluation
        if task_type == "classification":
            return self.evaluate_classification(y_true_flat, y_pred_flat, **kwargs)
        return self.evaluate_regression(y_true_flat, y_pred_flat)

    def evaluate_regression(
        self, y_true: ArrayFloat, y_pred: ArrayFloat
    ) -> dict[str, float]:
        """Evaluate regression accuracy metrics.

        Parameters
        ----------
        y_true : ArrayFloat
            Ground truth values.
        y_pred : ArrayFloat
            Predicted values.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "mae": Mean Absolute Error
            - "mse": Mean Squared Error
            - "rmse": Root Mean Squared Error
            - "r2": R-squared score
            - "correlation": Pearson correlation
            - "spearman_correlation": Spearman correlation
        """
        result: dict[str, float] = {}

        try:
            result["mae"] = float(mean_absolute_error(y_true, y_pred))
        except Exception:
            result["mae"] = np.nan

        try:
            result["mse"] = float(mean_squared_error(y_true, y_pred))
        except Exception:
            result["mse"] = np.nan

        try:
            result["rmse"] = float(np.sqrt(result["mse"]))
        except Exception:
            result["rmse"] = np.nan

        try:
            result["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            result["r2"] = np.nan

        try:
            corr, _ = pearsonr(y_true, y_pred)
            result["correlation"] = float(corr)
        except Exception:
            result["correlation"] = np.nan

        try:
            corr, _ = spearmanr(y_true, y_pred)
            result["spearman_correlation"] = float(corr)
        except Exception:
            result["spearman_correlation"] = np.nan

        return result

    def evaluate_classification(
        self,
        y_true: ArrayInt,
        y_pred: ArrayInt,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate classification accuracy metrics.

        Parameters
        ----------
        y_true : ArrayInt
            Ground truth class labels.
        y_pred : ArrayInt
            Predicted class labels.
        **kwargs
            Additional parameters including:
            - average: Averaging method for multiclass (default="weighted").

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "accuracy": Classification accuracy
            - "precision": Weighted precision score
            - "recall": Weighted recall score
            - "f1_score": Weighted F1 score
        """
        average = kwargs.get("average", "weighted")
        result: dict[str, float] = {}

        # Convert to integer for classification
        y_true_int = y_true.astype(int)
        y_pred_int = y_pred.astype(int)

        try:
            result["accuracy"] = float(accuracy_score(y_true_int, y_pred_int))
        except Exception:
            result["accuracy"] = np.nan

        try:
            unique_labels = np.unique(np.concatenate([y_true_int, y_pred_int]))
            if len(unique_labels) > 2:
                avg = average if average in ["micro", "macro", "weighted"] else "weighted"
            else:
                avg = "binary"

            result["precision"] = float(
                precision_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
        except Exception:
            result["precision"] = np.nan

        try:
            unique_labels = np.unique(np.concatenate([y_true_int, y_pred_int]))
            if len(unique_labels) > 2:
                avg = average if average in ["micro", "macro", "weighted"] else "weighted"
            else:
                avg = "binary"

            result["recall"] = float(
                recall_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
        except Exception:
            result["recall"] = np.nan

        try:
            unique_labels = np.unique(np.concatenate([y_true_int, y_pred_int]))
            if len(unique_labels) > 2:
                avg = average if average in ["micro", "macro", "weighted"] else "weighted"
            else:
                avg = "binary"

            result["f1_score"] = float(
                f1_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
        except Exception:
            result["f1_score"] = np.nan

        return result

    def evaluate_with_mask(
        self,
        y_true: ArrayFloat,
        y_pred: ArrayFloat,
        mask: ArrayInt,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate accuracy with explicit mask handling.

        Parameters
        ----------
        y_true : ArrayFloat
            Ground truth values.
        y_pred : ArrayFloat
            Predicted values.
        mask : ArrayInt
            Mask array where 0 indicates valid values.
        **kwargs
            Additional parameters passed to evaluate().

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.
        """
        return self.evaluate(
            y_true, y_pred, mask=mask, mask_sensitive=True, **kwargs
        )

    def get_detailed_result(
        self,
        y_true: ArrayFloat | ArrayInt,
        y_pred: ArrayFloat | ArrayInt,
        **kwargs,
    ) -> AccuracyResult:
        """Get detailed accuracy evaluation result.

        Parameters
        ----------
        y_true : ArrayFloat | ArrayInt
            Ground truth values.
        y_pred : ArrayFloat | ArrayInt
            Predicted values.
        **kwargs
            Additional parameters passed to evaluate().

        Returns
        -------
        AccuracyResult
            Detailed result with metrics and metadata.
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        n_samples = y_true_arr.size
        metrics = self.evaluate(y_true_arr, y_pred_arr, **kwargs)

        # Count valid (non-NaN) samples
        valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
        n_valid = int(np.sum(valid_mask))

        return AccuracyResult(
            task_type=kwargs.get("task_type", self.task_type),
            metrics=metrics,
            n_samples=n_samples,
            n_valid=n_valid,
        )

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_valid_mask(self, mask: ArrayInt) -> NDArray[np.bool_]:
        """Get boolean mask for valid (non-masked) values.

        Parameters
        ----------
        mask : ArrayInt
            Mask array where specific codes indicate masked values.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array where True indicates valid values.
        """
        mask_arr = np.asarray(mask)
        return mask_arr == self._VALID_MASK

    def _detect_task_type(
        self,
        y_true: ArrayFloat,
        y_pred: ArrayFloat,
        fallback: str = "auto",
    ) -> str:
        """Detect whether the task is regression or classification.

        Parameters
        ----------
        y_true : ArrayFloat
            Ground truth values.
        y_pred : ArrayFloat
            Predicted values.
        fallback : str, default="auto"
            Fallback task type if detection fails.

        Returns
        -------
        str
            Detected task type: "regression" or "classification".
        """
        # Check if values are all integers
        if y_true.size == 0:
            return fallback if fallback != "auto" else "regression"

        y_true_unique = np.unique(y_true)
        y_pred_unique = np.unique(y_pred)

        # If few unique values and they are integers, likely classification
        n_unique = len(y_true_unique)
        is_int_true = np.allclose(y_true_unique, y_true_unique.astype(int))
        is_int_pred = np.allclose(y_pred_unique, y_pred_unique.astype(int))

        if n_unique <= 20 and is_int_true and is_int_pred:
            return "classification"

        return fallback if fallback != "auto" else "regression"

    @staticmethod
    def _empty_result() -> dict[str, float]:
        """Return empty result dictionary with NaN values.

        Returns
        -------
        dict[str, float]
            Dictionary with NaN values for all metrics.
        """
        return {
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "correlation": np.nan,
            "spearman_correlation": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_accuracy(
    y_true: ArrayFloat | ArrayInt,
    y_pred: ArrayFloat | ArrayInt,
    task_type: str = "auto",
    mask_sensitive: bool = False,
    **kwargs,
) -> dict[str, float]:
    """Convenience function to evaluate accuracy metrics.

    Parameters
    ----------
    y_true : ArrayFloat | ArrayInt
        Ground truth values.
    y_pred : ArrayFloat | ArrayInt
        Predicted values.
    task_type : {"auto", "regression", "classification"}, default="auto"
        Type of evaluation task.
    mask_sensitive : bool, default=False
        Whether to handle masked values.
    **kwargs
        Additional parameters including:
        - mask: Optional mask array.
        - average: Averaging method for classification.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators.accuracy import evaluate_accuracy
    >>>
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    >>>
    >>> metrics = evaluate_accuracy(y_true, y_pred, task_type="regression")
    >>> print(f"MAE: {metrics['mae']:.3f}")
    >>> print(f"R2: {metrics['r2']:.3f}")
    """
    evaluator = AccuracyEvaluator(mask_sensitive=mask_sensitive)
    return evaluator.evaluate(y_true, y_pred, task_type=task_type, **kwargs)


def evaluate_regression_accuracy(
    y_true: ArrayFloat,
    y_pred: ArrayFloat,
) -> dict[str, float]:
    """Convenience function to evaluate regression metrics.

    Parameters
    ----------
    y_true : ArrayFloat
        Ground truth values.
    y_pred : ArrayFloat
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "mae": Mean Absolute Error
        - "mse": Mean Squared Error
        - "rmse": Root Mean Squared Error
        - "r2": R-squared score
        - "correlation": Pearson correlation
        - "spearman_correlation": Spearman correlation

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators.accuracy import evaluate_regression_accuracy
    >>>
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    >>>
    >>> metrics = evaluate_regression_accuracy(y_true, y_pred)
    """
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate_regression(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    )


def evaluate_classification_accuracy(
    y_true: ArrayInt,
    y_pred: ArrayInt,
    average: str = "weighted",
) -> dict[str, float]:
    """Convenience function to evaluate classification metrics.

    Parameters
    ----------
    y_true : ArrayInt
        Ground truth class labels.
    y_pred : ArrayInt
        Predicted class labels.
    average : str, default="weighted"
        Averaging method for multiclass.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "accuracy": Classification accuracy
        - "precision": Precision score
        - "recall": Recall score
        - "f1_score": F1 score

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators.accuracy import evaluate_classification_accuracy
    >>>
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 1, 0, 2, 2])
    >>>
    >>> metrics = evaluate_classification_accuracy(y_true, y_pred)
    """
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate_classification(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), average=average
    )


__all__ = [
    "BaseEvaluator",
    "AccuracyEvaluator",
    "AccuracyResult",
    "evaluate_accuracy",
    "evaluate_regression_accuracy",
    "evaluate_classification_accuracy",
]
