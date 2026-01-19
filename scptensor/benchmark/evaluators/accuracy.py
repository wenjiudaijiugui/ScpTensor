"""Accuracy evaluator for algorithm quality assessment."""

from __future__ import annotations

from dataclasses import dataclass

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

from .biological import BaseEvaluator

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


@dataclass(slots=True)
class AccuracyResult:
    """Result of accuracy evaluation."""
    task_type: str
    metrics: dict[str, float]
    n_samples: int = 0
    n_valid: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        return {"task_type": self.task_type, "n_samples": self.n_samples,
                "n_valid": self.n_valid, **self.metrics}


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator for accuracy metrics of algorithm results."""

    __slots__ = ("mask_sensitive", "task_type")
    _VALID_MASK = 0

    def __init__(self, mask_sensitive: bool = False, task_type: str = "auto") -> None:
        self.mask_sensitive = mask_sensitive
        self.task_type = task_type

    def evaluate(self, y_true: ArrayFloat | ArrayInt, y_pred: ArrayFloat | ArrayInt,
                 task_type: str = "auto", **kwargs) -> dict[str, float]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}")

        mask = kwargs.get("mask")
        if mask is not None and self.mask_sensitive:
            valid_mask = np.asarray(mask) == self._VALID_MASK
            y_true_arr = y_true_arr[valid_mask]
            y_pred_arr = y_pred_arr[valid_mask]

        y_true_flat = y_true_arr.ravel()
        y_pred_flat = y_pred_arr.ravel()

        valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_flat = y_true_flat[valid_idx]
        y_pred_flat = y_pred_flat[valid_idx]

        if len(y_true_flat) == 0:
            return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan,
                    "correlation": np.nan, "spearman_correlation": np.nan,
                    "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan}

        if task_type == "auto":
            task_type = self._detect_task_type(y_true_flat, y_pred_flat)

        if task_type == "classification":
            return self.evaluate_classification(y_true_flat, y_pred_flat, **kwargs)
        return self.evaluate_regression(y_true_flat, y_pred_flat)

    def evaluate_regression(self, y_true: ArrayFloat, y_pred: ArrayFloat) -> dict[str, float]:
        result = {}
        try:
            result["mae"] = float(mean_absolute_error(y_true, y_pred))
            result["mse"] = float(mean_squared_error(y_true, y_pred))
            result["rmse"] = float(np.sqrt(result["mse"]))
            result["r2"] = float(r2_score(y_true, y_pred))
            result["correlation"] = float(pearsonr(y_true, y_pred)[0])
            result["spearman_correlation"] = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            result.update({"mae": np.nan, "mse": np.nan, "rmse": np.nan,
                           "r2": np.nan, "correlation": np.nan, "spearman_correlation": np.nan})
        return result

    def evaluate_classification(self, y_true: ArrayInt, y_pred: ArrayInt,
                                 **kwargs) -> dict[str, float]:
        y_true_int, y_pred_int = y_true.astype(int), y_pred.astype(int)
        average = kwargs.get("average", "weighted")

        unique_labels = np.unique(np.concatenate([y_true_int, y_pred_int]))
        avg = "binary" if len(unique_labels) <= 2 else (
            average if average in ["micro", "macro", "weighted"] else "weighted")

        result = {}
        try:
            result["accuracy"] = float(accuracy_score(y_true_int, y_pred_int))
            result["precision"] = float(precision_score(y_true_int, y_pred_int, average=avg, zero_division=0))
            result["recall"] = float(recall_score(y_true_int, y_pred_int, average=avg, zero_division=0))
            result["f1_score"] = float(f1_score(y_true_int, y_pred_int, average=avg, zero_division=0))
        except Exception:
            result.update({"accuracy": np.nan, "precision": np.nan,
                           "recall": np.nan, "f1_score": np.nan})
        return result

    def evaluate_with_mask(self, y_true: ArrayFloat, y_pred: ArrayFloat,
                           mask: ArrayInt, **kwargs) -> dict[str, float]:
        return self.evaluate(y_true, y_pred, mask=mask, mask_sensitive=True, **kwargs)

    def get_detailed_result(self, y_true: ArrayFloat | ArrayInt,
                            y_pred: ArrayFloat | ArrayInt, **kwargs) -> AccuracyResult:
        y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
        metrics = self.evaluate(y_true_arr, y_pred_arr, **kwargs)
        return AccuracyResult(
            task_type=kwargs.get("task_type", self.task_type),
            metrics=metrics,
            n_samples=y_true_arr.size,
            n_valid=int(np.sum(~(np.isnan(y_true_arr) | np.isnan(y_pred_arr)))),
        )

    def _detect_task_type(self, y_true: ArrayFloat, y_pred: ArrayFloat) -> str:
        if y_true.size == 0:
            return "regression"
        y_true_unique, y_pred_unique = np.unique(y_true), np.unique(y_pred)
        n_unique = len(y_true_unique)
        is_int = np.allclose(y_true_unique, y_true_unique.astype(int)) and \
                 np.allclose(y_pred_unique, y_pred_unique.astype(int))
        return "classification" if n_unique <= 20 and is_int else "regression"


def evaluate_accuracy(y_true: ArrayFloat | ArrayInt, y_pred: ArrayFloat | ArrayInt,
                     task_type: str = "auto", mask_sensitive: bool = False,
                     **kwargs) -> dict[str, float]:
    return AccuracyEvaluator(mask_sensitive=mask_sensitive).evaluate(
        y_true, y_pred, task_type=task_type, **kwargs)


def evaluate_regression_accuracy(y_true: ArrayFloat, y_pred: ArrayFloat) -> dict[str, float]:
    return AccuracyEvaluator().evaluate_regression(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())


def evaluate_classification_accuracy(y_true: ArrayInt, y_pred: ArrayInt,
                                     average: str = "weighted") -> dict[str, float]:
    return AccuracyEvaluator().evaluate_classification(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), average=average)


__all__ = [
    "BaseEvaluator",
    "AccuracyEvaluator",
    "AccuracyResult",
    "evaluate_accuracy",
    "evaluate_regression_accuracy",
    "evaluate_classification_accuracy",
]
