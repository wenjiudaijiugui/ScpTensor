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
        return {
            "task_type": self.task_type,
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            **self.metrics,
        }


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator for accuracy metrics of algorithm results."""

    __slots__ = ("mask_sensitive", "task_type")
    _VALID_MASK = 0

    def __init__(self, mask_sensitive: bool = False, task_type: str = "auto") -> None:
        self.mask_sensitive = mask_sensitive
        self.task_type = task_type

    def evaluate(
        self,
        y_true: ArrayFloat | ArrayInt,
        y_pred: ArrayFloat | ArrayInt,
        task_type: str = "auto",
        **kwargs,
    ) -> dict[str, float]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}"
            )

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
            result.update(
                {
                    "mae": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "correlation": np.nan,
                    "spearman_correlation": np.nan,
                }
            )
        return result

    def evaluate_classification(
        self, y_true: ArrayInt, y_pred: ArrayInt, **kwargs
    ) -> dict[str, float]:
        y_true_int, y_pred_int = y_true.astype(int), y_pred.astype(int)
        average = kwargs.get("average", "weighted")

        unique_labels = np.unique(np.concatenate([y_true_int, y_pred_int]))
        avg = (
            "binary"
            if len(unique_labels) <= 2
            else (average if average in ["micro", "macro", "weighted"] else "weighted")
        )

        result = {}
        try:
            result["accuracy"] = float(accuracy_score(y_true_int, y_pred_int))
            result["precision"] = float(
                precision_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
            result["recall"] = float(
                recall_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
            result["f1_score"] = float(
                f1_score(y_true_int, y_pred_int, average=avg, zero_division=0)
            )
        except Exception:
            result.update(
                {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan}
            )
        return result

    def evaluate_with_mask(
        self, y_true: ArrayFloat, y_pred: ArrayFloat, mask: ArrayInt, **kwargs
    ) -> dict[str, float]:
        return self.evaluate(y_true, y_pred, mask=mask, mask_sensitive=True, **kwargs)

    def get_detailed_result(
        self, y_true: ArrayFloat | ArrayInt, y_pred: ArrayFloat | ArrayInt, **kwargs
    ) -> AccuracyResult:
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
        is_int = np.allclose(y_true_unique, y_true_unique.astype(int)) and np.allclose(
            y_pred_unique, y_pred_unique.astype(int)
        )
        return "classification" if n_unique <= 20 and is_int else "regression"


def evaluate_accuracy(
    y_true: ArrayFloat | ArrayInt,
    y_pred: ArrayFloat | ArrayInt,
    task_type: str = "auto",
    mask_sensitive: bool = False,
    **kwargs,
) -> dict[str, float]:
    return AccuracyEvaluator(mask_sensitive=mask_sensitive).evaluate(
        y_true, y_pred, task_type=task_type, **kwargs
    )


def evaluate_regression_accuracy(y_true: ArrayFloat, y_pred: ArrayFloat) -> dict[str, float]:
    return AccuracyEvaluator().evaluate_regression(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    )


def evaluate_classification_accuracy(
    y_true: ArrayInt, y_pred: ArrayInt, average: str = "weighted"
) -> dict[str, float]:
    return AccuracyEvaluator().evaluate_classification(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), average=average
    )


def compare_pca_variance(
    variance_scptensor: np.ndarray,
    variance_competitor: np.ndarray,
    n_components: int | None = None,
    method: str = "scptensor",
) -> dict[str, float | np.ndarray]:
    """比较PCA方差解释率.

    Parameters
    ----------
    variance_scptensor : np.ndarray
        ScpTensor的方差解释率数组
    variance_competitor : np.ndarray
        竞争对手的方差解释率数组
    n_components : Optional[int]
        比较的前n个主成分，None表示全部
    method : str
        方法标识符

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        包含以下键:
        - 'correlation': Pearson相关系数
        - 'spearman': Spearman相关系数
        - 'mse': 均方误差
        - 'mae': 平均绝对误差
        - 'max_diff': 最大差异
        - 'cumulative_diff': 累积方差差异
    """
    # 截取前n_components个
    if n_components is not None:
        var_scpt = variance_scptensor[:n_components]
        var_comp = variance_competitor[:n_components]
    else:
        var_scpt = variance_scptensor
        var_comp = variance_competitor

    # 计算各种指标
    result = {
        "correlation": float(np.corrcoef(var_scpt, var_comp)[0, 1]),
        "spearman": float(spearmanr(var_scpt, var_comp)[0]),
        "mse": float(np.mean((var_scpt - var_comp) ** 2)),
        "mae": float(np.mean(np.abs(var_scpt - var_comp))),
        "max_diff": float(np.max(np.abs(var_scpt - var_comp))),
        "cumulative_diff": float(np.abs(np.sum(var_scpt) - np.sum(var_comp))),
    }

    return result


def compare_dimensionality_reduction(
    X_scptensor: np.ndarray,
    X_competitor: np.ndarray,
    method: str = "pca",
    metric: str = "procrustes",
) -> dict[str, float]:
    """比较降维结果的质量.

    Parameters
    ----------
    X_scptensor : np.ndarray
        ScpTensor的降维结果 (n_samples x n_components)
    X_competitor : np.ndarray
        竞争对手的降维结果 (n_samples x n_components)
    method : str
        降维方法 ('pca', 'umap', 'tsne')
    metric : str
        比较指标 ('procrustes', 'correlation', 'rv_coefficient')

    Returns
    -------
    Dict[str, float]
        包含相似性指标的字典
    """
    from scipy.spatial import procrustes

    # 确保形状一致
    assert X_scptensor.shape == X_competitor.shape, (
        f"Shape mismatch: {X_scptensor.shape} vs {X_competitor.shape}"
    )

    if metric == "procrustes":
        # Procrustes分析
        _, _, disparity = procrustes(X_scptensor, X_competitor)
        return {
            "procrustes_disparity": float(disparity),
            "similarity": float(1 / (1 + disparity)),
        }
    elif metric == "correlation":
        # 计算每个维度的相关性
        corrs = [
            np.corrcoef(X_scptensor[:, i], X_competitor[:, i])[0, 1]
            for i in range(X_scptensor.shape[1])
        ]
        return {
            "mean_correlation": float(np.mean(corrs)),
            "min_correlation": float(np.min(corrs)),
            "max_correlation": float(np.max(corrs)),
        }
    elif metric == "rv_coefficient":
        # RV系数（矩阵相关性）
        # 确保都是2D数组
        X1 = X_scptensor.reshape(-1, 1) if X_scptensor.ndim == 1 else X_scptensor
        X2 = X_competitor.reshape(-1, 1) if X_competitor.ndim == 1 else X_competitor

        # 中心化
        X1_centered = X1 - X1.mean(axis=0)
        X2_centered = X2 - X2.mean(axis=0)

        # 计算RV系数
        rv = np.trace(X1_centered.T @ X2_centered @ X2_centered.T @ X1_centered) / np.sqrt(
            np.trace(X1_centered.T @ X1_centered @ X1_centered.T @ X1_centered)
            * np.trace(X2_centered.T @ X2_centered @ X2_centered.T @ X2_centered)
        )
        return {"rv_coefficient": float(rv)}
    else:
        raise ValueError(f"Unknown metric: {metric}")


__all__ = [
    "BaseEvaluator",
    "AccuracyEvaluator",
    "AccuracyResult",
    "evaluate_accuracy",
    "evaluate_regression_accuracy",
    "evaluate_classification_accuracy",
    "compare_pca_variance",
    "compare_dimensionality_reduction",
]
