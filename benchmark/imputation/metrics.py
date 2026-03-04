"""Metrics for protein-level imputation benchmarking."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

METRIC_DIRECTIONS: dict[str, bool] = {
    "holdout_coverage": True,
    "pearson_r": True,
    "spearman_r": True,
    "mae": False,
    "rmse": False,
    "nrmse": False,
    "within_group_cv_median": False,
    "runtime_sec": False,
    "post_missing_rate": False,
}


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    y_std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_reconstruction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute masked-entry reconstruction metrics.

    Metrics follow common proteomics-imputation benchmark practice
    (e.g., NRMSE + correlation-based diagnostics).
    """
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    eval_mask = np.isfinite(truth) & np.isfinite(pred)
    n_total = int(truth.size)
    n_eval = int(np.sum(eval_mask))
    coverage = float(n_eval / n_total) if n_total > 0 else float("nan")

    if n_eval == 0:
        return {
            "n_holdout": float(n_total),
            "n_eval": 0.0,
            "holdout_coverage": coverage,
            "mae": float("nan"),
            "rmse": float("nan"),
            "nrmse": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "median_abs_error": float("nan"),
        }

    t = truth[eval_mask]
    p = pred[eval_mask]
    err = p - t

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    denom = float(np.std(t, ddof=1)) if t.size > 1 else float(np.std(t, ddof=0))
    if not np.isfinite(denom) or denom <= 1e-12:
        nrmse = float("nan")
    else:
        nrmse = float(rmse / denom)

    pearson_r = _safe_pearson(t, p)
    t_std = float(np.std(t, ddof=1)) if t.size > 1 else float(np.std(t, ddof=0))
    p_std = float(np.std(p, ddof=1)) if p.size > 1 else float(np.std(p, ddof=0))
    if t_std <= 1e-12 or p_std <= 1e-12:
        spearman_r = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            spearman_r = float(spearmanr(t, p, nan_policy="omit").correlation)

    return {
        "n_holdout": float(n_total),
        "n_eval": float(n_eval),
        "holdout_coverage": coverage,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "median_abs_error": float(np.median(np.abs(err))),
    }


def compute_within_group_cv_median(
    matrix: np.ndarray,
    groups: Sequence[str] | None,
) -> float:
    """Median within-group CV across protein features.

    Inspired by CV-focused workflow metrics used in DIA preprocessing studies.
    """
    if groups is None:
        return float("nan")

    x = np.asarray(matrix, dtype=np.float64)
    g = np.asarray(list(groups), dtype=object)
    if g.shape[0] != x.shape[0]:
        return float("nan")

    per_group_medians: list[float] = []
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        if idx.size < 2:
            continue

        block = x[idx, :]
        finite_counts = np.sum(np.isfinite(block), axis=0)
        valid = finite_counts >= 2
        if not np.any(valid):
            continue

        block_valid = block[:, valid]
        mean_vals = np.nanmean(block_valid, axis=0)
        std_vals = np.nanstd(block_valid, axis=0, ddof=1)

        cv = np.full(mean_vals.shape, np.nan, dtype=np.float64)
        nz = np.isfinite(mean_vals) & np.isfinite(std_vals) & (np.abs(mean_vals) > 1e-12)
        cv[nz] = std_vals[nz] / np.abs(mean_vals[nz])

        cv = cv[np.isfinite(cv)]
        if cv.size > 0:
            per_group_medians.append(float(np.median(cv)))

    if not per_group_medians:
        return float("nan")
    return float(np.median(np.asarray(per_group_medians, dtype=np.float64)))


def score_methods(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-dataset normalized scores and overall ranking."""
    rows: list[dict[str, object]] = []

    for _dataset, block in summary_df.groupby("dataset", sort=False):
        work = block.copy()
        score_cols: list[str] = []

        for metric, higher_better in METRIC_DIRECTIONS.items():
            if metric not in work.columns:
                continue

            values = work[metric].astype(float)
            finite = np.isfinite(values.to_numpy(dtype=np.float64))
            score_col = f"score_{metric}"
            score_cols.append(score_col)

            if np.sum(finite) < 2:
                work[score_col] = np.nan
                continue

            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            if np.isclose(vmin, vmax):
                work[score_col] = np.where(np.isfinite(values), 1.0, np.nan)
                continue

            if higher_better:
                work[score_col] = (values - vmin) / (vmax - vmin)
            else:
                work[score_col] = (vmax - values) / (vmax - vmin)

        if score_cols:
            work["overall_score"] = np.nanmean(work[score_cols].to_numpy(dtype=np.float64), axis=1)
        else:
            work["overall_score"] = np.nan

        rows.extend(work.to_dict("records"))

    return pd.DataFrame(rows)
