"""Metrics utilities for integration benchmark outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

BALANCED_METRIC_DIRECTIONS: dict[str, bool] = {
    "between_batch_ratio": False,
    "delta_between_batch_ratio": False,
    "batch_asw": False,
    "batch_mixing": True,
    "lisi_approx": True,
    "between_condition_ratio": True,
    "condition_asw": True,
    "condition_ari": True,
    "condition_nmi": True,
    "condition_knn_purity": True,
    "runtime_sec": False,
}

CONFOUNDED_METRIC_DIRECTIONS: dict[str, bool] = {
    "between_batch_ratio": False,
    "delta_between_batch_ratio": False,
    "batch_asw": False,
    "batch_mixing": True,
    "lisi_approx": True,
    "runtime_sec": False,
}

# Default profile uses balanced settings.
METRIC_DIRECTIONS = BALANCED_METRIC_DIRECTIONS


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=1)) if x.size > 1 else float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=1)) if y.size > 1 else float(np.std(y, ddof=0))
    if np.allclose(x, y, equal_nan=True, rtol=1e-10, atol=1e-12):
        return 1.0
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_marker_consistency_metrics(
    reference_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    condition_labels: Sequence[str],
    *,
    top_k: int = 50,
) -> dict[str, float]:
    """Compare marker-style group contrasts before/after integration.

    This reports a third, read-only axis for benchmark interpretation. It does
    not change the current ranking contract, which remains driven by batch and
    biological-conservation scores.
    """
    x_ref = np.asarray(reference_matrix, dtype=np.float64)
    x_cmp = np.asarray(candidate_matrix, dtype=np.float64)
    labels = np.asarray(list(condition_labels), dtype=object)

    if x_ref.shape != x_cmp.shape or labels.shape[0] != x_ref.shape[0]:
        return {
            "marker_log2fc_pearson": float("nan"),
            "marker_topk_jaccard": float("nan"),
            "marker_topk_sign_agreement": float("nan"),
        }

    unique = np.unique(labels)
    if unique.size < 2:
        return {
            "marker_log2fc_pearson": float("nan"),
            "marker_topk_jaccard": float("nan"),
            "marker_topk_sign_agreement": float("nan"),
        }

    corr_vals: list[float] = []
    jacc_vals: list[float] = []
    sign_vals: list[float] = []

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            g1 = unique[i]
            g2 = unique[j]
            idx1 = np.where(labels == g1)[0]
            idx2 = np.where(labels == g2)[0]
            if idx1.size < 2 or idx2.size < 2:
                continue

            fc_ref = np.nanmean(x_ref[idx1, :], axis=0) - np.nanmean(x_ref[idx2, :], axis=0)
            fc_cmp = np.nanmean(x_cmp[idx1, :], axis=0) - np.nanmean(x_cmp[idx2, :], axis=0)

            valid = np.isfinite(fc_ref) & np.isfinite(fc_cmp)
            if np.sum(valid) < 3:
                continue

            ref = fc_ref[valid]
            cmp = fc_cmp[valid]
            corr_vals.append(_safe_pearson(ref, cmp))

            max_k = max(1, ref.size - 1)
            k = int(min(max(5, top_k), max_k))
            idx_top_ref = np.argsort(-np.abs(ref))[:k]
            idx_top_cmp = np.argsort(-np.abs(cmp))[:k]

            set_ref = set(idx_top_ref.tolist())
            set_cmp = set(idx_top_cmp.tolist())
            inter = len(set_ref & set_cmp)
            union = len(set_ref | set_cmp)
            jacc_vals.append(float(inter / union) if union > 0 else float("nan"))

            sign_vals.append(float(np.mean(np.sign(cmp[idx_top_ref]) == np.sign(ref[idx_top_ref]))))

    return {
        "marker_log2fc_pearson": (float(np.nanmean(corr_vals)) if corr_vals else float("nan")),
        "marker_topk_jaccard": (float(np.nanmean(jacc_vals)) if jacc_vals else float("nan")),
        "marker_topk_sign_agreement": (float(np.nanmean(sign_vals)) if sign_vals else float("nan")),
    }


def _score_block(
    block: pd.DataFrame,
    metric_directions: dict[str, bool],
) -> pd.DataFrame:
    work = block.copy()
    score_cols: list[str] = []

    for metric, higher_better in metric_directions.items():
        if metric not in work.columns:
            continue

        vals = pd.to_numeric(work[metric], errors="coerce")
        finite = np.isfinite(vals.to_numpy(dtype=np.float64))
        score_col = f"score_{metric}"
        score_cols.append(score_col)

        if np.sum(finite) < 2:
            work[score_col] = np.nan
            continue

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if np.isclose(vmin, vmax):
            work[score_col] = np.where(np.isfinite(vals), 1.0, np.nan)
            continue

        if higher_better:
            work[score_col] = (vals - vmin) / (vmax - vmin)
        else:
            work[score_col] = (vmax - vals) / (vmax - vmin)

    if score_cols:
        work["overall_score"] = np.nanmean(work[score_cols].to_numpy(dtype=np.float64), axis=1)
    else:
        work["overall_score"] = np.nan

    return work


def score_methods(
    summary_df: pd.DataFrame,
    *,
    metric_directions: dict[str, bool] | None = None,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute normalized per-metric scores and overall score."""
    if summary_df.empty:
        return pd.DataFrame()

    metric_directions = metric_directions or METRIC_DIRECTIONS
    group_by = list(group_by or ["dataset", "scenario"])
    valid_group_by = [c for c in group_by if c in summary_df.columns]
    if not valid_group_by:
        valid_group_by = ["dataset"] if "dataset" in summary_df.columns else []

    if not valid_group_by:
        return _score_block(summary_df, metric_directions)

    rows: list[dict[str, Any]] = []
    for _, block in summary_df.groupby(valid_group_by, sort=False, dropna=False):
        scored = _score_block(block, metric_directions)
        rows.extend(scored.to_dict("records"))
    return pd.DataFrame(rows)


__all__ = [
    "BALANCED_METRIC_DIRECTIONS",
    "CONFOUNDED_METRIC_DIRECTIONS",
    "METRIC_DIRECTIONS",
    "compute_marker_consistency_metrics",
    "score_methods",
]
