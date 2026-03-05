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
    "score_methods",
]
