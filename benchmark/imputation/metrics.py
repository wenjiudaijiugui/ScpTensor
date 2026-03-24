"""Metrics for protein-level imputation benchmarking."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from scptensor.core import MaskCode

METRIC_DIRECTIONS: dict[str, bool] = {
    "holdout_coverage": True,
    "pearson_r": True,
    "spearman_r": True,
    "mae": False,
    "rmse": False,
    "nrmse": False,
    "within_group_cv_median": False,
    "cluster_ari": True,
    "cluster_nmi": True,
    "cluster_asw": True,
    "cluster_knn_purity": True,
    "de_log2fc_pearson": True,
    "de_topk_jaccard": True,
    "de_topk_sign_agreement": True,
    "de_topk_f1": True,
    "de_pauc_01": True,
    "de_pauc_05": True,
    "de_pauc_10": True,
    "ratio_pairwise_auc_mean": True,
    "ratio_changed_vs_bg_auc": True,
    "ratio_mae": False,
    "ratio_rmse": False,
    "retained_proteins_ratio": True,
    "fully_observed_proteins_ratio": True,
    "runtime_sec": False,
    "post_missing_rate": False,
}


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    y_std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    if np.allclose(x, y, equal_nan=True, rtol=1e-10, atol=1e-12):
        return 1.0
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _nanmean_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


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


def summarize_holdout_state_profile(mask_codes: np.ndarray) -> dict[str, float]:
    """Summarize source-state composition for the evaluated holdout entries."""
    values = np.asarray(mask_codes, dtype=np.int8).ravel()
    if values.size == 0:
        return {
            "state_valid_fraction": float("nan"),
            "state_non_valid_fraction": float("nan"),
            "state_mbr_fraction": float("nan"),
            "state_lod_fraction": float("nan"),
            "state_filtered_fraction": float("nan"),
            "state_outlier_fraction": float("nan"),
            "state_imputed_fraction": float("nan"),
            "state_uncertain_fraction": float("nan"),
            "state_direct_observation_rate": float("nan"),
            "state_supported_observation_rate": float("nan"),
            "state_uncertainty_burden": float("nan"),
        }

    fractions = {
        MaskCode.VALID: float(np.mean(values == MaskCode.VALID.value)),
        MaskCode.MBR: float(np.mean(values == MaskCode.MBR.value)),
        MaskCode.LOD: float(np.mean(values == MaskCode.LOD.value)),
        MaskCode.FILTERED: float(np.mean(values == MaskCode.FILTERED.value)),
        MaskCode.OUTLIER: float(np.mean(values == MaskCode.OUTLIER.value)),
        MaskCode.IMPUTED: float(np.mean(values == MaskCode.IMPUTED.value)),
        MaskCode.UNCERTAIN: float(np.mean(values == MaskCode.UNCERTAIN.value)),
    }
    valid_fraction = fractions[MaskCode.VALID]
    supported_fraction = valid_fraction + fractions[MaskCode.MBR]
    uncertainty_burden = (
        fractions[MaskCode.FILTERED]
        + fractions[MaskCode.OUTLIER]
        + fractions[MaskCode.IMPUTED]
        + fractions[MaskCode.UNCERTAIN]
    )
    return {
        "state_valid_fraction": valid_fraction,
        "state_non_valid_fraction": float(1.0 - valid_fraction),
        "state_mbr_fraction": fractions[MaskCode.MBR],
        "state_lod_fraction": fractions[MaskCode.LOD],
        "state_filtered_fraction": fractions[MaskCode.FILTERED],
        "state_outlier_fraction": fractions[MaskCode.OUTLIER],
        "state_imputed_fraction": fractions[MaskCode.IMPUTED],
        "state_uncertain_fraction": fractions[MaskCode.UNCERTAIN],
        "state_direct_observation_rate": valid_fraction,
        "state_supported_observation_rate": float(supported_fraction),
        "state_uncertainty_burden": float(uncertainty_burden),
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


def _valid_group_array(
    groups: Sequence[str] | None,
    n_samples: int,
) -> np.ndarray | None:
    if groups is None:
        return None
    g = np.asarray(list(groups), dtype=object)
    if g.shape[0] != n_samples:
        return None
    uniq, counts = np.unique(g, return_counts=True)
    if uniq.size < 2 or np.min(counts) < 2:
        return None
    return g


def compute_cluster_metrics(
    matrix: np.ndarray,
    groups: Sequence[str] | None,
    *,
    n_neighbors: int = 10,
) -> dict[str, float]:
    """Compute cluster-structure preservation metrics against known group labels."""
    x = np.asarray(matrix, dtype=np.float64)
    labels = _valid_group_array(groups, x.shape[0])
    if labels is None:
        return {
            "cluster_ari": float("nan"),
            "cluster_nmi": float("nan"),
            "cluster_asw": float("nan"),
            "cluster_knn_purity": float("nan"),
        }

    n_clusters = int(np.unique(labels).size)
    if x.shape[0] <= n_clusters:
        return {
            "cluster_ari": float("nan"),
            "cluster_nmi": float("nan"),
            "cluster_asw": float("nan"),
            "cluster_knn_purity": float("nan"),
        }

    try:
        pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x)
        ari = float(adjusted_rand_score(labels, pred))
        nmi = float(normalized_mutual_info_score(labels, pred))
    except Exception:  # noqa: BLE001
        ari, nmi = float("nan"), float("nan")

    try:
        asw = float(silhouette_score(x, labels))
    except Exception:  # noqa: BLE001
        asw = float("nan")

    k = max(2, min(int(n_neighbors), x.shape[0] - 1))
    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x)
        _, idx = nbrs.kneighbors(x)
        scores: list[float] = []
        for neigh in idx:
            neigh_labels = labels[neigh[1:]]
            if neigh_labels.size == 0:
                continue
            purity = np.mean(neigh_labels == labels[neigh[0]])
            scores.append(float(purity))
        knn_purity = float(np.mean(scores)) if scores else float("nan")
    except Exception:  # noqa: BLE001
        knn_purity = float("nan")

    return {
        "cluster_ari": ari,
        "cluster_nmi": nmi,
        "cluster_asw": asw,
        "cluster_knn_purity": knn_purity,
    }


def compute_de_consistency_metrics(
    matrix_true: np.ndarray,
    matrix_imputed: np.ndarray,
    groups: Sequence[str] | None,
    *,
    top_k: int = 50,
) -> dict[str, float]:
    """Compare DE-like group-contrast signals before/after imputation.

    Uses pairwise group log2 fold-change vectors as a task-oriented proxy for DE consistency.
    """
    x_true = np.asarray(matrix_true, dtype=np.float64)
    x_imp = np.asarray(matrix_imputed, dtype=np.float64)
    labels = _valid_group_array(groups, x_true.shape[0])

    if labels is None or x_true.shape != x_imp.shape:
        return {
            "de_log2fc_pearson": float("nan"),
            "de_topk_jaccard": float("nan"),
            "de_topk_sign_agreement": float("nan"),
            "de_topk_f1": float("nan"),
            "de_pauc_01": float("nan"),
            "de_pauc_05": float("nan"),
            "de_pauc_10": float("nan"),
        }

    uniq = np.unique(labels)
    if uniq.size < 2:
        return {
            "de_log2fc_pearson": float("nan"),
            "de_topk_jaccard": float("nan"),
            "de_topk_sign_agreement": float("nan"),
            "de_topk_f1": float("nan"),
            "de_pauc_01": float("nan"),
            "de_pauc_05": float("nan"),
            "de_pauc_10": float("nan"),
        }

    corr_vals: list[float] = []
    jacc_vals: list[float] = []
    sign_vals: list[float] = []
    f1_vals: list[float] = []
    pauc_01_vals: list[float] = []
    pauc_05_vals: list[float] = []
    pauc_10_vals: list[float] = []

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            g1, g2 = uniq[i], uniq[j]
            idx1 = np.where(labels == g1)[0]
            idx2 = np.where(labels == g2)[0]
            if idx1.size < 2 or idx2.size < 2:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fc_true = np.nanmean(x_true[idx1, :], axis=0) - np.nanmean(x_true[idx2, :], axis=0)
                fc_imp = np.nanmean(x_imp[idx1, :], axis=0) - np.nanmean(x_imp[idx2, :], axis=0)

            valid = np.isfinite(fc_true) & np.isfinite(fc_imp)
            if np.sum(valid) < 3:
                continue

            t = fc_true[valid]
            p = fc_imp[valid]

            corr_vals.append(_safe_pearson(t, p))

            max_k = max(1, t.size - 1)
            k = int(min(max(5, top_k), max_k))
            idx_top_true = np.argsort(-np.abs(t))[:k]
            idx_top_imp = np.argsort(-np.abs(p))[:k]
            set_true = set(idx_top_true.tolist())
            set_imp = set(idx_top_imp.tolist())
            inter = len(set_true & set_imp)
            union = len(set_true | set_imp)
            jacc_vals.append(float(inter / union) if union > 0 else float("nan"))
            f1_vals.append(float(inter / k) if k > 0 else float("nan"))

            sign_match = np.mean(np.sign(p[idx_top_true]) == np.sign(t[idx_top_true]))
            sign_vals.append(float(sign_match))

            labels_binary = np.zeros(t.size, dtype=np.int8)
            labels_binary[idx_top_true] = 1
            scores_abs = np.abs(p)
            if np.unique(labels_binary).size == 2:
                for max_fpr, collector in (
                    (0.01, pauc_01_vals),
                    (0.05, pauc_05_vals),
                    (0.10, pauc_10_vals),
                ):
                    try:
                        collector.append(
                            float(roc_auc_score(labels_binary, scores_abs, max_fpr=max_fpr))
                        )
                    except ValueError:
                        collector.append(float("nan"))

    return {
        "de_log2fc_pearson": _nanmean_or_nan(corr_vals),
        "de_topk_jaccard": _nanmean_or_nan(jacc_vals),
        "de_topk_sign_agreement": _nanmean_or_nan(sign_vals),
        "de_topk_f1": _nanmean_or_nan(f1_vals),
        "de_pauc_01": _nanmean_or_nan(pauc_01_vals),
        "de_pauc_05": _nanmean_or_nan(pauc_05_vals),
        "de_pauc_10": _nanmean_or_nan(pauc_10_vals),
    }


def _score_block(
    block: pd.DataFrame,
    metric_directions: dict[str, bool],
) -> pd.DataFrame:
    """Generate normalized per-metric scores for one comparison block."""
    work = block.copy()
    score_cols: list[str] = []

    for metric, higher_better in metric_directions.items():
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
        score_mat = work[score_cols].to_numpy(dtype=np.float64)
        valid_counts = np.sum(np.isfinite(score_mat), axis=1)
        numer = np.nansum(score_mat, axis=1)
        overall = np.full(work.shape[0], np.nan, dtype=np.float64)
        mask = valid_counts > 0
        overall[mask] = numer[mask] / valid_counts[mask]
        work["overall_score"] = overall
    else:
        work["overall_score"] = np.nan

    if "success_rate" in work.columns:
        success = pd.to_numeric(work["success_rate"], errors="coerce").clip(0.0, 1.0)
        success = success.fillna(0.0)
        work["score_success_rate"] = success
        work["overall_score"] = work["overall_score"] * success.to_numpy(dtype=np.float64)

    return work


def score_methods(
    summary_df: pd.DataFrame,
    *,
    metric_directions: dict[str, bool] | None = None,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate normalized scores within each benchmark scenario block."""
    if summary_df.empty:
        return pd.DataFrame()

    metric_directions = metric_directions or METRIC_DIRECTIONS
    group_by = list(group_by or ["dataset"])
    valid_group_by = [column for column in group_by if column in summary_df.columns]
    if not valid_group_by:
        valid_group_by = ["dataset"] if "dataset" in summary_df.columns else []

    if not valid_group_by:
        return _score_block(summary_df, metric_directions)

    rows: list[dict[str, object]] = []
    for _, block in summary_df.groupby(valid_group_by, sort=False, dropna=False):
        scored = _score_block(block, metric_directions)
        rows.extend(scored.to_dict("records"))
    return pd.DataFrame(rows)
