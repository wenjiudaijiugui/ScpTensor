"""Metrics for peptide->protein aggregation benchmarking."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

EXPECTED_LOG2FC_HYE124: dict[str, float] = {
    "HUMAN": 0.0,
    "YEAST": 1.0,
    "ECOLI": -2.0,
}

BACKGROUND_SPECIES = "HUMAN"


def infer_species_from_protein_id(protein_id: str | None) -> str | None:
    """Infer benchmark species label from protein identifier string."""
    if not protein_id:
        return None

    token = str(protein_id).upper()
    if "HUMAN" in token:
        return "HUMAN"
    if "YEAS" in token or "YEAST" in token:
        return "YEAST"
    if "ECOLI" in token or "ESCHERICHIA COLI" in token:
        return "ECOLI"
    return None


def _nanmean_columns(values: np.ndarray) -> np.ndarray:
    """Column-wise nanmean without all-NaN warnings."""
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    has = np.isfinite(values).any(axis=0)
    if np.any(has):
        out[has] = np.nanmean(values[:, has], axis=0)
    return out


def _feature_cv(values: np.ndarray) -> np.ndarray:
    """Coefficient of variation for each feature across replicate samples."""
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    finite_counts = np.sum(np.isfinite(values), axis=0)
    mean_vals = _nanmean_columns(values)
    valid = (finite_counts >= 2) & np.isfinite(mean_vals) & (mean_vals > 0)
    if np.any(valid):
        std_vals = np.nanstd(values[:, valid], axis=0, ddof=1)
        out[valid] = std_vals / mean_vals[valid]
    return out


def compute_protein_level_table(
    *,
    method: str,
    protein_ids: list[str],
    matrix: np.ndarray,
    group_a_idx: np.ndarray,
    group_b_idx: np.ndarray,
) -> pd.DataFrame:
    """Compute per-protein A/B log2 fold-change and replicate CVs."""
    x = np.asarray(matrix, dtype=np.float64)
    group_a = x[group_a_idx, :]
    group_b = x[group_b_idx, :]

    mean_a = _nanmean_columns(group_a)
    mean_b = _nanmean_columns(group_b)

    log2_fc = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid_ratio = np.isfinite(mean_a) & np.isfinite(mean_b) & (mean_a > 0) & (mean_b > 0)
    log2_fc[valid_ratio] = np.log2(mean_a[valid_ratio] / mean_b[valid_ratio])

    cv_a = _feature_cv(group_a)
    cv_b = _feature_cv(group_b)
    cv_stack = np.vstack([cv_a, cv_b])
    cv_mean = np.full(cv_stack.shape[1], np.nan, dtype=np.float64)
    has_cv = np.isfinite(cv_stack).any(axis=0)
    if np.any(has_cv):
        cv_mean[has_cv] = np.nanmean(cv_stack[:, has_cv], axis=0)

    species = [infer_species_from_protein_id(pid) for pid in protein_ids]

    return pd.DataFrame(
        {
            "method": method,
            "protein_id": protein_ids,
            "species": species,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "log2_fc_ab": log2_fc,
            "cv_a": cv_a,
            "cv_b": cv_b,
            "cv_mean": cv_mean,
        }
    )


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    try:
        auc_val = float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")
    if np.isnan(auc_val):
        return float("nan")
    return max(auc_val, 1.0 - auc_val)


def _compute_pairwise_auc(
    quantified: pd.DataFrame,
    expected_log2fc: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    species_order = list(expected_log2fc.keys())

    for sp1, sp2 in combinations(species_order, 2):
        sub = quantified[quantified["species"].isin([sp1, sp2])]
        if sub.empty or sub["species"].nunique() < 2:
            auc_val = float("nan")
        else:
            pos = sp1 if expected_log2fc[sp1] >= expected_log2fc[sp2] else sp2
            labels = (sub["species"] == pos).astype(int).to_numpy()
            scores = sub["log2_fc_ab"].to_numpy(dtype=np.float64)
            auc_val = _safe_auc(labels, scores)

        rows.append(
            {
                "species_pair": f"{sp1}_vs_{sp2}",
                "auc": auc_val,
            }
        )

    return pd.DataFrame(rows)


def summarize_method(
    protein_table: pd.DataFrame,
    *,
    expected_log2fc: Mapping[str, float] = EXPECTED_LOG2FC_HYE124,
    background_species: str = BACKGROUND_SPECIES,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize one aggregation method with LFQ-style benchmark metrics."""
    method = str(protein_table["method"].iloc[0]) if not protein_table.empty else "unknown"

    scoped = protein_table[protein_table["species"].isin(expected_log2fc.keys())].copy()
    scoped["expected_log2_fc_ab"] = scoped["species"].map(expected_log2fc)
    scoped["is_quantified"] = np.isfinite(scoped["log2_fc_ab"])

    quantified = scoped[scoped["is_quantified"]].copy()
    quantified["error"] = quantified["log2_fc_ab"] - quantified["expected_log2_fc_ab"]

    if quantified.empty:
        raise ValueError(f"No quantified proteins remain for method '{method}'.")

    abs_err = np.abs(quantified["error"].to_numpy(dtype=np.float64))
    sq_err = np.square(quantified["error"].to_numpy(dtype=np.float64))

    pairwise_auc = _compute_pairwise_auc(quantified, expected_log2fc)
    pairwise_auc["method"] = method

    labels_changed = (quantified["species"] != background_species).astype(int).to_numpy()
    bg_expected = float(expected_log2fc[background_species])
    changed_scores = np.abs(quantified["log2_fc_ab"].to_numpy(dtype=np.float64) - bg_expected)
    changed_auc = _safe_auc(labels_changed, changed_scores)

    per_species_rows: list[dict[str, float | int | str]] = []
    for species in expected_log2fc:
        block_all = scoped[scoped["species"] == species]
        block_q = quantified[quantified["species"] == species]

        n_total = int(block_all.shape[0])
        n_quantified = int(block_q.shape[0])
        coverage = float(n_quantified / n_total) if n_total > 0 else float("nan")

        if block_q.empty:
            median_fc = float("nan")
            median_err = float("nan")
            sd_fc = float("nan")
        else:
            median_fc = float(np.nanmedian(block_q["log2_fc_ab"].to_numpy(dtype=np.float64)))
            median_err = float(np.nanmedian(block_q["error"].to_numpy(dtype=np.float64)))
            if block_q.shape[0] > 1:
                sd_fc = float(np.nanstd(block_q["log2_fc_ab"].to_numpy(dtype=np.float64), ddof=1))
            else:
                sd_fc = float("nan")

        per_species_rows.append(
            {
                "method": method,
                "species": species,
                "n_total": n_total,
                "n_quantified": n_quantified,
                "coverage_ratio": coverage,
                "median_log2_fc_ab": median_fc,
                "median_error": median_err,
                "sd_log2_fc_ab": sd_fc,
            }
        )

    species_summary = pd.DataFrame(per_species_rows)

    pair_auc_values = pairwise_auc["auc"].to_numpy(dtype=np.float64)
    pairwise_auc_mean = (
        float(np.nanmean(pair_auc_values)) if np.isfinite(pair_auc_values).any() else float("nan")
    )

    scoped_n = int(scoped.shape[0])
    quantified_n = int(quantified.shape[0])

    summary = {
        "method": method,
        "n_species_mapped": scoped_n,
        "n_quantified": quantified_n,
        "coverage_ratio": float(quantified_n / scoped_n) if scoped_n > 0 else float("nan"),
        "mae": float(np.nanmean(abs_err)),
        "median_abs_error": float(np.nanmedian(abs_err)),
        "rmse": float(np.sqrt(np.nanmean(sq_err))),
        "bias": float(np.nanmean(quantified["error"].to_numpy(dtype=np.float64))),
        "error_sd": (
            float(np.nanstd(quantified["error"].to_numpy(dtype=np.float64), ddof=1))
            if quantified_n > 1
            else float("nan")
        ),
        "log2_fc_sd": (
            float(np.nanstd(quantified["log2_fc_ab"].to_numpy(dtype=np.float64), ddof=1))
            if quantified_n > 1
            else float("nan")
        ),
        "technical_variance_bg_cv_median": float(
            np.nanmedian(
                quantified.loc[quantified["species"] == background_species, "cv_mean"].to_numpy(
                    dtype=np.float64
                )
            )
        ),
        "cv_median_all": float(np.nanmedian(quantified["cv_mean"].to_numpy(dtype=np.float64))),
        "species_overlap_auc_mean": pairwise_auc_mean,
        "changed_vs_background_auc": changed_auc,
    }

    return summary, quantified, species_summary, pairwise_auc


def summarize_mapping_burden(protein_assignments: list[str | None]) -> dict[str, float]:
    """Summarize precursor/peptide -> protein ambiguity burden."""
    multiplicities: list[int] = []

    for assignment in protein_assignments:
        if assignment is None:
            continue
        tokens = [
            token.strip()
            for token in str(assignment).replace("|", ";").replace(",", ";").split(";")
            if token.strip()
        ]
        if not tokens:
            continue
        multiplicities.append(len(set(tokens)))

    if not multiplicities:
        return {
            "ambiguous_mapping_fraction": float("nan"),
            "mapping_targets_per_peptide_mean": float("nan"),
            "mapping_targets_per_peptide_median": float("nan"),
        }

    values = np.asarray(multiplicities, dtype=np.float64)
    return {
        "ambiguous_mapping_fraction": float(np.mean(values > 1)),
        "mapping_targets_per_peptide_mean": float(np.mean(values)),
        "mapping_targets_per_peptide_median": float(np.median(values)),
    }


def summarize_state_burden(
    mask: np.ndarray | sp.spmatrix | None,
    *,
    shape: tuple[int, int],
) -> dict[str, float]:
    """Summarize aggregated matrix state burden."""
    total = float(shape[0] * shape[1])
    if total <= 0:
        return {
            "state_valid_fraction": float("nan"),
            "state_non_valid_fraction": float("nan"),
            "state_lod_fraction": float("nan"),
            "state_uncertain_fraction": float("nan"),
        }

    if mask is None:
        return {
            "state_valid_fraction": 1.0,
            "state_non_valid_fraction": 0.0,
            "state_lod_fraction": 0.0,
            "state_uncertain_fraction": 0.0,
        }

    mask_dense = mask.toarray() if sp.issparse(mask) else np.asarray(mask, dtype=np.int8)
    valid_fraction = float(np.mean(mask_dense == 0))
    lod_fraction = float(np.mean(mask_dense == 2))
    uncertain_fraction = float(np.mean(mask_dense == 6))
    return {
        "state_valid_fraction": valid_fraction,
        "state_non_valid_fraction": float(1.0 - valid_fraction),
        "state_lod_fraction": lod_fraction,
        "state_uncertain_fraction": uncertain_fraction,
    }


def summarize_de_consistency_proxy(
    quantified: pd.DataFrame,
    *,
    background_species: str = BACKGROUND_SPECIES,
    zero_tolerance: float = 0.5,
) -> dict[str, float]:
    """Summarize a task-style DE proxy from expected species fold changes."""
    if quantified.empty:
        return {
            "de_changed_direction_accuracy": float("nan"),
            "de_background_stability_rate": float("nan"),
            "de_consistency_score": float("nan"),
        }

    changed = quantified[quantified["species"] != background_species].copy()
    background = quantified[quantified["species"] == background_species].copy()

    changed_accuracy = float("nan")
    if not changed.empty:
        expected = changed["expected_log2_fc_ab"].to_numpy(dtype=np.float64)
        observed = changed["log2_fc_ab"].to_numpy(dtype=np.float64)
        valid = np.isfinite(expected) & np.isfinite(observed) & (np.abs(expected) > 1e-12)
        if np.any(valid):
            changed_accuracy = float(np.mean(np.sign(observed[valid]) == np.sign(expected[valid])))

    background_stability = float("nan")
    if not background.empty:
        observed_bg = background["log2_fc_ab"].to_numpy(dtype=np.float64)
        valid_bg = np.isfinite(observed_bg)
        if np.any(valid_bg):
            background_stability = float(np.mean(np.abs(observed_bg[valid_bg]) <= zero_tolerance))

    components = [value for value in (changed_accuracy, background_stability) if np.isfinite(value)]
    de_consistency = float(np.mean(components)) if components else float("nan")
    return {
        "de_changed_direction_accuracy": changed_accuracy,
        "de_background_stability_rate": background_stability,
        "de_consistency_score": de_consistency,
    }
