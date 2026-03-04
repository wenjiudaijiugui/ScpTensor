"""Metrics for protein-level normalization benchmarking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

EXPECTED_LOG2FC_HYE124: dict[str, float] = {
    "HUMAN": 0.0,
    "YEAST": 1.0,
    "ECOLI": -2.0,
}

BACKGROUND_SPECIES = "HUMAN"


def infer_species_from_protein_id(protein_id: str | None) -> str | None:
    """Infer benchmark species label from protein identifier."""
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
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    has = np.isfinite(values).any(axis=0)
    if np.any(has):
        out[has] = np.nanmean(values[:, has], axis=0)
    return out


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

        rows.append({"species_pair": f"{sp1}_vs_{sp2}", "auc": auc_val})

    return pd.DataFrame(rows)


def _sample_quantiles(matrix: np.ndarray, q: float) -> np.ndarray:
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for idx in range(matrix.shape[0]):
        row = matrix[idx, :]
        valid = row[np.isfinite(row)]
        if valid.size > 0:
            out[idx] = float(np.quantile(valid, q))
    return out


def _median_abs_deviation(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    center = float(np.median(valid))
    return float(np.median(np.abs(valid - center)))


def _rle_sample_mad(matrix: np.ndarray) -> np.ndarray:
    feature_medians = np.nanmedian(matrix, axis=0)
    rle = matrix - feature_medians[None, :]

    sample_mads = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for idx in range(rle.shape[0]):
        row = rle[idx, :]
        valid = row[np.isfinite(row)]
        if valid.size < 5:
            continue
        sample_mads[idx] = _median_abs_deviation(valid)
    return sample_mads


def compute_distribution_metrics(
    matrix: np.ndarray,
    *,
    max_wasserstein_pairs: int = 512,
    random_seed: int = 42,
) -> dict[str, float]:
    """Compute distribution-alignment metrics for one normalized matrix."""
    x = np.asarray(matrix, dtype=np.float64)
    n_samples, n_features = x.shape
    finite_mask = np.isfinite(x)

    sample_medians = _sample_quantiles(x, 0.5)
    q75 = _sample_quantiles(x, 0.75)
    q25 = _sample_quantiles(x, 0.25)
    sample_iqr = q75 - q25

    rle_sample_mad = _rle_sample_mad(x)

    pair_indices = list(combinations(range(n_samples), 2))
    if len(pair_indices) > max_wasserstein_pairs:
        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(len(pair_indices), size=max_wasserstein_pairs, replace=False)
        pair_indices = [pair_indices[idx] for idx in chosen.tolist()]

    wasserstein_values: list[float] = []
    for i, j in pair_indices:
        xi = x[i, :]
        xj = x[j, :]
        vi = xi[np.isfinite(xi)]
        vj = xj[np.isfinite(xj)]
        if vi.size < 20 or vj.size < 20:
            continue
        wasserstein_values.append(float(wasserstein_distance(vi, vj)))

    iqr_valid = sample_iqr[np.isfinite(sample_iqr) & (sample_iqr > 0)]
    if iqr_valid.size >= 2:
        iqr_dispersion_cv = float(np.std(iqr_valid, ddof=1) / np.mean(iqr_valid))
    else:
        iqr_dispersion_cv = float("nan")

    return {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "coverage_ratio": float(np.sum(finite_mask) / finite_mask.size),
        "feature_quantified_ratio": float(np.mean(np.any(finite_mask, axis=0))),
        "sample_median_mad": _median_abs_deviation(sample_medians),
        "sample_iqr_cv": iqr_dispersion_cv,
        "rle_mad_median": float(np.nanmedian(rle_sample_mad)),
        "pairwise_wasserstein_median": (
            float(np.nanmedian(np.asarray(wasserstein_values, dtype=np.float64)))
            if wasserstein_values
            else float("nan")
        ),
    }


def _group_indices(groups: Sequence[str]) -> dict[str, np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for idx, grp in enumerate(groups):
        mapping.setdefault(str(grp), []).append(idx)
    return {key: np.asarray(idxs, dtype=int) for key, idxs in mapping.items()}


def compute_group_metrics(
    matrix: np.ndarray,
    groups: Sequence[str] | None,
) -> dict[str, float]:
    """Compute group-aware metrics when replicate labels are available."""
    if groups is None:
        return {
            "group_count": float("nan"),
            "group_min_reps": float("nan"),
            "within_group_sd_median": float("nan"),
            "group_eta2_median": float("nan"),
        }

    x = np.asarray(matrix, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if g.shape[0] != x.shape[0]:
        raise ValueError(
            f"Group label count ({g.shape[0]}) does not match sample count ({x.shape[0]})."
        )

    idx_map = _group_indices([str(v) for v in g.tolist()])
    group_sizes = {k: int(v.size) for k, v in idx_map.items()}
    valid_groups = {k: v for k, v in idx_map.items() if v.size >= 2}

    if len(valid_groups) < 2:
        return {
            "group_count": float(len(idx_map)),
            "group_min_reps": float(min(group_sizes.values())) if group_sizes else float("nan"),
            "within_group_sd_median": float("nan"),
            "group_eta2_median": float("nan"),
        }

    per_group_medians: list[float] = []
    for idxs in valid_groups.values():
        block = x[idxs, :]
        finite_counts = np.sum(np.isfinite(block), axis=0)
        valid = finite_counts >= 2
        if not np.any(valid):
            continue
        sd_vals = np.nanstd(block[:, valid], axis=0, ddof=1)
        if np.isfinite(sd_vals).any():
            per_group_medians.append(float(np.nanmedian(sd_vals)))

    eta2_values: list[float] = []
    for feat_idx in range(x.shape[1]):
        col = x[:, feat_idx]
        finite = np.isfinite(col)
        if np.sum(finite) < 4:
            continue

        vals = col[finite]
        groups_f = g[finite]
        uniq, counts = np.unique(groups_f, return_counts=True)
        if uniq.size < 2 or np.min(counts) < 2:
            continue

        grand = float(np.mean(vals))
        ss_total = float(np.sum((vals - grand) ** 2))
        if ss_total <= 0:
            continue

        ss_between = 0.0
        for grp in uniq:
            grp_vals = vals[groups_f == grp]
            ss_between += grp_vals.size * float((np.mean(grp_vals) - grand) ** 2)

        eta2_values.append(ss_between / ss_total)

    return {
        "group_count": float(len(idx_map)),
        "group_min_reps": float(min(group_sizes.values())) if group_sizes else float("nan"),
        "within_group_sd_median": (
            float(np.nanmedian(np.asarray(per_group_medians, dtype=np.float64)))
            if per_group_medians
            else float("nan")
        ),
        "group_eta2_median": (
            float(np.nanmedian(np.asarray(eta2_values, dtype=np.float64)))
            if eta2_values
            else float("nan")
        ),
    }


def compute_ratio_metrics(
    matrix: np.ndarray,
    protein_ids: Sequence[str],
    groups: Sequence[str] | None,
    *,
    group_a: str,
    group_b: str,
    expected_log2fc: Mapping[str, float],
    background_species: str = BACKGROUND_SPECIES,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute ratio accuracy metrics against known species-level ground truth."""
    if groups is None:
        return (
            {
                "ratio_n_quantified": float("nan"),
                "ratio_mae": float("nan"),
                "ratio_rmse": float("nan"),
                "ratio_bias": float("nan"),
                "ratio_pairwise_auc_mean": float("nan"),
                "ratio_changed_vs_bg_auc": float("nan"),
            },
            pd.DataFrame(),
        )

    x = np.asarray(matrix, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    idx_a = np.where(g == group_a)[0]
    idx_b = np.where(g == group_b)[0]
    if idx_a.size < 2 or idx_b.size < 2:
        return (
            {
                "ratio_n_quantified": float("nan"),
                "ratio_mae": float("nan"),
                "ratio_rmse": float("nan"),
                "ratio_bias": float("nan"),
                "ratio_pairwise_auc_mean": float("nan"),
                "ratio_changed_vs_bg_auc": float("nan"),
            },
            pd.DataFrame(),
        )

    mean_a = _nanmean_columns(x[idx_a, :])
    mean_b = _nanmean_columns(x[idx_b, :])
    log2_fc = mean_a - mean_b

    species = [infer_species_from_protein_id(pid) for pid in protein_ids]
    ratio_df = pd.DataFrame(
        {
            "protein_id": list(protein_ids),
            "species": species,
            "log2_fc_ab": log2_fc,
        }
    )
    ratio_df["expected_log2_fc_ab"] = ratio_df["species"].map(expected_log2fc)
    ratio_df = ratio_df[ratio_df["species"].isin(expected_log2fc.keys())].copy()
    ratio_df["is_quantified"] = np.isfinite(ratio_df["log2_fc_ab"])
    quantified = ratio_df[ratio_df["is_quantified"]].copy()
    if quantified.empty:
        return (
            {
                "ratio_n_quantified": 0.0,
                "ratio_mae": float("nan"),
                "ratio_rmse": float("nan"),
                "ratio_bias": float("nan"),
                "ratio_pairwise_auc_mean": float("nan"),
                "ratio_changed_vs_bg_auc": float("nan"),
            },
            ratio_df,
        )

    quantified["error"] = quantified["log2_fc_ab"] - quantified["expected_log2_fc_ab"]
    abs_err = np.abs(quantified["error"].to_numpy(dtype=np.float64))
    sq_err = np.square(quantified["error"].to_numpy(dtype=np.float64))

    pairwise_auc = _compute_pairwise_auc(quantified, expected_log2fc)
    pair_auc_values = pairwise_auc["auc"].to_numpy(dtype=np.float64)
    pairwise_auc_mean = (
        float(np.nanmean(pair_auc_values)) if np.isfinite(pair_auc_values).any() else float("nan")
    )

    labels_changed = (quantified["species"] != background_species).astype(int).to_numpy()
    bg_expected = float(expected_log2fc[background_species])
    changed_scores = np.abs(quantified["log2_fc_ab"].to_numpy(dtype=np.float64) - bg_expected)
    changed_auc = _safe_auc(labels_changed, changed_scores)

    return (
        {
            "ratio_n_quantified": float(quantified.shape[0]),
            "ratio_mae": float(np.nanmean(abs_err)),
            "ratio_rmse": float(np.sqrt(np.nanmean(sq_err))),
            "ratio_bias": float(np.nanmean(quantified["error"].to_numpy(dtype=np.float64))),
            "ratio_pairwise_auc_mean": pairwise_auc_mean,
            "ratio_changed_vs_bg_auc": changed_auc,
        },
        ratio_df,
    )


def guess_groups_from_sample_ids(sample_ids: Sequence[str]) -> list[str] | None:
    """Infer replicate group labels from sample IDs with conservative rules."""
    tokens_by_sample: list[list[str]] = []
    for sample_id in sample_ids:
        parts = [tok for tok in str(sample_id).replace("-", "_").split("_") if tok]
        tokens_by_sample.append(parts)

    counter: Counter[str] = Counter()
    for parts in tokens_by_sample:
        for token in parts:
            if len(token) >= 2:
                counter[token] += 1

    candidates = [
        token
        for token, count in counter.items()
        if count >= 2 and (token.startswith("S") or token.lower().startswith("group"))
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda tok: (-counter[tok], tok))
    selected = candidates[0]

    groups: list[str] = []
    for parts in tokens_by_sample:
        grp = selected if selected in parts else "UNKNOWN"
        groups.append(grp)

    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return None

    counts = Counter(groups)
    non_unknown = [count for key, count in counts.items() if key != "UNKNOWN"]
    if not non_unknown or min(non_unknown) < 2:
        return None

    return groups
