"""Shared utilities for differential expression modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats

if TYPE_CHECKING:
    pass


def isna(arr: np.ndarray) -> np.ndarray:
    """Check for NaN/null values in an array.

    Handles both numeric and string arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Boolean mask indicating NA values.
    """
    if arr.dtype.kind in {"U", "S", "O"}:
        return (arr == None) | (arr == "NaN") | (arr == "nan") | (arr == "NA") | (arr == "")  # noqa: E711
    return np.isnan(arr)


def extract_group_indices(
    groups: np.ndarray,
    group_names: list | tuple,
    min_samples_per_group: int = 2,
) -> dict:
    """Extract indices for each group.

    Parameters
    ----------
    groups : np.ndarray
        Array of group labels.
    group_names : list or tuple
        Names of groups to extract.
    min_samples_per_group : int, default=2
        Minimum samples required per group.

    Returns
    -------
    dict
        Mapping from group name to indices array.

    Raises
    ------
    ValueError
        If insufficient samples in any group.
    """
    from scptensor.core.exceptions import ValidationError

    indices = {}
    for name in group_names:
        mask = groups == name
        n = np.sum(mask)
        if n < min_samples_per_group:
            raise ValidationError(
                f"Group '{name}' has only {n} samples, minimum {min_samples_per_group} required",
                field=name,
            )
        indices[name] = np.where(mask)[0]
    return indices


def extract_group_indices_from_obs(
    obs: pl.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract group indices from obs DataFrame.

    Parameters
    ----------
    obs : pl.DataFrame
        Sample metadata.
    group_col : str
        Column containing group labels.
    group1 : str
        First group name.
    group2 : str
        Second group name.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Indices for group1 and group2.
    """
    groups = obs[group_col].to_numpy()
    idx1 = np.where(groups == group1)[0]
    idx2 = np.where(groups == group2)[0]
    return idx1, idx2


def handle_missing_values(
    X: np.ndarray,
    strategy: str = "ignore",
) -> tuple[np.ndarray, np.ndarray]:
    """Handle missing values in data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features).
    strategy : str, default="ignore"
        Strategy: "ignore" (keep NaN), "mean" (impute), "drop" (remove rows).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Processed matrix and valid mask.
    """
    if strategy == "ignore":
        return X, ~np.isnan(X).all(axis=0)
    elif strategy == "mean":
        X_proc = X.copy()
        col_means = np.nanmean(X_proc, axis=0)
        for j in range(X_proc.shape[1]):
            mask = np.isnan(X_proc[:, j])
            X_proc[mask, j] = col_means[j]
        return X_proc, np.ones(X.shape[1], dtype=bool)
    elif strategy == "drop":
        valid_mask = ~np.isnan(X).any(axis=0)
        return X[:, valid_mask], valid_mask
    else:
        return X, np.ones(X.shape[1], dtype=bool)


def log2_fold_change(
    group1: np.ndarray,
    group2: np.ndarray,
    offset: float = 1.0,
) -> float:
    """Calculate log2 fold change between two groups.

    Parameters
    ----------
    group1 : np.ndarray
        Values for group 1.
    group2 : np.ndarray
        Values for group 2.
    offset : float, default=1.0
        Offset to avoid log(0).

    Returns
    -------
    float
        Log2 fold change (group1 / group2).
    """
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]

    if len(g1) == 0 or len(g2) == 0:
        return np.nan

    mean1 = np.mean(g1) + offset
    mean2 = np.mean(g2) + offset

    if mean2 <= 0:
        return np.nan

    return np.log2(mean1 / mean2)


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Calculate Cohen's d effect size.

    Parameters
    ----------
    group1 : np.ndarray
        Values for group 1.
    group2 : np.ndarray
        Values for group 2.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    n1, n2 = len(g1), len(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(g1) - np.mean(g2)) / pooled_std


def validate_pairing(
    obs: pl.DataFrame,
    group_col: str,
    pair_id_col: str,
    group1: str,
    group2: str,
) -> list:
    """Validate paired sample structure.

    Parameters
    ----------
    obs : pl.DataFrame
        Sample metadata.
    group_col : str
        Column containing group labels.
    pair_id_col : str
        Column containing pair identifiers.
    group1 : str
        First group name.
    group2 : str
        Second group name.

    Returns
    -------
    list
        List of valid pair tuples (idx1, idx2).

    Raises
    ------
    ValueError
        If pairing structure is invalid.
    """
    from scptensor.core.exceptions import ValidationError

    if pair_id_col not in obs.columns:
        raise ValidationError(
            f"Pair ID column '{pair_id_col}' not found in obs.",
            field=pair_id_col,
        )

    pairs = []
    pair_ids = obs[pair_id_col].unique().to_list()

    for pid in pair_ids:
        pair_mask = obs[pair_id_col] == pid
        pair_obs = obs.filter(pair_mask)

        groups_in_pair = pair_obs[group_col].unique().to_list()

        if group1 in groups_in_pair and group2 in groups_in_pair:
            idx1 = obs.filter((obs[pair_id_col] == pid) & (obs[group_col] == group1)).row(0)[0]
            idx2_row = obs.filter((obs[pair_id_col] == pid) & (obs[group_col] == group2))
            if len(idx2_row) > 0:
                idx2 = idx2_row.row(0)[0]
                pairs.append((idx1, idx2))

    return pairs


def rank_biserial_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate rank-biserial correlation effect size.

    Parameters
    ----------
    x : np.ndarray
        First group values.
    y : np.ndarray
        Second group values.

    Returns
    -------
    float
        Rank-biserial correlation (-1 to 1).
    """
    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]

    if len(x_clean) == 0 or len(y_clean) == 0:
        return np.nan

    # Use Mann-Whitney U statistic
    try:
        u_stat, _ = stats.mannwhitneyu(x_clean, y_clean, alternative="two-sided")
    except ValueError:
        return np.nan

    n1, n2 = len(x_clean), len(y_clean)
    r = 1 - (2 * u_stat) / (n1 * n2)

    return r


__all__ = [
    "isna",
    "extract_group_indices",
    "extract_group_indices_from_obs",
    "handle_missing_values",
    "log2_fold_change",
    "cohens_d",
    "validate_pairing",
    "rank_biserial_correlation",
]
