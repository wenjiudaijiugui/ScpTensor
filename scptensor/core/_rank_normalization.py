"""Shared internal kernels for quantile-style normalization routines."""

from __future__ import annotations

import numpy as np

from scptensor.core.exceptions import ValidationError


def map_reference_by_average_rank(
    row_valid: np.ndarray,
    reference_dist: np.ndarray,
) -> np.ndarray:
    """Map one valid row to the shared reference using average-rank ties."""
    n_valid = row_valid.size
    order = np.argsort(row_valid, kind="mergesort")
    mapped = np.empty(n_valid, dtype=float)

    if n_valid <= 1:
        mapped[order] = reference_dist[:n_valid]
        return mapped

    sorted_vals = row_valid[order]
    if not np.any(sorted_vals[1:] == sorted_vals[:-1]):
        mapped[order] = reference_dist[:n_valid]
        return mapped

    group_starts_mask = np.empty(n_valid, dtype=bool)
    group_starts_mask[0] = True
    group_starts_mask[1:] = sorted_vals[1:] != sorted_vals[:-1]

    group_starts = np.flatnonzero(group_starts_mask)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:] - 1
    group_ends[-1] = n_valid - 1

    avg_ranks = 0.5 * (group_starts + group_ends)
    left = avg_ranks.astype(np.intp)
    right = np.ceil(avg_ranks).astype(np.intp)
    weights = avg_ranks - left
    group_values = reference_dist[left] + weights * (reference_dist[right] - reference_dist[left])

    normalized_sorted = np.repeat(group_values, group_ends - group_starts + 1)
    mapped[order] = normalized_sorted
    return mapped


def quantile_normalize_dense_rows(x_dense: np.ndarray) -> np.ndarray:
    """Quantile-normalize a dense matrix row-wise."""
    x_dense = np.asarray(x_dense, dtype=float)
    if np.any(np.isinf(x_dense)):
        raise ValidationError(
            "Quantile normalization does not accept Inf values. "
            "Use NaN for missing entries or clean infinite intensities first.",
            field="X",
        )

    n_rows, n_cols = x_dense.shape
    rank_sums = np.zeros(n_cols, dtype=float)
    valid_counts = np.zeros(n_cols, dtype=np.int32)

    for row_idx in range(n_rows):
        row = x_dense[row_idx, :]
        row_valid = row[~np.isnan(row)]
        n_valid = row_valid.size

        if n_valid == 0:
            continue

        sorted_row = np.sort(row_valid)
        rank_sums[:n_valid] += sorted_row
        valid_counts[:n_valid] += 1

    reference_dist = np.divide(
        rank_sums,
        valid_counts,
        out=np.zeros_like(rank_sums, dtype=float),
        where=valid_counts > 0,
    )
    normalized = np.empty(x_dense.shape, dtype=float)

    for row_idx in range(n_rows):
        row = x_dense[row_idx, :]
        non_nan = ~np.isnan(row)

        if not np.any(non_nan):
            normalized[row_idx, :] = np.nan
            continue

        row_valid = row[non_nan]
        normalized[row_idx, non_nan] = map_reference_by_average_rank(row_valid, reference_dist)
        normalized[row_idx, ~non_nan] = np.nan

    return normalized


def rank_invariance_frequency(
    feature_sample: np.ndarray,
    qn_feature_sample: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-feature rank-invariance frequency."""
    n_features, n_samples = feature_sample.shape
    if n_features == 0 or n_samples == 0:
        return np.zeros(n_features, dtype=float)

    if qn_feature_sample is None:
        qn_feature_sample = quantile_normalize_dense_rows(feature_sample.T).T

    rank_positions = np.zeros((n_features, n_samples), dtype=np.int32)
    for sample_idx in range(n_samples):
        col = qn_feature_sample[:, sample_idx]
        valid_idx = np.where(np.isfinite(col))[0]
        if valid_idx.size == 0:
            continue

        order_valid = valid_idx[np.argsort(-col[valid_idx], kind="mergesort")]
        rank_positions[order_valid, sample_idx] = np.arange(1, valid_idx.size + 1, dtype=np.int32)

    frequencies = np.zeros(n_features, dtype=float)
    for feature_idx in range(n_features):
        row_ranks = rank_positions[feature_idx, :]
        positive = row_ranks > 0
        n_valid = int(np.sum(positive))
        if n_valid == 0:
            continue

        counts = np.bincount(row_ranks[positive])
        frequencies[feature_idx] = counts.max() / n_valid

    return frequencies


__all__ = [
    "map_reference_by_average_rank",
    "quantile_normalize_dense_rows",
    "rank_invariance_frequency",
]
