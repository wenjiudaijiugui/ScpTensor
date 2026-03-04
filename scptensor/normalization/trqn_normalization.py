"""Tail-Robust Quantile Normalization (TRQN) for protein-level matrices.

TRQN uses rank-invariant features to apply a mean/median-balanced variant of
quantile normalization on the selected subset, while the remaining features are
quantile-normalized with the standard procedure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer

from .base import create_result_layer, ensure_dense, log_operation, validate_assay_and_layer
from .quantile_normalization import _quantile_normalize_rows


def _rank_invariance_frequency(feature_sample: np.ndarray) -> np.ndarray:
    """Compute per-feature rank-invariance frequency.

    Parameters
    ----------
    feature_sample : np.ndarray
        Matrix of shape (n_features, n_samples).

    Returns
    -------
    np.ndarray
        Rank-invariance frequencies in [0, 1], one value per feature.
    """
    n_features, n_samples = feature_sample.shape
    ranks = np.full((n_features, n_samples), np.nan, dtype=float)

    for j in range(n_samples):
        col = feature_sample[:, j]
        valid = np.isfinite(col)
        if valid.sum() == 0:
            continue
        ranks[valid, j] = rankdata(col[valid], method="average")

    frequencies = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        row = ranks[i, :]
        valid = np.isfinite(row)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        _, counts = np.unique(row[valid], return_counts=True)
        frequencies[i] = counts.max() / n_valid

    return frequencies


def _validate_feature_indices(
    feature_indices: Sequence[int],
    n_features: int,
) -> np.ndarray:
    """Validate and normalize user-provided feature index list."""
    idx = np.asarray(feature_indices, dtype=int)
    if idx.size == 0:
        return idx

    invalid = idx[(idx < 0) | (idx >= n_features)]
    if invalid.size > 0:
        raise ScpValueError(
            f"feature_indices contains out-of-range indices: {invalid.tolist()}. "
            f"Valid range is [0, {n_features - 1}].",
            parameter="feature_indices",
            value=invalid.tolist(),
        )

    return np.unique(idx)


def norm_trqn(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "trqn_norm",
    low_thr: float = 0.5,
    balance_stat: str = "median",
    feature_indices: Sequence[int] | None = None,
) -> ScpContainer:
    """Apply Tail-Robust Quantile Normalization (TRQN) on protein matrix.

    Parameters
    ----------
    container : ScpContainer
        Input container with protein-level matrix.
    assay_name : str, default="protein"
        Target assay name.
    source_layer : str, default="raw"
        Source layer name.
    new_layer_name : str, default="trqn_norm"
        Destination layer name.
    low_thr : float, default=0.5
        Rank-invariance threshold in (0, 1]. Features with frequency >= low_thr
        are selected as rank-invariant if feature_indices is None.
    balance_stat : {"median", "mean"}, default="median"
        Statistic used to compute per-feature offset for balanced quantile step.
    feature_indices : Sequence[int] | None, default=None
        Optional explicit feature indices to use as rank-invariant set.
        If provided, automatic threshold-based detection is skipped.

    Returns
    -------
    ScpContainer
        Container with TRQN-normalized layer added.
    """
    if not (0.0 < low_thr <= 1.0):
        raise ScpValueError(
            f"low_thr must be in (0, 1], got {low_thr}.",
            parameter="low_thr",
            value=low_thr,
        )

    balance_key = balance_stat.strip().lower()
    if balance_key not in {"median", "mean"}:
        raise ScpValueError(
            f"balance_stat must be 'median' or 'mean', got '{balance_stat}'.",
            parameter="balance_stat",
            value=balance_stat,
        )

    assay, input_layer = validate_assay_and_layer(container, assay_name, source_layer)
    x_sample_feature = ensure_dense(input_layer.X).astype(float, copy=True)
    feature_sample = x_sample_feature.T
    n_features = feature_sample.shape[0]

    # Baseline quantile normalization for all features.
    qn_feature_sample = _quantile_normalize_rows(feature_sample.T).T

    if feature_indices is None:
        ri_freq = _rank_invariance_frequency(feature_sample)
        selected_idx = np.where(ri_freq >= low_thr)[0]
    else:
        ri_freq = np.full(n_features, np.nan, dtype=float)
        selected_idx = _validate_feature_indices(feature_indices, n_features)

    if selected_idx.size > 0:
        subset = feature_sample[selected_idx, :]
        if balance_key == "median":
            feature_offsets = np.nanmedian(subset, axis=1)
        else:
            feature_offsets = np.nanmean(subset, axis=1)

        balanced_subset = subset - feature_offsets[:, None]
        balanced_norm = _quantile_normalize_rows(balanced_subset.T).T + feature_offsets[:, None]
        qn_feature_sample[selected_idx, :] = balanced_norm

    x_trqn = qn_feature_sample.T
    new_matrix = create_result_layer(x_trqn, input_layer)
    assay.add_layer(new_layer_name, new_matrix)

    log_operation(
        container,
        action="normalization_trqn",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "low_thr": low_thr,
            "balance_stat": balance_key,
            "selected_features": int(selected_idx.size),
            "feature_indices_provided": feature_indices is not None,
        },
        description=f"TRQN normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container


__all__ = ["norm_trqn"]
