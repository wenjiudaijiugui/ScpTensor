"""Tail-Robust Quantile Normalization (TRQN) for protein-level matrices.

TRQN uses rank-invariant features to apply a mean/median-balanced variant of
quantile normalization on the selected subset, while the remaining features are
quantile-normalized with the standard procedure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer

from .base import ensure_dense, finalize_normalization_layer, validate_layer_context
from .quantile_normalization import _quantile_normalize_rows


def _rank_invariance_frequency(
    feature_sample: np.ndarray,
    qn_feature_sample: np.ndarray | None = None,
) -> np.ndarray:
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
    if n_features == 0 or n_samples == 0:
        return np.zeros(n_features, dtype=float)

    # MBQN reference computes RI frequencies after classical quantile
    # normalization and based on top-down rank positions.
    if qn_feature_sample is None:
        qn_feature_sample = _quantile_normalize_rows(feature_sample.T).T

    rank_positions = np.zeros((n_features, n_samples), dtype=np.int32)
    for j in range(n_samples):
        col = qn_feature_sample[:, j]
        valid_idx = np.where(np.isfinite(col))[0]
        if valid_idx.size == 0:
            continue

        # Descending order (top-down ranking). Missing entries remain at zero,
        # so we do not need to build an extra invalid-index concatenation.
        order_valid = valid_idx[np.argsort(-col[valid_idx], kind="mergesort")]
        rank_positions[order_valid, j] = np.arange(1, valid_idx.size + 1, dtype=np.int32)

    frequencies = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        row_ranks = rank_positions[i, :]
        positive = row_ranks > 0
        n_valid = int(np.sum(positive))
        if n_valid == 0:
            continue

        # Rank positions are positive contiguous integers per valid sample.
        # Counting directly from the assigned positions avoids re-scanning the
        # original matrix for finiteness and is faster than np.unique here.
        counts = np.bincount(row_ranks[positive])
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

    Notes
    -----
    ScpTensor AutoSelect only compares TRQN automatically on layers with
    explicit log provenance. Raw/unknown-scale layers remain limited to
    scale-weaker baselines until log transformation is recorded explicitly.
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

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer
    x_sample_feature = np.asarray(ensure_dense(input_layer.X), dtype=float)
    feature_sample = x_sample_feature.T
    n_features = feature_sample.shape[0]

    # Baseline quantile normalization for all features.
    qn_sample_feature = _quantile_normalize_rows(x_sample_feature)
    qn_feature_sample = qn_sample_feature.T

    if feature_indices is None:
        ri_freq = _rank_invariance_frequency(feature_sample, qn_feature_sample=qn_feature_sample)
        selected_idx = np.where(ri_freq >= low_thr)[0]
    else:
        ri_freq = np.full(n_features, np.nan, dtype=float)
        selected_idx = _validate_feature_indices(feature_indices, n_features)

    if selected_idx.size > 0:
        subset = np.array(feature_sample[selected_idx, :], dtype=float, copy=True)
        if balance_key == "median":
            feature_offsets = np.nanmedian(subset, axis=1)
        else:
            feature_offsets = np.nanmean(subset, axis=1)

        np.subtract(subset, feature_offsets[:, None], out=subset)
        balanced_norm = _quantile_normalize_rows(subset.T).T
        np.add(balanced_norm, feature_offsets[:, None], out=balanced_norm)
        qn_feature_sample[selected_idx, :] = balanced_norm

    x_trqn = qn_sample_feature if selected_idx.size == 0 else qn_feature_sample.T
    return finalize_normalization_layer(
        container,
        assay,
        input_layer,
        X=x_trqn,
        new_layer_name=new_layer_name,
        action="normalization_trqn",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "low_thr": low_thr,
            "balance_stat": balance_key,
            "selected_features": int(selected_idx.size),
            "feature_indices_provided": feature_indices is not None,
        },
        description=f"TRQN normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )


__all__ = ["norm_trqn"]
