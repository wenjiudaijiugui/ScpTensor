"""Linear-model batch correction (limma-style) for DIA proteomics matrices.

This method removes estimated batch coefficients from a feature matrix while
retaining intercept and optional biological covariate effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import scipy.linalg as la
import scipy.sparse as sp

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    preserve_sparsity,
    register_integrate_method,
    to_dense_array,
    validate_batch_integration_params,
    validate_layer_params,
)


@register_integrate_method("limma", integration_level="matrix", recommended_for_de=True)
def integrate_limma(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "limma",
    covariates: Sequence[str] | None = None,
    reference_batch: str | None = None,
) -> ScpContainer:
    """Remove batch effects using a limma-style linear model.

    Parameters
    ----------
    container : ScpContainer
        Input container with batch labels in ``obs``.
    batch_key : str
        Column in ``obs`` with batch labels.
    assay_name : str, default="protein"
        Assay name (supports ``protein``/``proteins`` aliasing).
    base_layer : str, default="raw"
        Source layer for correction.
    new_layer_name : str | None, default="limma"
        Output layer name.
    covariates : Sequence[str] | None, default=None
        Biological covariates to preserve in the fitted design.
    reference_batch : str | None, default=None
        Optional ScpTensor extension that keeps the chosen batch unchanged after
        correction. If None, batch effects are removed to limma's centered
        default rather than anchoring to the first observed batch.
    """
    assay, layer = validate_layer_params(container, assay_name, base_layer)
    obs_df, batches, unique_batches, _ = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2, min_samples_per_batch=2
    )

    input_was_sparse = sp.issparse(layer.X)
    x_dense = to_dense_array(layer.X, copy=not input_was_sparse)
    if np.isinf(x_dense).any():
        raise ScpValueError(
            "Limma integration supports NaN missing values but not Inf/-Inf values. "
            "Please replace or filter infinite values before batch correction.",
            parameter="X",
        )

    design_matrix, design_cols, batch_idx, ref_batch, batch_reference_row = (
        _build_limma_design_matrix(
            obs_df,
            batches,
            unique_batches,
            covariates=covariates,
            reference_batch=reference_batch,
        )
    )
    rank = np.linalg.matrix_rank(design_matrix)
    if rank < design_matrix.shape[1]:
        cols = ", ".join(design_cols)
        raise ValueError(
            f"Design matrix is rank deficient (rank={rank}, cols={design_matrix.shape[1]}). "
            "Batch and covariates are likely confounded. "
            f"Design columns: [{cols}]"
        )

    x_corrected = _remove_batch_effect_with_missing(
        x_dense,
        design_matrix,
        batch_idx,
        batch_reference_row=batch_reference_row,
    )

    x_out = preserve_sparsity(x_corrected, input_was_sparse)
    add_integrated_layer(assay, new_layer_name or "limma", x_out, layer)

    return log_integration_operation(
        container,
        action="integration_limma",
        method_name="limma",
        params={
            "batch_key": batch_key,
            "covariates": list(covariates) if covariates else None,
            "reference_batch": ref_batch,
            "n_batch_terms": len(batch_idx),
        },
        description=f"Limma-style linear batch correction (reference_batch={ref_batch}).",
    )


def _build_limma_design_matrix(
    obs_df: pl.DataFrame,
    batches: np.ndarray,
    unique_batches: np.ndarray,
    *,
    covariates: Sequence[str] | None,
    reference_batch: str | None,
) -> tuple[np.ndarray, list[str], list[int], str | None, np.ndarray | None]:
    """Build preserve-design plus limma-style sum-coded batch design."""
    n_samples = len(batches)
    batch_labels = [str(b) for b in unique_batches]
    ref = str(reference_batch) if reference_batch is not None else None
    if ref is not None and ref not in batch_labels:
        raise ScpValueError(
            f"reference_batch '{ref}' not found. Available batches: {batch_labels}",
            parameter="reference_batch",
            value=reference_batch,
        )

    covar_design = _build_covariate_design(obs_df, covariates, n_samples)
    design_matrix = covar_design.to_numpy().astype(float)
    if not np.isfinite(design_matrix).all():
        raise ScpValueError(
            "Covariates generated a design matrix containing missing or non-finite values. "
            "Limma-style batch correction requires fully observed covariates.",
            parameter="covariates",
            value=list(covariates) if covariates is not None else None,
        )

    batch_design, batch_cols, batch_reference_row = _build_batch_effect_design(
        batches=batches,
        batch_labels=batch_labels,
        reference_batch=ref,
    )
    if batch_design.shape[1] == 0:
        return design_matrix, covar_design.columns, [], ref, batch_reference_row

    combined = np.column_stack([design_matrix, batch_design])
    batch_idx = list(range(design_matrix.shape[1], combined.shape[1]))
    return combined, covar_design.columns + batch_cols, batch_idx, ref, batch_reference_row


def _build_batch_effect_design(
    *,
    batches: np.ndarray,
    batch_labels: list[str],
    reference_batch: str | None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Construct limma-style sum-coded batch contrasts and optional reference row."""
    n_batches = len(batch_labels)
    n_samples = len(batches)
    if n_batches <= 1:
        return np.empty((n_samples, 0), dtype=float), [], None

    contrast = np.zeros((n_batches, n_batches - 1), dtype=float)
    contrast[: n_batches - 1, :] = np.eye(n_batches - 1, dtype=float)
    contrast[-1, :] = -1.0

    batch_to_index = {label: idx for idx, label in enumerate(batch_labels)}
    batch_indices = np.array([batch_to_index[str(batch)] for batch in batches], dtype=int)
    batch_design = contrast[batch_indices]
    batch_cols = [f"batch_contr_{label}" for label in batch_labels[:-1]]
    reference_row = None if reference_batch is None else contrast[batch_to_index[reference_batch]]
    return batch_design, batch_cols, reference_row


def _build_covariate_design(
    obs_df: pl.DataFrame,
    covariates: Sequence[str] | None,
    n_samples: int,
) -> pl.DataFrame:
    """Build intercept + covariate design with dummy encoding for categoricals."""
    if not covariates:
        return pl.DataFrame({"intercept": np.ones(n_samples)})

    missing = [col for col in covariates if col not in obs_df.columns]
    if missing:
        raise ScpValueError(
            f"covariates contain missing columns: {missing}. Available: {obs_df.columns}",
            parameter="covariates",
            value=list(covariates),
        )

    covar_df = obs_df.select(covariates)
    cat_cols = [
        c for c, t in covar_df.schema.items() if t in (pl.String, pl.Categorical, pl.Object)
    ]
    enc = covar_df.to_dummies(columns=cat_cols, drop_first=True) if cat_cols else covar_df
    return pl.concat([pl.DataFrame({"intercept": np.ones(n_samples)}), enc], how="horizontal")


def _remove_batch_effect_with_missing(
    X: np.ndarray,
    design_matrix: np.ndarray,
    batch_idx: list[int],
    *,
    batch_reference_row: np.ndarray | None,
) -> np.ndarray:
    """Fit per-feature linear models on observed samples and preserve NaN."""
    if not batch_idx:
        return X.copy()

    out = X.copy()
    for feature_idx in range(X.shape[1]):
        y = X[:, feature_idx]
        valid = ~np.isnan(y)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            continue

        design_valid = design_matrix[valid]
        rank = np.linalg.matrix_rank(design_valid)
        if rank == 0:
            continue

        estimable = _estimable_columns(design_valid, rank)
        coef = np.zeros(design_matrix.shape[1], dtype=float)
        coef[estimable] = np.linalg.lstsq(design_valid[:, estimable], y[valid], rcond=None)[0]
        batch_effect = design_valid[:, batch_idx] @ coef[batch_idx]
        ref_effect = 0.0
        if batch_reference_row is not None:
            ref_effect = float(batch_reference_row @ coef[batch_idx])
        out[np.where(valid)[0], feature_idx] = y[valid] - batch_effect + ref_effect

    return out


def _estimable_columns(design_valid: np.ndarray, rank: int) -> np.ndarray:
    """Return indices of estimable columns using pivoted QR decomposition."""
    if rank >= design_valid.shape[1]:
        return np.arange(design_valid.shape[1], dtype=int)
    _, _, pivots = la.qr(design_valid, mode="economic", pivoting=True)
    return np.sort(np.asarray(pivots[:rank], dtype=int))


__all__ = ["integrate_limma"]
