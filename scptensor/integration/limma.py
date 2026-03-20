"""Linear-model batch correction (limma-style) for DIA proteomics matrices.

This method removes estimated batch coefficients from a feature matrix while
retaining intercept and optional biological covariate effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
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
        Reference batch with zero batch coefficient. If None, first observed
        batch is used.
    """
    assay, layer = validate_layer_params(container, assay_name, base_layer)
    obs_df, batches, unique_batches, _ = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2, min_samples_per_batch=2
    )

    input_was_sparse = sp.issparse(layer.X)
    x_dense = to_dense_array(layer.X, copy=not input_was_sparse)

    design_matrix, design_cols, batch_term_cols, ref_batch = _build_limma_design_matrix(
        obs_df,
        batches,
        unique_batches,
        covariates=covariates,
        reference_batch=reference_batch,
    )
    rank = np.linalg.matrix_rank(design_matrix)
    if rank < design_matrix.shape[1]:
        cols = ", ".join(design_cols)
        raise ValueError(
            f"Design matrix is rank deficient (rank={rank}, cols={design_matrix.shape[1]}). "
            "Batch and covariates are likely confounded. "
            f"Design columns: [{cols}]"
        )

    batch_idx = [i for i, col in enumerate(design_cols) if col in set(batch_term_cols)]
    x_corrected = _remove_batch_effect_with_missing(x_dense, design_matrix, batch_idx)

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
            "n_batch_terms": len(batch_term_cols),
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
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Build design matrix: intercept + covariates + batch dummies."""
    n_samples = len(batches)
    ref = str(reference_batch) if reference_batch is not None else str(unique_batches[0])
    batch_labels = [str(b) for b in unique_batches]
    if ref not in batch_labels:
        raise ScpValueError(
            f"reference_batch '{ref}' not found. Available batches: {batch_labels}",
            parameter="reference_batch",
            value=reference_batch,
        )

    covar_design = _build_covariate_design(obs_df, covariates, n_samples)
    batch_terms: dict[str, np.ndarray] = {}
    for b in batch_labels:
        if b == ref:
            continue
        batch_terms[f"batch_{b}"] = (batches.astype(str) == b).astype(float)

    batch_df = pl.DataFrame(batch_terms) if batch_terms else pl.DataFrame()
    design_df = pl.concat([covar_design, batch_df], how="horizontal")
    return (
        design_df.to_numpy().astype(float),
        design_df.columns,
        list(batch_terms.keys()),
        ref,
    )


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
) -> np.ndarray:
    """Fit per-feature linear models on observed samples and preserve NaN."""
    if not batch_idx:
        return X.copy()

    out = X.copy()
    n_terms = design_matrix.shape[1]
    for feature_idx in range(X.shape[1]):
        y = X[:, feature_idx]
        valid = np.isfinite(y)
        if int(np.sum(valid)) <= len(batch_idx):
            continue

        design_valid = design_matrix[valid]
        if design_valid.shape[0] < n_terms:
            continue
        if np.linalg.matrix_rank(design_valid) < n_terms:
            continue

        coef = np.linalg.lstsq(design_valid, y[valid], rcond=None)[0]
        batch_effect = design_valid[:, batch_idx] @ coef[batch_idx]
        out[np.where(valid)[0], feature_idx] = y[valid] - batch_effect

    return out


__all__ = ["integrate_limma"]
