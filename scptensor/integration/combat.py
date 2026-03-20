"""ComBat batch effect correction for DIA-based single-cell proteomics data.

ComBat uses empirical Bayes methods to correct for batch effects while
preserving biological signals of interest.

ScpTensor keeps this implementation on a complete-matrix contract. Official
``sva::ComBat`` contains NA-aware per-feature fitting branches, but high-missing
proteomics workflows typically need additional missing-aware wrappers to avoid
silently changing missingness semantics. In ScpTensor, ``integrate_combat``
remains an explicit manual method and is excluded from the stable
DE-oriented AutoSelect candidate set.

Algorithm
---------

ComBat models batch-specific location and scale parameters and shrinks them
towards global estimates using empirical Bayes:

.. math::

    Y_{ij}^{*} = \\frac{Y_{ij} - \\hat{\\gamma}_{\\gamma(i)} - \\hat{\\beta}_{\\gamma(i)} X_{ij}}{\\hat{\\delta}_{\\gamma(i)}}

where:
- :math:`Y_{ij}` is the expression value for feature i, sample j
- :math:`\\gamma(i)` is the batch for sample j
- :math:`\\hat{\\gamma}, \\hat{\\beta}, \\hat{\\delta}` are estimated using empirical Bayes
- :math:`X_{ij}` are covariate effects

The empirical Bayes shrinkage uses:

.. math::

    \\gamma^* = \\frac{n \\hat{\\tau}^2 \\hat{\\gamma} + \\hat{\\delta} \\bar{\\gamma}}{n \\hat{\\tau}^2 + \\hat{\\delta}}

References
----------
Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in
microarray expression data using empirical Bayes methods. Biostatistics.

Fortin, J.-P., et al. (2017). Correction of unwanted variation in
microarray data using empirical Bayes methods. bioRxiv.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import ScpValueError, ValidationError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import ScpContainer
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    register_integrate_method,
    to_dense_array,
    validate_batch_integration_params,
    validate_layer_params,
)

EbMode = Literal["parametric", "nonparametric"]


@register_integrate_method("combat", integration_level="matrix", recommended_for_de=False)
def integrate_combat(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "combat",
    covariates: Sequence[str] | None = None,
    eb_mode: EbMode = "parametric",
) -> ScpContainer:
    """Apply ComBat batch effect correction using empirical Bayes.

    Parameters
    ----------
    container : ScpContainer
        Input container with batch information
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, default="protein"
        Name of the assay to process
    base_layer : str, default="raw"
        Name of the layer to use as input
    new_layer_name : str | None, default="combat"
        Name for the new layer with corrected data
    covariates : Sequence[str] | None
        Optional list of covariate column names in obs to preserve
    eb_mode : {"parametric", "nonparametric"}, default="parametric"
        Empirical Bayes mode for ComBat shrinkage.
        ``parametric`` uses normal/inverse-gamma priors (classic ComBat).
        ``nonparametric`` uses robust, distribution-free shrinkage.

    Returns
    -------
    ScpContainer
        Container with batch-corrected layer added

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist
    LayerNotFoundError
        If the specified layer does not exist in the assay
    ScpValueError
        If batch_key not found in obs, fewer than 2 batches, or <2 samples per batch
    ValueError
        If design matrix is rank deficient

    Notes
    -----
    This implementation currently requires a complete input matrix. For
    high-missing single-cell DIA protein matrices, prefer
    :func:`scptensor.integration.integrate_limma` when preserving missing values
    is required, or perform explicit filtering/imputation before ComBat.

    Examples
    --------
    >>> container = integrate_combat(container, batch_key='batch')
    >>> container = integrate_combat(container, batch_key='batch', covariates=['condition'])
    """

    # Validate parameters
    if eb_mode not in ("parametric", "nonparametric"):
        raise ScpValueError(
            f"eb_mode must be one of ['parametric', 'nonparametric'], got '{eb_mode}'.",
            parameter="eb_mode",
            value=eb_mode,
        )

    assay, layer = validate_layer_params(container, assay_name, base_layer)
    obs_df, batches, unique_batches, batch_counts = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2, min_samples_per_batch=2
    )

    # Get data
    input_was_sparse = is_sparse_matrix(layer.X)
    X_dense = to_dense_array(layer.X, copy=not input_was_sparse)

    if np.isnan(X_dense).any():
        raise ValidationError(
            "ScpTensor's ComBat implementation currently requires a complete matrix "
            "(no NaN values). For high-missing single-cell DIA protein matrices, "
            "prefer integrate_limma() to preserve missing values, or perform explicit "
            "filtering/imputation before ComBat."
        )

    # Transpose: (n_features, n_samples) for feature-wise operations
    dat = X_dense.T
    n_sample = dat.shape[1]

    # Build design matrices
    design_matrix, design_columns, n_batch = _build_design_matrices(
        obs_df, batches, unique_batches, covariates, n_sample
    )

    # Check for rank deficiency
    rank_design = np.linalg.matrix_rank(design_matrix)
    if rank_design < design_matrix.shape[1]:
        col_text = ", ".join(design_columns)
        raise ValueError(
            f"Design matrix is rank deficient (rank={rank_design}, cols={design_matrix.shape[1]}). "
            "Batch and biological covariates are likely confounded. "
            f"Design columns: [{col_text}]"
        )

    # Fit ComBat model
    X_corrected = _fit_combat(
        dat,
        design_matrix,
        n_batch,
        batches,
        unique_batches,
        n_sample,
        eb_mode=eb_mode,
    )
    X_corrected = X_corrected.T

    # Preserve sparsity if appropriate
    if input_was_sparse:
        sparsity_ratio = 1.0 - (np.count_nonzero(X_corrected) / X_corrected.size)
        if sparsity_ratio > 0.5:
            X_corrected = sp.csr_matrix(X_corrected)

    # Create new layer and log
    add_integrated_layer(assay, new_layer_name or "combat", X_corrected, layer)
    return log_integration_operation(
        container,
        action="integration_combat",
        method_name="combat",
        params={
            "batch_key": batch_key,
            "covariates": list(covariates) if covariates else None,
            "eb_mode": eb_mode,
        },
        description=f"ComBat batch correction (eb_mode={eb_mode}).",
    )


def _build_design_matrices(
    obs_df: pl.DataFrame,
    batches: np.ndarray,
    unique_batches: np.ndarray,
    covariates: Sequence[str] | None,
    n_sample: int,
) -> tuple[np.ndarray, list[str], int]:
    """Build ComBat design matrices for batch and covariates."""
    # Batch design matrix
    batch_items = unique_batches
    n_batch = len(batch_items)

    # Create one-hot encoding for batches
    batch_dummies = pl.DataFrame(
        {f"batch_{i}": (batches == b).astype(int) for i, b in enumerate(batch_items)}
    )

    # Build covariate design matrix
    mod = _build_covariate_design(obs_df, covariates, n_sample)

    # Combine batch and covariate matrices
    mod_for_design = mod.drop("intercept") if "intercept" in mod.columns else mod
    design_matrix = pl.concat([batch_dummies, mod_for_design], how="horizontal")
    X_design = design_matrix.to_numpy().astype(float)

    return X_design, design_matrix.columns, n_batch


def _fit_combat(
    dat: np.ndarray,
    design_matrix: np.ndarray,
    n_batch: int,
    batches: np.ndarray,
    unique_batches: np.ndarray,
    n_sample: int,
    *,
    eb_mode: EbMode,
) -> np.ndarray:
    """Fit ComBat model and return corrected data."""
    # Fit model and extract coefficients
    B_hat = np.linalg.lstsq(design_matrix, dat.T, rcond=None)[0].T
    _B_batch, B_covar = B_hat[:, :n_batch], B_hat[:, n_batch:]
    X_covar = design_matrix[:, n_batch:]

    # Compute grand mean and standardize
    rank_design = np.linalg.matrix_rank(design_matrix)
    fitted_values = (design_matrix @ B_hat.T).T
    grand_mean = np.dot(fitted_values, np.ones(n_sample)) / n_sample
    residuals = dat - fitted_values
    sigma = np.sqrt(np.sum(residuals**2, axis=1) / (n_sample - rank_design))
    sigma[sigma == 0] = 1e-8

    # Standardize data
    covar_effect = (X_covar @ B_covar.T).T
    Z = (dat - grand_mean[:, None] - covar_effect) / sigma[:, None]

    # Empirical Bayes estimation
    gamma_hat, delta_hat = _compute_batch_moments(Z, batches, unique_batches, n_batch)
    if eb_mode == "parametric":
        gamma_bar, t2, a_prior, b_prior = _compute_eb_priors(gamma_hat, delta_hat)
        gamma_star, delta_star = _solve_eb_for_batches(
            gamma_hat, delta_hat, batches, unique_batches, gamma_bar, t2, a_prior, b_prior
        )
    else:
        gamma_star, delta_star = _solve_nonparametric_for_batches(
            Z,
            gamma_hat,
            delta_hat,
            batches,
            unique_batches,
        )

    # Apply correction
    out_data = _apply_combat_correction(Z, batches, unique_batches, gamma_star, delta_star)
    return out_data * sigma[:, None] + grand_mean[:, None] + covar_effect


def _build_covariate_design(
    obs_df: pl.DataFrame,
    covariates: Sequence[str] | None,
    n_sample: int,
) -> pl.DataFrame:
    """Build design matrix for covariates with dummy encoding for categoricals."""
    if not covariates:
        return pl.DataFrame({"intercept": np.ones(n_sample)})

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

    mod = covar_df.to_dummies(columns=cat_cols, drop_first=True) if cat_cols else covar_df
    return mod.with_columns(pl.lit(1.0).alias("intercept"))


def _compute_batch_moments(
    Z: np.ndarray,
    batches: np.ndarray,
    batch_items: np.ndarray,
    n_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for each batch."""
    gamma_hat = np.zeros((n_batch, Z.shape[0]))
    delta_hat = np.zeros((n_batch, Z.shape[0]))

    for i, b in enumerate(batch_items):
        idx = np.where(batches == b)[0]
        Z_batch = Z[:, idx]
        gamma_hat[i] = np.mean(Z_batch, axis=1)
        delta_hat[i] = np.var(Z_batch, axis=1, ddof=1)

    return gamma_hat, delta_hat


def _compute_eb_priors(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical Bayes priors using method of moments."""
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)

    delta_mean = np.mean(delta_hat, axis=1)
    delta_var = np.var(delta_hat, axis=1, ddof=1)
    delta_var[delta_var == 0] = 1e-8

    a_prior = (delta_mean**2 / delta_var) + 2
    b_prior = delta_mean * (a_prior - 1)

    return gamma_bar, t2, a_prior, b_prior


def _solve_eb_for_batches(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batches: np.ndarray,
    batch_items: np.ndarray,
    gamma_bar: np.ndarray,
    t2: np.ndarray,
    a_prior: np.ndarray,
    b_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve empirical Bayes for all batches."""
    n_batch, n_features = gamma_hat.shape
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i, b in enumerate(batch_items):
        idx = np.where(batches == b)[0]
        n_samples = len(idx)

        g_h = gamma_hat[i][:, None]
        d_h = delta_hat[i][:, None]

        g_s, d_s = _solve_eb(g_h, d_h, gamma_bar[i], t2[i], a_prior[i], b_prior[i], n_samples)
        gamma_star[i] = g_s.flatten()
        delta_star[i] = d_s.flatten()

    return gamma_star, delta_star


def _solve_eb(
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    n: int,
    conv: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve empirical Bayes using iterative posterior estimation."""
    g_old, d_old = g_hat.copy(), d_hat.copy()
    denom = max(a + n / 2 - 1, 1e-8)

    for _ in range(100):
        g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
        sum2 = (n - 1) * d_hat + n * (g_hat - g_new) ** 2
        d_new = (b + 0.5 * sum2) / denom

        change = np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)) + np.max(
            np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)
        )

        g_old, d_old = g_new, d_new
        if change <= conv:
            break

    return g_old, d_old


def _solve_nonparametric_for_batches(
    Z: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batches: np.ndarray,
    batch_items: np.ndarray,
    *,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve nonparametric ComBat posterior via empirical priors (int.eprior-style)."""
    n_batch, _ = gamma_hat.shape
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i, b in enumerate(batch_items):
        idx = np.where(batches == b)[0]
        g_s, d_s = _solve_nonparametric_batch(
            Z[:, idx],
            gamma_hat[i],
            delta_hat[i],
            block_size=block_size,
        )
        gamma_star[i] = g_s
        delta_star[i] = d_s

    return gamma_star, delta_star


def _solve_nonparametric_batch(
    Z_batch: np.ndarray,
    gamma_batch: np.ndarray,
    delta_batch: np.ndarray,
    *,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nonparametric posterior means for one batch with block vectorization."""
    n_features, n_samples = Z_batch.shape
    g = np.asarray(gamma_batch, dtype=float)
    d = np.clip(np.asarray(delta_batch, dtype=float), 1e-8, None)

    if n_features <= 1:
        return g.copy(), d.copy()

    g_star = np.empty_like(g)
    d_star = np.empty_like(d)
    g2 = g * g
    row_sum = np.sum(Z_batch, axis=1)
    row_sq_sum = np.sum(Z_batch * Z_batch, axis=1)
    log_pref = -(n_samples / 2.0) * np.log(2.0 * np.pi * d)
    inv_two_d = 0.5 / d

    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)
        rows = np.arange(start, end)

        sum2 = (
            row_sq_sum[start:end, None]
            - 2.0 * row_sum[start:end, None] * g[None, :]
            + float(n_samples) * g2[None, :]
        )
        log_lh = log_pref[None, :] - sum2 * inv_two_d[None, :]
        log_lh[np.arange(end - start), rows] = -np.inf

        row_max = np.max(log_lh, axis=1, keepdims=True)
        finite_rows = np.isfinite(row_max).ravel()
        weights = np.zeros_like(log_lh)
        if np.any(finite_rows):
            weights[finite_rows] = np.exp(log_lh[finite_rows] - row_max[finite_rows])

        weight_sum = np.sum(weights, axis=1)
        g_num = weights @ g
        d_num = weights @ d
        g_block = np.divide(g_num, weight_sum, out=g[rows].copy(), where=weight_sum > 0)
        d_block = np.divide(d_num, weight_sum, out=d[rows].copy(), where=weight_sum > 0)

        g_star[start:end] = g_block
        d_star[start:end] = np.clip(d_block, 1e-8, None)

    return g_star, d_star


def _apply_combat_correction(
    Z: np.ndarray,
    batches: np.ndarray,
    batch_items: np.ndarray,
    gamma_star: np.ndarray,
    delta_star: np.ndarray,
) -> np.ndarray:
    """Apply ComBat correction to standardized data."""
    out_data = np.zeros_like(Z)

    for i, b in enumerate(batch_items):
        idx = np.where(batches == b)[0]
        delta = np.clip(delta_star[i], 1e-8, None)
        out_data[:, idx] = (Z[:, idx] - gamma_star[i][:, None]) / np.sqrt(delta[:, None])

    return out_data


__all__ = ["integrate_combat"]
