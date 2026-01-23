"""ComBat batch effect correction for single-cell proteomics data.

Reference
---------
Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in
microarray expression data using empirical Bayes methods. Biostatistics.

Fortin, J.-P., et al. (2017). Correction of unwanted variation in
microarray data using empirical Bayes methods. bioRxiv.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import ScpContainer, ScpMatrix


def integrate_combat(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "combat",
    covariates: Sequence[str] | None = None,
) -> ScpContainer:
    """Apply ComBat batch effect correction using empirical Bayes.

    ComBat (ComBat for Batch Adjustment) corrects for batch effects using an
    empirical Bayes framework. It models batch-specific location and scale
    parameters and shrinks them towards global estimates.

    Parameters
    ----------
    container : ScpContainer
        Input container with batch information
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, optional
        Name of the assay to process (default: 'protein')
    base_layer : str, optional
        Name of the layer to use as input (default: 'raw')
    new_layer_name : str, optional
        Name for the new layer with corrected data (default: 'combat')
    covariates : Sequence[str] or None, optional
        Optional list of covariate column names in obs to preserve

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
    The algorithm:
        0. Impute missing values (NaNs) using batch-specific means (Pre-processing)
        1. Fit linear model with batch and covariates
        2. Estimate batch-specific mean and variance parameters
        3. Apply empirical Bayes shrinkage
        4. Adjust data using shrunken parameters

    Examples
    --------
    >>> container = integrate_combat(container, batch_key='batch')
    >>> container = integrate_combat(container, batch_key='batch', covariates=['condition'])
    """
    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    X = assay.layers[base_layer].X.copy()
    input_was_sparse = is_sparse_matrix(X)

    # Convert to dense for ComBat computation
    X_dense = X.toarray() if input_was_sparse else np.asarray(X)

    # Validate batch_key
    obs_df = container.obs
    if batch_key not in obs_df.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. Available columns: {list(obs_df.columns)}",
            parameter="batch_key",
            value=batch_key,
        )

    batch = obs_df[batch_key]

    # Impute NaN values with batch-specific means
    if np.isnan(X_dense).any():
        X_dense = _impute_nans_batchwise(X_dense, batch.to_numpy())

    # Transpose: (n_features, n_samples) for feature-wise operations
    dat = X_dense.T

    # Build design matrix
    batch_items = batch.unique().to_numpy()
    batch_info = batch.to_dummies()
    n_batch, n_sample = batch_info.shape[1], batch_info.shape[0]
    sample_per_batch = batch_info.sum().to_numpy().flatten()

    # Check minimum sample requirement
    if (sample_per_batch < 2).any():
        raise ScpValueError(
            "ComBat requires at least 2 samples per batch.",
            parameter="batch_key",
            value=batch_key,
        )

    # Build covariate design matrix
    mod = _build_covariate_design(obs_df, covariates, n_sample)

    # Combine batch and covariate matrices (drop intercept to avoid rank deficiency)
    mod_for_design = mod.drop("intercept") if "intercept" in mod.columns else mod
    design_matrix = pl.concat([batch_info, mod_for_design], how="horizontal")
    X_design = design_matrix.to_numpy().astype(float)

    # Check for rank deficiency
    rank_design = np.linalg.matrix_rank(X_design)
    if rank_design < X_design.shape[1]:
        raise ValueError(
            f"Design matrix is rank deficient (rank={rank_design}, cols={X_design.shape[1]}). "
            "Batch and biological covariates are confounded."
        )

    # Fit model and extract coefficients
    B_hat = np.linalg.lstsq(X_design, dat.T, rcond=None)[0].T
    _B_batch, B_covar = B_hat[:, :n_batch], B_hat[:, n_batch:]
    X_covar = X_design[:, n_batch:]

    # Compute grand mean and standardize
    fitted_values = (X_design @ B_hat.T).T  # (n_features, n_samples)
    grand_mean = np.dot(fitted_values, np.ones(n_sample)) / n_sample  # (n_features,)
    residuals = dat - fitted_values
    sigma = np.sqrt(np.sum(residuals**2, axis=1) / (n_sample - rank_design))
    sigma[sigma == 0] = 1e-8

    # Standardize data
    covar_effect = (X_covar @ B_covar.T).T
    Z = (dat - grand_mean[:, None] - covar_effect) / sigma[:, None]

    # Empirical Bayes estimation
    gamma_hat, delta_hat = _compute_batch_moments(Z, batch.to_numpy(), batch_items, n_batch)
    gamma_bar, t2, a_prior, b_prior = _compute_eb_priors(gamma_hat, delta_hat)
    gamma_star, delta_star = _solve_eb_for_batches(
        gamma_hat,
        delta_hat,
        batch.to_numpy(),
        batch_items,
        gamma_bar,
        t2,
        a_prior,
        b_prior,
    )

    # Apply correction
    out_data = _apply_combat_correction(Z, batch.to_numpy(), batch_items, gamma_star, delta_star)
    out_data = out_data * sigma[:, None] + grand_mean[:, None] + covar_effect
    X_corrected = out_data.T

    # Preserve sparsity if appropriate
    if input_was_sparse:
        sparsity_ratio = 1.0 - (np.count_nonzero(X_corrected) / X_corrected.size)
        if sparsity_ratio > 0.5:
            X_corrected = sp.csr_matrix(X_corrected)

    # Create new layer and log
    M_input = assay.layers[base_layer].M
    new_matrix = ScpMatrix(
        X=X_corrected,
        M=M_input.copy() if M_input is not None else None,
    )
    container.assays[assay_name].add_layer(new_layer_name or "combat", new_matrix)
    container.log_operation(
        action="integration_combat",
        params={"batch_key": batch_key, "covariates": list(covariates) if covariates else None},
        description="ComBat batch correction.",
    )

    return container


def _impute_nans_batchwise(X: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Impute NaN values using batch-specific column means (fully vectorized)."""
    X_imputed = X.copy()

    for b in np.unique(batches):
        batch_mask = batches == b
        batch_data = X_imputed[batch_mask]
        batch_mean = np.nanmean(batch_data, axis=0)

        nan_mask = np.isnan(batch_data)
        if nan_mask.any():
            batch_rows = np.where(batch_mask)[0]
            nan_rows, nan_cols = np.where(nan_mask)
            global_rows = batch_rows[nan_rows]
            X_imputed[global_rows, nan_cols] = batch_mean[nan_cols]

    if np.isnan(X_imputed).any():
        col_mean = np.nan_to_num(np.nanmean(X_imputed, axis=0), nan=0.0)
        nan_mask = np.isnan(X_imputed)
        X_imputed[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    return X_imputed


def _build_covariate_design(
    obs_df: pl.DataFrame,
    covariates: Sequence[str] | None,
    n_sample: int,
) -> pl.DataFrame:
    """Build design matrix for covariates with dummy encoding for categoricals."""
    if not covariates:
        return pl.DataFrame({"intercept": np.ones(n_sample)})

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

    for _ in range(100):
        g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
        sum2 = (n - 1) * d_hat + n * (g_hat - g_new) ** 2
        d_new = (b + 0.5 * sum2) / (a + n / 2)

        change = np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)) + np.max(
            np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)
        )

        g_old, d_old = g_new, d_new
        if change <= conv:
            break

    return g_old, d_old


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
        out_data[:, idx] = (Z[:, idx] - gamma_star[i][:, None]) / np.sqrt(delta_star[i][:, None])

    return out_data


__all__ = ["integrate_combat"]
