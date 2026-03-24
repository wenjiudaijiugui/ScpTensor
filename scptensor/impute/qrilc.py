"""QRILC imputation (Quantile Regression Imputation of Left-Censored Data).

Reference:
    Wei R, et al. "Missing Value Imputation Approach for Mass Spectrometry-based
    Metabolomics Data." Sci Rep 2018;8:663.

Designed specifically for MNAR (Missing Not At Random) data where missingness
is due to low abundance (left-censored) values.
"""

import numpy as np
from scipy.stats import norm, truncnorm

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ScpValueError
from scptensor.impute._utils import (
    add_imputed_layer,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method, validate_layer_context

# =============================================================================
# Core QRILC algorithm (pure function for registry)
# =============================================================================


def qrilc_impute(
    data: np.ndarray,
    q: float = 0.01,
    random_state: int | None = None,
) -> np.ndarray:
    """QRILC imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    q : float, default=0.01
        Upper-tail trim used for QRILC's quantile regression fit.
        The default ``q=0.01`` reproduces the original ``upper.q=0.99``
        fitting range from ``imputeLCMD::impute.QRILC``.
    random_state : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Data with imputed values.

    """
    X = np.asarray(data, dtype=np.float64).copy()
    n_samples, n_features = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X
    if not 0 < q < 1:
        raise ScpValueError(
            f"Quantile q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    invalid_observed = (~missing_mask) & ~np.isfinite(X)
    if np.any(invalid_observed):
        raise ScpValueError(
            "QRILC requires observed values to be finite; use NaN to mark missing entries.",
            parameter="data",
        )

    rng = np.random.default_rng(random_state)
    global_detected = X[np.isfinite(X)]
    global_low = float(np.quantile(global_detected, q)) if global_detected.size > 0 else 0.0
    fit_upper_q = float(np.clip(1.0 - q, 0.5, 0.999))

    # imputeLCMD::impute.QRILC is sample-wise. Under ScpTensor's
    # (n_samples, n_features) convention, that means iterating over rows.
    for sample_idx in range(n_samples):
        missing_in_sample = missing_mask[sample_idx]
        n_missing = int(np.sum(missing_in_sample))
        if n_missing == 0:
            continue

        observed_values = X[sample_idx, ~missing_in_sample]
        if observed_values.size == 0:
            fill_value = max(global_low, 0.0) if np.all(global_detected >= 0) else global_low
            X[sample_idx, missing_in_sample] = fill_value
            continue

        row_non_negative = bool(np.all(observed_values >= 0))
        fallback_low = float(np.quantile(observed_values, min(q, 0.5)))
        if observed_values.size < 3:
            fill_value = max(fallback_low, 0.0) if row_non_negative else fallback_low
            X[sample_idx, missing_in_sample] = fill_value
            continue

        p_missing = n_missing / n_features
        if (p_missing + 0.001) >= fit_upper_q:
            fill_value = max(fallback_low, 0.0) if row_non_negative else fallback_low
            X[sample_idx, missing_in_sample] = fill_value
            continue

        sample_probs = np.arange(0.001, fit_upper_q + 0.001 + 1e-12, 0.01, dtype=np.float64)
        normal_probs = np.linspace(
            p_missing + 0.001,
            fit_upper_q + 0.001,
            num=sample_probs.size,
            dtype=np.float64,
        )
        q_normal = norm.ppf(normal_probs)
        q_sample = np.quantile(observed_values, sample_probs)
        valid = np.isfinite(q_normal) & np.isfinite(q_sample)
        if np.count_nonzero(valid) < 2:
            fill_value = max(fallback_low, 0.0) if row_non_negative else fallback_low
            X[sample_idx, missing_in_sample] = fill_value
            continue

        slope, intercept = np.polyfit(q_normal[valid], q_sample[valid], deg=1)
        mu = float(intercept)
        sigma = float(abs(slope))

        if not np.isfinite(mu):
            mu = fallback_low
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(observed_values, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(float(np.ptp(observed_values)) / 4.0, 1e-8)

        upper = norm.ppf(min(p_missing + 0.001, 0.999), loc=mu, scale=sigma)
        b = (upper - mu) / sigma
        if not np.isfinite(b):
            b = 0.0

        imputed_values = truncnorm.rvs(
            -np.inf,
            b,
            loc=mu,
            scale=sigma,
            size=n_missing,
            random_state=rng,
        )

        if row_non_negative:
            imputed_values = np.maximum(imputed_values, 0.0)
        X[sample_idx, missing_in_sample] = imputed_values

    return X


def validate_qrilc(data: np.ndarray) -> bool:
    """Validate data for QRILC imputation."""
    return data.size > 0


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


def impute_qrilc(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "qrilc",
    q: float = 0.01,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using Quantile Regression Imputation of Left-Censored Data.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "qrilc"
        Name for the new layer with imputed data.
    q : float, default=0.01
        Upper-tail trim used for QRILC's quantile regression fit (0-1).
        The default maps to the original ``upper.q=0.99`` setting.
    random_state : int, optional
        Random seed.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    ScpValueError
        If parameters are invalid.
    AssayNotFoundError
        If the assay does not exist.
    LayerNotFoundError
        If the layer does not exist.

    Examples
    --------
    >>> from scptensor.impute import impute_qrilc
    >>> result = impute_qrilc(container, "proteins", "raw", q=0.01)
    >>> "qrilc" in result.assays["proteins"].layers
    True

    """
    # Validate parameters
    if not 0 < q < 1:
        raise ScpValueError(
            f"Quantile q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    X_dense = to_dense_float_copy(input_matrix.X)

    missing_mask = np.isnan(X_dense)
    layer_name = new_layer_name or "qrilc"
    if not np.any(missing_mask):
        add_imputed_layer(assay, layer_name, X_dense, input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_qrilc",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "q": q,
            },
            description=(
                f"QRILC imputation on assay '{ctx.resolved_assay_name}': no missing values found."
            ),
        )

    # Apply QRILC imputation
    X_imputed = qrilc_impute(X_dense, q=q, random_state=random_state)
    preserve_observed_values(X_imputed, X_dense, missing_mask)

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, input_matrix, missing_mask)

    # Log operation
    return log_imputation_operation(
        container,
        action="impute_qrilc",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "q": q,
        },
        description=f"QRILC imputation (q={q}) on assay '{ctx.resolved_assay_name}'.",
    )


register_impute_method(
    ImputeMethod(
        name="qrilc",
        supports_sparse=False,
        validate=validate_qrilc,
        apply=impute_qrilc,
    ),
)
