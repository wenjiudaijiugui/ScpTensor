"""MinProb imputation (probabilistic minimum imputation for MNAR data).

References:
    Wei R, et al. Sci Rep 2018;8:663.

Designed for left-censored MNAR (Missing Not At Random) data where
missingness is due to low abundance - values below detection limit.

"""

import numpy as np

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
# Core MinProb algorithm (pure function for registry)
# =============================================================================


def minprob_impute(
    data: np.ndarray,
    sigma: float = 2.0,
    random_state: int | None = None,
    q: float = 0.01,
) -> np.ndarray:
    """MinProb imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    sigma : float, default=2.0
        Standard deviation multiplier.
    random_state : int, optional
        Random seed.
    q : float, default=0.01
        Low quantile used to estimate each sample row's low-abundance center.

    Returns
    -------
    np.ndarray
        Data with imputed values.

    """
    X = np.asarray(data, dtype=np.float64).copy()
    n_samples, _ = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}.",
            parameter="sigma",
            value=sigma,
        )
    if not 0 < q < 1:
        raise ScpValueError(
            f"q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    invalid_observed = (~missing_mask) & ~np.isfinite(X)
    if np.any(invalid_observed):
        raise ScpValueError(
            "MinProb requires observed values to be finite; use NaN to mark missing entries.",
            parameter="data",
        )

    rng = np.random.default_rng(random_state)
    global_detected = X[np.isfinite(X)]
    global_non_negative = global_detected.size == 0 or bool(np.all(global_detected >= 0))
    global_q = float(np.quantile(global_detected, q)) if global_detected.size > 0 else 0.0
    global_std = float(np.std(global_detected, ddof=1)) if global_detected.size > 1 else 0.0

    # imputeLCMD MinProb estimates one shared SD from sufficiently observed features.
    observed_fraction = np.mean(~missing_mask, axis=0)
    filtered_cols = np.where(observed_fraction > 0.5)[0]
    if filtered_cols.size > 0 and n_samples > 1:
        prot_sd = np.nanstd(X[:, filtered_cols], axis=0, ddof=1)
        finite_sd = prot_sd[np.isfinite(prot_sd) & (prot_sd > 0)]
        if finite_sd.size > 0:
            sd_temp = float(np.median(finite_sd)) * sigma
        else:
            sd_temp = global_std * sigma
    else:
        sd_temp = global_std * sigma
    if not np.isfinite(sd_temp) or sd_temp <= 0:
        sd_temp = max(global_std * sigma, 1e-8)

    # imputeLCMD::impute.MinProb is sample-wise. Under ScpTensor's
    # (n_samples, n_features) convention, that means iterating over rows.
    min_samples = np.full(n_samples, global_q, dtype=np.float64)
    row_non_negative = np.full(n_samples, global_non_negative, dtype=bool)
    for sample_idx in range(n_samples):
        observed = X[sample_idx, ~missing_mask[sample_idx]]
        if observed.size > 0:
            min_samples[sample_idx] = float(np.quantile(observed, q))
            row_non_negative[sample_idx] = bool(np.all(observed >= 0))

    for sample_idx in range(n_samples):
        miss_idx = np.where(missing_mask[sample_idx])[0]
        if miss_idx.size == 0:
            continue

        draws = rng.normal(loc=min_samples[sample_idx], scale=sd_temp, size=miss_idx.size)
        if row_non_negative[sample_idx]:
            draws = np.maximum(draws, 0.0)
        X[sample_idx, miss_idx] = draws

    return X


def validate_minprob(data: np.ndarray) -> bool:
    """Validate data for MinProb imputation."""
    return data.size > 0


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


def impute_minprob(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_minprob",
    sigma: float = 2.0,
    random_state: int | None = None,
    q: float = 0.01,
) -> ScpContainer:
    """Impute missing values using probabilistic minimum imputation (MinProb).

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_minprob"
        Name for the new layer with imputed data.
    sigma : float, default=2.0
        Standard deviation multiplier for distribution width.
    random_state : int, optional
        Random seed.
    q : float, default=0.01
        Low quantile used for per-feature low-abundance estimate.

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
    >>> from scptensor.impute import impute_minprob
    >>> result = impute_minprob(container, "proteins", sigma=2.0)
    >>> "imputed_minprob" in result.assays["proteins"].layers
    True

    """
    # Validate parameters
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}.",
            parameter="sigma",
            value=sigma,
        )
    if not 0 < q < 1:
        raise ScpValueError(
            f"q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    X_dense = to_dense_float_copy(input_matrix.X)

    missing_mask = np.isnan(X_dense)
    layer_name = new_layer_name or "imputed_minprob"
    if not np.any(missing_mask):
        add_imputed_layer(assay, layer_name, X_dense, input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_minprob",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "sigma": sigma,
                "q": q,
            },
            description=(
                f"MinProb imputation on assay '{ctx.resolved_assay_name}': no missing values found."
            ),
        )

    # Apply MinProb imputation
    X_imputed = minprob_impute(X_dense, sigma=sigma, random_state=random_state, q=q)
    preserve_observed_values(X_imputed, X_dense, missing_mask)

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, input_matrix, missing_mask)

    # Log operation
    return log_imputation_operation(
        container,
        action="impute_minprob",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "sigma": sigma,
            "q": q,
        },
        description=(
            f"MinProb imputation (sigma={sigma}, q={q}) on assay '{ctx.resolved_assay_name}'."
        ),
    )


register_impute_method(
    ImputeMethod(
        name="minprob",
        supports_sparse=False,
        validate=validate_minprob,
        apply=impute_minprob,
    ),
)
