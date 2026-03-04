"""
MinProb imputation (probabilistic minimum imputation for MNAR data).

References:
    Wei R, et al. Sci Rep 2018;8:663.

Designed for left-censored MNAR (Missing Not At Random) data where
missingness is due to low abundance - values below detection limit.
"""

from typing import cast

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask
from scptensor.impute.base import ImputeMethod, register_impute_method

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
        Low quantile used to estimate per-feature low-abundance center.

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
            f"q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    rng = np.random.default_rng(random_state)
    global_detected = X[np.isfinite(X)]
    global_non_negative = global_detected.size == 0 or bool(np.all(global_detected >= 0))
    global_q = float(np.quantile(global_detected, q)) if global_detected.size > 0 else 0.0
    global_std = float(np.std(global_detected, ddof=0)) if global_detected.size > 1 else 0.0

    # imputeLCMD MinProb estimates one shared SD from sufficiently observed features.
    observed_fraction = np.mean(~missing_mask, axis=0)
    filtered_cols = np.where(observed_fraction > 0.5)[0]
    if filtered_cols.size > 0:
        prot_sd = np.nanstd(X[:, filtered_cols], axis=0, ddof=0)
        finite_sd = prot_sd[np.isfinite(prot_sd) & (prot_sd > 0)]
        if finite_sd.size > 0:
            sd_temp = float(np.median(finite_sd)) * sigma
        else:
            sd_temp = global_std * sigma
    else:
        sd_temp = global_std * sigma
    if not np.isfinite(sd_temp) or sd_temp <= 0:
        sd_temp = max(global_std, 1e-8)

    # Feature-wise q-th quantile estimate under ScpTensor matrix convention
    # (rows: observations/cells, columns: protein features).
    min_features = np.full(n_features, global_q, dtype=np.float64)
    upper_bounds = np.full(n_features, global_q, dtype=np.float64)
    for j in range(n_features):
        observed = X[~missing_mask[:, j], j]
        if observed.size > 0:
            min_features[j] = float(np.quantile(observed, q))
            upper_bounds[j] = float(np.min(observed))

    for j in range(n_features):
        miss_idx = np.where(missing_mask[:, j])[0]
        if miss_idx.size == 0:
            continue
        draws = rng.normal(loc=min_features[j], scale=sd_temp, size=miss_idx.size)
        # Left-censored MNAR: imputed values should not exceed the lowest detected value.
        draws = np.minimum(draws, upper_bounds[j])
        if global_non_negative:
            draws = np.maximum(draws, 0.0)
        X[miss_idx, j] = draws

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
    """
    Impute missing values using probabilistic minimum imputation (MinProb).

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
    >>> from scptensor import impute_minprob
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

    # Validate assay and layer
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers: {available}.",
        )

    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X

    # Convert sparse to dense
    if sp.issparse(X_original):
        X_dense = cast(sp.spmatrix, X_original).toarray()
    else:
        X_dense = X_original.copy()

    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_minprob"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_minprob",
            params={"assay": assay_name, "source_layer": source_layer, "sigma": sigma},
            description=f"MinProb imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Apply MinProb imputation
    X_imputed = minprob_impute(X_dense, sigma=sigma, random_state=random_state, q=q)
    X_imputed[~missing_mask] = X_dense[~missing_mask]

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_minprob"
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_minprob",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "sigma": sigma,
            "q": q,
        },
        description=f"MinProb imputation (sigma={sigma}, q={q}) on assay '{assay_name}'.",
    )

    return container


register_impute_method(
    ImputeMethod(
        name="minprob",
        supports_sparse=False,
        validate=validate_minprob,
        apply=impute_minprob,
    )
)
