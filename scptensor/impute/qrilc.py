"""
QRILC imputation (Quantile Regression Imputation of Left-Censored Data).

Reference:
    Wei R, et al. "Missing Value Imputation Approach for Mass Spectrometry-based
    Metabolomics Data." Sci Rep 2018;8:663.

Designed specifically for MNAR (Missing Not At Random) data where missingness
is due to low abundance (left-censored) values.
"""

from typing import cast

import numpy as np
import scipy.sparse as sp
from scipy.stats import truncnorm

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask
from scptensor.impute.base import ImputeMethod, register_impute_method

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
        Quantile for defining left-censoring threshold.
    random_state : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Data with imputed values.
    """
    X = np.asarray(data, dtype=np.float64).copy()
    _, n_features = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X
    if not 0 < q < 1:
        raise ScpValueError(
            f"Quantile q must be between 0 and 1, got {q}.",
            parameter="q",
            value=q,
        )

    rng = np.random.default_rng(random_state)
    global_detected = X[np.isfinite(X)]
    global_non_negative = global_detected.size == 0 or bool(np.all(global_detected >= 0))

    # For each feature, estimate distribution and impute
    for feat_idx in range(n_features):
        feature_col = X[:, feat_idx]
        missing_in_feat = missing_mask[:, feat_idx]
        detected_mask = ~missing_in_feat

        detected_values = feature_col[detected_mask]
        n_missing = np.sum(missing_in_feat)
        n_detected = len(detected_values)

        if n_missing == 0:
            continue

        if n_detected < 3:
            min_detected = np.min(detected_values) if n_detected > 0 else 0.0
            fill_value = max(min_detected, 0.0) if global_non_negative else min_detected
            X[missing_in_feat, feat_idx] = fill_value
            continue

        # Estimate detection limit
        detection_limit = np.quantile(detected_values, q)
        above_threshold = detected_values >= detection_limit
        values_for_fit = detected_values[above_threshold]

        if len(values_for_fit) < 3:
            values_for_fit = detected_values

        # Fit normal distribution
        mu = np.mean(values_for_fit)
        sigma = np.std(values_for_fit, ddof=1)

        if sigma <= 0 or not np.isfinite(sigma):
            sigma = np.std(detected_values, ddof=1)
            if sigma <= 0 or not np.isfinite(sigma):
                sigma = (np.max(detected_values) - np.min(detected_values)) / 4
                if sigma <= 0:
                    sigma = 1.0

        lower_bound = 0.0 if np.all(detected_values >= 0) else (mu - 8.0 * sigma)
        a = (lower_bound - mu) / sigma
        b = (detection_limit - mu) / sigma

        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            # Conservative fallback when truncation interval degenerates.
            lo = min(np.min(detected_values), detection_limit)
            hi = max(np.min(detected_values), detection_limit)
            if hi <= lo:
                hi = lo + 1e-8
            imputed_values = rng.uniform(lo, hi, size=n_missing)
        else:
            imputed_values = truncnorm.rvs(
                a, b, loc=mu, scale=sigma, size=n_missing, random_state=rng
            )

        if np.all(detected_values >= 0):
            imputed_values = np.maximum(imputed_values, 0.0)
        X[missing_in_feat, feat_idx] = imputed_values

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
    """
    Impute missing values using Quantile Regression Imputation of Left-Censored Data.

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
        Quantile for defining left-censoring threshold (0-1).
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
    >>> from scptensor import impute_qrilc
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
    X_original = input_matrix.X.copy()

    # Convert sparse to dense
    if sp.issparse(X_original):
        X_dense = cast(sp.spmatrix, X_original).toarray()
    else:
        X_dense = np.asarray(X_original)

    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "qrilc"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_qrilc",
            params={"assay": assay_name, "source_layer": source_layer, "q": q},
            description=f"QRILC imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Apply QRILC imputation
    X_imputed = qrilc_impute(X_dense, q=q, random_state=random_state)
    X_imputed[~missing_mask] = X_dense[~missing_mask]

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "qrilc"
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_qrilc",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "q": q,
        },
        description=f"QRILC imputation (q={q}) on assay '{assay_name}'.",
    )

    return container


register_impute_method(
    ImputeMethod(
        name="qrilc",
        supports_sparse=False,
        validate=validate_qrilc,
        apply=impute_qrilc,
    )
)
