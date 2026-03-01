"""
QRILC imputation (Quantile Regression Imputation of Left-Censored Data).

Reference:
    Wei R, et al. "Missing Value Imputation Approach for Mass Spectrometry-based
    Metabolomics Data." Sci Rep 2018;8:663.

Designed specifically for MNAR (Missing Not At Random) data where missingness
is due to low abundance (left-censored) values.
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


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
    X = data.copy()
    n_samples, n_features = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X

    if random_state is not None:
        np.random.seed(random_state)

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
            X[missing_in_feat, feat_idx] = min_detected
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

        # Sample from left-censored distribution
        n_samples_to_generate = n_missing * 2
        samples = np.random.randn(n_samples_to_generate) * sigma + mu
        left_censored = samples[samples < detection_limit]

        if len(left_censored) >= n_missing:
            imputed_values = left_censored[:n_missing]
        else:
            min_detected = np.min(detected_values)
            if min_detected < detection_limit:
                supplement = np.random.uniform(
                    min_detected,
                    detection_limit,
                    size=n_missing - len(left_censored),
                )
                imputed_values = np.concatenate([left_censored, supplement])
            else:
                from scipy.stats import truncnorm

                a = (min_detected - mu) / sigma
                b = (detection_limit - mu) / sigma
                imputed_values = truncnorm.rvs(
                    a, b, loc=mu, scale=sigma, size=n_missing, random_state=random_state
                )

        # Ensure non-negative
        imputed_values = np.maximum(imputed_values, 0)
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
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original

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


# Register with base interface
from scptensor.impute.base import register_impute_method, ImputeMethod

register_impute_method(
    ImputeMethod(
        name="qrilc",
        supports_sparse=False,
        validate=validate_qrilc,
        apply=qrilc_impute,
    )
)
