"""
MinProb imputation (probabilistic minimum imputation for MNAR data).

References:
    Wei R, et al. Sci Rep 2018;8:663.

Designed for left-censored MNAR (Missing Not At Random) data where
missingness is due to low abundance - values below detection limit.
"""

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask

# =============================================================================
# Core MinProb algorithm (pure function for registry)
# =============================================================================


def minprob_impute(
    data: np.ndarray,
    sigma: float = 2.0,
    random_state: int | None = None,
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

    # Impute each feature
    for j in range(n_features):
        feature_col = X[:, j]
        missing_in_col = missing_mask[:, j]

        if not np.any(missing_in_col):
            continue

        detected_values = feature_col[~missing_in_col]
        if len(detected_values) == 0:
            X[missing_in_col, j] = np.random.uniform(0.01, 0.1, size=np.sum(missing_in_col))
            continue

        min_detected = np.min(detected_values)
        spread = min_detected / sigma
        n_missing = np.sum(missing_in_col)

        # Sample from normal and bias toward values below min_detected
        samples = np.random.normal(loc=min_detected, scale=spread, size=n_missing)
        imputed_values = np.maximum(0.01, min_detected - np.abs(samples - min_detected) * 0.5)

        X[missing_in_col, j] = imputed_values

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
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original.copy()

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
    X_imputed = minprob_impute(X_dense, sigma=sigma, random_state=random_state)

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
        },
        description=f"MinProb imputation (sigma={sigma}) on assay '{assay_name}'.",
    )

    return container


# Register with base interface
from scptensor.impute.base import ImputeMethod, register_impute_method

register_impute_method(
    ImputeMethod(
        name="minprob",
        supports_sparse=False,
        validate=validate_minprob,
        apply=minprob_impute,
    )
)
