"""
Local Least Squares imputation.

Reference:
    Kim H, et al. Bioinformatics 2005;21(2):187-198.
    Missing value estimation for DNA microarray gene expression data:
    Local least squares imputation.
"""

from typing import cast

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask
from scptensor.impute.base import ImputeMethod, register_impute_method

# =============================================================================
# Core LLS algorithm (pure function for registry)
# =============================================================================


def lls_impute(
    data: np.ndarray,
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_n_iter: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Local Least Squares imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    k : int, default=10
        Number of nearest neighbors.
    max_iter : int, default=100
        Maximum iterations for convergence.
    tol : float, default=1e-6
        Convergence threshold.

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

    # Initialize with column means (safe for all-NaN columns).
    col_means = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        observed = X[~missing_mask[:, j], j]
        col_means[j] = float(np.mean(observed)) if observed.size > 0 else 0.0

    for j in range(n_features):
        X[missing_mask[:, j], j] = col_means[j]

    # Iterative imputation
    n_iterations = 0
    for iteration in range(max_iter):
        n_iterations = iteration + 1
        prev_X = X.copy()
        samples_with_missing = np.where(missing_mask.any(axis=1))[0]

        for i in samples_with_missing:
            sample_missing = missing_mask[i]
            sample_observed = ~sample_missing

            if not sample_missing.any():
                continue

            observed_idx = np.where(sample_observed)[0]
            missing_idx = np.where(sample_missing)[0]

            if len(observed_idx) == 0 or len(missing_idx) == 0:
                X[i, missing_idx] = col_means[missing_idx]
                continue

            # Find neighbors using observed features
            k_search = min(k + 1, n_samples)
            X_obs_features = X[:, observed_idx]

            # Clean NaN values for distance computation
            X_obs_clean = X_obs_features.copy()
            for feat_idx in range(X_obs_clean.shape[1]):
                nan_mask = np.isnan(X_obs_clean[:, feat_idx])
                if nan_mask.any():
                    X_obs_clean[nan_mask, feat_idx] = col_means[observed_idx[feat_idx]]

            nbrs = NearestNeighbors(n_neighbors=k_search, algorithm="auto").fit(X_obs_clean)
            distances, neighbor_indices = nbrs.kneighbors([X_obs_clean[i]])

            # Remove self
            mask_not_self = neighbor_indices[0] != i
            neighbor_indices = neighbor_indices[0][mask_not_self]
            distances = distances[0][mask_not_self]

            if len(neighbor_indices) > k:
                neighbor_indices = neighbor_indices[:k]
                distances = distances[:k]

            if len(neighbor_indices) == 0:
                X[i, missing_idx] = col_means[missing_idx]
                continue

            # Impute each missing feature
            for j in missing_idx:
                neighbor_values = X[neighbor_indices, j]
                valid_mask = np.isfinite(neighbor_values)

                if not valid_mask.any():
                    X[i, j] = col_means[j]
                    continue

                valid_neighbors = neighbor_indices[valid_mask]
                valid_neighbor_values = neighbor_values[valid_mask]

                target_obs_values = X[i, observed_idx]
                n_valid = len(valid_neighbors)
                n_obs = len(observed_idx)

                if n_valid <= n_obs:
                    # Not enough neighbors - use weighted average
                    weights = 1.0 / (distances[valid_mask] + 1e-10)
                    weights /= weights.sum()
                    X[i, j] = np.dot(valid_neighbor_values, weights)
                else:
                    # Build regression model
                    Z = np.zeros((n_valid, n_obs + 1))
                    Z[:, 0] = 1.0
                    Z[:, 1:] = X[valid_neighbors][:, observed_idx]
                    y = valid_neighbor_values

                    try:
                        beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
                        target_design = np.concatenate([[1.0], target_obs_values])
                        X[i, j] = np.dot(target_design, beta)
                    except np.linalg.LinAlgError:
                        weights = 1.0 / (distances[valid_mask] + 1e-10)
                        weights /= weights.sum()
                        X[i, j] = np.dot(valid_neighbor_values, weights)

        # Check convergence
        change = np.abs(X[missing_mask] - prev_X[missing_mask])
        max_change = np.max(change)
        mean_val = np.mean(np.abs(X[missing_mask])) + 1e-10

        if max_change / mean_val < tol:
            break

    if return_n_iter:
        return X, n_iterations
    return X


def validate_lls(data: np.ndarray) -> bool:
    """Validate data for LLS imputation."""
    return data.size > 0 and data.shape[0] > 1


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


def impute_lls(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_lls",
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ScpContainer:
    """
    Impute missing values using Local Least Squares.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_lls"
        Name for the new layer with imputed data.
    k : int, default=10
        Number of nearest neighbors.
    max_iter : int, default=100
        Maximum iterations for convergence.
    tol : float, default=1e-6
        Convergence threshold.

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
    >>> from scptensor import impute_lls
    >>> result = impute_lls(container, "proteins", "raw", k=10)
    >>> "imputed_lls" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if k <= 0:
        raise ScpValueError(
            f"Number of neighbors (k) must be positive, got {k}.",
            parameter="k",
            value=k,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )
    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}.",
            parameter="tol",
            value=tol,
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
    X_raw = input_matrix.X.copy()
    if sp.issparse(X_raw):
        X_original = cast(sp.spmatrix, X_raw).toarray().astype(np.float64, copy=False)
    else:
        X_original = np.asarray(X_raw, dtype=np.float64)

    missing_mask = np.isnan(X_original)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_original, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_lls"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_lls",
            params={"assay": assay_name, "source_layer": source_layer, "k": k},
            description=f"LLS imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Apply LLS imputation
    X_imputed, iterations = lls_impute(
        X_original,
        k=k,
        max_iter=max_iter,
        tol=tol,
        return_n_iter=True,
    )
    X_imputed[~missing_mask] = X_original[~missing_mask]

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_lls"
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_lls",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "k": k,
            "n_iterations": iterations,
        },
        description=f"LLS imputation (k={k}, iterations={iterations}) on assay '{assay_name}'.",
    )

    return container


register_impute_method(
    ImputeMethod(
        name="lls",
        supports_sparse=False,
        validate=validate_lls,
        apply=impute_lls,
    )
)
