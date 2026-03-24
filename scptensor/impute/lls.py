"""Local Least Squares imputation.

Reference:
    Kim H, et al. Bioinformatics 2005;21(2):187-198.
    Missing value estimation for DNA microarray gene expression data:
    Local least squares imputation.
"""

from typing import cast

import numpy as np
from sklearn.neighbors import NearestNeighbors

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
# Core LLS algorithm (pure function for registry)
# =============================================================================


def _mean_fill_matrix(data: np.ndarray) -> np.ndarray:
    """Fill missing entries with per-feature means in the original orientation."""
    X = np.asarray(data, dtype=np.float64).copy()
    if X.size == 0:
        return X

    missing_mask = np.isnan(X)
    for j in range(X.shape[1]):
        observed = X[~missing_mask[:, j], j]
        fill_value = float(np.mean(observed)) if observed.size > 0 else 0.0
        X[missing_mask[:, j], j] = fill_value
    return X


def _lls_impute_entities(
    data: np.ndarray,
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_n_iter: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Run the row-wise LLS update on a generic entity x context matrix."""
    X = np.asarray(data, dtype=np.float64).copy()
    n_entities, n_contexts = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        if return_n_iter:
            return X, 0
        return X

    if n_entities <= 1 or n_contexts == 0:
        filled = _mean_fill_matrix(X)
        if return_n_iter:
            return filled, 0
        return filled

    # Initialize with context means (safe for all-NaN contexts).
    context_means = np.zeros(n_contexts, dtype=np.float64)
    for j in range(n_contexts):
        observed = X[~missing_mask[:, j], j]
        context_means[j] = float(np.mean(observed)) if observed.size > 0 else 0.0

    for j in range(n_contexts):
        X[missing_mask[:, j], j] = context_means[j]

    n_iterations = 0
    for iteration in range(max_iter):
        n_iterations = iteration + 1
        prev_X = X.copy()
        entities_with_missing = np.where(missing_mask.any(axis=1))[0]

        for i in entities_with_missing:
            entity_missing = missing_mask[i]
            entity_observed = ~entity_missing

            if not entity_missing.any():
                continue

            observed_idx = np.where(entity_observed)[0]
            missing_idx = np.where(entity_missing)[0]

            if len(observed_idx) == 0 or len(missing_idx) == 0:
                X[i, missing_idx] = context_means[missing_idx]
                continue

            # Find neighbors using the target entity's observed contexts.
            k_search = min(k + 1, n_entities)
            X_obs_context = X[:, observed_idx]

            nbrs = NearestNeighbors(n_neighbors=k_search, algorithm="auto").fit(X_obs_context)
            distances, neighbor_indices = nbrs.kneighbors([X_obs_context[i]])

            mask_not_self = neighbor_indices[0] != i
            neighbor_indices = neighbor_indices[0][mask_not_self]
            distances = distances[0][mask_not_self]

            if len(neighbor_indices) > k:
                neighbor_indices = neighbor_indices[:k]
                distances = distances[:k]

            if len(neighbor_indices) == 0:
                X[i, missing_idx] = context_means[missing_idx]
                continue

            for j in missing_idx:
                neighbor_values = X[neighbor_indices, j]
                valid_mask = np.isfinite(neighbor_values)

                if not valid_mask.any():
                    X[i, j] = context_means[j]
                    continue

                valid_neighbors = neighbor_indices[valid_mask]
                valid_neighbor_values = neighbor_values[valid_mask]

                target_obs_values = X[i, observed_idx]
                n_valid = len(valid_neighbors)
                n_obs = len(observed_idx)

                if n_valid <= n_obs:
                    weights = 1.0 / (distances[valid_mask] + 1e-10)
                    weights /= weights.sum()
                    X[i, j] = np.dot(valid_neighbor_values, weights)
                else:
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

        change = np.abs(X[missing_mask] - prev_X[missing_mask])
        max_change = np.max(change)
        mean_val = np.mean(np.abs(X[missing_mask])) + 1e-10

        if max_change / mean_val < tol:
            break

    if return_n_iter:
        return X, n_iterations
    return X


def lls_impute(
    data: np.ndarray,
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_n_iter: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Feature-wise Local Least Squares imputation under a sample x feature contract.

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
    X = np.asarray(data, dtype=np.float64)
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        if return_n_iter:
            return X.copy(), 0
        return X.copy()

    # The cited LLSimpute algorithm searches for local neighbors in feature space.
    # Under ScpTensor's sample x feature contract, transpose to feature x sample,
    # run the generic row-wise update there, then transpose back.
    if X.shape[0] < 2 or X.shape[1] < 2:
        filled = _mean_fill_matrix(X)
        if return_n_iter:
            return filled, 0
        return filled

    if return_n_iter:
        X_imputed_t, n_iterations = _lls_impute_entities(
            X.T,
            k=k,
            max_iter=max_iter,
            tol=tol,
            return_n_iter=True,
        )
        return X_imputed_t.T, n_iterations
    X_imputed_t = cast(
        "np.ndarray",
        _lls_impute_entities(
            X.T,
            k=k,
            max_iter=max_iter,
            tol=tol,
            return_n_iter=False,
        ),
    )
    return X_imputed_t.T


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
    """Impute missing values using Local Least Squares.

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

    Examples
    --------
    >>> from scptensor.impute import impute_lls
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

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    X_original = to_dense_float_copy(input_matrix.X)

    missing_mask = np.isnan(X_original)
    layer_name = new_layer_name or "imputed_lls"
    if not np.any(missing_mask):
        add_imputed_layer(assay, layer_name, X_original, input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_lls",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "k": k,
            },
            description=(
                f"LLS imputation on assay '{ctx.resolved_assay_name}': no missing values found."
            ),
        )

    # Apply LLS imputation
    X_imputed, iterations = lls_impute(
        X_original,
        k=k,
        max_iter=max_iter,
        tol=tol,
        return_n_iter=True,
    )
    preserve_observed_values(X_imputed, X_original, missing_mask)

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, input_matrix, missing_mask)

    # Log operation
    return log_imputation_operation(
        container,
        action="impute_lls",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "k": k,
            "n_iterations": iterations,
        },
        description=(
            f"LLS imputation (k={k}, iterations={iterations}) on assay '{ctx.resolved_assay_name}'."
        ),
    )


register_impute_method(
    ImputeMethod(
        name="lls",
        supports_sparse=False,
        validate=validate_lls,
        apply=impute_lls,
    ),
)
