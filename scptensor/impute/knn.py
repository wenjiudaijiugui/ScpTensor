"""
K-nearest neighbors imputation.

.. math::

    X_{ij} = \\sum_{k \\in N_i} w_k X_{kj}

where:
- :math:`N_i` are the k nearest neighbors
- :math:`w_k` are the weights (uniform or distance-based)
"""

import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


# =============================================================================
# Core KNN algorithm (pure function for registry)
# =============================================================================


def knn_impute(
    data: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
    oversample_factor: int = 3,
    batch_size: int = 500,
) -> np.ndarray:
    """K-nearest neighbors imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : str, default="uniform"
        Weight function ("uniform" or "distance").
    oversample_factor : int, default=3
        Multiplier for search range to handle missingness.
    batch_size : int, default=500
        Number of rows to process at once.

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

    # Initialize with column means
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0

    samples_with_missing = np.where(np.any(missing_mask, axis=1))[0]
    k_search = min(n_neighbors * oversample_factor + 1, n_samples)

    for start_idx in range(0, len(samples_with_missing), batch_size):
        end_idx = min(start_idx + batch_size, len(samples_with_missing))
        batch_indices = samples_with_missing[start_idx:end_idx]
        X_batch = X[batch_indices]
        dist_matrix_batch = nan_euclidean_distances(X_batch, X)

        for i, global_idx in enumerate(batch_indices):
            dists = dist_matrix_batch[i]

            # Find k_search nearest neighbors
            neighbor_candidates_arg = np.argpartition(dists, k_search - 1)[:k_search]
            candidate_dists = dists[neighbor_candidates_arg]

            # Sort by distance
            sorted_order = np.argsort(candidate_dists)
            neighbor_candidates_sorted = neighbor_candidates_arg[sorted_order]
            candidate_dists_sorted = candidate_dists[sorted_order]

            # Exclude self
            mask_not_self = neighbor_candidates_sorted != global_idx
            potential_neighbors = neighbor_candidates_sorted[mask_not_self]
            potential_dists = candidate_dists_sorted[mask_not_self]

            # Get missing features for this sample
            sample_missing_feats = np.where(missing_mask[global_idx])[0]

            if len(sample_missing_feats) == 0:
                continue

            # Get neighbor values for missing features
            neighbor_values = X[np.ix_(potential_neighbors, sample_missing_feats)]
            valid_mask = ~np.isnan(neighbor_values)

            # For each feature, impute using top k valid neighbors
            for feat_idx, feat_pos in enumerate(sample_missing_feats):
                feat_mask = valid_mask[:, feat_idx]
                if not np.any(feat_mask):
                    X[global_idx, feat_pos] = col_means[feat_pos]
                    continue

                valid_vals = neighbor_values[feat_mask, feat_idx]
                valid_dists = potential_dists[feat_mask]

                if weights == "uniform":
                    X[global_idx, feat_pos] = np.mean(valid_vals[:n_neighbors])
                else:
                    # Inverse distance weighting
                    with np.errstate(divide="ignore", invalid="ignore"):
                        w = 1.0 / valid_dists
                    inf_mask = np.isinf(w)
                    if np.any(inf_mask):
                        w[inf_mask] = 1.0
                        w[~inf_mask] = 0.0
                    w_sum = np.sum(w)
                    if w_sum < 1e-10:
                        X[global_idx, feat_pos] = np.mean(valid_vals[:n_neighbors])
                    else:
                        n_use = min(len(valid_vals), n_neighbors)
                        X[global_idx, feat_pos] = np.sum(valid_vals[:n_use] * (w[:n_use] / w_sum))

    return X


def validate_knn(data: np.ndarray) -> bool:
    """Validate data for KNN imputation."""
    return data.size > 0


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


def impute_knn(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_knn",
    k: int = 5,
    weights: str = "uniform",
    batch_size: int = 500,
    oversample_factor: int = 3,
) -> ScpContainer:
    """
    Impute missing values using k-Nearest Neighbors with over-sampling.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_knn"
        Name for the new layer with imputed data.
    k : int, default=5
        Number of neighbors to use.
    weights : str, default="uniform"
        Weight function: 'uniform' or 'distance'.
    batch_size : int, default=500
        Number of rows to process at once.
    oversample_factor : int, default=3
        Multiplier for search range k_search = oversample_factor * k.

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
    >>> from scptensor import impute_knn
    >>> result = impute_knn(container, "proteins", "raw", k=5)
    >>> "imputed_knn" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if k <= 0:
        raise ScpValueError(
            f"Number of neighbors (k) must be positive, got {k}.",
            parameter="k",
            value=k,
        )
    if weights not in ("uniform", "distance"):
        raise ScpValueError(
            f"Weights must be 'uniform' or 'distance', got '{weights}'.",
            parameter="weights",
            value=weights,
        )
    if batch_size <= 0:
        raise ScpValueError(
            f"Batch size must be positive, got {batch_size}.",
            parameter="batch_size",
            value=batch_size,
        )
    if oversample_factor < 1:
        raise ScpValueError(
            f"Oversample factor must be at least 1, got {oversample_factor}.",
            parameter="oversample_factor",
            value=oversample_factor,
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

    matrix = assay.layers[source_layer]
    X = matrix.X.copy()
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X, M=_update_imputed_mask(matrix.M, missing_mask))
        assay.add_layer(new_layer_name or "imputed_knn", new_matrix)
        return container

    # Apply KNN imputation
    X_imputed = knn_impute(
        X,
        n_neighbors=k,
        weights=weights,
        oversample_factor=oversample_factor,
        batch_size=batch_size,
    )

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_knn"
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_knn",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "k": k,
            "weights": weights,
        },
        description=f"KNN imputation (k={k}, weights={weights}) on assay '{assay_name}'.",
    )

    return container


# Register with base interface
from scptensor.impute.base import ImputeMethod, register_impute_method

register_impute_method(
    ImputeMethod(
        name="knn",
        supports_sparse=False,
        validate=validate_knn,
        apply=knn_impute,
    )
)
