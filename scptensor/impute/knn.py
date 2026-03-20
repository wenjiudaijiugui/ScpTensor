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
from scptensor.core.structures import ScpContainer
from scptensor.impute._utils import (
    add_imputed_layer,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method

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
    X = np.asarray(data, dtype=np.float64).copy()
    n_samples, n_features = X.shape
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X

    if np.all(missing_mask):
        return np.zeros_like(X)

    # Initialize with column means
    col_means = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        observed = X[~missing_mask[:, j], j]
        col_means[j] = float(np.mean(observed)) if observed.size > 0 else 0.0

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
                finite_dist_mask = np.isfinite(valid_dists)
                valid_vals = valid_vals[finite_dist_mask]
                valid_dists = valid_dists[finite_dist_mask]

                if len(valid_vals) == 0:
                    X[global_idx, feat_pos] = col_means[feat_pos]
                    continue

                n_use = min(len(valid_vals), n_neighbors)
                vals_use = valid_vals[:n_use]
                dists_use = valid_dists[:n_use]

                if weights == "uniform":
                    X[global_idx, feat_pos] = float(np.mean(vals_use))
                else:
                    # Inverse distance weighting
                    zero_dist_mask = dists_use <= 1e-12
                    if np.any(zero_dist_mask):
                        X[global_idx, feat_pos] = float(np.mean(vals_use[zero_dist_mask]))
                        continue

                    with np.errstate(divide="ignore", invalid="ignore"):
                        w = 1.0 / dists_use
                    w[~np.isfinite(w)] = 0.0
                    w_sum = float(np.sum(w))
                    if w_sum < 1e-12:
                        X[global_idx, feat_pos] = float(np.mean(vals_use))
                    else:
                        X[global_idx, feat_pos] = float(np.dot(vals_use, w / w_sum))

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
    X = to_dense_float_copy(matrix.X)
    missing_mask = np.isnan(X)
    layer_name = new_layer_name or "imputed_knn"

    if not np.any(missing_mask):
        add_imputed_layer(assay, layer_name, X, matrix, missing_mask)
        return container

    # Apply KNN imputation
    X_imputed = knn_impute(
        X,
        n_neighbors=k,
        weights=weights,
        oversample_factor=oversample_factor,
        batch_size=batch_size,
    )
    preserve_observed_values(X_imputed, X, missing_mask)

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, matrix, missing_mask)

    # Log operation
    return log_imputation_operation(
        container,
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


register_impute_method(
    ImputeMethod(
        name="knn",
        supports_sparse=False,
        validate=validate_knn,
        apply=impute_knn,
    )
)
