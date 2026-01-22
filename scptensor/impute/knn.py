"""KNN imputation for single-cell proteomics data."""

import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


def _compute_weighted_imputation(
    neighbor_values: np.ndarray,
    neighbor_dists: np.ndarray,
    weights: str,
    k: int,
) -> np.ndarray:
    """Compute weighted imputation for multiple features efficiently.

    Parameters
    ----------
    neighbor_values : np.ndarray
        Shape (n_neighbors, n_features) - values from neighbors
    neighbor_dists : np.ndarray
        Shape (n_neighbors,) - distances to neighbors
    weights : str
        'uniform' or 'distance'
    k : int
        Number of neighbors to use

    Returns
    -------
    np.ndarray
        Shape (n_features,) - imputed values
    """
    n_use = min(neighbor_values.shape[0], k)
    vals = neighbor_values[:n_use]
    dists = neighbor_dists[:n_use]

    if weights == "uniform":
        return np.mean(vals, axis=0)

    # Inverse distance weighting with zero-distance handling
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / dists

    inf_mask = np.isinf(w)
    if np.any(inf_mask):
        w[inf_mask] = 1.0
        w[~inf_mask] = 0.0

    w_sum = np.sum(w)
    if w_sum < 1e-10:
        return np.mean(vals, axis=0)

    w_normalized = w / w_sum
    return vals.T @ w_normalized


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

    Searches k_search = oversample_factor * k neighbors to handle missing
    values, then filters to valid (non-missing) neighbors per feature.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str, default "raw"
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_knn"
        Name for the new layer with imputed data.
    k : int, default 5
        Number of neighbors to use.
    weights : str, default "uniform"
        Weight function: 'uniform' or 'distance'.
    batch_size : int, default 500
        Number of rows to process at once.
    oversample_factor : int, default 3
        Multiplier for search range k_search = oversample_factor * k.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ScpValueError
        If parameters are invalid.

    Notes
    -----
    Time complexity: O(M * N * D) for distance calculation.
    Space complexity: O(batch_size * N) for batch processing.

    Examples
    --------
    >>> from scptensor import impute_knn
    >>> result = impute_knn(container, "proteins", k=5, weights="distance")
    >>> "imputed_knn" in result.assays["proteins"].layers
    True
    """
    if k <= 0:
        raise ScpValueError(
            f"Number of neighbors (k) must be positive, got {k}. "
            "Use k >= 1 for nearest neighbor imputation.",
            parameter="k",
            value=k,
        )
    if weights not in ("uniform", "distance"):
        raise ScpValueError(
            f"Weights must be 'uniform' or 'distance', got '{weights}'. "
            "Use 'uniform' for equal weighting or 'distance' for inverse distance weighting.",
            parameter="weights",
            value=weights,
        )
    if batch_size <= 0:
        raise ScpValueError(
            f"Batch size must be positive, got {batch_size}. "
            "Use batch_size >= 1 to process samples in batches.",
            parameter="batch_size",
            value=batch_size,
        )
    if oversample_factor < 1:
        raise ScpValueError(
            f"Oversample factor must be at least 1, got {oversample_factor}. "
            "Use oversample_factor >= 1 to control neighbor search range.",
            parameter="oversample_factor",
            value=oversample_factor,
        )

    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    matrix = assay.layers[source_layer]
    X = matrix.X.copy()
    n_samples, n_features = X.shape

    missing_mask = np.isnan(X)
    if not np.any(missing_mask):
        print("No missing values found. Copying layer.")
        new_matrix = ScpMatrix(X=X, M=_update_imputed_mask(matrix.M, missing_mask))
        assay.add_layer(new_layer_name or "imputed_knn", new_matrix)
        return container

    X_imputed = X.copy()
    samples_with_missing = np.where(np.any(missing_mask, axis=1))[0]
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0

    # Search k_search neighbors to buffer against missingness
    k_search = min(k * oversample_factor + 1, n_samples)

    for start_idx in range(0, len(samples_with_missing), batch_size):
        end_idx = min(start_idx + batch_size, len(samples_with_missing))
        batch_indices = samples_with_missing[start_idx:end_idx]
        X_batch = X[batch_indices]
        dist_matrix_batch = nan_euclidean_distances(X_batch, X)

        for i, global_idx in enumerate(batch_indices):
            dists = dist_matrix_batch[i]

            # Find k_search nearest neighbors using argpartition (O(N))
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

            # Get all neighbor values for missing features at once: (n_potential, n_missing)
            neighbor_values = X[np.ix_(potential_neighbors, sample_missing_feats)]

            # Filter to valid (non-NaN) neighbors per feature
            valid_mask = ~np.isnan(neighbor_values)

            # For each feature, count valid neighbors and select top k
            n_valid_per_feat = np.sum(valid_mask, axis=0)

            # Find features with no valid neighbors (fallback to column mean)
            no_valid_mask = n_valid_per_feat == 0

            if np.any(no_valid_mask):
                fallback_feats = sample_missing_feats[no_valid_mask]
                X_imputed[global_idx, fallback_feats] = col_means[fallback_feats]

                # Keep only features with at least one valid neighbor
                has_valid_mask = ~no_valid_mask
                if not np.any(has_valid_mask):
                    continue

                sample_missing_feats = sample_missing_feats[has_valid_mask]
                neighbor_values = neighbor_values[:, has_valid_mask]
                valid_mask = valid_mask[:, has_valid_mask]

            # For each feature, impute using top k valid neighbors
            for feat_idx, feat_pos in enumerate(sample_missing_feats):
                feat_mask = valid_mask[:, feat_idx]
                if not np.any(feat_mask):
                    X_imputed[global_idx, feat_pos] = col_means[feat_pos]
                    continue

                # Get valid neighbors for this feature
                valid_vals = neighbor_values[feat_mask, feat_idx]
                valid_dists = potential_dists[feat_mask]

                if len(valid_vals) > 0:
                    imputed_val = _compute_weighted_imputation(
                        valid_vals.reshape(-1, 1),
                        valid_dists,
                        weights,
                        k,
                    )[0]
                    X_imputed[global_idx, feat_pos] = imputed_val

    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_knn"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_knn",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "k": k,
            "weights": weights,
            "oversample_factor": oversample_factor,
        },
        description=f"KNN imputation (k={k}, weights={weights}) on assay '{assay_name}'.",
    )

    return container
