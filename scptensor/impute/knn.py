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


def knn(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str | None = "imputed_knn",
    k: int = 5,
    weights: str = "uniform",
    batch_size: int = 500,
    oversample_factor: int = 3,
) -> ScpContainer:
    """
    Impute missing values using k-Nearest Neighbors with over-sampling.

    This implementation addresses "effective K decay" by searching for
    k_search = oversample_factor * k neighbors, then filtering to find
    valid (non-missing) neighbors per feature.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values
    assay_name : str
        Name of the assay to use
    base_layer : str
        Name of the layer containing data with missing values
    new_layer_name : str, optional
        Name for the new layer with imputed data (default: 'imputed_knn')
    k : int, optional
        Number of neighbors to use (default: 5)
    weights : str, optional
        Weight function: 'uniform' or 'distance' (default: 'uniform')
    batch_size : int, optional
        Number of rows to process at once (default: 500)
    oversample_factor : int, optional
        Multiplier for search range k_search = oversample_factor * k (default: 3)

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist
    LayerNotFoundError
        If the specified layer does not exist
    ScpValueError
        If parameters are invalid

    Notes
    -----
    Time complexity: O(M * N * D) for distance calculation.
    Space complexity: O(batch_size * N) for batch processing.
    """
    # Parameter validation
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
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    # Get data
    matrix = assay.layers[base_layer]
    X = matrix.X.copy()
    n_samples, n_features = X.shape

    # Check for missing values
    missing_mask = np.isnan(X)
    if not np.any(missing_mask):
        print("No missing values found. Copying layer.")
        new_matrix = ScpMatrix(X=X, M=_update_imputed_mask(matrix.M, missing_mask))
        assay.add_layer(new_layer_name or "imputed_knn", new_matrix)
        return container

    # Imputation setup
    X_imputed = X.copy()
    samples_with_missing = np.where(np.any(missing_mask, axis=1))[0]
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0

    # Search k_search neighbors to buffer against missingness
    k_search = min(k * oversample_factor + 1, n_samples)

    # Process in batches to save memory
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

            # Impute each missing feature
            sample_missing_feats = np.where(missing_mask[global_idx])[0]

            for f_idx in sample_missing_feats:
                # Get valid neighbors for this feature
                neighbor_values = X[potential_neighbors, f_idx]
                valid_val_mask = ~np.isnan(neighbor_values)

                valid_neighbors_val = neighbor_values[valid_val_mask]
                valid_neighbors_dist = potential_dists[valid_val_mask]

                # Use top k valid neighbors
                if len(valid_neighbors_val) > k:
                    vals = valid_neighbors_val[:k]
                    ds = valid_neighbors_dist[:k]
                else:
                    vals = valid_neighbors_val
                    ds = valid_neighbors_dist

                # Fallback to column mean if no valid neighbors
                if len(vals) == 0:
                    X_imputed[global_idx, f_idx] = col_means[f_idx]
                    continue

                # Compute weighted average
                if weights == "distance":
                    with np.errstate(divide="ignore"):
                        w = 1.0 / ds

                    inf_mask = np.isinf(w)
                    if np.any(inf_mask):
                        w[inf_mask] = 1.0
                        w[~inf_mask] = 0.0

                    w_sum = np.sum(w)
                    imputed_val = (
                        np.dot(vals, w / w_sum) if w_sum > 1e-10 else np.mean(vals)
                    )
                else:
                    imputed_val = np.mean(vals)

                X_imputed[global_idx, f_idx] = imputed_val

    # Create new layer with updated mask
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_knn"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="knn",
        params={
            "assay": assay_name,
            "base_layer": base_layer,
            "new_layer": layer_name,
            "k": k,
            "weights": weights,
            "oversample_factor": oversample_factor,
        },
        description=f"KNN imputation (k={k}, weights={weights}) on assay '{assay_name}'.",
    )

    return container
