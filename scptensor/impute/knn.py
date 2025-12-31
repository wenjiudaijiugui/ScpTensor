from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay

def knn(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: Optional[str] = 'imputed_knn',
    k: int = 5,
    weights: str = 'uniform',
    batch_size: int = 500,
    oversample_factor: int = 3
) -> ScpContainer:
    """
    Impute missing values using k-Nearest Neighbors with Over-sampling and Filtering.
    
    Refactored V4:
    1. Time Complexity: Dominated by distance calculation O(M * N * D).
    2. Space Complexity: O(Batch * N) due to batch processing.
    3. Algorithm Correctness: Addresses "Effective K Decay" by over-sampling neighbors.
       We search for k_search = oversample_factor * k neighbors first.
       Then, for each missing feature, we filter these candidates to find valid (non-missing) ones.
       We select the top k valid neighbors for imputation.
    4. Numerical Stability: Adds protection for small weights sum in weighted KNN.
    
    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        base_layer: Name of the layer containing data with missing values.
        new_layer_name: Name for the new layer with imputed data.
        k: Number of neighbors to use for calculation.
        weights: Weight function used in prediction. 'uniform' or 'distance'.
        batch_size: Number of rows to process at once to control memory usage.
        oversample_factor: Factor to determine search range (k_search = oversample_factor * k).
                           Higher values reduce the chance of effective k decay but increase computation slightly.
        
    Returns:
        ScpContainer: The updated container with the new layer.
    """
    # 1. Validation
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")
        
    # 2. Get Data
    matrix = assay.layers[base_layer]
    X = matrix.X.copy() # (N_samples, N_features)
    n_samples, n_features = X.shape
    
    # Identify missing values
    missing_mask = np.isnan(X)
    if not np.any(missing_mask):
        print("No missing values found. Copying layer.")
        new_matrix = ScpMatrix(X=X, M=matrix.M.copy())
        assay.add_layer(new_layer_name, new_matrix)
        return container

    # 3. Imputation with Batch Processing
    X_imputed = X.copy()
    
    # Identify samples that need imputation
    samples_with_missing = np.where(np.any(missing_mask, axis=1))[0]
    n_missing_samples = len(samples_with_missing)
    
    # Pre-calculate column means for fallback
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0 # Handle all-NaN columns
    
    # ---------------------------------------------------------
    # NOTE: Over-sampling Strategy
    # We search for k_search potential neighbors to buffer against missingness.
    # ---------------------------------------------------------
    k_search = min(k * oversample_factor + 1, n_samples)
    
    # Process in batches to save memory
    for start_idx in range(0, n_missing_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_missing_samples)
        batch_indices = samples_with_missing[start_idx:end_idx]
        
        # X_batch: (batch_size, n_features)
        X_batch = X[batch_indices]
        
        # Compute distances between batch samples and ALL samples
        # dist_matrix_batch: (batch_size, n_samples)
        dist_matrix_batch = nan_euclidean_distances(X_batch, X)
        
        # Iterate through each sample in the batch
        for i, global_idx in enumerate(batch_indices):
            dists = dist_matrix_batch[i]
            
            # Find k_search nearest neighbors (candidates)
            # Use argpartition for O(N) complexity
            neighbor_candidates_arg = np.argpartition(dists, k_search - 1)[:k_search]
            
            # Retrieve distances for these candidates
            candidate_dists = dists[neighbor_candidates_arg]
            
            # Sort candidates to prioritize nearest ones
            sorted_order = np.argsort(candidate_dists)
            neighbor_candidates_sorted = neighbor_candidates_arg[sorted_order]
            candidate_dists_sorted = candidate_dists[sorted_order]
            
            # Filter out the sample itself
            mask_not_self = neighbor_candidates_sorted != global_idx
            potential_neighbors = neighbor_candidates_sorted[mask_not_self]
            potential_dists = candidate_dists_sorted[mask_not_self]
            
            # Impute missing features for this sample
            sample_missing_feats = np.where(missing_mask[global_idx])[0]
            
            for f_idx in sample_missing_feats:
                # ---------------------------------------------------------
                # NOTE: Dynamic Selection per Feature
                # From potential_neighbors, pick top k that have valid value at f_idx
                # ---------------------------------------------------------
                
                # Get values of potential neighbors at this feature
                neighbor_values = X[potential_neighbors, f_idx]
                
                # Identify which neighbors have valid values
                valid_val_mask = ~np.isnan(neighbor_values)
                
                # Select valid neighbors and their distances
                valid_neighbors_val = neighbor_values[valid_val_mask]
                valid_neighbors_dist = potential_dists[valid_val_mask]
                
                # If we have more than k valid neighbors, take top k
                if len(valid_neighbors_val) > k:
                    vals = valid_neighbors_val[:k]
                    ds = valid_neighbors_dist[:k]
                else:
                    # If fewer than k, use all available (graceful degradation)
                    vals = valid_neighbors_val
                    ds = valid_neighbors_dist
                
                # Check if we have ANY valid neighbors
                if len(vals) == 0:
                    # Fallback to column mean
                    X_imputed[global_idx, f_idx] = col_means[f_idx]
                    continue
                
                # Compute weighted average
                if weights == 'distance':
                    # Handle division by zero for distance
                    with np.errstate(divide='ignore'):
                        w = 1.0 / ds
                    
                    # If distance is 0 (duplicate sample), infinite weight
                    inf_mask = np.isinf(w)
                    if np.any(inf_mask):
                        w[inf_mask] = 1.0
                        w[~inf_mask] = 0.0
                    
                    w_sum = np.sum(w)
                    # Add epsilon protection for very small weights sum
                    if w_sum > 1e-10:
                        imputed_val = np.dot(vals, w / w_sum)
                    else:
                        imputed_val = np.mean(vals)
                else:
                    imputed_val = np.mean(vals)
                    
                X_imputed[global_idx, f_idx] = imputed_val

    # 4. Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=matrix.M.copy())
    assay.add_layer(new_layer_name, new_matrix)
    
    # 5. Log
    container.log_operation(
        action="knn",
        params={
            "assay": assay_name,
            "base_layer": base_layer,
            "new_layer": new_layer_name,
            "k": k,
            "weights": weights,
            "oversample_factor": oversample_factor
        },
        description=f"KNN imputation (k={k}, weights={weights}, oversample={oversample_factor}) on assay '{assay_name}'."
    )
    
    return container
