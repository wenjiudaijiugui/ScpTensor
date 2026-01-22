"""
Batch effect evaluation metrics.

This module implements metrics to assess the effectiveness of batch
effect removal in single-cell proteomics data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass


def compute_kbet(container: Any, k: int = 25) -> float:
    """
    Compute kBET score to measure local batch mixing.

    kBET (k-nearest neighbour batch effect test) measures whether the
    local neighborhood of each cell is well-mixed with respect to batches.

    Parameters
    ----------
    container : ScpContainer
        Data container with batch information in obs
    k : int, default 25
        Number of nearest neighbors to consider

    Returns
    -------
    float
        kBET score between 0 and 1, where higher values indicate
        better batch mixing

    References
    ----------
    Büttner, M. et al. (2019) Nature Methods
    """
    # Get data matrix and batch labels
    X = _get_embeddings(container)
    batch_labels = _get_batch_labels(container)

    n_cells = X.shape[0]

    # Handle edge cases
    if n_cells <= k + 1:
        return 0.0

    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches <= 1:
        # Only one batch - perfect mixing by definition
        return 1.0

    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_cells - 1)).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1 : k + 1]

    # Compute expected batch frequencies
    batch_counts = np.array([np.sum(batch_labels == batch) for batch in unique_batches])
    expected_freq = batch_counts / n_cells

    # Calculate kBET rejection rate
    n_rejected = 0
    for i in range(n_cells):
        neighbor_batches = batch_labels[neighbor_indices[i]]

        # Count observed frequencies
        observed_freq = np.array(
            [np.sum(neighbor_batches == batch) / k for batch in unique_batches],
        )

        # Chi-square-like test: check if any batch is significantly overrepresented
        # Using a threshold of 0.2 (20% deviation)
        if np.any(np.abs(observed_freq - expected_freq) > 0.2):
            n_rejected += 1

    # kBET score = 1 - rejection rate
    kbet_score = 1.0 - (n_rejected / n_cells)
    return float(kbet_score)


def compute_lisi(container: Any, k: int = 25, perplexity: float = 30.0) -> float:
    """
    Compute LISI (Local Inverse Simpson Index).

    LISI measures local diversity in the neighborhood, adapted for
    batch mixing assessment.

    Parameters
    ----------
    container : ScpContainer
        Data container with batch information
    k : int, default 25
        Number of nearest neighbors
    perplexity : float, default 30.0
        Perplexity parameter for LISI calculation (currently unused)

    Returns
    -------
    float
        Average LISI score across all cells

    References
    ----------
    Korsunsky, I. et al. (2019) Nature Methods
    """
    X = _get_embeddings(container)
    batch_labels = _get_batch_labels(container)

    n_cells = X.shape[0]
    n_batches = len(np.unique(batch_labels))

    # Handle edge cases
    if n_cells <= k + 1:
        return float(n_batches)

    if n_batches <= 1:
        return 1.0

    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_cells - 1)).fit(X)
    distances, indices = nbrs.kneighbors(X)

    neighbor_indices = indices[:, 1 : k + 1]

    # Compute inverse Simpson index for each cell
    lisi_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[neighbor_indices[i]]
        unique_batches, counts = np.unique(neighbor_batches, return_counts=True)

        if len(counts) == 0:
            lisi_scores.append(1.0)
            continue

        # Simpson's diversity index
        probs = counts / k
        simpson = np.sum(probs**2)

        # Inverse Simpson index
        inverse_simpson = 1.0 / simpson if simpson > 0 else 1.0
        lisi_scores.append(inverse_simpson)

    return float(np.mean(lisi_scores))


def compute_mixing_entropy(container: Any, k_neighbors: int = 25) -> float:
    """
    Compute batch mixing entropy based on local neighborhood composition.

    Parameters
    ----------
    container : ScpContainer
        Data container with batch information
    k_neighbors : int, default 25
        Number of neighbors to consider

    Returns
    -------
    float
        Average mixing entropy (higher is better, normalized to [0, 1])
    """
    X = _get_embeddings(container)
    batch_labels = _get_batch_labels(container)

    n_cells = X.shape[0]
    n_batches = len(np.unique(batch_labels))

    # Handle edge cases
    if n_cells <= k_neighbors + 1:
        return 1.0 if n_batches > 1 else 0.0

    if n_batches <= 1:
        return 0.0

    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_cells - 1)).fit(X)
    distances, indices = nbrs.kneighbors(X)

    neighbor_indices = indices[:, 1 : k_neighbors + 1]

    # Compute Shannon entropy for each cell
    entropies = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[neighbor_indices[i]]

        if len(neighbor_batches) == 0:
            entropies.append(0.0)
            continue

        unique_batches, counts = np.unique(neighbor_batches, return_counts=True)

        # Normalize to get probabilities
        probs = counts / k_neighbors

        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    # Normalize by max possible entropy (log(n_batches))
    max_entropy = np.log(n_batches) if n_batches > 1 else 1.0
    normalized_entropy = np.mean(entropies) / max_entropy

    return float(normalized_entropy)


def compute_variance_ratio(container: Any) -> float:
    """
    Compute ratio of within-batch variance to between-batch variance.

    Lower values indicate better batch correction (less between-batch variance
    relative to within-batch variance). Higher values suggest that batch effects
    are still present.

    Parameters
    ----------
    container : ScpContainer
        Data container with batch information

    Returns
    -------
    float
        Variance ratio (within/between). Higher is better for batch correction.
    """
    X = _get_data_matrix(container)
    batch_labels = _get_batch_labels(container)

    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    # Handle edge case: only one batch
    if n_batches <= 1:
        return 0.0

    # Compute overall mean
    overall_mean = np.mean(X, axis=0)

    # Within-batch and between-batch variance
    within_var = 0.0
    between_var = 0.0
    n_total = X.shape[0]

    for batch in unique_batches:
        batch_mask = batch_labels == batch
        batch_data = X[batch_mask]
        n_batch = np.sum(batch_mask)

        if n_batch == 0:
            continue

        batch_mean = np.mean(batch_data, axis=0)

        # Within: variance from batch mean
        within_var += np.sum((batch_data - batch_mean) ** 2)

        # Between: variance of batch mean from overall mean
        between_var += n_batch * np.sum((batch_mean - overall_mean) ** 2)

    # Avoid division by zero
    if between_var == 0:
        return float("inf")

    # F-statistic: ratio of between to within variance
    # Higher F-score = more batch effect (bad)
    # We return within/between for intuitive interpretation
    variance_ratio = within_var / between_var
    return float(variance_ratio)


def _get_embeddings(container: Any) -> np.ndarray:
    """
    Helper to get embeddings for evaluation.

    Priority order:
    1. PCA embeddings in obs['X_pca']
    2. UMAP embeddings in obs['X_umap']
    3. Raw data from first assay

    Returns
    -------
    np.ndarray
        Embeddings matrix of shape (n_cells, n_features)
    """
    # Try PCA embeddings first
    if hasattr(container, "obs") and "X_pca" in container.obs.columns:
        pca_data = container.obs["X_pca"].to_numpy()
        # Stack if it's an array of arrays
        if pca_data.ndim == 1 and len(pca_data) > 0:
            return np.vstack(pca_data)
        return pca_data

    # Try UMAP embeddings
    if hasattr(container, "obs") and "X_umap" in container.obs.columns:
        umap_data = container.obs["X_umap"].to_numpy()
        if umap_data.ndim == 1 and len(umap_data) > 0:
            return np.vstack(umap_data)
        return umap_data

    # Fallback to raw data
    return _get_data_matrix(container)


def _get_data_matrix(container: Any) -> np.ndarray:
    """
    Helper to get dense data matrix.

    Returns
    -------
    np.ndarray
        Dense data matrix of shape (n_cells, n_features)
    """
    if not hasattr(container, "assays") or not container.assays:
        raise ValueError("Container has no assays")

    # Get first assay
    assay_name = list(container.assays.keys())[0]
    assay = container.assays[assay_name]

    # Prefer normalized layer over raw
    layer_name = "log" if "log" in assay.layers else "X"
    if layer_name not in assay.layers:
        # Try to get any available layer
        if assay.layers:
            layer_name = list(assay.layers.keys())[0]
        else:
            raise ValueError("Assay has no layers")

    X = assay.layers[layer_name].X

    # Convert sparse to dense
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def _get_batch_labels(container: Any) -> np.ndarray:
    """
    Helper to get batch labels from container.

    Returns
    -------
    np.ndarray
        Batch labels as integer codes
    """
    if not hasattr(container, "obs"):
        # Return single batch if no obs
        return np.zeros(_get_embeddings(container).shape[0], dtype=int)

    # Check for 'batch' column
    if "batch" in container.obs.columns:
        batch_col = container.obs["batch"]
        # Convert to numpy and then to integer codes
        batch_values = batch_col.to_numpy()
        # Convert string batches to integer codes
        unique_batches = np.unique(batch_values)
        batch_to_code = {batch: idx for idx, batch in enumerate(unique_batches)}
        return np.array([batch_to_code[b] for b in batch_values], dtype=int)

    # Check for other common batch column names
    for col_name in ["batch_id", "Batch"]:
        if col_name in container.obs.columns:
            batch_col = container.obs[col_name]
            batch_values = batch_col.to_numpy()
            unique_batches = np.unique(batch_values)
            batch_to_code = {batch: idx for idx, batch in enumerate(unique_batches)}
            return np.array([batch_to_code[b] for b in batch_values], dtype=int)

    # Return single batch if no batch information found
    return np.zeros(_get_embeddings(container).shape[0], dtype=int)
