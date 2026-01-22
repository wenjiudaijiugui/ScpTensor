"""
Data structure preservation evaluation metrics.

This module implements metrics to assess how well the pipeline
preserves the underlying data structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass


def compute_pca_variance(
    container: Any,
    n_components: int = 10,
) -> np.ndarray:
    """
    Compute PCA variance explained for each component.

    Parameters
    ----------
    container : ScpContainer
        Data container
    n_components : int, default 10
        Number of PCA components to compute

    Returns
    -------
    np.ndarray
        Array of variance explained for each component

    Examples
    --------
    >>> variance = compute_pca_variance(container, n_components=10)
    >>> print(f"Total variance explained: {np.sum(variance):.2%}")
    """
    X = _get_data_matrix(container)

    # Limit n_components to min(n_samples, n_features)
    n_components = min(n_components, X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components)
    pca.fit(X)

    return pca.explained_variance_ratio_


def compute_nn_consistency(
    original: Any,
    result: Any,
    k: int = 10,
) -> float:
    """
    Compute nearest neighbor consistency (Jaccard similarity).

    Measures how many of the k nearest neighbors are preserved
    after processing. Higher values indicate better preservation
    of local structure.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container
    k : int, default 10
        Number of nearest neighbors

    Returns
    -------
    float
        Average Jaccard similarity between 0 and 1

    Examples
    --------
    >>> consistency = compute_nn_consistency(original, processed, k=10)
    >>> print(f"NN consistency: {consistency:.2%}")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    n_samples = min(X_orig.shape[0], X_result.shape[0])
    X_orig = X_orig[:n_samples]
    X_result = X_result[:n_samples]

    # Handle edge cases
    if n_samples <= k + 1:
        return 0.0

    # Find k-nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=min(k + 1, n_samples - 1)).fit(X_orig)
    _, indices_orig = nbrs_orig.kneighbors(X_orig)

    # Find k-nearest neighbors in result space
    nbrs_result = NearestNeighbors(n_neighbors=min(k + 1, n_samples - 1)).fit(X_result)
    _, indices_result = nbrs_result.kneighbors(X_result)

    # Exclude self (first neighbor)
    indices_orig = indices_orig[:, 1 : k + 1]
    indices_result = indices_result[:, 1 : k + 1]

    # Compute Jaccard similarity for each sample
    jaccard_scores = []
    for i in range(n_samples):
        set_orig = set(indices_orig[i])
        set_result = set(indices_result[i])

        intersection = len(set_orig & set_result)
        union = len(set_orig | set_result)

        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

    return float(np.mean(jaccard_scores))


def compute_distance_preservation(
    original: Any,
    result: Any,
    method: str = "spearman",
    sample_size: int | None = None,
) -> float:
    """
    Compute distance preservation using correlation of pairwise distances.

    Measures whether the relative distances between cells are preserved
    after processing.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container
    method : str, default "spearman"
        Correlation method ('spearman' or 'pearson')
    sample_size : int, optional
        Number of distance pairs to sample (for large datasets)

    Returns
    -------
    float
        Correlation coefficient between -1 and 1
        (higher = better distance preservation)

    Examples
    --------
    >>> # Use Spearman correlation (rank-based)
    >>> corr = compute_distance_preservation(orig, proc, method="spearman")
    >>> print(f"Distance correlation: {corr:.3f}")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    n_samples = min(X_orig.shape[0], X_result.shape[0])
    X_orig = X_orig[:n_samples]
    X_result = X_result[:n_samples]

    # Compute pairwise distances
    dist_orig = pdist(X_orig, metric="euclidean")
    dist_result = pdist(X_result, metric="euclidean")

    # Sample if needed (for large datasets)
    if sample_size is not None and len(dist_orig) > sample_size:
        idx = np.random.choice(len(dist_orig), sample_size, replace=False)
        dist_orig = dist_orig[idx]
        dist_result = dist_result[idx]

    # Compute correlation
    if method == "spearman":
        corr, _ = spearmanr(dist_orig, dist_result)
    else:  # pearson
        corr, _ = pearsonr(dist_orig, dist_result)

    return float(corr)


def compute_global_structure(
    original: Any,
    result: Any,
) -> dict[str, float]:
    """
    Compute global structure preservation metrics.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container

    Returns
    -------
    dict[str, float]
        Dictionary with global structure metrics:
        - centroid_distance: Distance between centroids
        - variance_ratio: Ratio of variances (result/original)
        - covariance_alignment: Alignment of covariance structures

    Examples
    --------
    >>> global_metrics = compute_global_structure(original, processed)
    >>> print(f"Centroid distance: {global_metrics['centroid_distance']:.4f}")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    n_samples = min(X_orig.shape[0], X_result.shape[0])
    X_orig = X_orig[:n_samples]
    X_result = X_result[:n_samples]

    # Compute centroid distances
    centroid_orig = np.mean(X_orig, axis=0)
    centroid_result = np.mean(X_result, axis=0)

    centroid_distance = np.linalg.norm(centroid_orig - centroid_result)

    # Compute variance ratio
    var_orig = np.var(X_orig)
    var_result = np.var(X_result)

    variance_ratio = var_result / var_orig if var_orig > 0 else 0.0

    # Compute covariance alignment (using subsampled PCA)
    n_features = X_orig.shape[1]
    n_components_pca = min(10, n_samples, n_features)

    try:
        pca_orig = PCA(n_components=n_components_pca)
        pca_result = PCA(n_components=n_components_pca)

        pca_orig.fit(X_orig)
        pca_result.fit(X_result)

        # Compare principal components using vector cosine similarity
        components_orig = pca_orig.components_.flatten()
        components_result = pca_result.components_.flatten()

        # Cosine similarity
        dot_product = np.dot(components_orig, components_result)
        norm_orig = np.linalg.norm(components_orig)
        norm_result = np.linalg.norm(components_result)

        covariance_alignment = (
            dot_product / (norm_orig * norm_result) if norm_orig > 0 and norm_result > 0 else 0.0
        )
    except Exception:
        covariance_alignment = 0.0

    return {
        "centroid_distance": float(centroid_distance),
        "variance_ratio": float(variance_ratio),
        "covariance_alignment": float(covariance_alignment),
    }


def compute_density_preservation(
    original: Any,
    result: Any,
    k: int = 10,
) -> dict[str, float]:
    """
    Compute local density preservation metrics.

    Measures whether local density patterns are preserved after processing.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container
    k : int, default 10
        Number of neighbors for density estimation

    Returns
    -------
    dict[str, float]
        Dictionary with density preservation metrics:
        - density_correlation: Correlation of local densities
        - density_rmse: RMSE between densities

    Examples
    --------
    >>> density = compute_density_preservation(original, processed, k=10)
    >>> print(f"Density correlation: {density['density_correlation']:.3f}")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    n_samples = min(X_orig.shape[0], X_result.shape[0])
    X_orig = X_orig[:n_samples]
    X_result = X_result[:n_samples]

    # Compute local density using k-NN distances
    nbrs_orig = NearestNeighbors(n_neighbors=min(k + 1, n_samples - 1)).fit(X_orig)
    dists_orig, _ = nbrs_orig.kneighbors(X_orig)

    nbrs_result = NearestNeighbors(n_neighbors=min(k + 1, n_samples - 1)).fit(X_result)
    dists_result, _ = nbrs_result.kneighbors(X_result)

    # Exclude self (first neighbor)
    dists_orig = dists_orig[:, 1:]
    dists_result = dists_result[:, 1:]

    # Compute density as inverse of mean distance
    density_orig = 1.0 / (np.mean(dists_orig, axis=1) + 1e-10)
    density_result = 1.0 / (np.mean(dists_result, axis=1) + 1e-10)

    # Compute correlation
    corr, _ = spearmanr(density_orig, density_result)

    # Compute RMSE
    rmse = np.sqrt(np.mean((density_orig - density_result) ** 2))

    return {
        "density_correlation": float(corr),
        "density_rmse": float(rmse),
    }


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

    # Prefer normalized layer
    layer_name = "log" if "log" in assay.layers else "X"
    if layer_name not in assay.layers:
        if assay.layers:
            layer_name = list(assay.layers.keys())[0]
        else:
            raise ValueError("Assay has no layers")

    X = assay.layers[layer_name].X

    # Convert sparse to dense
    if hasattr(X, "toarray"):
        return X.toarray()
    return X
