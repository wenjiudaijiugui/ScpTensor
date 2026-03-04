"""Clustering evaluation metrics for automatic method selection.

This module provides functions to compute clustering quality metrics
for evaluating clustering effectiveness in DIA-based single-cell proteomics analysis.

All metrics return values in the range [0, 1], where higher values indicate
better clustering quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score as sk_calinski_harabasz_score,
)
from sklearn.metrics import (
    davies_bouldin_score as sk_davies_bouldin_score,
)
from sklearn.metrics import silhouette_score as sk_silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Numerical stability constant
_EPS = 1e-10


def silhouette_score(X: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Calculate silhouette coefficient (higher is better).

    Measures how well samples are clustered with similar samples.
    Higher values indicate better-defined clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features).
    labels : NDArray[np.int_]
        Cluster labels for each sample, shape (n_samples,).

    Returns
    -------
    float
        Silhouette score in range [0, 1]. Higher values indicate better
        clustering. Returns 0.0 for edge cases.

    Notes
    -----
    The silhouette coefficient measures how similar a sample is to its own
    cluster compared to other clusters. Values range from -1 to 1, where:
    - 1: Sample is well-matched to its own cluster and poorly-matched to
      neighboring clusters
    - 0: Sample is on the boundary between two clusters
    - -1: Sample might be in the wrong cluster

    The returned value is clipped to [0, 1] since negative values typically
    indicate poor clustering.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = silhouette_score(X, labels)
    >>> 0.0 <= score <= 1.0
    True
    """
    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but labels has {len(labels)} elements"
        )

    # Check for minimum number of clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    labels_clean = labels[valid_mask]

    # Check if we still have at least 2 clusters after filtering
    unique_labels_clean = np.unique(labels_clean)
    if len(unique_labels_clean) < 2:
        return 0.0

    # Check minimum samples per cluster
    cluster_counts = np.bincount(labels_clean)
    if len(cluster_counts) > 0 and np.min(cluster_counts[cluster_counts > 0]) < 2:
        return 0.0

    try:
        score = sk_silhouette_score(X_clean, labels_clean)
        # Clip to [0, 1] - negative values indicate poor clustering
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def calinski_harabasz_score(X: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Calculate Calinski-Harabasz index (higher is better).

    Measures the ratio of between-cluster dispersion to within-cluster
    dispersion. Higher values indicate better-defined clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features).
    labels : NDArray[np.int_]
        Cluster labels for each sample, shape (n_samples,).

    Returns
    -------
    float
        Normalized Calinski-Harabasz score in range [0, 1]. Higher values
        indicate better clustering. Returns 0.0 for edge cases.

    Notes
    -----
    The Calinski-Harabasz index is defined as:
    CH = (BG / (k - 1)) / (WG / (n - k))

    where BG is between-group dispersion, WG is within-group dispersion,
    k is the number of clusters, and n is the number of samples.

    The score is normalized to [0, 1] using a sigmoid-like transformation
    for comparability with other metrics.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = calinski_harabasz_score(X, labels)
    >>> 0.0 <= score <= 1.0
    True
    """
    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but labels has {len(labels)} elements"
        )

    # Check for minimum number of clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    labels_clean = labels[valid_mask]
    n_samples = X_clean.shape[0]

    # Check if we still have at least 2 clusters after filtering
    unique_labels_clean = np.unique(labels_clean)
    n_clusters = len(unique_labels_clean)

    if n_clusters < 2:
        return 0.0

    if n_samples <= n_clusters:
        return 0.0

    try:
        ch_score = sk_calinski_harabasz_score(X_clean, labels_clean)

        # Normalize using sigmoid-like transformation
        # CH scores can vary widely, so we use a soft normalization
        # The transformation maps [0, inf) to [0, 1)
        # Using: score / (score + k) where k is a scaling factor
        # A typical good CH score is around 100-1000
        scaling_factor = 100.0
        normalized = ch_score / (ch_score + scaling_factor)

        return float(np.clip(normalized, 0.0, 1.0))
    except Exception:
        return 0.0


def davies_bouldin_score(X: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Calculate Davies-Bouldin index (lower is better, returns 1-db).

    Measures the average similarity between clusters. Lower values indicate
    better separation between clusters. The function returns 1 - DB so that
    higher values indicate better clustering.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features).
    labels : NDArray[np.int_]
        Cluster labels for each sample, shape (n_samples,).

    Returns
    -------
    float
        Normalized Davies-Bouldin score in range [0, 1]. Higher values
        indicate better clustering. Returns 0.0 for edge cases.

    Notes
    -----
    The Davies-Bouldin index is defined as the average similarity measure
    of each cluster with its most similar cluster. Lower values indicate
    better clustering.

    The DB score is typically >= 0. To normalize to [0, 1], we use:
    score = 1 / (1 + db)

    This maps db=0 (perfect) to 1.0 and db->inf to 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = davies_bouldin_score(X, labels)
    >>> 0.0 <= score <= 1.0
    True
    """
    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but labels has {len(labels)} elements"
        )

    # Check for minimum number of clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    labels_clean = labels[valid_mask]

    # Check if we still have at least 2 clusters after filtering
    unique_labels_clean = np.unique(labels_clean)
    if len(unique_labels_clean) < 2:
        return 0.0

    try:
        db_score = sk_davies_bouldin_score(X_clean, labels_clean)

        # Normalize: 1 / (1 + db) maps [0, inf) to (0, 1]
        # db = 0 (perfect) -> score = 1.0
        # db = inf -> score = 0.0
        normalized = 1.0 / (1.0 + db_score)

        return float(np.clip(normalized, 0.0, 1.0))
    except Exception:
        return 0.0


def clustering_stability(
    X: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_subsamples: int = 10,
    subsample_ratio: float = 0.8,
    random_state: int | None = None,
) -> float:
    """Calculate clustering stability through subsampling.

    Evaluates the stability of clustering results by comparing clusterings
    on different subsamples of the data. Higher stability indicates more
    robust clustering.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features).
    labels : NDArray[np.int_]
        Original cluster labels for each sample, shape (n_samples,).
    n_subsamples : int, optional
        Number of subsampling iterations, by default 10.
    subsample_ratio : float, optional
        Ratio of samples to include in each subsample, by default 0.8.
    random_state : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    float
        Stability score in range [0, 1]. Higher values indicate more stable
        clustering. Returns 0.0 for edge cases.

    Notes
    -----
    Stability is measured by:
    1. Creating multiple subsamples of the data
    2. Re-clustering each subsample using K-means with the same number of clusters
    3. Comparing the subsample clustering to the original labels using
       adjusted Rand index (ARI)

    ARI measures similarity between two clusterings, adjusted for chance:
    - ARI = 1: Perfect agreement
    - ARI = 0: Random labeling
    - ARI < 0: Less agreement than expected by chance

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = clustering_stability(X, labels, n_subsamples=5)
    >>> 0.0 <= score <= 1.0
    True
    """
    from sklearn.metrics import adjusted_rand_score

    # Handle edge cases
    if X.size == 0 or X.shape[0] < 4:
        return 0.0

    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but labels has {len(labels)} elements"
        )

    # Check for minimum number of clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    labels_clean = labels[valid_mask]
    n_samples = X_clean.shape[0]

    # Check minimum requirements for subsampling
    min_samples_for_subsample = max(4, int(1 / (1 - subsample_ratio + _EPS)))
    if n_samples < min_samples_for_subsample:
        return 0.0

    subsample_size = max(int(n_samples * subsample_ratio), 2)

    if subsample_size <= n_clusters:
        return 0.0

    # Set random state
    rng = np.random.RandomState(random_state)

    try:
        ari_scores = []

        for _ in range(n_subsamples):
            # Create random subsample indices
            subsample_indices = rng.choice(n_samples, size=subsample_size, replace=False)

            X_sub = X_clean[subsample_indices]
            labels_sub_original = labels_clean[subsample_indices]

            # Re-cluster using K-means with same number of clusters
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=rng.randint(0, 2**31),
                n_init=10,
            )
            labels_sub_new = kmeans.fit_predict(X_sub)

            # Calculate ARI between original and new clustering
            ari = adjusted_rand_score(labels_sub_original, labels_sub_new)
            ari_scores.append(ari)

        # Return average ARI
        avg_ari = np.mean(ari_scores)

        # Clip to [0, 1] - negative ARI indicates instability
        return float(np.clip(avg_ari, 0.0, 1.0))
    except Exception:
        return 0.0
