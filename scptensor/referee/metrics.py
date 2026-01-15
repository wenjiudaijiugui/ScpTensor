"""Metrics for evaluating integration quality.

This module provides metrics to assess batch correction and integration
methods for single-cell proteomics data.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, silhouette_score

if TYPE_CHECKING:
    pass


# Constants
SILHOUETTE_DOWNSAMPLE_THRESHOLD: int = 10000
SILHOUETTE_DOWNSAMPLE_SIZE: int = 10000
RANDOM_STATE: int = 42  # For reproducible downsampling


def compute_silhouette(X: NDArray[np.float64], labels: NDArray[np.intp]) -> float:
    """
    Compute Silhouette Score for clustering evaluation.

    The silhouette score measures how well-separated clusters are.
    Range: [-1, 1], where higher values indicate better-defined clusters.

    For large datasets (>10000 samples), downsampling is applied for performance.

    Parameters
    ----------
    X : NDArray[np.float64]
        Data matrix of shape (n_samples, n_features).
    labels : NDArray[np.intp]
        Cluster labels of shape (n_samples,).

    Returns
    -------
    float
        Silhouette score in [-1, 1]. Returns NaN if fewer than 2 unique labels.

    Notes
    -----
    - Score > 0.5: Well-separated clusters
    - Score ~ 0: Overlapping clusters
    - Score < 0: Incorrectly assigned clusters

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> labels = np.array([0] * 50 + [1] * 50)
    >>> score = compute_silhouette(X, labels)
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return np.nan

    n_samples = X.shape[0]

    # Downsample for speed on large datasets
    if n_samples > SILHOUETTE_DOWNSAMPLE_THRESHOLD:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(n_samples, SILHOUETTE_DOWNSAMPLE_SIZE, replace=False)
        X = X[indices]
        labels = labels[indices]

    return silhouette_score(X, labels)


def compute_batch_mixing(X: NDArray[np.float64], batch_labels: NDArray[np.intp]) -> float:
    """
    Evaluate batch mixing quality using inverse silhouette score.

    For batch correction, we want batches to be well-mixed (low silhouette
    separation between batches). This metric converts silhouette to a
    "higher is better" score.

    Parameters
    ----------
    X : NDArray[np.float64]
        Data matrix of shape (n_samples, n_features).
    batch_labels : NDArray[np.intp]
        Batch assignments of shape (n_samples,).

    Returns
    -------
    float
        Batch mixing score in [0, 1], where higher is better.
        - 1.0: Perfectly mixed batches
        - 0.5: Partially mixed
        - 0.0: Completely separated batches

    Notes
    -----
    The score is computed as ``1 - |silhouette_score|``:
    - Silhouette near 0 (mixed) -> score near 1 (good)
    - Silhouette near 1 (separated) -> score near 0 (bad)

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> batches = np.array([0] * 50 + [1] * 50)
    >>> mixing = compute_batch_mixing(X, batches)
    """
    sil = compute_silhouette(X, batch_labels)
    if np.isnan(sil):
        return 0.0

    return 1.0 - abs(sil)


def compute_cluster_separation(
    X: NDArray[np.float64],
    cluster_labels: NDArray[np.intp],
) -> float:
    """
    Evaluate cluster separation using Silhouette Score.

    Unlike batch mixing, for biological clusters we want high separation.
    This directly returns the silhouette score (higher is better).

    Parameters
    ----------
    X : NDArray[np.float64]
        Data matrix of shape (n_samples, n_features).
    cluster_labels : NDArray[np.intp]
        Cluster labels of shape (n_samples,).

    Returns
    -------
    float
        Cluster separation score in [-1, 1], where higher is better.
        Returns 0.0 if fewer than 2 unique labels.

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> clusters = np.array([0] * 50 + [1] * 50)
    >>> separation = compute_cluster_separation(X, clusters)
    """
    sil = compute_silhouette(X, cluster_labels)
    return sil if not np.isnan(sil) else 0.0


def compute_ari(
    labels_true: NDArray[np.intp],
    labels_pred: NDArray[np.intp],
) -> float:
    """
    Compute Adjusted Rand Index for clustering similarity.

    ARI measures the similarity between two clusterings, adjusted for chance.
    Range: [-1, 1], where 1 indicates perfect agreement.

    Parameters
    ----------
    labels_true : NDArray[np.intp]
        Ground truth cluster labels.
    labels_pred : NDArray[np.intp]
        Predicted cluster labels.

    Returns
    -------
    float
        Adjusted Rand Index in [-1, 1].

    Examples
    --------
    >>> true = np.array([0, 0, 1, 1])
    >>> pred = np.array([0, 0, 1, 1])
    >>> compute_ari(true, pred)
    1.0
    """
    return adjusted_rand_score(labels_true, labels_pred)


def compute_reconstruction_error(
    X_orig: NDArray[np.float64],
    X_recon: NDArray[np.float64],
) -> float:
    """
    Compute Root Mean Squared Error between original and reconstructed data.

    Parameters
    ----------
    X_orig : NDArray[np.float64]
        Original data matrix.
    X_recon : NDArray[np.float64]
        Reconstructed data matrix (same shape as X_orig).

    Returns
    -------
    float
        RMSE value. Lower values indicate better reconstruction.

    Notes
    -----
    NaN values are ignored in the computation.

    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X_recon = np.array([[1.1, 2.1], [2.9, 3.9]])
    >>> rmse = compute_reconstruction_error(X, X_recon)
    """
    return np.sqrt(np.nanmean((X_orig - X_recon) ** 2))
