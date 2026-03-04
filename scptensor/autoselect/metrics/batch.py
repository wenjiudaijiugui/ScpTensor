"""Batch effect metrics for automatic method selection.

This module provides functions to compute batch effect related metrics
for evaluating batch correction effectiveness in DIA-based single-cell proteomics analysis.

All metrics return values in the range [0, 1], where higher values indicate
better batch mixing or biological signal preservation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Numerical stability constant
_EPS = 1e-10


def batch_asw(X: NDArray[np.float64], batch_labels: NDArray[np.int_]) -> float:
    """Calculate batch average silhouette width (lower is better, returns 1-asw).

    Measures the degree of batch mixing. Lower ASW indicates better batch
    integration (batches are well-mixed). The function returns 1 - ASW so
    that higher values indicate better batch mixing.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features), typically
        dimensionality-reduced embeddings.
    batch_labels : NDArray[np.int_]
        Batch labels for each sample, shape (n_samples,).

    Returns
    -------
    float
        Batch mixing score in range [0, 1]. Higher values indicate better
        batch mixing (lower batch effect). Returns 0.0 for edge cases.

    Notes
    -----
    The silhouette score measures how similar a sample is to its own cluster
    (batch) compared to other clusters. For batch effect evaluation, we want
    samples from the same batch to NOT cluster together, so a lower ASW
    indicates better mixing.

    The returned value is 1 - ASW, so higher values are better.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> batch_labels = np.repeat([0, 1], 50)
    >>> score = batch_asw(X, batch_labels)
    >>> 0.0 <= score <= 1.0
    True
    """
    from sklearn.metrics import silhouette_score

    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(batch_labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but "
            f"batch_labels has {len(batch_labels)} elements"
        )

    # Check for minimum number of batches
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    batch_labels_clean = batch_labels[valid_mask]

    # Check if we still have at least 2 batches after filtering
    unique_batches_clean = np.unique(batch_labels_clean)
    if len(unique_batches_clean) < 2:
        return 0.0

    # Check minimum samples per batch
    batch_counts = np.bincount(batch_labels_clean)
    if len(batch_counts) > 0 and np.min(batch_counts[batch_counts > 0]) < 2:
        return 0.0

    try:
        # Calculate silhouette score for batch labels
        asw = silhouette_score(X_clean, batch_labels_clean)

        # Return 1 - ASW so higher is better (better mixing = lower ASW)
        score = 1.0 - asw

        # Clamp to [0, 1]
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        # Handle any sklearn errors
        return 0.0


def bio_asw(X: NDArray[np.float64], bio_labels: NDArray[np.int_]) -> float:
    """Calculate biological group average silhouette width (higher is better).

    Measures the preservation of biological signal after batch correction.
    Higher ASW indicates that biological groups are well-separated,
    meaning the biological signal is preserved.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features), typically
        dimensionality-reduced embeddings.
    bio_labels : NDArray[np.int_]
        Biological group labels for each sample (e.g., cell types),
        shape (n_samples,).

    Returns
    -------
    float
        Biological signal preservation score in range [0, 1]. Higher values
        indicate better preservation of biological structure.
        Returns 0.0 for edge cases.

    Notes
    -----
    The silhouette score measures how well biological groups are separated.
    Higher values indicate that samples cluster well with their biological
    group, which is desirable.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> bio_labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = bio_asw(X, bio_labels)
    >>> 0.0 <= score <= 1.0
    True
    """
    from sklearn.metrics import silhouette_score

    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(bio_labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but "
            f"bio_labels has {len(bio_labels)} elements"
        )

    # Check for minimum number of biological groups
    unique_bio = np.unique(bio_labels)
    if len(unique_bio) < 2:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    bio_labels_clean = bio_labels[valid_mask]

    # Check if we still have at least 2 groups after filtering
    unique_bio_clean = np.unique(bio_labels_clean)
    if len(unique_bio_clean) < 2:
        return 0.0

    # Check minimum samples per group
    group_counts = np.bincount(bio_labels_clean)
    if len(group_counts) > 0 and np.min(group_counts[group_counts > 0]) < 2:
        return 0.0

    try:
        # Calculate silhouette score for biological labels
        asw = silhouette_score(X_clean, bio_labels_clean)

        # Clamp to [0, 1]
        return float(np.clip(asw, 0.0, 1.0))
    except Exception:
        # Handle any sklearn errors
        return 0.0


def batch_mixing_score(
    X: NDArray[np.float64],
    batch_labels: NDArray[np.int_],
    n_neighbors: int = 50,
) -> float:
    """Calculate batch mixing score (simplified LISI).

    Evaluates how well samples from different batches are mixed in the
    local neighborhood of each sample. This is a simplified version of
    the Local Inverse Simpson's Index (LISI).

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features), typically
        dimensionality-reduced embeddings.
    batch_labels : NDArray[np.int_]
        Batch labels for each sample, shape (n_samples,).
    n_neighbors : int, optional
        Number of neighbors to consider, by default 50.

    Returns
    -------
    float
        Batch mixing score in range [0, 1]. Higher values indicate better
        batch mixing. Returns 0.0 for edge cases.

    Notes
    -----
    For each sample, we look at its k nearest neighbors and calculate
    the Simpson's diversity index based on batch proportions. The score
    is then averaged across all samples.

    A score of 1.0 indicates perfect mixing (all batches equally represented
    in each neighborhood), while lower scores indicate batch clustering.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> batch_labels = np.repeat([0, 1], 50)
    >>> score = batch_mixing_score(X, batch_labels, n_neighbors=10)
    >>> 0.0 <= score <= 1.0
    True
    """
    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    if len(batch_labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but "
            f"batch_labels has {len(batch_labels)} elements"
        )

    n_samples = X.shape[0]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 0.0

    # Adjust n_neighbors if necessary
    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors < 1:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    X_clean = X[valid_mask]
    batch_labels_clean = batch_labels[valid_mask]
    n_samples_clean = X_clean.shape[0]

    if n_samples_clean < 2:
        return 0.0

    # Re-check batches after filtering
    unique_batches_clean = np.unique(batch_labels_clean)
    if len(unique_batches_clean) < 2:
        return 0.0

    # Re-adjust n_neighbors
    n_neighbors = min(n_neighbors, n_samples_clean - 1)

    try:
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto").fit(X_clean)
        distances, indices = nbrs.kneighbors(X_clean)

        # Calculate mixing score for each sample
        scores = []
        for i in range(n_samples_clean):
            # Get neighbor batch labels (excluding self)
            neighbor_indices = indices[i, 1:]  # Exclude self (first neighbor)
            neighbor_batches = batch_labels_clean[neighbor_indices]

            # Count batch proportions
            batch_counts = np.bincount(neighbor_batches, minlength=len(unique_batches_clean))
            proportions = batch_counts / len(neighbor_batches)

            # Simpson's diversity index: 1 - sum(p^2)
            # Higher value = more diverse (better mixing)
            simpson = 1.0 - np.sum(proportions**2)
            scores.append(simpson)

        # Average score, normalized to [0, 1]
        # Simpson's index ranges from 0 (no diversity) to 1 - 1/n_batches (max diversity)
        max_simpson = 1.0 - 1.0 / n_batches
        if max_simpson < _EPS:
            return 0.0

        avg_score = np.mean(scores)
        normalized_score = avg_score / max_simpson

        return float(np.clip(normalized_score, 0.0, 1.0))
    except Exception:
        return 0.0
