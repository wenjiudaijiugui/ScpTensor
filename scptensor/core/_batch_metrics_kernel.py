"""Shared numerical kernels for batch-mixing metrics.

This module only provides stage-agnostic math primitives. Higher-level
modules are responsible for input selection and score interpretation.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def validate_n_neighbors(n_neighbors: int) -> None:
    """Validate kNN neighborhood size."""
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")


def validate_alpha(alpha: float) -> None:
    """Validate chi-square acceptance threshold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def validate_perplexity(perplexity: float) -> None:
    """Validate target perplexity."""
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")


def compute_self_excluded_knn(
    x: np.ndarray,
    n_neighbors: int,
    *,
    algorithm: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a self-excluded kNN graph.

    The implementation does not assume that ``kneighbors`` always places the
    query point in the first column.
    """
    n_samples = int(x.shape[0])
    local_k = min(int(n_neighbors), n_samples - 1)
    if local_k < 1:
        return (
            np.empty((n_samples, 0), dtype=float),
            np.empty((n_samples, 0), dtype=int),
        )

    nbrs = NearestNeighbors(n_neighbors=local_k + 1, algorithm=algorithm).fit(x)
    distances, indices = nbrs.kneighbors(x)

    filtered_distances: list[np.ndarray] = []
    filtered_indices: list[np.ndarray] = []
    for row_idx, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=False)):
        keep_mask = row_indices != row_idx
        row_distances = row_distances[keep_mask]
        row_indices = row_indices[keep_mask]

        if row_indices.size < local_k:
            row_distances = distances[row_idx, 1 : local_k + 1]
            row_indices = indices[row_idx, 1 : local_k + 1]

        filtered_distances.append(np.asarray(row_distances[:local_k], dtype=float))
        filtered_indices.append(np.asarray(row_indices[:local_k], dtype=int))

    return np.vstack(filtered_distances), np.vstack(filtered_indices)


def entropy_and_probabilities(distances: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Return Shannon entropy and normalized neighbor probabilities."""
    probabilities = np.exp(-distances * beta)
    sum_probabilities = float(np.sum(probabilities))
    if sum_probabilities <= 0.0:
        if distances.size == 0:
            return 0.0, probabilities
        uniform = np.full(distances.shape, 1.0 / distances.size, dtype=float)
        return float(np.log(distances.size)), uniform

    weighted_distance_sum = float(np.sum(distances * probabilities))
    entropy = np.log(sum_probabilities) + beta * weighted_distance_sum / sum_probabilities
    return float(entropy), probabilities / sum_probabilities


def compute_perplexity_probabilities(
    distances: np.ndarray,
    perplexity: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    """Binary-search per-row probabilities to match a target perplexity."""
    target_entropy = float(np.log(perplexity))
    probabilities = np.zeros_like(distances, dtype=float)

    for row_idx, row_distances in enumerate(distances):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        entropy, row_probabilities = entropy_and_probabilities(row_distances, beta)
        entropy_diff = entropy - target_entropy

        n_iter = 0
        while abs(entropy_diff) > tol and n_iter < max_iter:
            if entropy_diff > 0.0:
                beta_min = beta
                beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

            entropy, row_probabilities = entropy_and_probabilities(row_distances, beta)
            entropy_diff = entropy - target_entropy
            n_iter += 1

        probabilities[row_idx] = row_probabilities

    return probabilities


def compute_inverse_simpson_scores(
    label_codes: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_probabilities: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Compute weighted inverse Simpson diversity per sample."""
    scores = np.ones(neighbor_indices.shape[0], dtype=float)
    for row_idx, neighbors in enumerate(neighbor_indices):
        weights = neighbor_probabilities[row_idx]
        local_probabilities = np.bincount(
            label_codes[neighbors],
            weights=weights,
            minlength=n_labels,
        )
        simpson_index = float(np.sum(local_probabilities**2))
        if simpson_index > 0.0:
            scores[row_idx] = 1.0 / simpson_index
    return scores
