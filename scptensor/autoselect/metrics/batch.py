"""Batch-effect metrics for automatic method selection.

This module intentionally exposes two metric families:

- backward-compatible proxy scores used by current selection logic
- more standardized kBET / iLISI-style scores for diagnostics and reporting

All public scores are normalized to ``[0, 1]`` unless explicitly documented
otherwise. Higher values indicate better batch mixing or biological signal
preservation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.typing import NDArray


_EPS = 1e-10


def _prepare_labeled_matrix(
    X: NDArray[np.float64],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate shape, drop non-finite rows, and factorize labels."""
    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} samples but "
            f"batch_labels has {len(labels)} elements"
        )

    valid_mask = np.isfinite(X).all(axis=1)
    if not np.any(valid_mask):
        return (
            np.empty((0, X.shape[1]), dtype=float),
            np.empty((0,), dtype=labels.dtype),
            np.empty((0,), dtype=int),
        )

    x_clean = np.asarray(X[valid_mask], dtype=float)
    labels_clean = np.asarray(labels)[valid_mask]
    _, label_codes = np.unique(labels_clean, return_inverse=True)
    return x_clean, labels_clean, label_codes


def _validate_neighbors(n_neighbors: int) -> None:
    """Validate kNN neighborhood size."""
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")


def _validate_alpha(alpha: float) -> None:
    """Validate kBET acceptance threshold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def _validate_perplexity(perplexity: float) -> None:
    """Validate target perplexity."""
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")


def _compute_knn(
    X: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a self-excluded kNN graph."""
    local_k = min(n_neighbors, X.shape[0] - 1)
    if local_k < 1:
        return (
            np.empty((X.shape[0], 0), dtype=float),
            np.empty((X.shape[0], 0), dtype=int),
        )

    nbrs = NearestNeighbors(n_neighbors=local_k + 1, algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors(X)

    return distances[:, 1 : local_k + 1], indices[:, 1 : local_k + 1]


def _hbeta(distances: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Return Shannon entropy and normalized probabilities for one row."""
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


def _compute_perplexity_probabilities(
    distances: np.ndarray,
    perplexity: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    """Binary-search neighbor probabilities to match a target perplexity."""
    target_entropy = float(np.log(perplexity))
    probabilities = np.zeros_like(distances, dtype=float)

    for row_idx, row_distances in enumerate(distances):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        entropy, row_probabilities = _hbeta(row_distances, beta)
        entropy_diff = entropy - target_entropy

        n_iter = 0
        while abs(entropy_diff) > tol and n_iter < max_iter:
            if entropy_diff > 0.0:
                beta_min = beta
                beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

            entropy, row_probabilities = _hbeta(row_distances, beta)
            entropy_diff = entropy - target_entropy
            n_iter += 1

        probabilities[row_idx] = row_probabilities

    return probabilities


def _compute_inverse_simpson_scores(
    label_codes: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_probabilities: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Compute weighted inverse Simpson diversity for each sample."""
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


def batch_asw(X: NDArray[np.float64], batch_labels: np.ndarray) -> float:
    """Calculate 1-ASW batch mixing score.

    Lower raw batch ASW indicates better mixing. This helper keeps the
    historical AutoSelect convention and returns ``1 - ASW`` clipped to
    ``[0, 1]`` so higher remains better.
    """
    from sklearn.metrics import silhouette_score

    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    x_clean, _, batch_codes = _prepare_labeled_matrix(X, batch_labels)
    if x_clean.shape[0] < 2 or len(np.unique(batch_codes)) < 2:
        return 0.0

    _, counts = np.unique(batch_codes, return_counts=True)
    if counts.size == 0 or int(np.min(counts)) < 2:
        return 0.0

    try:
        asw = float(silhouette_score(x_clean, batch_codes))
        return float(np.clip(1.0 - asw, 0.0, 1.0))
    except Exception:
        return 0.0


def bio_asw(X: NDArray[np.float64], bio_labels: np.ndarray) -> float:
    """Calculate biological-group ASW score."""
    from sklearn.metrics import silhouette_score

    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    x_clean, _, bio_codes = _prepare_labeled_matrix(X, bio_labels)
    if x_clean.shape[0] < 2 or len(np.unique(bio_codes)) < 2:
        return 0.0

    _, counts = np.unique(bio_codes, return_counts=True)
    if counts.size == 0 or int(np.min(counts)) < 2:
        return 0.0

    try:
        asw = float(silhouette_score(x_clean, bio_codes))
        return float(np.clip(asw, 0.0, 1.0))
    except Exception:
        return 0.0


def batch_mixing_score(
    X: NDArray[np.float64],
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
) -> float:
    """Calculate a heuristic local batch-mixing proxy.

    This is the historical AutoSelect proxy score. It is not kBET and not
    original LISI. It uses normalized Simpson diversity on a fixed-k
    neighborhood and remains the current batch-mixing score consumed by
    AutoSelect selection logic.
    """
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    _validate_neighbors(n_neighbors)
    x_clean, _, batch_codes = _prepare_labeled_matrix(X, batch_labels)
    n_samples = x_clean.shape[0]
    n_batches = len(np.unique(batch_codes))

    if n_samples < 2 or n_batches < 2:
        return 0.0

    neighbor_distances, neighbor_indices = _compute_knn(x_clean, n_neighbors)
    if neighbor_distances.shape[1] == 0:
        return 0.0

    try:
        scores = []
        for neighbors in neighbor_indices:
            neighbor_codes = batch_codes[neighbors]
            batch_counts = np.bincount(neighbor_codes, minlength=n_batches)
            proportions = batch_counts / len(neighbor_codes)
            simpson = 1.0 - np.sum(proportions**2)
            scores.append(simpson)

        max_simpson = 1.0 - 1.0 / n_batches
        if max_simpson < _EPS:
            return 0.0

        avg_score = float(np.mean(scores))
        return float(np.clip(avg_score / max_simpson, 0.0, 1.0))
    except Exception:
        return 0.0


def kbet_score(
    X: NDArray[np.float64],
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
    alpha: float = 0.05,
) -> float:
    """Calculate a fixed-k kBET acceptance rate."""
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    _validate_neighbors(n_neighbors)
    _validate_alpha(alpha)
    x_clean, _, batch_codes = _prepare_labeled_matrix(X, batch_labels)

    n_samples = x_clean.shape[0]
    n_batches = len(np.unique(batch_codes))
    if n_samples < 2 or n_batches < 2:
        return 0.0

    _, neighbor_indices = _compute_knn(x_clean, n_neighbors)
    if neighbor_indices.shape[1] == 0:
        return 0.0

    local_k = neighbor_indices.shape[1]
    expected_counts = np.bincount(batch_codes, minlength=n_batches).astype(float)
    expected_counts = expected_counts / expected_counts.sum() * local_k
    if np.any(expected_counts <= 0.0):
        return 0.0

    degrees_of_freedom = n_batches - 1
    if degrees_of_freedom <= 0:
        return 0.0

    acceptance = np.ones(neighbor_indices.shape[0], dtype=bool)
    for row_idx, neighbors in enumerate(neighbor_indices):
        observed_counts = np.bincount(batch_codes[neighbors], minlength=n_batches).astype(float)
        chi_square = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_value = float(chi2.sf(chi_square, degrees_of_freedom))
        acceptance[row_idx] = p_value >= alpha

    return float(np.mean(acceptance))


def ilisi_score(
    X: NDArray[np.float64],
    batch_labels: np.ndarray,
    n_neighbors: int = 90,
    perplexity: float = 30.0,
    *,
    scale: bool = True,
) -> float:
    """Calculate a standardized iLISI summary.

    This follows the perplexity-weighted neighborhood style used by modern
    LISI implementations more closely than the historical fixed-k proxy.
    """
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    _validate_neighbors(n_neighbors)
    _validate_perplexity(perplexity)
    x_clean, _, batch_codes = _prepare_labeled_matrix(X, batch_labels)

    n_samples = x_clean.shape[0]
    n_batches = len(np.unique(batch_codes))
    if n_samples < 2 or n_batches < 2:
        return 0.0

    neighbor_distances, neighbor_indices = _compute_knn(x_clean, n_neighbors)
    local_k = neighbor_indices.shape[1]
    if local_k == 0:
        return 0.0

    effective_perplexity = min(float(perplexity), float(local_k))
    neighbor_probabilities = _compute_perplexity_probabilities(
        neighbor_distances,
        effective_perplexity,
    )
    ilisi_scores = _compute_inverse_simpson_scores(
        batch_codes,
        neighbor_indices,
        neighbor_probabilities,
        n_batches,
    )
    median_ilisi = float(np.median(ilisi_scores))

    if not scale:
        return median_ilisi

    return float(np.clip((median_ilisi - 1.0) / (n_batches - 1.0), 0.0, 1.0))


__all__ = [
    "batch_asw",
    "bio_asw",
    "batch_mixing_score",
    "kbet_score",
    "ilisi_score",
]
