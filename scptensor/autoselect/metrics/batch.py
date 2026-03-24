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

from scptensor.core._batch_metrics_kernel import (
    compute_inverse_simpson_scores as _compute_inverse_simpson_scores,
)
from scptensor.core._batch_metrics_kernel import (
    compute_perplexity_probabilities as _compute_perplexity_probabilities,
)
from scptensor.core._batch_metrics_kernel import (
    compute_self_excluded_knn as _compute_knn,
)
from scptensor.core._batch_metrics_kernel import (
    validate_alpha as _validate_alpha,
)
from scptensor.core._batch_metrics_kernel import (
    validate_n_neighbors as _validate_neighbors,
)
from scptensor.core._batch_metrics_kernel import (
    validate_perplexity as _validate_perplexity,
)

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
            f"batch_labels has {len(labels)} elements",
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


def _try_raw_label_silhouette(
    X: NDArray[np.float64],
    labels: np.ndarray,
) -> tuple[float, bool]:
    """Return raw silhouette width and whether the input was evaluable."""
    from sklearn.metrics import silhouette_score

    if X.size == 0 or X.shape[0] < 2:
        return 0.0, False

    x_clean, _, label_codes = _prepare_labeled_matrix(X, labels)
    if x_clean.shape[0] < 2 or len(np.unique(label_codes)) < 2:
        return 0.0, False

    _, counts = np.unique(label_codes, return_counts=True)
    if counts.size == 0 or int(np.min(counts)) < 2:
        return 0.0, False

    try:
        score = float(np.clip(silhouette_score(x_clean, label_codes), -1.0, 1.0))
        return score, True
    except Exception:
        return 0.0, False


def _raw_batch_asw(X: NDArray[np.float64], batch_labels: np.ndarray) -> float:
    """Return raw batch silhouette width with fail-closed semantics."""
    score, evaluable = _try_raw_label_silhouette(X, batch_labels)
    return score if evaluable else 0.0


def batch_asw(X: NDArray[np.float64], batch_labels: np.ndarray) -> float:
    """Calculate 1-ASW batch mixing score.

    Lower raw batch ASW indicates better mixing. This helper derives the
    historical AutoSelect score from the authoritative raw silhouette result
    and returns ``1 - ASW`` clipped to ``[0, 1]`` so higher remains better.
    """
    raw_asw, evaluable = _try_raw_label_silhouette(X, batch_labels)
    if not evaluable:
        return 0.0
    return float(np.clip(1.0 - raw_asw, 0.0, 1.0))


def bio_asw(X: NDArray[np.float64], bio_labels: np.ndarray) -> float:
    """Calculate biological-group ASW score."""
    raw_asw, evaluable = _try_raw_label_silhouette(X, bio_labels)
    if not evaluable:
        return 0.0
    return float(np.clip(raw_asw, 0.0, 1.0))


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


def lisi_approx_score(
    X: NDArray[np.float64],
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
) -> float:
    """Calculate fixed-k approximate LISI from unweighted neighborhoods.

    This preserves the historical ScpTensor approximation used in integration
    diagnostics and serves as the shared implementation for that API surface.
    """
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    _validate_neighbors(n_neighbors)
    x_clean, _, batch_codes = _prepare_labeled_matrix(X, batch_labels)
    n_samples = x_clean.shape[0]
    n_batches = len(np.unique(batch_codes))

    if n_samples < 2 or n_batches < 2:
        return 0.0

    _, neighbor_indices = _compute_knn(x_clean, n_neighbors)
    if neighbor_indices.shape[1] == 0:
        return 0.0

    try:
        lisi_scores: list[float] = []
        for neighbors in neighbor_indices:
            neighbor_codes = batch_codes[neighbors]
            batch_counts = np.bincount(neighbor_codes, minlength=n_batches)
            proportions = batch_counts / len(neighbor_codes)
            simpson = float(np.sum(proportions**2))
            lisi_scores.append(0.0 if simpson <= 0.0 else 1.0 / simpson)

        return float(np.mean(lisi_scores))
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
    "lisi_approx_score",
    "kbet_score",
    "ilisi_score",
]
