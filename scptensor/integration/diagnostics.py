"""Diagnostics for batch integration quality assessment.

Provides both backward-compatible proxy metrics and more standardized
integration metrics for evaluating batch effect correction results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def _prepare_valid_rows(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    batch_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an embedding/matrix layer and keep only fully finite sample rows."""
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found")

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise ValueError(f"Layer '{layer_name}' not found")

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    if batch_key not in container.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")

    batches = container.obs[batch_key].to_numpy()
    valid_mask = np.isfinite(X).all(axis=1)
    return X[valid_mask], batches[valid_mask]


def _validate_neighbors(n_neighbors: int) -> None:
    """Validate neighborhood size for local mixing metrics."""
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")


def _validate_alpha(alpha: float) -> None:
    """Validate acceptance threshold for hypothesis-test metrics."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def _validate_perplexity(perplexity: float) -> None:
    """Validate target perplexity for LISI-style metrics."""
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")


def _compute_knn(
    X: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a self-excluded kNN graph for the given samples."""
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(n_neighbors, len(X) - 1)
    if n_neighbors < 1:
        return (
            np.empty((len(X), 0), dtype=float),
            np.empty((len(X), 0), dtype=int),
        )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    filtered_distances: list[np.ndarray] = []
    filtered_indices: list[np.ndarray] = []
    for row_idx, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=False)):
        keep_mask = row_indices != row_idx
        row_distances = row_distances[keep_mask]
        row_indices = row_indices[keep_mask]

        if row_indices.size < n_neighbors:
            row_distances = distances[row_idx, 1 : n_neighbors + 1]
            row_indices = indices[row_idx, 1 : n_neighbors + 1]

        filtered_distances.append(row_distances[:n_neighbors])
        filtered_indices.append(row_indices[:n_neighbors])

    return np.vstack(filtered_distances), np.vstack(filtered_indices)


def _hbeta(distances: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Compute Shannon entropy and neighbor probabilities for one sample."""
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
    """Match a target perplexity with per-row binary search over beta."""
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
    batch_codes: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_probabilities: np.ndarray,
    n_batches: int,
) -> np.ndarray:
    """Compute per-sample inverse Simpson diversity from weighted neighborhoods."""
    scores = np.ones(neighbor_indices.shape[0], dtype=float)

    for row_idx, neighbors in enumerate(neighbor_indices):
        weights = neighbor_probabilities[row_idx]
        local_probabilities = np.bincount(
            batch_codes[neighbors],
            weights=weights,
            minlength=n_batches,
        )
        simpson_index = float(np.sum(local_probabilities**2))
        if simpson_index > 0.0:
            scores[row_idx] = 1.0 / simpson_index

    return scores


def compute_batch_asw(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
) -> float:
    """Compute batch Average Silhouette Width (ASW).

    Lower values indicate better batch mixing.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.

    Returns
    -------
    float
        Batch ASW score (range -1 to 1, lower is better mixing).
    """
    X_valid, batches_valid = _prepare_valid_rows(container, assay_name, layer_name, batch_key)

    n_valid = len(X_valid)
    n_labels = len(np.unique(batches_valid))
    if n_labels < 2 or n_valid < 3 or n_labels >= n_valid:
        return 0.0

    # Raw silhouette on batch labels: lower indicates better mixing.
    asw = silhouette_score(X_valid, batches_valid)
    return float(asw)


def compute_batch_mixing_metric(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
) -> float:
    """Compute a heuristic local batch-mixing proxy.

    Measures how closely each local neighborhood matches the global batch
    composition. This is a ScpTensor-specific heuristic proxy, not kBET or
    original LISI.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.
    n_neighbors : int, default=50
        Number of neighbors for local mixing.

    Returns
    -------
    float
        Heuristic local-composition score (higher is better).
    """
    from sklearn.neighbors import NearestNeighbors

    _validate_neighbors(n_neighbors)
    X, batches = _prepare_valid_rows(container, assay_name, layer_name, batch_key)

    if len(X) < 2:
        return 1.0

    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    # Compute kNN
    n_neighbors = min(n_neighbors, len(X) - 1)
    if n_neighbors < 1:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Compute mixing score
    batch_frequencies = np.array([np.sum(batches == b) for b in unique_batches]) / len(batches)

    scores = []
    for idx in indices:
        # Skip self
        neighbor_batches = batches[idx[1:]]
        n_local = len(neighbor_batches)
        if n_local == 0:
            scores.append(1.0)
            continue
        # Expected frequency under global batch composition.
        expected = batch_frequencies
        # Observed local composition.
        observed = np.array([np.sum(neighbor_batches == b) for b in unique_batches]) / n_local
        # One minus total-variation distance between local and global composition.
        score = 1 - np.sum(np.abs(observed - expected)) / 2
        scores.append(score)

    return float(np.mean(scores))


def compute_lisi_approx(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
) -> float:
    """Compute approximate Local Inverse Simpson's Index (LISI).

    Measures batch diversity in local neighborhoods.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.
    n_neighbors : int, default=50
        Number of neighbors.

    Returns
    -------
    float
        Approximate LISI score (higher = better mixing).
    """
    from sklearn.neighbors import NearestNeighbors

    _validate_neighbors(n_neighbors)
    X, batches = _prepare_valid_rows(container, assay_name, layer_name, batch_key)

    if len(X) < 2:
        return 1.0

    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return float(n_batches)

    # Compute kNN
    n_neighbors = min(n_neighbors, len(X) - 1)
    if n_neighbors < 1:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Compute approximate LISI
    lisi_scores: list[float] = []
    for idx in indices:
        neighbor_batches = batches[idx[1:]]  # Skip self
        n_local = len(neighbor_batches)
        if n_local == 0:
            lisi_scores.append(1.0)
            continue
        # Count per batch
        counts = np.array([np.sum(neighbor_batches == b) for b in unique_batches])
        # Simpson's index
        proportions = counts / n_local
        simpson = np.sum(proportions**2)
        # Inverse Simpson's index
        if simpson > 0:
            lisi_scores.append(1.0 / simpson)
        else:
            lisi_scores.append(1.0)

    return float(np.mean(lisi_scores))


def compute_kbet(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
    alpha: float = 0.05,
) -> float:
    """Compute a fixed-k kBET acceptance rate.

    This implements the core kBET goodness-of-fit test: compare each sample's
    local batch composition against the global batch composition with a
    chi-square test, then report the acceptance rate ``P(p >= alpha)``.

    The function intentionally keeps a fixed neighborhood size and therefore
    does not reproduce every adaptive heuristic from the original R package.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.
    n_neighbors : int, default=50
        Number of nearest neighbors per sample.
    alpha : float, default=0.05
        Acceptance threshold for the chi-square p-value.

    Returns
    -------
    float
        Acceptance rate in ``[0, 1]``. Higher values indicate better mixing.
    """
    _validate_neighbors(n_neighbors)
    _validate_alpha(alpha)
    X, batches = _prepare_valid_rows(container, assay_name, layer_name, batch_key)

    if len(X) < 2:
        return 1.0

    _, batch_codes = np.unique(batches, return_inverse=True)
    n_batches = len(np.unique(batch_codes))
    if n_batches < 2:
        return 1.0

    _, neighbor_indices = _compute_knn(X, n_neighbors)
    if neighbor_indices.shape[1] == 0:
        return 1.0

    local_k = neighbor_indices.shape[1]
    expected_counts = np.bincount(batch_codes, minlength=n_batches).astype(float)
    expected_counts = expected_counts / expected_counts.sum() * local_k

    degrees_of_freedom = n_batches - 1
    if degrees_of_freedom <= 0:
        return 1.0

    p_values = np.ones(neighbor_indices.shape[0], dtype=float)
    for row_idx, neighbors in enumerate(neighbor_indices):
        observed_counts = np.bincount(batch_codes[neighbors], minlength=n_batches).astype(float)
        chi_square = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_values[row_idx] = float(chi2.sf(chi_square, degrees_of_freedom))

    return float(np.mean(p_values >= alpha))


def compute_ilisi(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 90,
    perplexity: float = 30.0,
    *,
    scale: bool = True,
) -> float:
    """Compute a more standardized iLISI summary.

    Per sample, this function estimates a perplexity-matched neighbor
    probability distribution, computes the inverse Simpson diversity of local
    batch labels, and summarizes the result with the median across samples.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.
    n_neighbors : int, default=90
        Number of nearest neighbors used to build the local neighborhood.
    perplexity : float, default=30.0
        Target perplexity used to weight each neighborhood. On small datasets,
        the effective perplexity is clipped to the available neighbor count.
    scale : bool, default=True
        If ``True``, linearly scale the median iLISI from ``[1, n_batches]``
        to ``[0, 1]``. If ``False``, return the raw median iLISI.

    Returns
    -------
    float
        Median iLISI summary. Higher values indicate better mixing.
    """
    _validate_neighbors(n_neighbors)
    _validate_perplexity(perplexity)
    X, batches = _prepare_valid_rows(container, assay_name, layer_name, batch_key)

    if len(X) < 2:
        return 1.0

    _, batch_codes = np.unique(batches, return_inverse=True)
    n_batches = len(np.unique(batch_codes))
    if n_batches < 2:
        return 1.0

    neighbor_distances, neighbor_indices = _compute_knn(X, n_neighbors)
    local_k = neighbor_indices.shape[1]
    if local_k == 0:
        return 1.0

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


def integration_quality_report(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
) -> dict:
    """Generate comprehensive integration quality report.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.

    Returns
    -------
    dict
        Dictionary with quality metrics.
    """
    X_valid, batches_valid = _prepare_valid_rows(container, assay_name, layer_name, batch_key)
    n_valid_batches = len(np.unique(batches_valid)) if len(X_valid) else 0

    report: dict[str, float | dict[str, str]] = {
        "batch_asw": compute_batch_asw(container, assay_name, layer_name, batch_key),
        "batch_mixing": compute_batch_mixing_metric(container, assay_name, layer_name, batch_key),
        "lisi_approx": compute_lisi_approx(container, assay_name, layer_name, batch_key),
    }

    # Interpretation
    report["interpretation"] = {
        "batch_asw": "Lower is better (good mixing)",
        "batch_mixing": "Higher is better (0-1 scale)",
        "lisi_approx": f"Higher is better (max = n_valid_batches, here: {n_valid_batches})",
    }

    return report


__all__ = [
    "compute_batch_asw",
    "compute_batch_mixing_metric",
    "compute_lisi_approx",
    "compute_kbet",
    "compute_ilisi",
    "integration_quality_report",
]
