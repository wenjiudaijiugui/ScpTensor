"""Diagnostics for batch integration quality assessment.

Provides metrics for evaluating batch effect correction results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def compute_batch_asw(
    container: "ScpContainer",
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
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found")

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise ValueError(f"Layer '{layer_name}' not found")

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    if batch_key not in container.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")

    batches = container.obs[batch_key].to_numpy()

    # Filter to valid samples
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    batches_valid = batches[valid_mask]

    if len(np.unique(batches_valid)) < 2:
        return 0.0

    # ASW for batch labels (lower = better mixing)
    asw = silhouette_score(X_valid, batches_valid)

    # Convert to batch mixing score (1 - |ASW|, higher = better)
    # Return raw ASW (lower = better)
    return float(asw)


def compute_batch_mixing_metric(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
) -> float:
    """Compute batch mixing metric.

    Measures how well batches are mixed in local neighborhoods.

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
        Batch mixing score (higher is better).
    """
    from sklearn.neighbors import NearestNeighbors

    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found")

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise ValueError(f"Layer '{layer_name}' not found")

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    if batch_key not in container.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")

    batches = container.obs[batch_key].to_numpy()
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    # Compute kNN
    n_neighbors = min(n_neighbors, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Compute mixing score
    batch_frequencies = np.array([np.sum(batches == b) for b in unique_batches]) / len(batches)

    scores = []
    for i, idx in enumerate(indices):
        # Skip self
        neighbor_batches = batches[idx[1:]]
        # Expected frequency if perfectly mixed
        expected = batch_frequencies
        # Observed frequency
        observed = np.array([np.sum(neighbor_batches == b) for b in unique_batches]) / n_neighbors
        # Entropy-based mixing score
        score = 1 - np.sum(np.abs(observed - expected)) / 2
        scores.append(score)

    return float(np.mean(scores))


def compute_lisi_approx(
    container: "ScpContainer",
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

    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found")

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise ValueError(f"Layer '{layer_name}' not found")

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    if batch_key not in container.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")

    batches = container.obs[batch_key].to_numpy()
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return float(n_batches)

    # Compute kNN
    n_neighbors = min(n_neighbors, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Compute approximate LISI
    lisi_scores = []
    for idx in indices:
        neighbor_batches = batches[idx[1:]]  # Skip self
        # Count per batch
        counts = np.array([np.sum(neighbor_batches == b) for b in unique_batches])
        # Simpson's index
        proportions = counts / n_neighbors
        simpson = np.sum(proportions ** 2)
        # Inverse Simpson's index
        if simpson > 0:
            lisi_scores.append(1.0 / simpson)
        else:
            lisi_scores.append(1.0)

    return float(np.mean(lisi_scores))


def integration_quality_report(
    container: "ScpContainer",
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
    report = {
        "batch_asw": compute_batch_asw(container, assay_name, layer_name, batch_key),
        "batch_mixing": compute_batch_mixing_metric(container, assay_name, layer_name, batch_key),
        "lisi_approx": compute_lisi_approx(container, assay_name, layer_name, batch_key),
    }

    # Interpretation
    report["interpretation"] = {
        "batch_asw": "Lower is better (good mixing)",
        "batch_mixing": "Higher is better (0-1 scale)",
        "lisi_approx": f"Higher is better (max = n_batches, here: {len(container.obs[batch_key].unique())})",
    }

    return report


__all__ = [
    "compute_batch_asw",
    "compute_batch_mixing_metric",
    "compute_lisi_approx",
    "integration_quality_report",
]
