"""Diagnostics for batch integration quality assessment.

This module is intentionally a container-facing wrapper layer. Numerical batch
metrics live in ``scptensor.autoselect.metrics.batch`` so integration
diagnostics and AutoSelect use a single source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scptensor.autoselect.metrics.batch import (
    _prepare_labeled_matrix,
    _raw_batch_asw,
    batch_mixing_score,
    ilisi_score,
    kbet_score,
    lisi_approx_score,
)

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def _load_metric_inputs(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    batch_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a matrix layer and aligned batch labels from a container."""
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found")

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise ValueError(f"Layer '{layer_name}' not found")

    if batch_key not in container.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=float), container.obs[batch_key].to_numpy()


def compute_batch_asw(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
) -> float:
    """Compute raw batch Average Silhouette Width (ASW).

    Lower values indicate better batch mixing.
    """
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    return _raw_batch_asw(X, batches)


def compute_batch_mixing_metric(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
) -> float:
    """Compute the historical local batch-mixing proxy score."""
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    return batch_mixing_score(X, batches, n_neighbors=n_neighbors)


def compute_lisi_approx(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
) -> float:
    """Compute the historical fixed-k approximate LISI score."""
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    return lisi_approx_score(X, batches, n_neighbors=n_neighbors)


def compute_kbet(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    n_neighbors: int = 50,
    alpha: float = 0.05,
) -> float:
    """Compute fixed-k kBET acceptance rate."""
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    return kbet_score(X, batches, n_neighbors=n_neighbors, alpha=alpha)


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
    """Compute a standardized iLISI summary."""
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    return ilisi_score(
        X,
        batches,
        n_neighbors=n_neighbors,
        perplexity=perplexity,
        scale=scale,
    )


def integration_quality_report(
    container: ScpContainer,
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
) -> dict:
    """Generate comprehensive integration quality report."""
    X, batches = _load_metric_inputs(container, assay_name, layer_name, batch_key)
    x_valid, batches_valid, _ = _prepare_labeled_matrix(X, batches)
    n_valid_batches = len(np.unique(batches_valid)) if len(x_valid) else 0

    report: dict[str, float | dict[str, str]] = {
        "batch_asw": _raw_batch_asw(X, batches),
        "batch_mixing": batch_mixing_score(X, batches),
        "lisi_approx": lisi_approx_score(X, batches),
    }
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
