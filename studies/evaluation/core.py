"""Evaluation metrics for pipeline comparison.

This module provides essential evaluation metrics for assessing
pipeline performance without over-abstraction.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import ks_2samp

from scptensor.core.structures import ScpContainer


def _get_matrix(assay, layer_name: str | None = None):
    """Extract matrix from assay layer."""
    if layer_name is None:
        layer_name = "raw" if "raw" in assay.layers else list(assay.layers.keys())[0]
    return assay.layers[layer_name].X


def compute_sparsity(container: ScpContainer, assay_name: str = "proteins") -> float:
    """Compute fraction of missing values."""
    assay = container.assays[assay_name]
    matrix = _get_matrix(assay)

    if sparse.issparse(matrix):
        return 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    return float(np.mean(np.isnan(matrix)))


def compute_statistics(container: ScpContainer, assay_name: str = "proteins") -> dict[str, float]:
    """Compute basic statistics."""
    assay = container.assays[assay_name]
    matrix = _get_matrix(assay)

    if sparse.issparse(matrix):
        matrix = matrix.toarray()

    valid = matrix[~np.isnan(matrix)]

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "cv": float(np.std(valid) / np.mean(valid)) if np.mean(valid) != 0 else 0.0,
    }


def compute_distribution_change(
    original: ScpContainer, result: ScpContainer, assay_name: str = "proteins"
) -> dict[str, float]:
    """Compute distribution change metrics."""
    stats_orig = compute_statistics(original, assay_name)
    stats_res = compute_statistics(result, assay_name)

    # Get data for KS test
    matrix_orig = _get_matrix(original.assays[assay_name])
    matrix_res = _get_matrix(result.assays[assay_name])

    if sparse.issparse(matrix_orig):
        matrix_orig = matrix_orig.toarray()
    if sparse.issparse(matrix_res):
        matrix_res = matrix_res.toarray()

    ks_stat, ks_pvalue = ks_2samp(
        matrix_orig[~np.isnan(matrix_orig)], matrix_res[~np.isnan(matrix_res)]
    )

    return {
        "mean_change": abs(stats_res["mean"] - stats_orig["mean"]),
        "std_change": abs(stats_res["std"] - stats_orig["std"]),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def compute_pca_variance(
    container: ScpContainer, assay_name: str = "proteins", n_components: int = 10
) -> dict[str, Any]:
    """Compute PCA variance explained."""
    from sklearn.decomposition import PCA

    assay = container.assays[assay_name]
    layer_name = "pca" if "pca" in assay.layers else list(assay.layers.keys())[-1]
    matrix = _get_matrix(assay, layer_name)

    if sparse.issparse(matrix):
        matrix = matrix.toarray()

    # Handle missing values
    matrix = np.nan_to_num(matrix, nan=0.0)

    n_components = min(n_components, min(matrix.shape[0], matrix.shape[1]))
    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    return {
        "variance_ratios": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
    }


@contextmanager
def monitor_performance() -> Generator[dict[str, float], None, None]:
    """Context manager to monitor runtime and memory."""
    import tracemalloc

    result = {"runtime": 0.0, "memory_peak": 0.0}

    tracemalloc.start()
    start_time = time.time()

    try:
        yield result
    finally:
        result["runtime"] = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        result["memory_peak"] = peak / (1024**3)  # GB
        tracemalloc.stop()


def evaluate_pipeline(
    original: ScpContainer,
    result: ScpContainer,
    runtime: float,
    memory_peak: float,
    pipeline_name: str = "unknown",
    dataset_name: str = "unknown",
    assay_name: str = "proteins",
) -> dict[str, Any]:
    """
    Comprehensive pipeline evaluation.

    Parameters
    ----------
    original : ScpContainer
        Original data before processing
    result : ScpContainer
        Processed data
    runtime : float
        Execution time in seconds
    memory_peak : float
        Peak memory usage in GB
    pipeline_name : str
        Name of pipeline
    dataset_name : str
        Name of dataset
    assay_name : str
        Name of assay to evaluate

    Returns
    -------
    dict
        Evaluation metrics
    """
    # Sparsity
    sparsity_orig = compute_sparsity(original, assay_name)
    sparsity_res = compute_sparsity(result, assay_name)

    # Distribution
    dist_change = compute_distribution_change(original, result, assay_name)

    # PCA variance
    try:
        pca_var = compute_pca_variance(result, assay_name)
    except Exception:
        pca_var = {"variance_ratios": [], "cumulative_variance": 0.0, "pc1_variance": 0.0}

    return {
        "pipeline_name": pipeline_name,
        "dataset_name": dataset_name,
        "runtime_seconds": runtime,
        "memory_gb": memory_peak,
        "sparsity_original": sparsity_orig,
        "sparsity_result": sparsity_res,
        "sparsity_change": sparsity_res - sparsity_orig,
        "distribution": dist_change,
        "pca_variance": pca_var,
    }
