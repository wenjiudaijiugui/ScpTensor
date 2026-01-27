"""Streamlined evaluation metrics for single-cell data analysis."""

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


def calculate_kbet(x: ArrayFloat, batch_labels: ArrayInt, k: int = 25) -> float:
    """Calculate kBET score for batch effect assessment."""
    n = x.shape[0]
    k = min(k + 1, n)
    if n < k + 1:
        return 0.0

    batches = np.asarray(batch_labels)
    unique_batches = np.unique(batches)
    if len(unique_batches) < 2:
        return 0.0

    global_freq = np.array([np.mean(batches == b) for b in unique_batches])

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)

    neighbor_batches = batches[indices[:, 1:]]
    local_freq = np.stack([np.mean(neighbor_batches == b, axis=1) for b in unique_batches], axis=1)
    chi2 = np.sum((local_freq - global_freq) ** 2, axis=1)
    return float(np.mean(chi2 < 0.1))


def calculate_ilisi(x: ArrayFloat, batch_labels: ArrayInt, k: int = 20) -> float:
    """Calculate iLISI score for batch mixing."""
    n = x.shape[0]
    k = min(k + 1, n)
    if n < k + 1:
        return 0.0

    labels = np.asarray(batch_labels)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)

    neighbor_labels = labels[indices[:, 1:]]
    simpson_vals = np.array(
        [
            1.0
            / np.sum(
                (
                    np.bincount(neighbor_labels[i], minlength=labels.max() + 1)
                    / len(neighbor_labels[i])
                )
                ** 2
            )
            for i in range(len(neighbor_labels))
        ]
    )
    return float(np.mean(simpson_vals))


def calculate_clisi(x: ArrayFloat, cell_labels: ArrayInt, k: int = 20) -> float:
    """Calculate cLISI score for cell type separation."""
    n = x.shape[0]
    k = min(k + 1, n)
    if n < k + 1:
        return 0.0

    labels = np.asarray(cell_labels)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)

    neighbor_labels = labels[indices[:, 1:]]
    simpson_vals = np.array(
        [
            1.0
            / np.sum(
                (
                    np.bincount(neighbor_labels[i], minlength=labels.max() + 1)
                    / len(neighbor_labels[i])
                )
                ** 2
            )
            for i in range(len(neighbor_labels))
        ]
    )
    return float(np.mean(simpson_vals))


def calculate_asw(x: ArrayFloat, labels: ArrayInt, metric: str = "euclidean") -> float:
    """Calculate average silhouette width for clustering quality."""
    return float(silhouette_score(x, labels, metric=metric))


def calculate_mse(x1: ArrayFloat, x2: ArrayFloat) -> float:
    """Calculate mean squared error between two arrays."""
    return float(np.mean((np.asarray(x1) - np.asarray(x2)) ** 2))


def calculate_mae(x1: ArrayFloat, x2: ArrayFloat) -> float:
    """Calculate mean absolute error between two arrays."""
    return float(np.mean(np.abs(np.asarray(x1) - np.asarray(x2))))


def calculate_correlation(x1: ArrayFloat, x2: ArrayFloat) -> float:
    """Calculate Pearson correlation coefficient between two arrays."""
    arr1 = np.asarray(x1).ravel()
    arr2 = np.asarray(x2).ravel()
    return float(np.corrcoef(arr1, arr2)[0, 1])


def measure_runtime(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, float]:
    """Measure function execution time."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return {"runtime": end - start, "result": result}


def measure_memory_usage(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, float]:
    """Measure peak memory usage during function execution."""
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]

    result = func(*args, **kwargs)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"memory_mb": (peak_mem - start_mem) / (1024 * 1024), "result": result}


def calculate_all_metrics(
    x: ArrayFloat, batch_labels: ArrayInt, cell_labels: ArrayInt, k: int = 25
) -> dict[str, float]:
    """Calculate all batch correction and biological preservation metrics."""
    results = {}

    # Batch correction metrics
    results["kbet"] = calculate_kbet(x, batch_labels, k)
    results["ilisi"] = calculate_ilisi(x, batch_labels, k)

    # Biological preservation metrics
    results["clisi"] = calculate_clisi(x, cell_labels, k)
    results["asw"] = calculate_asw(x, cell_labels)

    return results


def calculate_integration_metrics(
    x_orig: ArrayFloat,
    x_corrected: ArrayFloat,
    batch_labels: ArrayInt,
    cell_labels: ArrayInt,
    k: int = 25,
) -> dict[str, float]:
    """Compare metrics before and after batch correction."""
    orig_metrics = calculate_all_metrics(x_orig, batch_labels, cell_labels, k)
    corr_metrics = calculate_all_metrics(x_corrected, batch_labels, cell_labels, k)

    return {
        "kbet_orig": orig_metrics["kbet"],
        "kbet_corr": corr_metrics["kbet"],
        "kbet_delta": corr_metrics["kbet"] - orig_metrics["kbet"],
        "ilisi_orig": orig_metrics["ilisi"],
        "ilisi_corr": corr_metrics["ilisi"],
        "ilisi_delta": corr_metrics["ilisi"] - orig_metrics["ilisi"],
        "clisi_corr": corr_metrics["clisi"],
        "asw_corr": corr_metrics["asw"],
    }
