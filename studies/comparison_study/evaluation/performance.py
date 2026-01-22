"""
Computational performance evaluation metrics.

This module implements metrics to assess the computational efficiency
of different analysis pipelines.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

if TYPE_CHECKING:
    from collections.abc import Callable


@contextmanager
def monitor_performance():
    """
    Context manager to monitor runtime and memory usage.

    Yields
    ------
    dict
        Dictionary with 'runtime', 'memory_peak', and 'memory_delta' keys

    Examples
    --------
    >>> with monitor_performance() as perf:
    ...     # Run pipeline
    ...     result = run_pipeline(data)
    >>> print(f"Runtime: {perf['runtime']:.2f}s")
    >>> print(f"Peak memory: {perf['memory_peak']:.2f} GB")
    """
    process = psutil.Process()

    # Record initial state
    start_time = psutil.time.perf_counter()
    start_memory = process.memory_info().rss / (1024**3)  # GB

    # Track peak memory
    peak_memory = start_memory

    try:
        # Create result dictionary
        result: dict[str, float] = {
            "runtime": 0.0,
            "memory_peak": start_memory,
            "memory_delta": 0.0,
        }

        yield result

    finally:
        # Record final state
        end_time = psutil.time.perf_counter()
        end_memory = process.memory_info().rss / (1024**3)

        result["runtime"] = end_time - start_time
        result["memory_peak"] = max(peak_memory, end_memory)
        result["memory_delta"] = end_memory - start_memory


def compute_efficiency_score(
    runtime: float,
    memory: float,
    n_cells: int,
    n_features: int,
) -> dict[str, float]:
    """
    Compute efficiency scores normalized by data size.

    Parameters
    ----------
    runtime : float
        Runtime in seconds
    memory : float
        Peak memory usage in GB
    n_cells : int
        Number of cells
    n_features : int
        Number of features

    Returns
    -------
    dict[str, float]
        Dictionary containing normalized efficiency metrics:
        - time_per_cell: Time per cell in seconds
        - time_per_feature: Time per feature in seconds
        - time_per_million_entries: Time per million data entries
        - memory_per_cell: Memory per cell in GB
        - memory_per_feature: Memory per feature in GB
        - memory_per_million_entries: Memory per million entries in GB
    """
    data_size = n_cells * n_features

    return {
        "time_per_cell": runtime / n_cells if n_cells > 0 else 0.0,
        "time_per_feature": runtime / n_features if n_features > 0 else 0.0,
        "time_per_million_entries": runtime / (data_size / 1e6) if data_size > 0 else 0.0,
        "memory_per_cell": memory / n_cells if n_cells > 0 else 0.0,
        "memory_per_feature": memory / n_features if n_features > 0 else 0.0,
        "memory_per_million_entries": memory / (data_size / 1e6) if data_size > 0 else 0.0,
    }


def estimate_complexity(
    runtimes: list[float] | tuple[float, ...],
    data_sizes: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """
    Estimate time and space complexity by fitting to different complexity classes.

    Parameters
    ----------
    runtimes : list[float] or tuple[float, ...]
        List of runtimes for different data sizes
    data_sizes : list[int] or tuple[int, ...]
        List of data sizes (n_cells * n_features)

    Returns
    -------
    dict[str, Any]
        Dictionary with complexity estimates:
        - linear_r2: R² for linear fit O(n)
        - quadratic_r2: R² for quadratic fit O(n²)
        - loglinear_r2: R² for log-linear fit O(n log n)
        - estimated_complexity: Best fitting complexity class

    Examples
    --------
    >>> runtimes = [0.1, 0.2, 0.4, 0.8]  # Doubling data doubles time
    >>> sizes = [1000, 2000, 4000, 8000]
    >>> complexity = estimate_complexity(runtimes, sizes)
    >>> print(complexity['estimated_complexity'])
    'linear'
    """
    from scipy.optimize import curve_fit

    def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:  # noqa: ARG001
        """Linear complexity: O(n)"""
        return a * x

    def quadratic(x: np.ndarray, a: float, b: float) -> np.ndarray:  # noqa: ARG001
        """Quadratic complexity: O(n²)"""
        return a * x**2

    def loglinear(x: np.ndarray, a: float, b: float) -> np.ndarray:  # noqa: ARG001
        """Log-linear complexity: O(n log n)"""
        return a * x * np.log(x + 1)

    data_sizes_arr = np.array(data_sizes, dtype=float)
    runtimes_arr = np.array(runtimes, dtype=float)

    results: dict[str, Any] = {}

    # Fit different complexity models
    try:
        popt_linear, _ = curve_fit(linear, data_sizes_arr, runtimes_arr, maxfev=10000)
        results["linear_r2"] = _compute_r2(runtimes_arr, linear(data_sizes_arr, *popt_linear))
    except RuntimeError:
        results["linear_r2"] = -np.inf

    try:
        popt_quadratic, _ = curve_fit(quadratic, data_sizes_arr, runtimes_arr, maxfev=10000)
        results["quadratic_r2"] = _compute_r2(
            runtimes_arr,
            quadratic(data_sizes_arr, *popt_quadratic),
        )
    except RuntimeError:
        results["quadratic_r2"] = -np.inf

    try:
        popt_loglinear, _ = curve_fit(loglinear, data_sizes_arr, runtimes_arr, maxfev=10000)
        results["loglinear_r2"] = _compute_r2(
            runtimes_arr,
            loglinear(data_sizes_arr, *popt_loglinear),
        )
    except RuntimeError:
        results["loglinear_r2"] = -np.inf

    # Determine best fit
    complexity_types = [
        ("linear", results.get("linear_r2", -np.inf)),
        ("quadratic", results.get("quadratic_r2", -np.inf)),
        ("loglinear", results.get("loglinear_r2", -np.inf)),
    ]

    best_fit = max(complexity_types, key=lambda x: x[1])
    results["estimated_complexity"] = best_fit[0]

    return results


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² score.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)


def profile_memory_usage(
    func: Callable[[], Any],
    n_samples: int = 10,
) -> dict[str, float]:
    """
    Profile memory usage of a function over multiple runs.

    Parameters
    ----------
    func : Callable[[], Any]
        Function to profile (should take no arguments)
    n_samples : int, default 10
        Number of times to run the function

    Returns
    -------
    dict[str, float]
        Dictionary with memory statistics:
        - mean_memory_gb: Mean memory usage in GB
        - std_memory_gb: Standard deviation of memory usage
        - min_memory_gb: Minimum memory usage
        - max_memory_gb: Maximum memory usage

    Examples
    --------
    >>> def process_data():
    ...     return large_computation()
    >>> stats = profile_memory_usage(process_data, n_samples=5)
    >>> print(f"Mean memory: {stats['mean_memory_gb']:.2f} GB")
    """
    process = psutil.Process()
    memory_samples = []

    for _ in range(n_samples):
        # Measure memory before
        mem_before = process.memory_info().rss / (1024**3)

        # Run function
        func()

        # Measure memory after
        mem_after = process.memory_info().rss / (1024**3)
        memory_samples.append(mem_after - mem_before)

    if not memory_samples:
        return {
            "mean_memory_gb": 0.0,
            "std_memory_gb": 0.0,
            "min_memory_gb": 0.0,
            "max_memory_gb": 0.0,
        }

    return {
        "mean_memory_gb": float(np.mean(memory_samples)),
        "std_memory_gb": float(np.std(memory_samples)),
        "min_memory_gb": float(np.min(memory_samples)),
        "max_memory_gb": float(np.max(memory_samples)),
    }
