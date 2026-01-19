"""Performance evaluator for benchmarking algorithm runtime characteristics.

This module provides tools for measuring and analyzing the computational
performance of analysis methods, including runtime, memory usage, throughput,
and CPU time.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
import tracemalloc
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import Any

# =============================================================================
# Type Aliases
# =============================================================================

ArrayFloat = NDArray[np.float64]
T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceResult:
    """Result of a performance evaluation.

    Attributes
    ----------
    runtime : float
        Mean execution time in seconds.
    runtime_std : float
        Standard deviation of execution time in seconds.
    memory_usage : float
        Mean memory usage in megabytes (MB).
    memory_usage_std : float
        Standard deviation of memory usage in MB.
    throughput : float
        Mean throughput in samples per second.
    throughput_std : float
        Standard deviation of throughput.
    cpu_time : float
        Mean CPU time in seconds.
    cpu_time_std : float
        Standard deviation of CPU time.
    n_runs : int
        Number of benchmark runs performed.
    n_samples : int | None
        Number of samples processed (for throughput calculation).
    """

    runtime: float
    runtime_std: float
    memory_usage: float
    memory_usage_std: float
    throughput: float
    throughput_std: float
    cpu_time: float
    cpu_time_std: float
    n_runs: int
    n_samples: int | None

    def to_dict(self) -> dict[str, float]:
        """Convert result to a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary with metric names as keys and values as values.
        """
        return {
            "runtime": self.runtime,
            "runtime_std": self.runtime_std,
            "memory_usage": self.memory_usage,
            "memory_usage_std": self.memory_usage_std,
            "throughput": self.throughput,
            "throughput_std": self.throughput_std,
            "cpu_time": self.cpu_time,
            "cpu_time_std": self.cpu_time_std,
        }


# =============================================================================
# Performance Evaluator
# =============================================================================


class PerformanceEvaluator:
    """Evaluator for computational performance metrics.

    This evaluator measures the runtime performance of algorithms including
    execution time, memory usage, throughput, and CPU time. It supports
    multiple runs with warmup for accurate measurements.

    Parameters
    ----------
    n_runs : int, default=3
        Number of benchmark runs to perform for averaging.
    warmup_runs : int, default=1
        Number of warmup runs before measurement (not included in results).
    track_memory : bool, default=True
        Whether to track memory usage using tracemalloc.
    track_cpu : bool, default=True
        Whether to track CPU time separately from wall time.

    Attributes
    ----------
    n_runs : int
        Number of benchmark runs.
    warmup_runs : int
        Number of warmup runs.
    track_memory : bool
        Whether memory tracking is enabled.
    track_cpu : bool
        Whether CPU time tracking is enabled.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import PerformanceEvaluator
    >>>
    >>> def my_algorithm(X):
    ...     return X * 2
    >>>
    >>> X = np.random.randn(100, 50)
    >>> evaluator = PerformanceEvaluator(n_runs=5)
    >>> result = evaluator.evaluate(my_algorithm, X)
    >>> print(f"Runtime: {result.runtime:.4f}s")
    >>> print(f"Memory: {result.memory_usage:.2f}MB")
    """

    __slots__ = ("n_runs", "warmup_runs", "track_memory", "track_cpu")

    def __init__(
        self,
        n_runs: int = 3,
        warmup_runs: int = 1,
        track_memory: bool = True,
        track_cpu: bool = True,
    ) -> None:
        """Initialize the performance evaluator.

        Parameters
        ----------
        n_runs : int, default=3
            Number of benchmark runs to perform.
        warmup_runs : int, default=1
            Number of warmup runs before measurement.
        track_memory : bool, default=True
            Whether to track memory usage.
        track_cpu : bool, default=True
            Whether to track CPU time.
        """
        self.n_runs = max(1, n_runs)
        self.warmup_runs = max(0, warmup_runs)
        self.track_memory = track_memory
        self.track_cpu = track_cpu

    def evaluate(
        self,
        func: Callable[..., T_co],
        X: ArrayFloat | Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate the performance of a function on the given data.

        Parameters
        ----------
        func : Callable[..., T_co]
            Function to benchmark. Should accept X as first argument.
        X : ArrayFloat | Any
            Input data. If array-like, used for throughput calculation.
        *args : Any
            Additional positional arguments to pass to func.
        **kwargs : Any
            Additional keyword arguments to pass to func.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "runtime": Mean execution time (seconds)
            - "runtime_std": Standard deviation of execution time
            - "memory_usage": Mean memory usage (MB)
            - "memory_usage_std": Standard deviation of memory usage
            - "throughput": Mean throughput (samples/second)
            - "throughput_std": Standard deviation of throughput
            - "cpu_time": Mean CPU time (seconds)
            - "cpu_time_std": Standard deviation of CPU time
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            with contextlib.suppress(Exception):
                _ = func(X, *args, **kwargs)

        # Storage for run results
        runtimes: list[float] = []
        memory_usages: list[float] = []
        cpu_times: list[float] = []

        # Benchmark runs
        for _ in range(self.n_runs):
            if self.track_memory:
                tracemalloc.start()
                start_mem = tracemalloc.get_traced_memory()[0]

            start_time = time.perf_counter()
            start_cpu = time.process_time()

            try:
                _ = func(X, *args, **kwargs)
            except Exception as e:
                if self.track_memory:
                    tracemalloc.stop()
                raise RuntimeError(f"Function execution failed: {e}") from e

            end_time = time.perf_counter()
            end_cpu = time.process_time()

            if self.track_memory:
                current_mem = tracemalloc.get_traced_memory()[0]
                tracemalloc.stop()
                mem_diff = (current_mem - start_mem) / (1024 * 1024)  # Convert to MB
                memory_usages.append(max(0.0, mem_diff))

            runtimes.append(end_time - start_time)
            if self.track_cpu:
                cpu_times.append(end_cpu - start_cpu)

        # Calculate statistics
        runtime_mean = float(np.mean(runtimes))
        runtime_std = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0

        if self.track_memory and memory_usages:
            memory_mean = float(np.mean(memory_usages))
            memory_std = float(np.std(memory_usages, ddof=1)) if len(memory_usages) > 1 else 0.0
        else:
            memory_mean = 0.0
            memory_std = 0.0

        if self.track_cpu and cpu_times:
            cpu_mean = float(np.mean(cpu_times))
            cpu_std = float(np.std(cpu_times, ddof=1)) if len(cpu_times) > 1 else 0.0
        else:
            cpu_mean = 0.0
            cpu_std = 0.0

        # Calculate throughput
        n_samples = self._get_n_samples(X)
        throughput_mean = n_samples / runtime_mean if runtime_mean > 0 and n_samples else 0.0
        throughput_std = (
            (n_samples / np.array(runtimes)).std(ddof=1) if len(runtimes) > 1 and n_samples else 0.0
        )

        return {
            "runtime": runtime_mean,
            "runtime_std": runtime_std,
            "memory_usage": memory_mean,
            "memory_usage_std": memory_std,
            "throughput": throughput_mean,
            "throughput_std": float(throughput_std),
            "cpu_time": cpu_mean,
            "cpu_time_std": cpu_std,
        }

    def evaluate_scalability(
        self,
        func: Callable[..., T_co],
        sizes: list[int],
        n_features: int = 50,
        **kwargs: Any,
    ) -> dict[int, dict[str, float]]:
        """Evaluate performance across different data sizes.

        Parameters
        ----------
        func : Callable[..., T_co]
            Function to benchmark.
        sizes : list[int]
            List of sample sizes to test.
        n_features : int, default=50
            Number of features for synthetic data.
        **kwargs : Any
            Additional arguments to pass to func.

        Returns
        -------
        dict[int, dict[str, float]]
            Dictionary mapping size to performance metrics.
        """
        results: dict[int, dict[str, float]] = {}

        for size in sizes:
            X = np.random.randn(size, n_features)
            metrics = self.evaluate(func, X, **kwargs)
            results[size] = metrics

        return results

    def compare(
        self,
        funcs: dict[str, Callable[..., T_co]],
        X: ArrayFloat | Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """Compare performance of multiple functions.

        Parameters
        ----------
        funcs : dict[str, Callable[..., T_co]]
            Dictionary mapping names to functions.
        X : ArrayFloat | Any
            Input data.
        *args : Any
            Additional positional arguments to pass to functions.
        **kwargs : Any
            Additional keyword arguments to pass to functions.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping function names to their performance metrics.
        """
        results: dict[str, dict[str, float]] = {}

        for name, func in funcs.items():
            try:
                metrics = self.evaluate(func, X, *args, **kwargs)
                results[name] = metrics
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "runtime": np.nan,
                    "runtime_std": np.nan,
                    "memory_usage": np.nan,
                    "memory_usage_std": np.nan,
                    "throughput": np.nan,
                    "throughput_std": np.nan,
                    "cpu_time": np.nan,
                    "cpu_time_std": np.nan,
                }

        return results

    def to_performance_result(
        self, metrics: dict[str, float], n_samples: int | None = None
    ) -> PerformanceResult:
        """Convert metrics dictionary to PerformanceResult.

        Parameters
        ----------
        metrics : dict[str, float]
            Performance metrics from evaluate().
        n_samples : int | None, default=None
            Number of samples (for throughput context).

        Returns
        -------
        PerformanceResult
            Frozen dataclass with performance results.
        """
        return PerformanceResult(
            runtime=metrics.get("runtime", 0.0),
            runtime_std=metrics.get("runtime_std", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            memory_usage_std=metrics.get("memory_usage_std", 0.0),
            throughput=metrics.get("throughput", 0.0),
            throughput_std=metrics.get("throughput_std", 0.0),
            cpu_time=metrics.get("cpu_time", 0.0),
            cpu_time_std=metrics.get("cpu_time_std", 0.0),
            n_runs=self.n_runs,
            n_samples=n_samples,
        )

    @staticmethod
    def _get_n_samples(X: Any) -> int | None:
        """Extract number of samples from input data.

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        int | None
            Number of samples if available, None otherwise.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return len(X)
            return X.shape[0]
        if hasattr(X, "__len__"):
            return len(X)
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_performance(
    func: Callable[..., T_co],
    X: ArrayFloat | Any,
    *args: Any,
    n_runs: int = 3,
    warmup_runs: int = 1,
    track_memory: bool = True,
    track_cpu: bool = True,
    **kwargs: Any,
) -> dict[str, float]:
    """Convenience function to evaluate performance of a function.

    Parameters
    ----------
    func : Callable[..., T_co]
        Function to benchmark.
    X : ArrayFloat | Any
        Input data.
    *args : Any
        Additional positional arguments to pass to func.
    n_runs : int, default=3
        Number of benchmark runs.
    warmup_runs : int, default=1
        Number of warmup runs.
    track_memory : bool, default=True
        Whether to track memory usage.
    track_cpu : bool, default=True
        Whether to track CPU time.
    **kwargs : Any
        Additional keyword arguments to pass to func.

    Returns
    -------
    dict[str, float]
        Dictionary of performance metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import evaluate_performance
    >>>
    >>> def my_function(X):
    ...     return X @ X.T
    >>>
    >>> X = np.random.randn(100, 50)
    >>> metrics = evaluate_performance(my_function, X, n_runs=5)
    >>> print(f"Runtime: {metrics['runtime']:.4f}s")
    """
    evaluator = PerformanceEvaluator(
        n_runs=n_runs,
        warmup_runs=warmup_runs,
        track_memory=track_memory,
        track_cpu=track_cpu,
    )
    return evaluator.evaluate(func, X, *args, **kwargs)


def benchmark_scalability(
    func: Callable[..., T_co],
    sizes: list[int],
    n_features: int = 50,
    n_runs: int = 3,
    **kwargs: Any,
) -> dict[int, dict[str, float]]:
    """Benchmark function across different data sizes.

    Parameters
    ----------
    func : Callable[..., T_co]
        Function to benchmark.
    sizes : list[int]
        List of sample sizes to test.
    n_features : int, default=50
        Number of features for synthetic data.
    n_runs : int, default=3
        Number of benchmark runs per size.
    **kwargs : Any
        Additional arguments to pass to func.

    Returns
    -------
    dict[int, dict[str, float]]
        Dictionary mapping size to performance metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import benchmark_scalability
    >>>
    >>> def pca(X):
    ...     # Simple PCA-like operation
    ...     cov = X.T @ X
    ...     return np.linalg.eigvalsh(cov)
    >>>
    >>> results = benchmark_scalability(pca, [50, 100, 200, 500])
    >>> for size, metrics in results.items():
    ...     print(f"{size}: {metrics['runtime']:.4f}s")
    """
    evaluator = PerformanceEvaluator(n_runs=n_runs)
    return evaluator.evaluate_scalability(func, sizes, n_features, **kwargs)


__all__ = [
    "PerformanceEvaluator",
    "PerformanceResult",
    "evaluate_performance",
    "benchmark_scalability",
]
