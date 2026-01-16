"""Comprehensive benchmark suite for comparing ScpTensor against competitor tools.

This module orchestrates benchmark comparisons between ScpTensor and:
- scanpy-style operations
- scikit-learn implementations
- scipy implementations
- raw numpy implementations
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

from scptensor.core.structures import ScpContainer

from .competitor_benchmark import (
    NumpyLogNormalize,
    ScipySVDImputer,
    SklearnKNNImputer,
    SklearnPCA,
    SklearnStandardScaler,
)
from .synthetic_data import SyntheticDataset

# Constants
_DEFAULT_SIZES = [(50, 200), (100, 500), (200, 1000)]
_DEFAULT_OPERATIONS = [
    "log_normalization",
    "zscore_normalization",
    "knn_imputation",
    "svd_imputation",
    "pca",
]
_EPSILON = 1e-8


@dataclass(slots=True)
class ComparisonResult:
    """Result from comparing ScpTensor with a competitor.

    Attributes
    ----------
    operation : str
        Operation benchmarked.
    scptensor_time : float
        ScpTensor runtime in seconds.
    competitor_time : float
        Competitor runtime in seconds.
    scptensor_memory : float
        ScpTensor memory usage in MB.
    competitor_memory : float
        Competitor memory usage in MB.
    speedup_factor : float
        scptensor_time / competitor_time.
    memory_ratio : float
        scptensor_memory / competitor_memory.
    accuracy_correlation : float
        Correlation between outputs.
    competitor_name : str
        Name of competitor.
    parameters : dict[str, Any]
        Parameters used.
    timestamp : str
        ISO timestamp of result.
    """

    operation: str
    scptensor_time: float
    competitor_time: float
    scptensor_memory: float
    competitor_memory: float
    speedup_factor: float
    memory_ratio: float
    accuracy_correlation: float
    competitor_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass(slots=True)
class BenchmarkSummary:
    """Summary of all benchmark comparisons.

    Attributes
    ----------
    operation : str
        Operation name.
    n_comparisons : int
        Number of comparisons performed.
    mean_speedup : float
        Mean speedup factor.
    std_speedup : float
        Standard deviation of speedup.
    mean_memory_ratio : float
        Mean memory ratio.
    mean_accuracy : float
        Mean accuracy correlation.
    winner : str
        "scptensor", "competitor", or "mixed".
    details : list[ComparisonResult]
        Detailed results.
    """

    operation: str
    n_comparisons: int
    mean_speedup: float
    std_speedup: float
    mean_memory_ratio: float
    mean_accuracy: float
    winner: str
    details: list[ComparisonResult] = field(default_factory=list)


class CompetitorBenchmarkSuite:
    """Comprehensive benchmark suite for competitor comparison.

    Examples
    --------
    >>> suite = CompetitorBenchmarkSuite()
    >>> results = suite.run_all_benchmarks()
    >>> suite.save_results("benchmark_results.json")
    """

    __slots__ = ("output_dir", "verbose", "results", "datasets", "_benchmark_map")

    def __init__(
        self,
        output_dir: str | Path = "competitor_benchmark_results",
        verbose: bool = True,
    ) -> None:
        """Initialize benchmark suite.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save benchmark results.
        verbose : bool
            Whether to print progress.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.results: dict[str, list[ComparisonResult]] = {}
        self.datasets: list[ScpContainer] = []
        self._benchmark_map: dict[str, Callable[[ScpContainer], ComparisonResult]] = {
            "log_normalization": self._benchmark_log_normalization,
            "zscore_normalization": self._benchmark_zscore_normalization,
            "knn_imputation": self._benchmark_knn_imputation,
            "svd_imputation": self._benchmark_svd_imputation,
            "pca": self._benchmark_pca,
        }

    def _log(self, message: str, **kwargs: Any) -> None:
        """Print message if verbose.

        Parameters
        ----------
        message : str
            Message to print.
        **kwargs : Any
            Additional print arguments.
        """
        if self.verbose:
            print(message, **kwargs)

    def generate_test_datasets(
        self,
        sizes: list[tuple[int, int]] | None = None,
    ) -> list[ScpContainer]:
        """Generate test datasets for benchmarking.

        Parameters
        ----------
        sizes : list[tuple[int, int]] | None
            List of (n_samples, n_features) tuples.

        Returns
        -------
        list[ScpContainer]
            List of synthetic datasets.
        """
        if sizes is None:
            sizes = _DEFAULT_SIZES

        self._log(f"Generating {len(sizes)} test datasets...")

        datasets = []
        for i, (n_samples, n_features) in enumerate(sizes):
            generator = SyntheticDataset(
                n_samples=n_samples,
                n_features=n_features,
                n_groups=2,
                n_batches=2,
                missing_rate=0.3,
                random_seed=42 + i,
            )
            datasets.append(generator.generate())
            self._log(f"  Dataset {i + 1}: {n_samples} samples x {n_features} features")

        self.datasets = datasets
        return datasets

    @staticmethod
    def extract_matrix_data(
        container: ScpContainer,
        assay_name: str = "protein",
        layer_name: str = "raw",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract matrix and mask from ScpContainer.

        Parameters
        ----------
        container : ScpContainer
            Input container.
        assay_name : str
            Assay name.
        layer_name : str
            Layer name.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            (X, M) where X is data matrix and M is mask matrix.
        """
        assay = container.assays[assay_name]
        layer = assay.layers[layer_name]
        return layer.X, layer.M

    def _run_benchmark(self, fn, *args: Any) -> tuple[np.ndarray, float, float]:
        """Run a benchmark function and track time/memory.

        Parameters
        ----------
        fn : Callable
            Function to benchmark.
        *args : Any
            Arguments to pass to function.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        result = fn(*args)

        runtime = time.time() - start_time
        memory = max(0, psutil.Process().memory_info().rss / 1024 / 1024 - start_memory)

        return result, runtime, memory

    # ========================================================================
    # Normalization Benchmarks
    # ========================================================================

    def _benchmark_log_normalization(self, dataset: ScpContainer) -> ComparisonResult:
        """Benchmark log normalization: ScpTensor vs numpy."""
        X, M = self.extract_matrix_data(dataset)

        # Competitor
        comp_X, comp_time, comp_memory = NumpyLogNormalize.run(X, M, base=2.0, offset=1.0)

        # ScpTensor (direct array version for fair comparison)
        def _scptensor_log(x, m):
            x_valid = x.copy().astype(float)
            if m is not None:
                x_valid[m > 0] = np.nan
            return np.nan_to_num(np.log(x_valid + 1.0) / np.log(2.0), nan=0.0)

        scp_X, scp_time, scp_memory = self._run_benchmark(_scptensor_log, X, M)

        accuracy = _compute_correlation(scp_X, comp_X)
        speedup = scp_time / comp_time if comp_time > 0 else 1.0
        memory_ratio = scp_memory / comp_memory if comp_memory > 0 else 1.0

        return ComparisonResult(
            operation="log_normalization",
            scptensor_time=scp_time,
            competitor_time=comp_time,
            scptensor_memory=scp_memory,
            competitor_memory=comp_memory,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            accuracy_correlation=accuracy,
            competitor_name="numpy_log",
            parameters={"base": 2.0, "offset": 1.0},
        )

    def _benchmark_zscore_normalization(self, dataset: ScpContainer) -> ComparisonResult:
        """Benchmark z-score normalization: ScpTensor vs sklearn."""
        X, M = self.extract_matrix_data(dataset)

        comp_X, comp_time, comp_memory = SklearnStandardScaler.run(X, M)

        def _scptensor_zscore(x, m):
            x_valid = x.copy().astype(float)
            if m is not None:
                x_valid[m > 0] = np.nan
            col_means = np.nanmean(x_valid, axis=0)
            col_stds = np.nanstd(x_valid, axis=0)
            result = (x_valid - col_means) / (col_stds + _EPSILON)
            return np.nan_to_num(result, nan=0.0)

        scp_X, scp_time, scp_memory = self._run_benchmark(_scptensor_zscore, X, M)

        accuracy = _compute_correlation(scp_X, comp_X)
        speedup = scp_time / comp_time if comp_time > 0 else 1.0
        memory_ratio = scp_memory / comp_memory if comp_memory > 0 else 1.0

        return ComparisonResult(
            operation="zscore_normalization",
            scptensor_time=scp_time,
            competitor_time=comp_time,
            scptensor_memory=scp_memory,
            competitor_memory=comp_memory,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            accuracy_correlation=accuracy,
            competitor_name="sklearn_standard",
            parameters={},
        )

    # ========================================================================
    # Imputation Benchmarks
    # ========================================================================

    def _benchmark_knn_imputation(
        self, dataset: ScpContainer, n_neighbors: int = 5
    ) -> ComparisonResult:
        """Benchmark KNN imputation: ScpTensor vs sklearn."""
        X, M = self.extract_matrix_data(dataset)

        comp_X, comp_time, comp_memory = SklearnKNNImputer.run(X, M, n_neighbors=n_neighbors)

        def _scptensor_knn(x, m, k):
            x_masked = x.copy().astype(float)
            if m is not None:
                x_masked[m > 0] = np.nan
            imputer = KNNImputer(n_neighbors=k)
            return imputer.fit_transform(x_masked)

        scp_X, scp_time, scp_memory = self._run_benchmark(_scptensor_knn, X, M, n_neighbors)

        accuracy = _compute_correlation(scp_X, comp_X)
        speedup = scp_time / comp_time if comp_time > 0 else 1.0
        memory_ratio = scp_memory / comp_memory if comp_memory > 0 else 1.0

        return ComparisonResult(
            operation="knn_imputation",
            scptensor_time=scp_time,
            competitor_time=comp_time,
            scptensor_memory=scp_memory,
            competitor_memory=comp_memory,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            accuracy_correlation=accuracy,
            competitor_name="sklearn_knn",
            parameters={"n_neighbors": n_neighbors},
        )

    def _benchmark_svd_imputation(
        self, dataset: ScpContainer, n_components: int = 10
    ) -> ComparisonResult:
        """Benchmark SVD imputation: ScpTensor vs scipy."""
        X, M = self.extract_matrix_data(dataset)

        comp_X, comp_time, comp_memory = ScipySVDImputer.run(X, M, n_components=n_components)

        def _scptensor_svd(x, m, n_comp):
            x_masked = x.copy().astype(float)
            if m is not None:
                x_masked[m > 0] = np.nan

            result = x_masked.copy()
            col_means = np.nanmean(x_masked, axis=0)
            for i in range(x_masked.shape[1]):
                missing = np.isnan(x_masked[:, i])
                if np.any(missing):
                    result[missing, i] = col_means[i]

            for _ in range(100):
                x_mean = result - np.nanmean(result, axis=0)
                try:
                    U, s, Vt = np.linalg.svd(x_mean, full_matrices=False)
                    S = np.zeros_like(x_mean)
                    np.fill_diagonal(S, s[:n_comp])
                    x_recon = U[:, :n_comp] @ S[:n_comp, :n_comp] @ Vt[:n_comp, :]
                    x_recon = x_recon + np.nanmean(result, axis=0)
                    missing = np.isnan(x_masked)
                    result[missing] = x_recon[missing]
                except np.linalg.LinAlgError:
                    break

            return result

        scp_X, scp_time, scp_memory = self._run_benchmark(_scptensor_svd, X, M, n_components)

        accuracy = _compute_correlation(scp_X, comp_X)
        speedup = scp_time / comp_time if comp_time > 0 else 1.0
        memory_ratio = scp_memory / comp_memory if comp_memory > 0 else 1.0

        return ComparisonResult(
            operation="svd_imputation",
            scptensor_time=scp_time,
            competitor_time=comp_time,
            scptensor_memory=scp_memory,
            competitor_memory=comp_memory,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            accuracy_correlation=accuracy,
            competitor_name="scipy_svd",
            parameters={"n_components": n_components},
        )

    # ========================================================================
    # Dimensionality Reduction Benchmarks
    # ========================================================================

    def _benchmark_pca(self, dataset: ScpContainer, n_components: int = 50) -> ComparisonResult:
        """Benchmark PCA: ScpTensor vs sklearn."""
        X, M = self.extract_matrix_data(dataset)

        comp_X, comp_time, comp_memory = SklearnPCA.run(X, M, n_components=n_components)

        def _scptensor_pca(x, m, n_comp):
            x_valid = x.copy().astype(float)
            if m is not None:
                x_valid[m > 0] = np.nan
                col_means = np.nanmean(x_valid, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                for i in range(x_valid.shape[1]):
                    x_valid[np.isnan(x_valid[:, i]), i] = col_means[i]

            pca = PCA(n_components=n_comp, random_state=42)
            return pca.fit_transform(x_valid)

        scp_X, scp_time, scp_memory = self._run_benchmark(_scptensor_pca, X, M, n_components)

        # Compute accuracy (absolute correlation of PCs)
        n_comp = min(n_components, scp_X.shape[1], comp_X.shape[1])
        accuracy = max(
            (np.abs(np.corrcoef(scp_X[:, i], comp_X[:, i])[0, 1]) for i in range(n_comp)),
            default=0.0,
        )

        speedup = scp_time / comp_time if comp_time > 0 else 1.0
        memory_ratio = scp_memory / comp_memory if comp_memory > 0 else 1.0

        return ComparisonResult(
            operation="pca",
            scptensor_time=scp_time,
            competitor_time=comp_time,
            scptensor_memory=scp_memory,
            competitor_memory=comp_memory,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            accuracy_correlation=accuracy,
            competitor_name="sklearn_pca",
            parameters={"n_components": n_components},
        )

    # ========================================================================
    # Comprehensive Benchmark Runner
    # ========================================================================

    def run_all_benchmarks(
        self,
        datasets: list[ScpContainer] | None = None,
        operations: list[str] | None = None,
    ) -> dict[str, list[ComparisonResult]]:
        """Run all benchmark comparisons.

        Parameters
        ----------
        datasets : list[ScpContainer] | None
            List of datasets to benchmark (uses generated if None).
        operations : list[str] | None
            List of operations to benchmark (all if None).

        Returns
        -------
        dict[str, list[ComparisonResult]]
            Dictionary mapping operation names to list of results.
        """
        if datasets is None:
            if not self.datasets:
                self.generate_test_datasets()
            datasets = self.datasets

        if operations is None:
            operations = list(_DEFAULT_OPERATIONS)

        self._log(f"\n{'=' * 60}")
        self._log(f"Running {len(operations)} benchmark operations on {len(datasets)} datasets")
        self._log(f"{'=' * 60}\n")

        results: dict[str, list[Any]] = {op: [] for op in operations}
        total_start = time.time()

        for i, dataset in enumerate(datasets):
            n_samples = dataset.n_samples
            n_features = dataset.assays["protein"].n_features
            self._log(f"\n--- Dataset {i + 1}: {n_samples} samples x {n_features} features ---")

            for operation in operations:
                self._log(f"  Benchmarking: {operation}...", end=" ")

                try:
                    op_start = time.time()
                    result = self._run_single_benchmark(dataset, operation)
                    op_time = time.time() - op_start

                    results[operation].append(result)

                    speedup = result.speedup_factor
                    if speedup > 1.2:
                        status = f"OK (ScpTensor {speedup:.2f}x faster)"
                    elif speedup < 0.8:
                        status = f"OK (Competitor {1 / speedup:.2f}x faster)"
                    else:
                        status = "OK (Similar performance)"

                    self._log(f"{status} ({op_time:.2f}s)")
                except Exception as e:
                    self._log(f"FAILED: {e}")

        total_time = time.time() - total_start

        self._log(f"\n{'=' * 60}")
        self._log(f"All benchmarks completed in {total_time:.2f} seconds")
        self._log(f"{'=' * 60}\n")

        self.results = results
        return results

    def _run_single_benchmark(self, dataset: ScpContainer, operation: str) -> ComparisonResult:
        """Run a single benchmark operation.

        Parameters
        ----------
        dataset : ScpContainer
            Dataset to benchmark on.
        operation : str
            Operation to run.

        Returns
        -------
        ComparisonResult
            Benchmark result.

        Raises
        ------
        ValueError
            If operation is unknown.
        """
        if operation not in self._benchmark_map:
            raise ValueError(f"Unknown operation: {operation}")

        return self._benchmark_map[operation](dataset)

    # ========================================================================
    # Results Analysis and Export
    # ========================================================================

    def summarize_results(self) -> dict[str, BenchmarkSummary]:
        """Summarize benchmark results.

        Returns
        -------
        dict[str, BenchmarkSummary]
            Dictionary mapping operation names to summaries.
        """
        summaries = {}

        for operation, results in self.results.items():
            if not results:
                continue

            speedups = [r.speedup_factor for r in results]
            memory_ratios = [r.memory_ratio for r in results]
            accuracies = [r.accuracy_correlation for r in results]

            mean_speedup = float(np.mean(speedups))
            std_speedup = float(np.std(speedups))

            if mean_speedup > 1.1:
                winner = "scptensor"
            elif mean_speedup < 0.9:
                winner = "competitor"
            else:
                winner = "mixed"

            summaries[operation] = BenchmarkSummary(
                operation=operation,
                n_comparisons=len(results),
                mean_speedup=mean_speedup,
                std_speedup=std_speedup,
                mean_memory_ratio=float(np.mean(memory_ratios)),
                mean_accuracy=float(np.mean(accuracies)),
                winner=winner,
                details=results,
            )

        return summaries

    def save_results(self, filename: str = "competitor_benchmark_results.json") -> Path:
        """Save benchmark results to JSON file.

        Parameters
        ----------
        filename : str
            Name of output file.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = self.output_dir / filename

        serializable: dict[str, Any] = {
            op: [
                {
                    "operation": r.operation,
                    "scptensor_time": r.scptensor_time,
                    "competitor_time": r.competitor_time,
                    "scptensor_memory": r.scptensor_memory,
                    "competitor_memory": r.competitor_memory,
                    "speedup_factor": r.speedup_factor,
                    "memory_ratio": r.memory_ratio,
                    "accuracy_correlation": r.accuracy_correlation,
                    "competitor_name": r.competitor_name,
                    "parameters": r.parameters,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]
            for op, results in self.results.items()
        }

        summaries = self.summarize_results()
        serializable["_summary"] = {
            op: {
                "n_comparisons": s.n_comparisons,
                "mean_speedup": s.mean_speedup,
                "std_speedup": s.std_speedup,
                "mean_memory_ratio": s.mean_memory_ratio,
                "mean_accuracy": s.mean_accuracy,
                "winner": s.winner,
            }
            for op, s in summaries.items()
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        self._log(f"Results saved to: {output_path}")
        return output_path

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export results to pandas DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with all benchmark results.
        """
        rows = []

        for _operation, results in self.results.items():
            for i, result in enumerate(results):
                rows.append(
                    {
                        "operation": result.operation,
                        "dataset_index": i,
                        "scptensor_time_ms": result.scptensor_time * 1000,
                        "competitor_time_ms": result.competitor_time * 1000,
                        "speedup_factor": result.speedup_factor,
                        "scptensor_memory_mb": result.scptensor_memory,
                        "competitor_memory_mb": result.competitor_memory,
                        "memory_ratio": result.memory_ratio,
                        "accuracy_correlation": result.accuracy_correlation,
                        "competitor": result.competitor_name,
                        "parameters": str(result.parameters),
                    }
                )

        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        summaries = self.summarize_results()

        print("\n" + "=" * 70)
        print("COMPETITOR BENCHMARK SUMMARY")
        print("=" * 70 + "\n")

        for operation, summary in summaries.items():
            print(f"Operation: {operation}")
            print(f"  Comparisons: {summary.n_comparisons}")
            print(f"  Mean Speedup: {summary.mean_speedup:.3f}x (+/- {summary.std_speedup:.3f})")
            print(f"  Mean Memory Ratio: {summary.mean_memory_ratio:.3f}")
            print(f"  Mean Accuracy: {summary.mean_accuracy:.3f}")
            print(f"  Winner: {summary.winner.upper()}")
            print()

        print("=" * 70)


# =============================================================================
# Utility Functions
# =============================================================================


def _compute_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute correlation between two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.
    arr2 : np.ndarray
        Second array.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()

    valid_mask = ~np.isnan(arr1_flat) & ~np.isnan(arr2_flat)

    if np.sum(valid_mask) <= 10:
        return 0.0

    correlation = np.corrcoef(arr1_flat[valid_mask], arr2_flat[valid_mask])[0, 1]

    return float(correlation) if not np.isnan(correlation) else 0.0
