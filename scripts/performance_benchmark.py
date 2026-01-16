#!/usr/bin/env python
"""Performance benchmarking script for ScpTensor.

This script benchmarks key operations across the ScpTensor library
to identify performance bottlenecks and track optimizations.

Run with:
    uv run python scripts/performance_benchmark.py

Or with HTML output:
    uv run python scripts/performance_benchmark.py --html output.html
"""

import argparse
import gc
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Set random seed for reproducibility
np.random.seed(42)

try:
    import polars as pl
except ImportError:
    print("Polars not installed. Installing...")
    import subprocess

    subprocess.check_call(["uv", "pip", "install", "polars", "-q"])
    import polars as pl

# Import ScpTensor modules
from scptensor.core import jit_ops, sparse_utils
from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single operation."""

    name: str
    operation: str
    size: tuple[int, int]
    time_ms: float
    memory_mb: float
    sparsity: float = 0.0
    iterations: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Benchmark suite for running tests and collecting results."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def get_results_by_operation(self, operation: str) -> list[BenchmarkResult]:
        """Get all results for a specific operation."""
        return [r for r in self.results if r.operation == operation]

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK SUITE: {self.name}")
        print(f"{'=' * 80}")

        # Group by operation
        ops = defaultdict(list)
        for r in self.results:
            ops[r.operation].append(r)

        for op_name, results in sorted(ops.items()):
            print(f"\n{op_name}:")
            print(f"  {'Size':<20} {'Sparsity':<12} {'Time (ms)':<15} {'Memory (MB)':<15}")
            print(f"  {'-' * 60}")
            for r in results:
                if len(r.size) >= 2:
                    size_str = f"{r.size[0]}x{r.size[1]}"
                else:
                    size_str = "N/A"
                sparsity_str = f"{r.sparsity:.1%}" if r.sparsity > 0 else "dense"
                print(
                    f"  {size_str:<20} {sparsity_str:<12} {r.time_ms:<15.3f} {r.memory_mb:<15.2f}"
                )

    def save_json(self, path: str | Path) -> None:
        """Save results to JSON file."""
        data = {
            "suite_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "operation": r.operation,
                    "size": r.size,
                    "time_ms": r.time_ms,
                    "memory_mb": r.memory_mb,
                    "sparsity": r.sparsity,
                    "iterations": r.iterations,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()
        return 0.0

    current, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


def benchmark_operation(
    name: str,
    operation: str,
    func,
    *args,
    iterations: int = 3,
    warmup: int = 1,
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a single operation.

    Args:
        name: Descriptive name for the benchmark
        operation: Category of operation (e.g., 'distance', 'log_transform')
        func: Function to benchmark
        *args: Positional arguments for func
        iterations: Number of times to run the benchmark
        warmup: Number of warmup runs (not timed)
        **kwargs: Keyword arguments for func

    Returns:
        BenchmarkResult with timing and memory information
    """
    # Warmup runs
    for _ in range(warmup):
        result = func(*args, **kwargs)
        del result
        gc.collect()

    # Timed runs
    times = []
    import tracemalloc

    tracemalloc.start()

    for _ in range(iterations):
        gc.collect()
        tracemalloc.reset_peak()

        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms
        _, peak_mb = tracemalloc.get_traced_memory()

        del result
        gc.collect()

    tracemalloc.stop()

    avg_time = np.mean(times)
    memory_mb = peak_mb / 1024 / 1024

    return BenchmarkResult(
        name=name,
        operation=operation,
        size=args[0].shape if hasattr(args[0], "shape") else (0, 0),
        time_ms=avg_time,
        memory_mb=memory_mb,
        iterations=iterations,
    )


def generate_test_data(
    n_samples: int,
    n_features: int,
    sparsity: float = 0.5,
    missing_rate: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate test data for benchmarking.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        sparsity: Ratio of zero values
        missing_rate: Ratio of NaN values

    Returns:
        (X, missing_mask) tuple
    """
    X = np.random.randn(n_samples, n_features)

    # Apply sparsity
    if sparsity > 0:
        mask = np.random.rand(n_samples, n_features) < sparsity
        X[mask] = 0

    # Apply missing values
    missing_mask = np.random.rand(n_samples, n_features) < missing_rate
    X[missing_mask] = np.nan

    return X, missing_mask


def benchmark_jit_operations(suite: BenchmarkSuite) -> None:
    """Benchmark JIT-compiled operations."""
    print("\nBenchmarking JIT operations...")

    sizes = [(100, 50), (500, 100), (1000, 200), (5000, 500)]

    for n_samples, n_features in sizes:
        X, _ = generate_test_data(n_samples, n_features, sparsity=0.0, missing_rate=0.3)

        # Benchmark euclidean distance
        if jit_ops.NUMBA_AVAILABLE:
            result = benchmark_operation(
                f"euclidean_distance_{n_samples}x{n_features}",
                "euclidean_distance",
                jit_ops.euclidean_distance_no_nan,
                X[0],
                X[1],
                iterations=10,
            )
            result.sparsity = 0.0
            suite.add_result(result)

        # Benchmark pairwise distances
        if jit_ops.NUMBA_AVAILABLE:
            X_small = X[: min(100, n_samples)]
            result = benchmark_operation(
                f"pairwise_distances_{n_samples}x{n_features}",
                "pairwise_distances",
                jit_ops.nan_euclidean_distance_row_to_matrix,
                X[0],
                X_small,
                iterations=5,
            )
            result.sparsity = 0.0
            suite.add_result(result)

        # Benchmark mean_no_nan
        if jit_ops.NUMBA_AVAILABLE:
            result = benchmark_operation(
                f"mean_no_nan_{n_samples}x{n_features}",
                "mean_no_nan",
                jit_ops.mean_no_nan,
                X[0],
                iterations=100,
            )
            result.sparsity = 0.0
            suite.add_result(result)

        # Benchmark mean_axis_no_nan
        if jit_ops.NUMBA_AVAILABLE:
            result = benchmark_operation(
                f"mean_axis_{n_samples}x{n_features}",
                "mean_axis_no_nan",
                jit_ops.mean_axis_no_nan,
                X,
                0,
                iterations=20,
            )
            result.sparsity = 0.0
            suite.add_result(result)

        # Benchmark count_mask_codes
        if jit_ops.NUMBA_AVAILABLE:
            M = np.random.randint(0, 6, (n_samples, n_features), dtype=np.int8)
            result = benchmark_operation(
                f"count_mask_codes_{n_samples}x{n_features}",
                "count_mask_codes",
                jit_ops.count_mask_codes,
                M,
                iterations=50,
            )
            result.sparsity = 0.0
            suite.add_result(result)


def benchmark_sparse_operations(suite: BenchmarkSuite) -> None:
    """Benchmark sparse matrix operations."""
    print("\nBenchmarking sparse matrix operations...")

    sizes = [(100, 50), (500, 100), (1000, 200), (5000, 500)]
    sparsities = [0.5, 0.7, 0.9]

    for n_samples, n_features in sizes:
        for sparsity in sparsities:
            X_dense, _ = generate_test_data(
                n_samples, n_features, sparsity=sparsity, missing_rate=0.0
            )
            X_sparse = sp.csr_matrix(X_dense)

            # Benchmark sparse_safe_log1p
            result = benchmark_operation(
                f"sparse_log1p_{n_samples}x{n_features}_sparse{sparsity}",
                "sparse_log1p",
                sparse_utils.sparse_safe_log1p,
                X_sparse,
                iterations=10,
            )
            result.sparsity = sparsity
            suite.add_result(result)

            # Benchmark sparse_safe_log1p_with_scale
            result = benchmark_operation(
                f"sparse_log1p_scale_{n_samples}x{n_features}_sparse{sparsity}",
                "sparse_log1p_scale",
                sparse_utils.sparse_safe_log1p_with_scale,
                X_sparse,
                1.0,
                np.log(2.0),
                iterations=10,
            )
            result.sparsity = sparsity
            suite.add_result(result)

            # Benchmark sparse_multiply_rowwise
            factors = np.random.randn(n_samples)
            result = benchmark_operation(
                f"sparse_mul_row_{n_samples}x{n_features}_sparse{sparsity}",
                "sparse_multiply_rowwise",
                sparse_utils.sparse_multiply_rowwise,
                X_sparse,
                factors,
                iterations=10,
            )
            result.sparsity = sparsity
            suite.add_result(result)

            # Benchmark sparse_multiply_colwise
            factors_col = np.random.randn(n_features)
            result = benchmark_operation(
                f"sparse_mul_col_{n_samples}x{n_features}_sparse{sparsity}",
                "sparse_multiply_colwise",
                sparse_utils.sparse_multiply_colwise,
                X_sparse,
                factors_col,
                iterations=10,
            )
            result.sparsity = sparsity
            suite.add_result(result)


def benchmark_normalization_operations(suite: BenchmarkSuite) -> None:
    """Benchmark normalization operations."""
    print("\nBenchmarking normalization operations...")

    from scptensor.normalization.log import log_normalize

    sizes = [(100, 50), (500, 100), (1000, 200), (5000, 500)]

    for n_samples, n_features in sizes:
        # Create test container
        X_dense = np.abs(np.random.randn(n_samples, n_features)) * 100 + 1
        X_sparse = sp.csr_matrix(X_dense)

        obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
        var = pl.DataFrame({"_index": [f"f{i}" for i in range(n_features)]})

        # Test with sparse matrix
        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_sparse))
        container = ScpContainer(obs=obs, assays={"test": assay})

        result = benchmark_operation(
            f"log_normalize_sparse_{n_samples}x{n_features}",
            "log_normalize",
            log_normalize,
            container,
            "test",
            "raw",
            "log",
            iterations=5,
        )
        result.sparsity = 0.0
        suite.add_result(result)


def benchmark_imputation_operations(suite: BenchmarkSuite) -> None:
    """Benchmark imputation operations."""
    print("\nBenchmarking imputation operations...")

    from scptensor.impute.knn import knn
    from scptensor.impute.ppca import ppca

    # Smaller sizes for imputation (more expensive)
    sizes = [(50, 30), (100, 50), (200, 100)]

    for n_samples, n_features in sizes:
        X, missing_mask = generate_test_data(n_samples, n_features, sparsity=0.0, missing_rate=0.2)

        obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
        var = pl.DataFrame({"_index": [f"f{i}" for i in range(n_features)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X))
        container = ScpContainer(obs=obs, assays={"test": assay})

        # Benchmark KNN imputation (small k for speed)
        result = benchmark_operation(
            f"knn_impute_{n_samples}x{n_features}",
            "knn_impute",
            knn,
            container,
            "test",
            "raw",
            "imputed",
            3,
            "uniform",
            100,
            iterations=2,
            warmup=0,
        )
        result.sparsity = 0.0
        suite.add_result(result)

        # Benchmark PPCA imputation (fewer components for speed)
        assay2 = Assay(var=var)
        assay2.add_layer("raw", ScpMatrix(X=X))
        container2 = ScpContainer(obs=obs, assays={"test": assay2})

        result = benchmark_operation(
            f"ppca_impute_{n_samples}x{n_features}",
            "ppca_impute",
            ppca,
            container2,
            "test",
            "raw",
            "imputed",
            5,
            20,
            1e-4,
            42,
            iterations=2,
            warmup=0,
        )
        result.sparsity = 0.0
        suite.add_result(result)


def benchmark_matrix_operations(suite: BenchmarkSuite) -> None:
    """Benchmark core matrix operations."""
    print("\nBenchmarking matrix operations...")

    from scptensor.core.matrix_ops import MatrixOps

    sizes = [(100, 50), (500, 100), (1000, 200), (5000, 500)]

    for n_samples, n_features in sizes:
        X, missing_mask = generate_test_data(n_samples, n_features, sparsity=0.0, missing_rate=0.2)
        M = np.zeros((n_samples, n_features), dtype=np.int8)
        M[missing_mask] = MaskCode.MBR.value

        matrix = ScpMatrix(X=X, M=M)

        # Benchmark get_valid_mask
        result = benchmark_operation(
            f"get_valid_mask_{n_samples}x{n_features}",
            "get_valid_mask",
            MatrixOps.get_valid_mask,
            matrix,
            iterations=50,
        )
        result.sparsity = 0.0
        suite.add_result(result)

        # Benchmark get_mask_statistics
        result = benchmark_operation(
            f"mask_stats_{n_samples}x{n_features}",
            "mask_statistics",
            MatrixOps.get_mask_statistics,
            matrix,
            iterations=20,
        )
        result.sparsity = 0.0
        suite.add_result(result)

        # Benchmark mark_values
        indices = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        result = benchmark_operation(
            f"mark_values_{n_samples}x{n_features}",
            "mark_values",
            MatrixOps.mark_values,
            matrix,
            indices,
            MaskCode.IMPUTED,
            iterations=50,
        )
        result.sparsity = 0.0
        suite.add_result(result)


def run_all_benchmarks() -> BenchmarkSuite:
    """Run all benchmark suites."""
    suite = BenchmarkSuite(name="ScpTensor Performance Benchmark")

    print(f"\n{'=' * 80}")
    print("SCP TENSOR PERFORMANCE BENCHMARK")
    print(f"{'=' * 80}")
    print(f"Numba available: {jit_ops.NUMBA_AVAILABLE}")
    print(f"Scipy version: {sp.__version__}")
    print(f"NumPy version: {np.__version__}")

    benchmark_jit_operations(suite)
    benchmark_sparse_operations(suite)
    benchmark_matrix_operations(suite)
    benchmark_normalization_operations(suite)
    benchmark_imputation_operations(suite)

    return suite


def generate_comparison_report(suite: BenchmarkSuite) -> None:
    """Generate a comparison report for optimization tracking."""
    print(f"\n{'=' * 80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")

    # Calculate average times per operation category
    ops = defaultdict(list)
    for r in suite.results:
        ops[r.operation].append(r.time_ms)

    print("\nAverage execution time by operation category:")
    for op_name, times in sorted(ops.items()):
        avg = np.mean(times)
        std = np.std(times)
        print(f"  {op_name:<30} {avg:>10.3f} ms (std: {std:.3f})")

    # Identify slowest operations
    all_results = sorted(suite.results, key=lambda r: r.time_ms, reverse=True)
    print("\nTop 10 slowest benchmarks:")
    for i, r in enumerate(all_results[:10], 1):
        if len(r.size) >= 2:
            size_str = f"{r.size[0]}x{r.size[1]}"
        else:
            size_str = "N/A"
        print(f"  {i}. {r.name:<50} {r.time_ms:>10.3f} ms ({size_str})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ScpTensor Performance Benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument(
        "--suite",
        "-s",
        type=str,
        default="all",
        choices=["all", "jit", "sparse", "matrix", "normalize", "impute"],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick benchmark (smaller sizes, fewer iterations)",
    )

    args = parser.parse_args()

    suite = BenchmarkSuite(name="ScpTensor Performance Benchmark")

    if args.suite == "all":
        suite = run_all_benchmarks()
    elif args.suite == "jit":
        benchmark_jit_operations(suite)
    elif args.suite == "sparse":
        benchmark_sparse_operations(suite)
    elif args.suite == "matrix":
        benchmark_matrix_operations(suite)
    elif args.suite == "normalize":
        benchmark_normalization_operations(suite)
    elif args.suite == "impute":
        benchmark_imputation_operations(suite)

    # Print summary
    suite.print_summary()
    generate_comparison_report(suite)

    # Save results
    if args.output:
        suite.save_json(args.output)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"benchmark_results_{timestamp}.json"
        suite.save_json(default_path)
        print(f"\nResults saved to {default_path}")

    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
