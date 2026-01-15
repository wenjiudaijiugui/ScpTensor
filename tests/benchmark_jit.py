"""
Benchmark script to compare JIT-compiled vs pure NumPy performance.

This script measures the speedup provided by Numba JIT compilation
for various operations in ScpTensor.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scptensor.core import jit_ops


def benchmark_function(func, *args, warmup=3, runs=10, **kwargs):
    """Benchmark a function with warmup and multiple runs.

    Returns
    -------
    dict
        Contains mean_time, std_time, and speedup if applicable
    """
    # Warmup runs to allow JIT compilation
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "result": result,
    }


def print_benchmark(name: str, jit_result: dict, fallback_result: dict = None):
    """Print benchmark results in a formatted way."""
    print(f"\n{name}:")
    print(f"  JIT:      {jit_result['mean'] * 1000:.4f} ms (std: {jit_result['std'] * 1000:.4f})")

    if fallback_result:
        speedup = fallback_result["mean"] / jit_result["mean"]
        print(
            f"  Fallback: {fallback_result['mean'] * 1000:.4f} ms (std: {fallback_result['std'] * 1000:.4f})"
        )
        print(f"  Speedup:  {speedup:.2f}x")
    else:
        print("  (No fallback comparison available)")


def benchmark_distance_functions():
    """Benchmark distance calculation functions."""
    print("\n" + "=" * 60)
    print("DISTANCE CALCULATION BENCHMARKS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    missing_rate = 0.2

    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Add NaN values
    X_nan = X.copy()
    mask = np.random.rand(n_samples, n_features) < missing_rate
    X_nan[mask] = np.nan

    Y_nan = Y.copy()
    mask = np.random.rand(n_samples, n_features) < missing_rate
    Y_nan[mask] = np.nan

    # Benchmark: nan_euclidean_distance_matrix_to_matrix
    # Use smaller size for this one
    n_small = 100
    X_small = X_nan[:n_small]
    Y_small = Y_nan[:n_small]

    if jit_ops.NUMBA_AVAILABLE:
        # JIT version
        jit_result = benchmark_function(
            jit_ops.nan_euclidean_distance_matrix_to_matrix, X_small, Y_small, warmup=2, runs=5
        )

        # Fallback version (sklearn)
        fallback_result = benchmark_function(
            nan_euclidean_distance_matrix_to_matrix_fallback, X_small, Y_small, warmup=0, runs=5
        )

        print_benchmark(
            "nan_euclidean_distance_matrix_to_matrix (100x100)", jit_result, fallback_result
        )
    else:
        print("  Numba not available, skipping JIT benchmarks")

    # Benchmark: euclidean_distance_no_nan
    x = X_nan[0]
    y = Y_nan[0]

    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(
            jit_ops.euclidean_distance_no_nan, x, y, warmup=2, runs=1000
        )

        # Pure NumPy fallback
        def numpy_euclidean_distance_no_nan(x, y):
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_mask) == 0:
                return np.inf
            return np.sqrt(np.sum((x[valid_mask] - y[valid_mask]) ** 2) / np.sum(valid_mask))

        fallback_result = benchmark_function(
            numpy_euclidean_distance_no_nan, x, y, warmup=0, runs=1000
        )

        print_benchmark("euclidean_distance_no_nan (vector)", jit_result, fallback_result)


def nan_euclidean_distance_matrix_to_matrix_fallback(X, Y):
    """Pure NumPy fallback for distance matrix calculation."""
    from sklearn.metrics.pairwise import nan_euclidean_distances

    return nan_euclidean_distances(X, Y)


def benchmark_statistical_functions():
    """Benchmark statistical functions."""
    print("\n" + "=" * 60)
    print("STATISTICAL FUNCTION BENCHMARKS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_rows = 1000
    n_cols = 200
    missing_rate = 0.2

    X = np.random.randn(n_rows, n_cols)
    mask = np.random.rand(n_rows, n_cols) < missing_rate
    X[mask] = np.nan

    # Benchmark: mean_axis_no_nan
    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(
            jit_ops.mean_axis_no_nan,
            X,
            0,  # axis=0 for column means
            warmup=2,
            runs=20,
        )

        fallback_result = benchmark_function(np.nanmean, X, 0, warmup=0, runs=20)

        print_benchmark(
            f"mean_axis_no_nan (axis=0, {n_rows}x{n_cols})", jit_result, fallback_result
        )

    # Benchmark: var_axis_no_nan
    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(jit_ops.var_axis_no_nan, X, 0, warmup=2, runs=20)

        fallback_result = benchmark_function(np.nanvar, X, 0, ddof=1, warmup=0, runs=20)

        print_benchmark(f"var_axis_no_nan (axis=0, {n_rows}x{n_cols})", jit_result, fallback_result)


def benchmark_threshold_functions():
    """Benchmark threshold and filtering functions."""
    print("\n" + "=" * 60)
    print("THRESHOLD/FILTERING BENCHMARKS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_rows = 1000
    n_cols = 500
    threshold = 0.5

    X = np.random.randn(n_rows, n_cols)

    # Benchmark: count_above_threshold
    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(
            jit_ops.count_above_threshold,
            X,
            threshold,
            1,  # axis=1 for row-wise counts
            warmup=2,
            runs=50,
        )

        # NumPy fallback
        def numpy_count_above_threshold(X, threshold, axis):
            return np.sum(threshold < X, axis=axis)

        fallback_result = benchmark_function(
            numpy_count_above_threshold, X, threshold, 1, warmup=0, runs=50
        )

        print_benchmark(
            f"count_above_threshold (axis=1, {n_rows}x{n_cols})", jit_result, fallback_result
        )

    # Benchmark: filter_by_threshold_count
    min_count = 100

    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(
            jit_ops.filter_by_threshold_count, X, threshold, min_count, 1, warmup=2, runs=50
        )

        # NumPy fallback
        def numpy_filter_by_threshold_count(X, threshold, min_count, axis):
            counts = np.sum(threshold < X, axis=axis)
            return counts >= min_count

        fallback_result = benchmark_function(
            numpy_filter_by_threshold_count, X, threshold, min_count, 1, warmup=0, runs=50
        )

        print_benchmark(
            f"filter_by_threshold_count (axis=1, {n_rows}x{n_cols})", jit_result, fallback_result
        )


def benchmark_imputation_functions():
    """Benchmark imputation functions."""
    print("\n" + "=" * 60)
    print("IMPUTATION BENCHMARKS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_rows = 500
    n_cols = 200
    missing_rate = 0.3

    X = np.random.randn(n_rows, n_cols)
    mask = np.random.rand(n_rows, n_cols) < missing_rate
    X[mask] = np.nan

    # Benchmark: impute_nan_with_col_means
    if jit_ops.NUMBA_AVAILABLE:
        # Need fresh copies for each run
        def jit_impute_col():
            X_copy = X.copy()
            jit_ops.impute_nan_with_col_means(X_copy)
            return X_copy

        jit_result = benchmark_function(jit_impute_col, warmup=2, runs=20)

        # NumPy fallback
        def numpy_impute_col():
            X_copy = X.copy()
            col_means = np.nanmean(X_copy, axis=0, keepdims=True)
            np.copyto(X_copy, col_means, where=np.isnan(X_copy))
            return X_copy

        fallback_result = benchmark_function(numpy_impute_col, warmup=0, runs=20)

        print_benchmark(
            f"impute_nan_with_col_means ({n_rows}x{n_cols}, {missing_rate * 100:.0f}% NaN)",
            jit_result,
            fallback_result,
        )

    # Benchmark: impute_nan_with_row_means
    if jit_ops.NUMBA_AVAILABLE:

        def jit_impute_row():
            X_copy = X.copy()
            jit_ops.impute_nan_with_row_means(X_copy)
            return X_copy

        jit_result = benchmark_function(jit_impute_row, warmup=2, runs=20)

        # NumPy fallback
        def numpy_impute_row():
            X_copy = X.copy()
            row_means = np.nanmean(X_copy, axis=1, keepdims=True)
            np.copyto(X_copy, row_means, where=np.isnan(X_copy))
            return X_copy

        fallback_result = benchmark_function(numpy_impute_row, warmup=0, runs=20)

        print_benchmark(
            f"impute_nan_with_row_means ({n_rows}x{n_cols}, {missing_rate * 100:.0f}% NaN)",
            jit_result,
            fallback_result,
        )


def benchmark_mask_functions():
    """Benchmark mask code functions."""
    print("\n" + "=" * 60)
    print("MASK CODE BENCHMARKS")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n_rows = 1000
    n_cols = 500

    M = np.random.randint(0, 6, (n_rows, n_cols), dtype=np.int64)

    # Benchmark: count_mask_codes
    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(jit_ops.count_mask_codes, M, warmup=2, runs=100)

        # NumPy fallback
        def numpy_count_mask_codes(M):
            counts = np.zeros(7, dtype=np.int64)
            for code in M.flat:
                if 0 <= code < 7:
                    counts[code] += 1
            return counts

        fallback_result = benchmark_function(numpy_count_mask_codes, M, warmup=0, runs=100)

        print_benchmark(f"count_mask_codes ({n_rows}x{n_cols})", jit_result, fallback_result)

    # Benchmark: find_missing_indices
    mask_codes = (1, 2)  # MBR and LOD

    if jit_ops.NUMBA_AVAILABLE:
        jit_result = benchmark_function(
            jit_ops.find_missing_indices, M, mask_codes, warmup=2, runs=100
        )

        # NumPy fallback
        def numpy_find_missing_indices(M, mask_codes):
            mask = np.isin(M, list(mask_codes))
            return np.where(mask)

        fallback_result = benchmark_function(
            numpy_find_missing_indices, M, mask_codes, warmup=0, runs=100
        )

        print_benchmark(f"find_missing_indices ({n_rows}x{n_cols})", jit_result, fallback_result)


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("ScpTensor JIT Performance Benchmark")
    print("=" * 60)
    print(f"\nNumba available: {jit_ops.NUMBA_AVAILABLE}")

    if not jit_ops.NUMBA_AVAILABLE:
        print("\nNumba is not installed. JIT functions will use fallback implementations.")
        print("Install numba for better performance: pip install numba")
        return

    print("\nBenchmark configuration:")
    print("  - Warmup runs: included in each benchmark")
    print("  - Timed runs: varies per test")
    print("  - Time shown: mean execution time")

    # Run all benchmarks
    benchmark_mask_functions()
    benchmark_statistical_functions()
    benchmark_threshold_functions()
    benchmark_imputation_functions()
    benchmark_distance_functions()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
