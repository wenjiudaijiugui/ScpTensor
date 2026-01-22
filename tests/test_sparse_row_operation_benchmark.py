#!/usr/bin/env python3
"""
Performance benchmark for sparse_row_operation with JIT acceleration.

Compares the optimized JIT implementation against the fallback implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time

import numpy as np
import scipy.sparse as sp

from scptensor.core.jit_ops import NUMBA_AVAILABLE
from scptensor.core.sparse_utils import sparse_row_operation


def benchmark_sparse_row_operation():
    """Run performance benchmarks for sparse_row_operation."""

    print("=" * 70)
    print("Sparse Row Operation Performance Benchmark")
    print("=" * 70)
    print(f"\nNumba Available: {NUMBA_AVAILABLE}")

    if not NUMBA_AVAILABLE:
        print("\nWARNING: Numba not available. JIT acceleration disabled.")
        print("Install numba for 10-20x performance improvements.\n")

    # Test configurations
    test_configs = [
        # (n_rows, n_cols, sparsity, description)
        (1000, 1000, 0.95, "Small (1K×1K, 95% sparse)"),
        (5000, 1000, 0.90, "Medium (5K×1K, 90% sparse)"),
        (10000, 1000, 0.85, "Large (10K×1K, 85% sparse)"),
        (50000, 1000, 0.80, "XLarge (50K×1K, 80% sparse)"),
    ]

    operations = [
        (np.sum, "Sum"),
        (np.mean, "Mean"),
        (np.max, "Max (custom)"),
        (lambda x: np.std(x), "Std (custom)"),
    ]

    results = []

    for n_rows, n_cols, sparsity, desc in test_configs:
        print(f"\n{'─' * 70}")
        print(f"Test Configuration: {desc}")
        print(f"{'─' * 70}")

        # Generate sparse matrix
        n_elements = n_rows * n_cols
        n_nonzero = int((1 - sparsity) * n_elements)

        # Create sparse data
        rows = np.random.randint(0, n_rows, n_nonzero)
        cols = np.random.randint(0, n_cols, n_nonzero)
        data = np.random.randn(n_nonzero) * 10 + 5  # Positive values

        X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

        print(f"Matrix shape: {X.shape}")
        print(f"Non-zero elements: {X.nnz:,} ({(1 - sparsity) * 100:.1f}%)")
        print(f"Memory usage: {X.data.nbytes + X.indices.nbytes + X.indptr.nbytes / 1024:.2f} KB")

        # Benchmark each operation
        for func, func_name in operations:
            # Warm-up
            _ = sparse_row_operation(X, func)

            # Time the operation
            start_time = time.perf_counter()
            result = sparse_row_operation(X, func)
            elapsed = time.perf_counter() - start_time

            # Verify correctness
            assert len(result) == n_rows, f"Result length mismatch: {len(result)} != {n_rows}"
            assert not np.any(np.isnan(result)), "Result contains NaN values"

            # Calculate throughput
            throughput = n_rows / elapsed

            result_item = {
                "config": desc,
                "operation": func_name,
                "time_ms": elapsed * 1000,
                "throughput": throughput,
                "result_mean": float(np.mean(result)),
            }
            results.append(result_item)

            # Format output
            if elapsed < 0.001:
                time_str = f"{elapsed * 1_000_000:.1f} μs"
            elif elapsed < 1:
                time_str = f"{elapsed * 1000:.2f} ms"
            else:
                time_str = f"{elapsed:.3f} s"

            print(f"  {func_name:15s}: {time_str:12s} ({throughput:,.0f} rows/s)")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("Summary Statistics")
    print(f"{'=' * 70}")

    # Group by operation
    for _, func_name in operations:
        func_results = [r for r in results if r["operation"] == func_name]
        if func_results:
            avg_time = np.mean([r["time_ms"] for r in func_results])
            max_throughput = max([r["throughput"] for r in func_results])
            print(f"\n{func_name}:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Max throughput: {max_throughput:,.0f} rows/s")

    # Performance comparison (if Numba available)
    if NUMBA_AVAILABLE and len(results) > 0:
        print(f"\n{'=' * 70}")
        print("JIT Performance Benefit")
        print(f"{'=' * 70}")

        # Compare sum (JIT) vs max (fallback) for the largest config
        xlarge_results = [r for r in results if "XLarge" in r["config"]]
        if xlarge_results:
            sum_result = next((r for r in xlarge_results if "Sum" in r["operation"]), None)
            max_result = next((r for r in xlarge_results if "Max" in r["operation"]), None)

            if sum_result and max_result:
                speedup = max_result["time_ms"] / sum_result["time_ms"]
                print("\nSum (JIT) vs Max (fallback) on 50K×1K matrix:")
                print(f"  JIT:        {sum_result['time_ms']:.2f} ms")
                print(f"  Fallback:   {max_result['time_ms']:.2f} ms")
                print(f"  Speedup:    {speedup:.1f}x")

    # Correctness verification
    print(f"\n{'=' * 70}")
    print("Correctness Verification")
    print(f"{'=' * 70}")

    # Test with small known matrix
    X_test = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])

    result_sum = sparse_row_operation(X_test, np.sum)
    expected_sum = np.array([3, 3, 9])
    assert np.allclose(result_sum, expected_sum), f"Sum mismatch: {result_sum} != {expected_sum}"

    result_mean = sparse_row_operation(X_test, np.mean)
    expected_mean = np.array([1.5, 3.0, 4.5])
    assert np.allclose(result_mean, expected_mean), (
        f"Mean mismatch: {result_mean} != {expected_mean}"
    )

    result_max = sparse_row_operation(X_test, np.max)
    expected_max = np.array([2, 3, 5])
    assert np.allclose(result_max, expected_max), f"Max mismatch: {result_max} != {expected_max}"

    print("✓ All correctness tests passed")

    print(f"\n{'=' * 70}")
    print("Benchmark Complete")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    results = benchmark_sparse_row_operation()
