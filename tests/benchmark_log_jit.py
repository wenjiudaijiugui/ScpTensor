"""
Benchmark script for JIT-accelerated log normalization.

This script compares the performance of:
1. Original NumPy-only implementation
2. JIT-accelerated implementation (for large matrices)
3. Combined log+scale operation vs separate operations
"""

import os
import time
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple

# Set seed for reproducibility
np.random.seed(42)


def sparse_log_original(X: sp.csr_matrix, offset: float = 1.0, scale: float = 1.0) -> sp.csr_matrix:
    """Original implementation - separate log and scale operations."""
    result = X.copy()
    if result.data.dtype.kind != 'f':
        result.data = result.data.astype(float)
    result.data = np.log1p(result.data + offset - 1.0)
    result.data /= scale
    return result


def sparse_log_combined_numpy(X: sp.csr_matrix, offset: float = 1.0, scale: float = 1.0) -> sp.csr_matrix:
    """Combined operations using NumPy (still separate internally)."""
    result = X.copy()
    if result.data.dtype.kind != 'f':
        result.data = result.data.astype(float)
    # Single operation but still two passes over data
    result.data = np.log1p(result.data + offset - 1.0) / scale
    return result


def sparse_log_jit(X: sp.csr_matrix, offset: float = 1.0, scale: float = 1.0) -> sp.csr_matrix:
    """JIT-accelerated combined operation (from sparse_utils)."""
    from scptensor.core.sparse_utils import sparse_safe_log1p_with_scale
    return sparse_safe_log1p_with_scale(X, offset=offset, scale=scale, use_jit=True)


def sparse_log_no_jit(X: sp.csr_matrix, offset: float = 1.0, scale: float = 1.0) -> sp.csr_matrix:
    """Explicitly disable JIT."""
    from scptensor.core.sparse_utils import sparse_safe_log1p_with_scale
    return sparse_safe_log1p_with_scale(X, offset=offset, scale=scale, use_jit=False)


def benchmark_function(
    func: callable,
    X: sp.csr_matrix,
    offset: float,
    scale: float,
    n_repeats: int = 10,
    warmup: int = 2
) -> Dict[str, float]:
    """Benchmark a function and return timing statistics."""
    # Warmup runs
    for _ in range(warmup):
        _ = func(X, offset, scale)

    # Timed runs
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = func(X, offset, scale)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def generate_sparse_matrix(n_rows: int, n_cols: int, sparsity: float) -> sp.csr_matrix:
    """Generate a random sparse matrix with given sparsity."""
    nnz = int(n_rows * n_cols * (1 - sparsity))
    data = np.random.uniform(0.1, 100.0, nnz)
    rows = np.random.randint(0, n_rows, nnz)
    cols = np.random.randint(0, n_cols, nnz)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def run_benchmark_suite() -> None:
    """Run comprehensive benchmark suite."""
    print("=" * 80)
    print("JIT-Accelerated Log Normalization Benchmark")
    print("=" * 80)
    print()

    # Test configurations
    test_configs = [
        # (n_rows, n_cols, sparsity, description)
        (1000, 1000, 0.90, "Small (1M cells, 90% sparse)"),
        (5000, 1000, 0.90, "Medium (5M cells, 90% sparse)"),
        (10000, 1000, 0.90, "Large (10M cells, 90% sparse)"),
        (10000, 2000, 0.95, "Large High-Sparse (20M cells, 95% sparse)"),
    ]

    offset = 1.0
    scale = np.log(2.0)  # log2 normalization

    functions = [
        ('Original (separate ops)', sparse_log_original),
        ('Combined NumPy', sparse_log_combined_numpy),
        ('No JIT (use_jit=False)', sparse_log_no_jit),
        ('JIT (use_jit=True)', sparse_log_jit),
    ]

    results = []

    for n_rows, n_cols, sparsity, desc in test_configs:
        print(f"\nTest: {desc}")
        print(f"Shape: ({n_rows}, {n_cols}), Sparsity: {sparsity:.1%}")

        X = generate_sparse_matrix(n_rows, n_cols, sparsity)
        nnz = X.nnz
        print(f"Non-zero elements: {nnz:,}")

        # Determine JIT threshold
        jit_threshold = int(os.getenv('SCPTENSOR_JIT_THRESHOLD', '50000'))
        will_use_jit = nnz > jit_threshold
        print(f"JIT threshold: {jit_threshold:,}, Will use JIT: {will_use_jit}")

        # Verify results are identical
        result_ref = functions[0][1](X, offset, scale)
        for name, func in functions[1:]:
            result = func(X, offset, scale)
            diff = np.abs(result.data - result_ref.data).max()
            assert diff < 1e-10, f"Results differ for {name}: max diff = {diff}"
        print("All implementations produce identical results")

        # Benchmark each function
        print(f"\n{'Implementation':<30} {'Mean (ms)':<12} {'Speedup':<10}")
        print("-" * 55)

        baseline_time = None
        for name, func in functions:
            stats = benchmark_function(func, X, offset, scale, n_repeats=10)
            mean_ms = stats['mean'] * 1000

            if baseline_time is None:
                baseline_time = stats['mean']
                speedup = 1.0
            else:
                speedup = baseline_time / stats['mean']

            print(f"{name:<30} {mean_ms:<12.3f} {speedup:<10.2f}x")
            results.append({
                'config': desc,
                'nnz': nnz,
                'function': name,
                'mean_ms': mean_ms,
                'speedup': speedup,
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate speedups for large matrices (where JIT is most effective)
    large_results = [r for r in results if r['nnz'] > 50000]
    if large_results:
        jit_result = [r for r in large_results if 'JIT (use_jit=True)' in r['function']]
        original_result = [r for r in large_results if 'Original' in r['function']]

        if jit_result and original_result:
            avg_jit_time = np.mean([r['mean_ms'] for r in jit_result])
            avg_original_time = np.mean([r['mean_ms'] for r in original_result])
            avg_speedup = avg_original_time / avg_jit_time

            print(f"\nAverage speedup for large matrices (JIT vs Original): {avg_speedup:.2f}x")
            print(f"Average time (Original): {avg_original_time:.3f} ms")
            print(f"Average time (JIT): {avg_jit_time:.3f} ms")

    print("\nNote: First run includes JIT compilation overhead.")
    print("Subsequent runs benefit from cached compilation.")


if __name__ == "__main__":
    run_benchmark_suite()
