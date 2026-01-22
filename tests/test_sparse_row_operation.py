#!/usr/bin/env python3
"""
Unit tests for optimized sparse_row_operation function.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scipy.sparse as sp

from scptensor.core.jit_ops import NUMBA_AVAILABLE
from scptensor.core.sparse_utils import sparse_row_operation


def test_sparse_row_operation_sum():
    """Test sparse row operation with sum."""
    X = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ sparse_row_operation sum test passed")


def test_sparse_row_operation_mean():
    """Test sparse row operation with mean."""
    X = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X, np.mean)

    expected = np.array([1.5, 3.0, 4.5])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ sparse_row_operation mean test passed")


def test_sparse_row_operation_custom_func():
    """Test sparse row operation with custom function."""
    X = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])

    # Test with max
    result_max = sparse_row_operation(X, np.max)
    expected_max = np.array([2.0, 3.0, 5.0])
    assert np.allclose(result_max, expected_max), f"Expected {expected_max}, got {result_max}"

    # Test with min
    result_min = sparse_row_operation(X, np.min)
    expected_min = np.array([1.0, 3.0, 4.0])
    assert np.allclose(result_min, expected_min), f"Expected {expected_min}, got {result_min}"

    # Test with custom lambda
    result_custom = sparse_row_operation(X, lambda x: np.std(x) if len(x) > 0 else 0.0)
    assert len(result_custom) == 3, f"Result length mismatch: {len(result_custom)} != 3"

    print("✓ sparse_row_operation custom function test passed")


def test_sparse_row_operation_empty_rows():
    """Test sparse row operation with empty rows."""
    X = sp.csr_matrix([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    result = sparse_row_operation(X, np.sum)

    expected = np.array([0.0, 6.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ sparse_row_operation empty rows test passed")


def test_sparse_row_operation_large_matrix():
    """Test sparse row operation with large matrix."""
    # Create large sparse matrix
    n_rows, n_cols = 10000, 1000
    n_nonzero = 50000

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Test sum
    result_sum = sparse_row_operation(X, np.sum)
    assert len(result_sum) == n_rows, f"Result length mismatch: {len(result_sum)} != {n_rows}"
    assert not np.any(np.isnan(result_sum)), "Result contains NaN values"

    # Test mean
    result_mean = sparse_row_operation(X, np.mean)
    assert len(result_mean) == n_rows, f"Result length mismatch: {len(result_mean)} != {n_rows}"
    assert not np.any(np.isnan(result_mean)), "Result contains NaN values"

    print("✓ sparse_row_operation large matrix test passed")


def test_sparse_row_operation_csr_format():
    """Test that CSR format is handled correctly."""
    # Create CSR matrix
    X_csr = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X_csr, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ sparse_row_operation CSR format test passed")


def test_sparse_row_operation_csc_format():
    """Test that CSC format is converted to CSR correctly."""
    # Create CSC matrix (should be converted to CSR internally)
    X_csc = sp.csc_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X_csc, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ sparse_row_operation CSC format test passed")


def test_performance_comparison():
    """Compare JIT vs fallback performance."""
    if not NUMBA_AVAILABLE:
        print("⚠ Numba not available, skipping performance comparison")
        return

    import time

    # Create test matrix
    n_rows, n_cols = 10000, 1000
    n_nonzero = 100000

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Benchmark sum (uses JIT)
    start = time.perf_counter()
    result_sum = sparse_row_operation(X, np.sum)
    time_sum = time.perf_counter() - start

    # Benchmark max (uses fallback)
    start = time.perf_counter()
    result_max = sparse_row_operation(X, np.max)
    time_max = time.perf_counter() - start

    speedup = time_max / time_sum

    print("\nPerformance Comparison (10K×1K matrix):")
    print(f"  Sum (JIT):      {time_sum * 1000:.2f} ms")
    print(f"  Max (fallback): {time_max * 1000:.2f} ms")
    print(f"  Speedup:        {speedup:.1f}x")

    # JIT should be significantly faster
    assert speedup > 2.0, f"JIT should be at least 2x faster, got {speedup:.1f}x"
    print("✓ Performance comparison test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Optimized sparse_row_operation")
    print("=" * 70)
    print(f"\nNumba Available: {NUMBA_AVAILABLE}\n")

    test_sparse_row_operation_sum()
    test_sparse_row_operation_mean()
    test_sparse_row_operation_custom_func()
    test_sparse_row_operation_empty_rows()
    test_sparse_row_operation_large_matrix()
    test_sparse_row_operation_csr_format()
    test_sparse_row_operation_csc_format()
    test_performance_comparison()

    print("\n" + "=" * 70)
    print("✅ All sparse_row_operation tests passed!")
    print("=" * 70)
