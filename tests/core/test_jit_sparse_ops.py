"""Tests for JIT-accelerated sparse matrix operations.

This module tests the JIT kernels for sparse matrix row operations,
including basic correctness tests and performance benchmarks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import sparse

from scptensor.core.jit_ops import (
    NUMBA_AVAILABLE,
    _sparse_row_mean_jit,
    _sparse_row_sum_jit,
)
from scptensor.core.sparse_utils import sparse_row_operation


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_sum_jit_basic() -> None:
    """Test basic row sum with JIT.

    Creates a simple CSR matrix and verifies that the JIT kernel
    correctly computes row sums.
    """
    # Create simple CSR matrix: [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    n_rows = 3

    result = _sparse_row_sum_jit(indptr, data, n_rows)

    # Row 0: 1+2=3, Row 1: 3=3, Row 2: 4+5=9
    expected = np.array([3.0, 3.0, 9.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_mean_jit_basic() -> None:
    """Test basic row mean with JIT.

    Creates a simple CSR matrix and verifies that the JIT kernel
    correctly computes row means, including handling of empty rows.
    """
    # Create simple CSR matrix: [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    n_rows = 3

    result = _sparse_row_mean_jit(indptr, data, n_rows)

    # Row 0: (1+2)/2=1.5, Row 1: 3/1=3, Row 2: (4+5)/2=4.5
    expected = np.array([1.5, 3.0, 4.5])
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_sum_jit_with_empty_rows() -> None:
    """Test JIT kernel handles empty rows correctly.

    Verifies that rows with no non-zero values are handled properly
    (sum should be 0.0).
    """
    # Matrix with an empty row: [[1, 2], [0, 0], [3, 4]]
    indptr = np.array([0, 2, 2, 4], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    n_rows = 3

    result = _sparse_row_sum_jit(indptr, data, n_rows)

    # Row 0: 1+2=3, Row 1: 0 (empty), Row 2: 3+4=7
    expected = np.array([3.0, 0.0, 7.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_mean_jit_with_empty_rows() -> None:
    """Test JIT kernel handles empty rows correctly for mean.

    Verifies that rows with no non-zero values return 0.0 as the mean.
    """
    # Matrix with an empty row: [[1, 2], [0, 0], [3, 4]]
    indptr = np.array([0, 2, 2, 4], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    n_rows = 3

    result = _sparse_row_mean_jit(indptr, data, n_rows)

    # Row 0: (1+2)/2=1.5, Row 1: 0 (empty), Row 2: (3+4)/2=3.5
    expected = np.array([1.5, 0.0, 3.5])
    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_sum_jit_large() -> None:
    """Test JIT with larger sparse matrix.

    Compares JIT results with scipy's built-in sum operation
    to ensure correctness on a larger random sparse matrix.
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = 1000, 500

    # Create random sparse matrix (90% sparse)
    X_dense = rng.random((n_rows, n_cols))
    X_dense[X_dense < 0.9] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    # JIT result
    result_jit = _sparse_row_sum_jit(X_sparse.indptr, X_sparse.data, n_rows)

    # NumPy result for comparison
    result_np = X_sparse.sum(axis=1).A1

    np.testing.assert_allclose(result_jit, result_np, rtol=1e-10)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_sparse_row_mean_jit_large() -> None:
    """Test JIT mean with larger sparse matrix.

    Compares JIT results with manual mean computation
    to ensure correctness on a larger random sparse matrix.
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = 1000, 500

    # Create random sparse matrix (90% sparse)
    X_dense = rng.random((n_rows, n_cols))
    X_dense[X_dense < 0.9] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    # JIT result
    result_jit = _sparse_row_mean_jit(X_sparse.indptr, X_sparse.data, n_rows)

    # Compute expected manually (accounting for sparse structure)
    expected = np.zeros(n_rows)
    for i in range(n_rows):
        start, end = X_sparse.indptr[i], X_sparse.indptr[i + 1]
        if end > start:
            expected[i] = np.mean(X_sparse.data[start:end])
        else:
            expected[i] = 0.0

    np.testing.assert_allclose(result_jit, expected, rtol=1e-10)


def test_sparse_row_operation_sum() -> None:
    """Test sparse_row_operation integration for sum.

    Verifies that the high-level sparse_row_operation function
    correctly uses JIT acceleration for np.sum.
    """
    X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])

    result_sum = sparse_row_operation(X, np.sum)
    expected_sum = np.array([3.0, 3.0, 9.0])

    np.testing.assert_allclose(result_sum, expected_sum)


def test_sparse_row_operation_mean() -> None:
    """Test sparse_row_operation integration for mean.

    Verifies that the high-level sparse_row_operation function
    correctly uses JIT acceleration for np.mean.
    """
    X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])

    result_mean = sparse_row_operation(X, np.mean)
    expected_mean = np.array([1.5, 3.0, 4.5])

    np.testing.assert_allclose(result_mean, expected_mean)


def test_sparse_row_operation_custom_function() -> None:
    """Test sparse_row_operation with custom function.

    Verifies that the fallback path works correctly for
    custom functions that don't have JIT kernels.
    """
    X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])

    # Custom function: compute max
    result_max = sparse_row_operation(X, np.max)
    expected_max = np.array([2.0, 3.0, 5.0])

    np.testing.assert_allclose(result_max, expected_max)


@pytest.mark.benchmark
@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_jit_performance_sum() -> None:
    """Benchmark JIT vs pure Python performance for sum.

    Measures the speedup of JIT kernel over pure Python loop.
    Requires pytest with benchmark marker.
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = 5000, 2000
    X_dense = rng.random((n_rows, n_cols))
    X_dense[X_dense < 0.85] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    # JIT version
    start = time.time()
    result_jit = _sparse_row_sum_jit(X_sparse.indptr, X_sparse.data, n_rows)
    jit_time = time.time() - start

    # Pure Python version
    start = time.time()
    result_python = np.array([np.sum(X_sparse[i].toarray()) for i in range(n_rows)])
    python_time = time.time() - start

    # Verify correctness
    np.testing.assert_allclose(result_jit, result_python.flatten(), rtol=1e-10)

    # JIT should be faster
    speedup = python_time / jit_time
    print(f"\nJIT time: {jit_time:.4f}s, Python time: {python_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    assert jit_time < python_time, "JIT should be faster than Python"
    assert speedup >= 5.0, f"Expected at least 5x speedup, got {speedup:.2f}x"


@pytest.mark.benchmark
@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_jit_performance_mean() -> None:
    """Benchmark JIT vs pure Python performance for mean.

    Measures the speedup of JIT kernel over pure Python loop.
    Requires pytest with benchmark marker.
    """
    rng = np.random.default_rng(42)
    n_rows, n_cols = 5000, 2000
    X_dense = rng.random((n_rows, n_cols))
    X_dense[X_dense < 0.85] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    # JIT version
    start = time.time()
    result_jit = _sparse_row_mean_jit(X_sparse.indptr, X_sparse.data, n_rows)
    jit_time = time.time() - start

    # Pure Python version (mean of non-zero values only, to match JIT)
    start = time.time()
    result_python = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        start_idx, end_idx = X_sparse.indptr[i], X_sparse.indptr[i + 1]
        if end_idx > start_idx:
            result_python[i] = np.mean(X_sparse.data[start_idx:end_idx])
        else:
            result_python[i] = 0.0
    python_time = time.time() - start

    # Verify correctness
    np.testing.assert_allclose(result_jit, result_python, rtol=1e-10)

    # JIT should be faster
    speedup = python_time / jit_time
    print(f"\nJIT time: {jit_time:.4f}s, Python time: {python_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    assert jit_time < python_time, "JIT should be faster than Python"
    # Mean operation is simpler, so we expect at least 1.2x speedup
    assert speedup >= 1.2, f"Expected at least 1.2x speedup, got {speedup:.2f}x"


def test_fallback_implementation() -> None:
    """Test fallback implementation when Numba is not available.

    This test always runs, but only tests the fallback path
    when Numba is not installed.
    """
    if not NUMBA_AVAILABLE:
        # Simple test to ensure fallback works
        indptr = np.array([0, 2, 3, 5], dtype=np.int64)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        n_rows = 3

        result_sum = _sparse_row_sum_jit(indptr, data, n_rows)
        result_mean = _sparse_row_mean_jit(indptr, data, n_rows)

        np.testing.assert_allclose(result_sum, [3.0, 3.0, 9.0])
        np.testing.assert_allclose(result_mean, [1.5, 3.0, 4.5])
    else:
        # When Numba is available, we skip this test
        pytest.skip("Numba is available, fallback not tested")
