"""Unit tests for ``sparse_row_operation``."""

import numpy as np
import pytest
import scipy.sparse as sp

from scptensor.core.jit_ops import NUMBA_AVAILABLE
from scptensor.core.sparse_utils import sparse_row_operation


def test_sparse_row_operation_sum():
    """Test sparse row operation with sum."""
    X = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_sparse_row_operation_mean():
    """Test sparse row operation with mean."""
    X = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X, np.mean)

    expected = np.array([1.5, 3.0, 4.5])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


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


def test_sparse_row_operation_empty_rows():
    """Test sparse row operation with empty rows."""
    X = sp.csr_matrix([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    result = sparse_row_operation(X, np.sum)

    expected = np.array([0.0, 6.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_sparse_row_operation_large_matrix():
    """Test sparse row operation with large matrix."""
    # Create large sparse matrix
    n_rows, n_cols = 10000, 1000
    n_nonzero = 50000

    rng = np.random.default_rng(42)
    rows = rng.integers(0, n_rows, n_nonzero)
    cols = rng.integers(0, n_cols, n_nonzero)
    data = rng.standard_normal(n_nonzero)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Test sum
    result_sum = sparse_row_operation(X, np.sum)
    assert len(result_sum) == n_rows, f"Result length mismatch: {len(result_sum)} != {n_rows}"
    assert not np.any(np.isnan(result_sum)), "Result contains NaN values"

    # Test mean
    result_mean = sparse_row_operation(X, np.mean)
    assert len(result_mean) == n_rows, f"Result length mismatch: {len(result_mean)} != {n_rows}"
    assert not np.any(np.isnan(result_mean)), "Result contains NaN values"


def test_sparse_row_operation_csr_format():
    """Test that CSR format is handled correctly."""
    # Create CSR matrix
    X_csr = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X_csr, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_sparse_row_operation_csc_format():
    """Test that CSC format is converted to CSR correctly."""
    # Create CSC matrix (should be converted to CSR internally)
    X_csc = sp.csc_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    result = sparse_row_operation(X_csc, np.sum)

    expected = np.array([3.0, 3.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_performance_comparison():
    """Compare JIT vs fallback performance."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available, skipping JIT performance comparison")

    import time

    rng = np.random.default_rng(42)

    # Create test matrix
    n_rows, n_cols = 10000, 1000
    n_nonzero = 100000

    rows = rng.integers(0, n_rows, n_nonzero)
    cols = rng.integers(0, n_cols, n_nonzero)
    data = rng.standard_normal(n_nonzero)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Benchmark sum (uses JIT)
    start = time.perf_counter()
    sparse_row_operation(X, np.sum)
    time_sum = time.perf_counter() - start

    # Benchmark max (uses fallback)
    start = time.perf_counter()
    sparse_row_operation(X, np.max)
    time_max = time.perf_counter() - start

    speedup = time_max / time_sum if time_sum > 0 else np.inf
    assert np.isfinite(speedup)
    assert speedup > 0
