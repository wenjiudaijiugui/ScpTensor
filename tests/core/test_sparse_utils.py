"""Tests for sparse utilities module."""

import numpy as np
import scipy.sparse as sp

from scptensor.core.sparse_utils import (
    ensure_sparse_format,
    get_memory_usage,
    get_sparsity_ratio,
    is_sparse_matrix,
    sparse_col_operation,
    sparse_copy,
    sparse_row_operation,
    to_sparse_if_beneficial,
)


def test_is_sparse_matrix():
    """Test sparse matrix detection."""
    dense = np.array([[1, 2], [3, 4]])
    sparse = sp.csr_matrix([[1, 0], [0, 4]])

    assert not is_sparse_matrix(dense), "Dense array should not be detected as sparse"
    assert is_sparse_matrix(sparse), "CSR matrix should be detected as sparse"


def test_get_sparsity_ratio():
    """Test sparsity ratio calculation."""
    # Test with dense array (7 zeros out of 9 elements = 7/9)
    X_test = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    expected = 7 / 9  # 7 zeros out of 9 elements
    result = get_sparsity_ratio(X_test)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    # Test with sparse matrix
    X_sparse = sp.csr_matrix(X_test)
    result_sparse = get_sparsity_ratio(X_sparse)
    assert abs(result_sparse - expected) < 1e-10, f"Expected {expected}, got {result_sparse}"


def test_to_sparse_if_beneficial():
    """Test conditional sparse conversion."""
    # Should convert (66% sparse > 50% threshold)
    X_sparse_input = np.array([[1, 0, 0], [0, 0, 2]])
    result = to_sparse_if_beneficial(X_sparse_input, threshold=0.5)
    assert is_sparse_matrix(result), "Should convert to sparse above threshold"

    # Should NOT convert (20% sparse < 50% threshold)
    X_dense_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    result2 = to_sparse_if_beneficial(X_dense_input, threshold=0.5)
    assert not is_sparse_matrix(result2), "Should NOT convert to sparse below threshold"


def test_ensure_sparse_format():
    """Test sparse format conversion."""
    # CSC to CSR
    X_csc = sp.csc_matrix([[1, 0], [0, 4]])
    X_csr = ensure_sparse_format(X_csc, format="csr")
    assert isinstance(X_csr, sp.csr_matrix), "Should convert CSC to CSR"

    # Dense to CSR
    X_dense = np.array([[1, 0], [0, 4]])
    X_csr2 = ensure_sparse_format(X_dense, format="csr")
    assert isinstance(X_csr2, sp.csr_matrix), "Should convert dense to CSR"


def test_sparse_copy():
    """Test sparse copy."""
    X = sp.csr_matrix([[1, 0], [0, 4]])
    X_copy = sparse_copy(X)

    # Verify it's a copy
    X_copy[0, 0] = 99
    assert X[0, 0] == 1, "Original should be unchanged after copy modification"


def test_get_memory_usage():
    """Test memory usage calculation."""
    # Test with sparse matrix
    X_sparse = sp.csr_matrix([[1, 0], [0, 4]])
    stats = get_memory_usage(X_sparse)
    assert stats["is_sparse"], "Should detect as sparse"
    assert stats["shape"] == (2, 2), "Shape should be (2, 2)"
    assert "nbytes" in stats, "Should include nbytes"

    # Test with dense matrix
    X_dense = np.array([[1, 2], [3, 4]])
    stats_dense = get_memory_usage(X_dense)
    assert not stats_dense["is_sparse"], "Should detect as dense"
    assert stats_dense["shape"] == (2, 2), "Shape should be (2, 2)"


def test_sparse_vs_dense_memory():
    """Compare memory usage between sparse and dense."""
    # Create a sparse matrix (90% zeros)
    n_rows, n_cols = 1000, 1000
    n_elements = n_rows * n_cols
    n_nonzero = int(0.1 * n_elements)  # 10% non-zero

    # Create sparse data
    rng = np.random.default_rng(42)
    rows = rng.integers(0, n_rows, n_nonzero)
    cols = rng.integers(0, n_cols, n_nonzero)
    data = rng.standard_normal(n_nonzero)

    X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    X_dense = X_sparse.toarray()

    stats_sparse = get_memory_usage(X_sparse)
    stats_dense = get_memory_usage(X_dense)

    compression_ratio = stats_dense["nbytes"] / stats_sparse["nbytes"]

    assert compression_ratio > 2, "Sparse should use significantly less memory"


def test_sparse_col_operation_sum_and_mean():
    """Column reductions should use explicit stored values only."""
    X = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]])

    result_sum = sparse_col_operation(X, np.sum)
    result_mean = sparse_col_operation(X, np.mean)

    np.testing.assert_allclose(result_sum, np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(result_mean, np.array([2.5, 5.0, 2.5]))


def test_sparse_col_operation_empty_columns_return_zero():
    """Empty sparse columns should follow the stable 0.0 fallback convention."""
    X = sp.csr_matrix([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    result_sum = sparse_col_operation(X, np.sum)
    result_mean = sparse_col_operation(X, np.mean)

    np.testing.assert_allclose(result_sum, np.array([4.0, 0.0, 2.0]))
    np.testing.assert_allclose(result_mean, np.array([4.0, 0.0, 2.0]))


def test_sparse_row_and_col_custom_functions_still_use_fallback_semantics():
    """Custom reducers should still see only explicit stored values."""
    X = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0], [4.0, 5.0, 0.0]])

    row_max = sparse_row_operation(X, np.max)
    col_max = sparse_col_operation(X, np.max)

    np.testing.assert_allclose(row_max, np.array([2.0, 7.0, 5.0]))
    np.testing.assert_allclose(col_max, np.array([4.0, 7.0, 2.0]))
