#!/usr/bin/env python3
"""
Test matrix_ops sparse optimizations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scipy.sparse as sp
from scptensor.core.structures import ScpMatrix, MaskCode
from scptensor.core.matrix_ops import MatrixOps


def create_test_matrix(sparse=True, missing_rate=0.5):
    """Create a test matrix with some missing values."""
    n_rows, n_cols = 100, 50

    if sparse:
        # Create sparse matrix with specified missing rate
        n_elements = n_rows * n_cols
        n_nonzero = int((1 - missing_rate) * n_elements)

        rows = np.random.randint(0, n_rows, n_nonzero)
        cols = np.random.randint(0, n_cols, n_nonzero)
        data = np.random.randn(n_nonzero) * 10 + 20

        X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        M = sp.csr_matrix(X != 0, dtype=np.int8)  # Mark non-zeros as VALID
    else:
        X = np.random.randn(n_rows, n_cols) * 10 + 20
        M = np.ones((n_rows, n_cols), dtype=np.int8) * MaskCode.VALID.value

    return ScpMatrix(X=X, M=M)


def test_mark_values_sparse():
    """Test mark_values with sparse matrix."""
    matrix = create_test_matrix(sparse=True, missing_rate=0.7)

    # Mark some random positions
    rows = np.array([0, 5, 10])
    cols = np.array([0, 5, 10])

    result = MatrixOps.mark_values(matrix, (rows, cols), MaskCode.IMPUTED)

    # Verify sparsity preserved
    assert sp.issparse(result.M), "Result should preserve sparsity"

    # Verify values marked correctly
    M_dense = result.M.toarray()
    assert M_dense[0, 0] == MaskCode.IMPUTED.value
    assert M_dense[5, 5] == MaskCode.IMPUTED.value
    assert M_dense[10, 10] == MaskCode.IMPUTED.value

    print("✓ mark_values sparse tests passed")


def test_filter_by_mask_sparse():
    """Test filter_by_mask with sparse matrix."""
    matrix = create_test_matrix(sparse=True, missing_rate=0.6)

    result = MatrixOps.filter_by_mask(matrix, [MaskCode.VALID])

    # Verify sparsity preserved
    assert sp.issparse(result.M), "Mask should preserve sparsity"

    # Verify all entries are either VALID or FILTERED
    M_unique = np.unique(result.M.data)
    valid_values = {MaskCode.VALID.value, MaskCode.FILTERED.value}
    assert all(v in valid_values for v in M_unique), "All values should be VALID or FILTERED"

    print("✓ filter_by_mask sparse tests passed")


def test_apply_mask_to_values_sparse():
    """Test apply_mask_to_values with sparse matrix."""
    matrix = create_test_matrix(sparse=True, missing_rate=0.6)

    # Test 'zero' operation
    result_zero = MatrixOps.apply_mask_to_values(matrix, operation='zero')
    assert sp.issparse(result_zero.X), "Result should preserve sparsity for 'zero' operation"

    # Verify invalid values are zero
    X_dense = result_zero.X.toarray()
    M_dense = result_zero.M.toarray()
    invalid_mask = M_dense != MaskCode.VALID.value
    assert np.all(X_dense[invalid_mask] == 0), "Invalid values should be zero"

    print("✓ apply_mask_to_values sparse tests passed")


def test_no_densification():
    """Verify that sparse operations don't unnecessarily densify."""
    matrix = create_test_matrix(sparse=True, missing_rate=0.8)

    # Track memory before
    from scptensor.core.sparse_utils import get_memory_usage
    mem_before = get_memory_usage(matrix.X)

    # Apply operations
    result = MatrixOps.mark_values(matrix, (np.array([0]), np.array([0])), MaskCode.IMPUTED)
    mem_after = get_memory_usage(result.X)

    # Memory usage should be similar (within 20% tolerance)
    ratio = mem_after['nbytes'] / mem_before['nbytes']
    assert ratio < 1.2, f"Memory usage increased too much: {ratio:.2f}x (indicates densification)"

    print(f"✓ No densification test passed (memory ratio: {ratio:.2f}x)")


def test_performance_comparison():
    """Compare performance of sparse vs dense operations."""
    import time

    # Create matrices
    n_rows, n_cols = 1000, 500
    missing_rate = 0.8

    # Sparse matrix
    n_nonzero = int((1 - missing_rate) * n_rows * n_cols)
    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero) * 10 + 20

    X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    M_sparse = sp.csr_matrix(X_sparse != 0, dtype=np.int8)
    matrix_sparse = ScpMatrix(X=X_sparse, M=M_sparse)

    # Dense matrix
    X_dense = X_sparse.toarray()
    M_dense = M_sparse.toarray()
    matrix_dense = ScpMatrix(X=X_dense, M=M_dense)

    # Test mark_values performance
    rows_idx = np.array([0, 10, 20])
    cols_idx = np.array([0, 10, 20])

    start = time.time()
    for _ in range(100):
        _ = MatrixOps.mark_values(matrix_sparse, (rows_idx, cols_idx), MaskCode.IMPUTED)
    time_sparse = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = MatrixOps.mark_values(matrix_dense, (rows_idx, cols_idx), MaskCode.IMPUTED)
    time_dense = time.time() - start

    print(f"\nPerformance comparison (1000x500 matrix, 80% sparse):")
    print(f"  Sparse operations: {time_sparse*1000:.2f} ms")
    print(f"  Dense operations:  {time_dense*1000:.2f} ms")
    print(f"  Speedup: {time_dense/time_sparse:.2f}x")

    print("✓ Performance comparison completed")


if __name__ == "__main__":
    print("Testing matrix_ops sparse optimizations...\n")

    test_mark_values_sparse()
    test_filter_by_mask_sparse()
    test_apply_mask_to_values_sparse()
    test_no_densification()
    test_performance_comparison()

    print("\n" + "="*60)
    print("✅ All matrix_ops sparse optimization tests passed!")
    print("="*60)
