"""Regression tests for sparse mask handling in MatrixOps."""

import numpy as np
import scipy.sparse as sp

from scptensor.core.matrix_ops import MatrixOps
from scptensor.core.structures import MaskCode, ScpMatrix


def test_get_mask_statistics_sparse_uses_full_shape():
    """Sparse statistics should include implicit VALID entries."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    matrix = ScpMatrix(X=X)  # M is implicit all-VALID

    stats = MatrixOps.get_mask_statistics(matrix)

    assert stats["VALID"]["count"] == 4
    assert stats["VALID"]["percentage"] == 100.0
    assert stats["IMPUTED"]["count"] == 0


def test_apply_mask_to_values_sparse_zero_only_invalid_entries():
    """Sparse zeroing should not clear valid entries."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    M = sp.csr_matrix(np.array([[0, 0, 2], [0, 0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    out = MatrixOps.apply_mask_to_values(matrix, operation="zero")

    np.testing.assert_array_equal(out.X.toarray(), np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))


def test_apply_mask_to_values_sparse_nan_marks_invalid_entries():
    """Sparse NaN marking should work for invalid sparse mask entries."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    M = sp.csr_matrix(np.array([[0, 0, 2], [0, 0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    out = MatrixOps.apply_mask_to_values(matrix, operation="nan")
    out_dense = out.X.toarray()

    assert np.isnan(out_dense[0, 2])
    np.testing.assert_array_equal(
        np.nan_to_num(out_dense, nan=-1.0),
        np.array([[1.0, 0.0, -1.0], [0.0, 3.0, 0.0]]),
    )


def test_filter_by_mask_sparse_when_valid_not_kept_filters_implicit_valids():
    """If VALID is not kept, implicit sparse VALID entries must be filtered too."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    M = sp.csr_matrix(np.array([[0, 5], [0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    out = MatrixOps.filter_by_mask(matrix, [MaskCode.IMPUTED])

    np.testing.assert_array_equal(out.M.toarray(), np.array([[3, 5], [3, 3]], dtype=np.int8))
    assert np.isnan(out.X[0, 0])
    assert out.X[0, 1] == 2.0
    assert np.isnan(out.X[1, 0])
    assert np.isnan(out.X[1, 1])
