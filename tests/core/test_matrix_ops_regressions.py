"""Regression tests for sparse mask handling in MatrixOps."""

import warnings

import numpy as np
import scipy.sparse as sp

from scptensor.core.matrix_ops import MatrixOps
from scptensor.core.structures import MaskCode, MatrixMetadata, ScpMatrix


def test_get_mask_statistics_sparse_uses_full_shape():
    """Sparse statistics should include implicit VALID entries."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    matrix = ScpMatrix(X=X)  # M is implicit all-VALID

    stats = MatrixOps.get_mask_statistics(matrix)

    assert stats["VALID"]["count"] == 4
    assert stats["VALID"]["percentage"] == 100.0
    assert stats["IMPUTED"]["count"] == 0


def test_get_mask_statistics_dense_counts_each_code_once() -> None:
    """Dense statistics should count all codes with full-matrix percentages."""
    X = np.arange(12, dtype=float).reshape(3, 4)
    M = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 0],
            [0, 0, 5, 2],
        ],
        dtype=np.int8,
    )
    matrix = ScpMatrix(X=X, M=M)

    stats = MatrixOps.get_mask_statistics(matrix)

    assert stats["VALID"]["count"] == 4
    assert stats["LOD"]["count"] == 2
    assert stats["IMPUTED"]["count"] == 2
    assert stats["UNCERTAIN"]["count"] == 1
    assert stats["VALID"]["percentage"] == (4 / 12) * 100.0


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


def test_apply_mask_to_values_sparse_nan_creates_explicit_nan_for_structural_zero():
    """Invalid sparse structural zeros should still become explicit NaN entries."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    M = sp.csr_matrix(np.array([[0, 2], [0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    out = MatrixOps.apply_mask_to_values(matrix, operation="nan")
    out_dense = out.X.toarray()

    assert np.isnan(out_dense[0, 1])
    np.testing.assert_array_equal(
        np.nan_to_num(out_dense, nan=-1.0),
        np.array([[1.0, -1.0], [0.0, 2.0]]),
    )


def test_mark_values_sparse_creates_explicit_mask_entry_for_structural_zero():
    """Sparse mask writes must materialize previously implicit VALID coordinates."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    matrix = ScpMatrix(X=X)

    out = MatrixOps.mark_values(
        matrix,
        (np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)),
        MaskCode.IMPUTED,
    )

    assert matrix.M is None
    assert sp.issparse(out.M)
    np.testing.assert_array_equal(out.M.toarray(), np.array([[0, 5], [0, 0]], dtype=np.int8))


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


def test_filter_by_mask_sparse_when_valid_not_kept_does_not_mutate_input():
    """The densify path must still return a detached matrix and deep-copy metadata."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [3.0, 4.0]]))
    M = sp.csr_matrix(np.array([[0, 5], [0, 0]], dtype=np.int8))
    matrix = ScpMatrix(
        X=X,
        M=M,
        metadata=MatrixMetadata(creation_info={"nested": {"state": "original"}}),
    )
    x_before = matrix.X.copy()
    m_before = matrix.M.copy()

    out = MatrixOps.filter_by_mask(matrix, [MaskCode.IMPUTED])

    assert sp.issparse(matrix.X)
    assert sp.issparse(matrix.M)
    np.testing.assert_array_equal(matrix.X.toarray(), x_before.toarray())
    np.testing.assert_array_equal(matrix.M.toarray(), m_before.toarray())
    assert out.metadata is not None
    assert matrix.metadata is not None
    assert out.metadata is not matrix.metadata
    out.metadata.creation_info["nested"]["state"] = "changed"
    assert matrix.metadata.creation_info["nested"]["state"] == "original"


def test_filter_by_mask_sparse_valid_path_marks_structural_zero_as_nan():
    """Sparse filter writes must materialize NaN at invalid structural-zero coordinates."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    M = sp.csr_matrix(np.array([[0, 4], [0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    out = MatrixOps.filter_by_mask(matrix, [MaskCode.VALID])
    out_dense = out.X.toarray()

    assert np.isnan(out_dense[0, 1])
    np.testing.assert_array_equal(
        np.nan_to_num(out_dense, nan=-1.0),
        np.array([[1.0, -1.0], [0.0, 2.0]]),
    )
    np.testing.assert_array_equal(out.M.toarray(), np.array([[0, 3], [0, 0]], dtype=np.int8))


def test_sparse_mask_queries_preserve_boolean_semantics_without_warning():
    """Sparse mask query helpers should avoid sparse comparison warnings."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    M = sp.csr_matrix(np.array([[0, 2], [0, 0]], dtype=np.int8))
    matrix = ScpMatrix(X=X, M=M)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        valid = MatrixOps.get_valid_mask(matrix)
        missing = MatrixOps.get_missing_mask(matrix)
        lod = MatrixOps.get_missing_type_mask(matrix, MaskCode.LOD)

    assert sp.issparse(valid)
    assert sp.issparse(missing)
    assert sp.issparse(lod)
    np.testing.assert_array_equal(valid.toarray(), np.array([[True, False], [True, True]]))
    np.testing.assert_array_equal(missing.toarray(), np.array([[False, True], [False, False]]))
    np.testing.assert_array_equal(lod.toarray(), np.array([[False, True], [False, False]]))
    assert not any(issubclass(w.category, sp.SparseEfficiencyWarning) for w in caught)
