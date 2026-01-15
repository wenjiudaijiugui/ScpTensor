"""
Edge case tests for ScpMatrix.

This module tests edge cases, boundary conditions, and error handling for ScpMatrix.
"""

import numpy as np
import pytest
from scipy import sparse

from scptensor.core import MaskCode, MatrixMetadata, ScpMatrix


class TestScpMatrixEdgeCases:
    """Test edge cases for ScpMatrix."""

    def test_empty_matrix(self):
        """Test ScpMatrix with empty (0x0) matrix."""
        X = np.array([]).reshape(0, 0)
        matrix = ScpMatrix(X=X)
        assert matrix.X.shape == (0, 0)
        assert matrix.get_m().shape == (0, 0)

    def test_single_element_matrix(self):
        """Test ScpMatrix with single element (1x1)."""
        X = np.array([[42.0]])
        M = np.array([[MaskCode.VALID]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.X.shape == (1, 1)
        assert matrix.X[0, 0] == 42.0
        assert matrix.M[0, 0] == MaskCode.VALID

    def test_single_row_matrix(self):
        """Test ScpMatrix with single row (1 sample, many features)."""
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        matrix = ScpMatrix(X=X)
        assert matrix.X.shape == (1, 5)
        assert matrix.get_m().shape == (1, 5)

    def test_single_column_matrix(self):
        """Test ScpMatrix with single column (many samples, 1 feature)."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        matrix = ScpMatrix(X=X)
        assert matrix.X.shape == (5, 1)
        assert matrix.get_m().shape == (5, 1)

    def test_very_large_matrix(self):
        """Test ScpMatrix with large dimensions."""
        # 1000 samples x 5000 features (but smaller for test speed)
        X = np.random.rand(100, 500)
        M = np.zeros((100, 500), dtype=np.int8)
        M[np.random.rand(100, 500) < 0.1] = MaskCode.MBR  # 10% missing
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.X.shape == (100, 500)
        assert np.sum(matrix.M == MaskCode.MBR) > 0

    def test_matrix_with_nan_values(self):
        """Test ScpMatrix with NaN values in X."""
        X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        matrix = ScpMatrix(X=X)
        assert np.isnan(matrix.X[0, 1])
        assert np.isnan(matrix.X[1, 2])

    def test_matrix_with_inf_values(self):
        """Test ScpMatrix with inf values in X."""
        X = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, -np.inf]])
        matrix = ScpMatrix(X=X)
        assert np.isinf(matrix.X[0, 1])
        assert np.isinf(matrix.X[1, 2])

    def test_matrix_with_all_zeros(self):
        """Test ScpMatrix with all zero values."""
        X = np.zeros((5, 5))
        matrix = ScpMatrix(X=X)
        assert np.all(matrix.X == 0)

    def test_matrix_with_negative_values(self):
        """Test ScpMatrix with negative values."""
        X = np.array([[-1.0, -2.0, -3.0]])
        matrix = ScpMatrix(X=X)
        assert matrix.X[0, 0] == -1.0
        assert matrix.X[0, 1] == -2.0

    def test_matrix_with_mixed_positive_negative(self):
        """Test ScpMatrix with mixed positive and negative values."""
        X = np.array([[1.0, -1.0, 2.0, -2.0, 0.0]])
        matrix = ScpMatrix(X=X)
        np.testing.assert_array_equal(matrix.X, X)


class TestScpMatrixTypeConversions:
    """Test type conversions in ScpMatrix."""

    def test_integer_matrix_converted_to_float(self):
        """Test that integer matrix is converted to float."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        matrix = ScpMatrix(X=X)
        assert np.issubdtype(matrix.X.dtype, np.floating)
        assert matrix.X.dtype == np.float64

    def test_bool_matrix_converted_to_float(self):
        """Test that boolean matrix is converted to float."""
        X = np.array([[True, False], [False, True]])
        matrix = ScpMatrix(X=X)
        assert np.issubdtype(matrix.X.dtype, np.floating)

    def test_uint_matrix_converted_to_float(self):
        """Test that unsigned integer matrix is converted to float."""
        X = np.array([[1, 2, 3]], dtype=np.uint8)
        matrix = ScpMatrix(X=X)
        assert np.issubdtype(matrix.X.dtype, np.floating)


class TestScpMatrixShapeMismatch:
    """Test shape mismatch validation."""

    def test_mask_shape_mismatch_raises_error(self):
        """Test that mismatched X and M shapes raise ValueError."""
        X = np.random.rand(5, 3)
        M = np.zeros((3, 5), dtype=np.int8)  # Wrong shape
        with pytest.raises(ValueError, match="Shape mismatch"):
            ScpMatrix(X=X, M=M)

    def test_mask_shape_mismatch_different_rows(self):
        """Test shape mismatch with different number of rows."""
        X = np.random.rand(5, 3)
        M = np.zeros((4, 3), dtype=np.int8)  # Wrong rows
        with pytest.raises(ValueError, match="Shape mismatch"):
            ScpMatrix(X=X, M=M)

    def test_mask_shape_mismatch_different_cols(self):
        """Test shape mismatch with different number of columns."""
        X = np.random.rand(5, 3)
        M = np.zeros((5, 2), dtype=np.int8)  # Wrong columns
        with pytest.raises(ValueError, match="Shape mismatch"):
            ScpMatrix(X=X, M=M)


class TestScpMatrixSparseEdgeCases:
    """Test sparse matrix edge cases."""

    def test_sparse_matrix_all_zeros(self):
        """Test sparse matrix with all zeros."""
        X = sparse.csr_matrix((5, 5))
        matrix = ScpMatrix(X=X)
        assert sparse.issparse(matrix.X)
        assert matrix.X.nnz == 0

    def test_sparse_matrix_single_nonzero(self):
        """Test sparse matrix with single non-zero value."""
        X = sparse.csr_matrix((5, 5))
        X[0, 0] = 42.0
        matrix = ScpMatrix(X=X)
        assert matrix.X.nnz == 1
        assert matrix.X[0, 0] == 42.0

    def test_sparse_matrix_with_dense_mask(self):
        """Test sparse X with dense M."""
        X = sparse.csr_matrix(np.random.rand(5, 5))
        M = np.zeros((5, 5), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        assert sparse.issparse(matrix.X)
        assert isinstance(matrix.M, np.ndarray)

    def test_sparse_matrix_with_sparse_mask(self):
        """Test sparse X with sparse M."""
        X = sparse.csr_matrix(np.random.rand(5, 5))
        M = sparse.csr_matrix(np.zeros((5, 5), dtype=np.int8))
        matrix = ScpMatrix(X=X, M=M)
        assert sparse.issparse(matrix.X)
        assert sparse.issparse(matrix.M)

    def test_csr_matrix_format_preserved(self):
        """Test that CSR format is preserved."""
        X = sparse.csr_matrix(np.random.rand(5, 5))
        matrix = ScpMatrix(X=X)
        assert sparse.issparse(matrix.X)
        assert isinstance(matrix.X, sparse.csr_matrix)

    def test_csc_matrix_format(self):
        """Test that CSC format works."""
        X = sparse.csc_matrix(np.random.rand(5, 5))
        matrix = ScpMatrix(X=X)
        assert sparse.issparse(matrix.X)


class TestScpMatrixMetadata:
    """Test MatrixMetadata functionality."""

    def test_matrix_with_confidence_scores(self):
        """Test matrix with confidence scores metadata."""
        X = np.random.rand(3, 3)
        confidence = np.random.rand(3, 3)
        metadata = MatrixMetadata(confidence_scores=confidence)
        matrix = ScpMatrix(X=X, metadata=metadata)
        assert matrix.metadata.confidence_scores is not None
        np.testing.assert_array_equal(matrix.metadata.confidence_scores, confidence)

    def test_matrix_with_detection_limits(self):
        """Test matrix with detection limits metadata."""
        X = np.random.rand(3, 3)
        limits = np.ones((3, 3)) * 0.5
        metadata = MatrixMetadata(detection_limits=limits)
        matrix = ScpMatrix(X=X, metadata=metadata)
        np.testing.assert_array_equal(matrix.metadata.detection_limits, limits)

    def test_matrix_with_imputation_quality(self):
        """Test matrix with imputation quality scores."""
        X = np.random.rand(3, 3)
        quality = np.random.rand(3, 3)
        metadata = MatrixMetadata(imputation_quality=quality)
        matrix = ScpMatrix(X=X, metadata=metadata)
        np.testing.assert_array_equal(matrix.metadata.imputation_quality, quality)

    def test_matrix_with_outlier_scores(self):
        """Test matrix with outlier scores."""
        X = np.random.rand(3, 3)
        scores = np.random.rand(3, 3)
        metadata = MatrixMetadata(outlier_scores=scores)
        matrix = ScpMatrix(X=X, metadata=metadata)
        np.testing.assert_array_equal(matrix.metadata.outlier_scores, scores)

    def test_matrix_with_creation_info(self):
        """Test matrix with creation info."""
        X = np.random.rand(3, 3)
        info = {"method": "test", "params": {"k": 5}}
        metadata = MatrixMetadata(creation_info=info)
        matrix = ScpMatrix(X=X, metadata=metadata)
        assert matrix.metadata.creation_info == info

    def test_matrix_with_all_metadata_fields(self):
        """Test matrix with all metadata fields populated."""
        X = np.random.rand(3, 3)
        metadata = MatrixMetadata(
            confidence_scores=np.random.rand(3, 3),
            detection_limits=np.ones((3, 3)) * 0.5,
            imputation_quality=np.random.rand(3, 3),
            outlier_scores=np.random.rand(3, 3),
            creation_info={"method": "full_test"},
        )
        matrix = ScpMatrix(X=X, metadata=metadata)
        assert matrix.metadata.confidence_scores is not None
        assert matrix.metadata.detection_limits is not None
        assert matrix.metadata.imputation_quality is not None
        assert matrix.metadata.outlier_scores is not None
        assert matrix.metadata.creation_info is not None

    def test_matrix_with_sparse_metadata(self):
        """Test matrix with sparse metadata arrays."""
        X = sparse.csr_matrix(np.random.rand(5, 5))
        confidence = sparse.csr_matrix(np.random.rand(5, 5))
        metadata = MatrixMetadata(confidence_scores=confidence)
        matrix = ScpMatrix(X=X, metadata=metadata)
        assert sparse.issparse(matrix.metadata.confidence_scores)


class TestScpMatrixCopyEdgeCases:
    """Test copy functionality with edge cases."""

    def test_copy_preserves_sparse_format(self):
        """Test that copy preserves sparse format."""
        X = sparse.csr_matrix(np.random.rand(5, 5))
        matrix = ScpMatrix(X=X)
        matrix_copy = matrix.copy()
        assert sparse.issparse(matrix_copy.X)

    def test_copy_preserves_mask(self):
        """Test that copy preserves mask correctly."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 4, 5], [6, 0, 0]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        matrix_copy = matrix.copy()
        np.testing.assert_array_equal(matrix_copy.M, matrix.M)

    def test_copy_independence(self):
        """Test that copy creates independent data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = np.array([[0, 1], [2, 0]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        matrix_copy = matrix.copy()

        # Modify original
        matrix.X[0, 0] = 999.0
        matrix.M[0, 1] = MaskCode.IMPUTED

        # Copy should not be affected
        assert matrix_copy.X[0, 0] == 1.0
        assert matrix_copy.M[0, 1] == MaskCode.MBR

    def test_copy_does_not_preserve_metadata(self):
        """Test that copy does not preserve metadata (current behavior)."""
        X = np.random.rand(3, 3)
        metadata = MatrixMetadata(creation_info={"test": "value"})
        matrix = ScpMatrix(X=X, metadata=metadata)
        matrix_copy = matrix.copy()
        # Note: The current implementation of copy() does not preserve metadata
        # This test documents current behavior - metadata is lost during copy
        assert matrix_copy.metadata is None


class TestScpMatrixGetM:
    """Test get_m method variations."""

    def test_get_m_returns_zeros_when_no_mask(self):
        """Test that get_m returns zero matrix when M is None."""
        X = np.random.rand(3, 3)
        matrix = ScpMatrix(X=X)
        result = matrix.get_m()
        expected = np.zeros((3, 3), dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_get_m_returns_sparse_zeros_when_x_sparse(self):
        """Test that get_m returns sparse zeros when X is sparse and M is None."""
        X = sparse.csr_matrix(np.random.rand(3, 3))
        matrix = ScpMatrix(X=X)
        result = matrix.get_m()
        assert sparse.issparse(result)
        assert result.shape == (3, 3)

    def test_get_m_returns_mask_when_present(self):
        """Test that get_m returns M when present."""
        X = np.random.rand(3, 3)
        M = np.ones((3, 3), dtype=np.int8) * MaskCode.IMPUTED
        matrix = ScpMatrix(X=X, M=M)
        result = matrix.get_m()
        np.testing.assert_array_equal(result, M)
