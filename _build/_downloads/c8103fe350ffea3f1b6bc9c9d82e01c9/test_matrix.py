"""
Tests for ScpMatrix core structure.

This module contains tests for ScpMatrix functionality.
"""

import pytest
import numpy as np
from scipy import sparse


class TestScpMatrixBasic:
    """Test basic ScpMatrix functionality."""

    def test_matrix_creation_with_dense_array(self):
        """Test creating ScpMatrix with dense numpy array."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(10, 5)
        matrix = ScpMatrix(X=X)

        assert matrix.X.shape == (10, 5)
        np.testing.assert_array_equal(matrix.X, X)

    def test_matrix_creation_with_sparse_matrix(self):
        """Test creating ScpMatrix with sparse matrix."""
        from scptensor.core import ScpMatrix

        X_dense = np.random.rand(10, 5)
        X_dense[X_dense < 0.7] = 0  # Make it sparse
        X_sparse = sparse.csr_matrix(X_dense)

        matrix = ScpMatrix(X=X_sparse)

        assert sparse.issparse(matrix.X)
        assert matrix.X.shape == (10, 5)

    def test_matrix_creation_with_mask(self):
        """Test creating ScpMatrix with mask matrix."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(5, 3)
        M = np.zeros((5, 3), dtype=np.int8)
        M[0, 0] = 1  # Missing value
        M[2, 1] = 2  # LOD

        matrix = ScpMatrix(X=X, M=M)

        assert matrix.X.shape == (5, 3)
        assert matrix.M is not None
        assert matrix.M.shape == (5, 3)
        assert matrix.M[0, 0] == 1
        assert matrix.M[2, 1] == 2

    def test_matrix_creation_without_mask(self):
        """Test creating ScpMatrix without mask (M=None)."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(5, 3)
        matrix = ScpMatrix(X=X)

        assert matrix.M is None

    def test_get_m_with_no_mask(self):
        """Test get_m when M is None."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(5, 3)
        matrix = ScpMatrix(X=X)

        result = matrix.get_m()

        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result, np.zeros((5, 3), dtype=np.int8))

    def test_get_m_with_mask(self):
        """Test get_m when M is provided."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(5, 3)
        M = np.ones((5, 3), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        result = matrix.get_m()

        np.testing.assert_array_equal(result, M)

    def test_get_m_with_sparse_mask(self):
        """Test get_m with sparse mask matrix."""
        from scptensor.core import ScpMatrix

        X = np.random.rand(5, 3)
        M_sparse = sparse.csr_matrix(np.ones((5, 3), dtype=np.int8))
        matrix = ScpMatrix(X=X, M=M_sparse)

        result = matrix.get_m()

        # Should return sparse matrix
        assert sparse.issparse(result)
        assert result.shape == (5, 3)

    def test_matrix_copy(self):
        """Test copy method creates independent copy."""
        from scptensor.core import ScpMatrix

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = np.array([[0, 1], [2, 0]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        matrix_copy = matrix.copy()

        # Verify it's a different object
        assert matrix_copy is not matrix

        # Verify data is the same
        np.testing.assert_array_equal(matrix_copy.X, matrix.X)
        np.testing.assert_array_equal(matrix_copy.get_m(), matrix.get_m())

        # Modify original
        matrix.X[0, 0] = 999.0

        # Copy should not be affected
        assert matrix_copy.X[0, 0] == 1.0

    def test_matrix_with_metadata(self):
        """Test ScpMatrix with metadata."""
        from scptensor.core import ScpMatrix, MatrixMetadata

        X = np.random.rand(5, 3)
        confidence_scores = np.random.rand(5, 3)
        metadata = MatrixMetadata(
            confidence_scores=confidence_scores,
            detection_limits=np.zeros((5, 3)),
            creation_info={"method": "test"}
        )
        matrix = ScpMatrix(X=X, metadata=metadata)

        assert matrix.metadata is not None
        assert matrix.metadata.confidence_scores is not None
        assert matrix.metadata.confidence_scores.shape == (5, 3)
        assert matrix.metadata.creation_info == {"method": "test"}

    def test_matrix_large_dimensions(self):
        """Test ScpMatrix with large dimensions (stress test)."""
        from scptensor.core import ScpMatrix

        # Large matrix: 1000 samples x 5000 features
        X = np.random.rand(100, 500)  # Smaller for test speed
        M = np.zeros((100, 500), dtype=np.int8)
        M[np.random.rand(100, 500) < 0.3] = 1  # 30% missing

        matrix = ScpMatrix(X=X, M=M)

        assert matrix.X.shape == (100, 500)
        assert matrix.M.shape == (100, 500)
        assert np.sum(matrix.M == 1) > 0  # Some missing values
