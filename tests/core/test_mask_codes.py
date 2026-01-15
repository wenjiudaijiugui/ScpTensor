"""
Tests for MaskCode enum and mask code operations.

This module tests the provenance tracking system through mask codes.
"""

import numpy as np
import pytest
from scipy import sparse

from scptensor.core import MaskCode, ScpMatrix


class TestMaskCodeEnum:
    """Test MaskCode enum functionality."""

    def test_mask_code_values(self):
        """Test that all mask codes have correct integer values."""
        assert MaskCode.VALID == 0
        assert MaskCode.MBR == 1
        assert MaskCode.LOD == 2
        assert MaskCode.FILTERED == 3
        assert MaskCode.OUTLIER == 4
        assert MaskCode.IMPUTED == 5
        assert MaskCode.UNCERTAIN == 6

    def test_mask_code_names(self):
        """Test that mask codes have correct names."""
        assert MaskCode.VALID.name == "VALID"
        assert MaskCode.MBR.name == "MBR"
        assert MaskCode.LOD.name == "LOD"
        assert MaskCode.FILTERED.name == "FILTERED"
        assert MaskCode.OUTLIER.name == "OUTLIER"
        assert MaskCode.IMPUTED.name == "IMPUTED"
        assert MaskCode.UNCERTAIN.name == "UNCERTAIN"

    def test_mask_code_iteration(self):
        """Test iterating over all mask codes."""
        codes = list(MaskCode)
        assert len(codes) == 7
        assert MaskCode.VALID in codes
        assert MaskCode.IMPUTED in codes

    def test_mask_code_comparison(self):
        """Test mask code comparison operations."""
        assert MaskCode.VALID < MaskCode.IMPUTED
        assert MaskCode.MBR <= MaskCode.LOD
        assert MaskCode.IMPUTED > MaskCode.FILTERED
        assert MaskCode.UNCERTAIN >= MaskCode.OUTLIER


class TestMaskCodeValidation:
    """Test mask code validation in ScpMatrix."""

    def test_valid_mask_codes_accepted(self):
        """Test that all valid mask codes are accepted."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 4, 5], [6, 0, 1]], dtype=np.int8)
        # Should not raise
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M is not None

    def test_invalid_mask_code_raises_error(self):
        """Test that invalid mask codes raise ValueError."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 99, 5], [6, 0, 1]], dtype=np.int8)  # 99 is invalid
        with pytest.raises(ValueError, match="Invalid mask codes"):
            ScpMatrix(X=X, M=M)

    def test_negative_mask_code_raises_error(self):
        """Test that negative mask codes raise ValueError."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, -1], [3, 4, 5], [6, 0, 1]], dtype=np.int8)  # -1 is invalid
        with pytest.raises(ValueError, match="Invalid mask codes"):
            ScpMatrix(X=X, M=M)

    def test_mask_code_auto_cast_to_int8(self):
        """Test that mask codes are auto-cast to int8."""
        X = np.random.rand(1, 3)
        M = np.array([[0, 1, 2]], dtype=np.int32)  # Wrong type
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M.dtype == np.int8

    def test_sparse_mask_codes_accepted(self):
        """Test that sparse mask matrices work with valid codes."""
        X = np.random.rand(5, 5)
        M_dense = np.zeros((5, 5), dtype=np.int8)
        M_dense[0, 0] = 1
        M_dense[1, 1] = 5
        M_sparse = sparse.csr_matrix(M_dense)
        # Should not raise
        matrix = ScpMatrix(X=X, M=M_sparse)
        assert sparse.issparse(matrix.M)


class TestMaskCodeOperations:
    """Test operations with mask codes."""

    def test_get_valid_value_mask(self):
        """Test getting mask for only valid values."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 0, 5], [6, 0, 0]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        mask = matrix.M == MaskCode.VALID
        expected = np.array([[True, False, False], [False, True, False], [False, True, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_get_missing_value_mask(self):
        """Test getting mask for all missing value types."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 0, 5], [6, 0, 0]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        # Missing: MBR(1), LOD(2), FILTERED(3)
        missing_mask = np.isin(matrix.M, [MaskCode.MBR, MaskCode.LOD, MaskCode.FILTERED])
        expected = np.array([[False, True, True], [True, False, False], [False, False, False]])
        np.testing.assert_array_equal(missing_mask, expected)

    def test_imputed_value_mask(self):
        """Test getting mask for imputed values."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 5], [3, 5, 0], [6, 0, 5]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        imputed_mask = matrix.M == MaskCode.IMPUTED
        expected = np.array([[False, False, True], [False, True, False], [False, False, True]])
        np.testing.assert_array_equal(imputed_mask, expected)

    def test_mask_code_preservation_in_copy(self):
        """Test that mask codes are preserved during copy."""
        X = np.random.rand(3, 3)
        M = np.array([[0, 1, 2], [3, 4, 5], [6, 0, 1]], dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        matrix_copy = matrix.copy()
        np.testing.assert_array_equal(matrix_copy.M, matrix.M)


class TestMaskCodeProvenanceTracking:
    """Test provenance tracking through mask codes."""

    def test_mark_value_as_missing_mbr(self):
        """Test marking a value as MBR (Match Between Runs)."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 1] = MaskCode.MBR
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 1] == MaskCode.MBR
        assert matrix.M[0, 0] == MaskCode.VALID

    def test_mark_value_as_lod(self):
        """Test marking a value as LOD (Limit of Detection)."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 2] = MaskCode.LOD
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 2] == MaskCode.LOD

    def test_mark_value_as_filtered(self):
        """Test marking a value as filtered (QC removed)."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 0] = MaskCode.FILTERED
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 0] == MaskCode.FILTERED

    def test_mark_value_as_outlier(self):
        """Test marking a value as statistical outlier."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 1] = MaskCode.OUTLIER
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 1] == MaskCode.OUTLIER

    def test_mark_value_as_imputed(self):
        """Test marking a value as imputed."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 2] = MaskCode.IMPUTED
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 2] == MaskCode.IMPUTED

    def test_mark_value_as_uncertain(self):
        """Test marking a value as uncertain quality."""
        X = np.array([[1.0, 2.0, 3.0]])
        M = np.zeros((1, 3), dtype=np.int8)
        M[0, 0] = MaskCode.UNCERTAIN
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.M[0, 0] == MaskCode.UNCERTAIN

    def test_count_by_mask_code(self):
        """Test counting values by mask code."""
        X = np.random.rand(10, 10)
        M = np.zeros((10, 10), dtype=np.int8)
        M[0, 0] = MaskCode.MBR
        M[0, 1] = MaskCode.LOD
        M[0, 2] = MaskCode.FILTERED
        M[0, 3] = MaskCode.OUTLIER
        M[0, 4] = MaskCode.IMPUTED
        M[0, 5] = MaskCode.UNCERTAIN
        matrix = ScpMatrix(X=X, M=M)

        # Count each type
        valid_count = np.sum(matrix.M == MaskCode.VALID)
        mbr_count = np.sum(matrix.M == MaskCode.MBR)
        lod_count = np.sum(matrix.M == MaskCode.LOD)
        filtered_count = np.sum(matrix.M == MaskCode.FILTERED)
        outlier_count = np.sum(matrix.M == MaskCode.OUTLIER)
        imputed_count = np.sum(matrix.M == MaskCode.IMPUTED)
        uncertain_count = np.sum(matrix.M == MaskCode.UNCERTAIN)

        assert valid_count == 94  # 100 - 6 marked
        assert mbr_count == 1
        assert lod_count == 1
        assert filtered_count == 1
        assert outlier_count == 1
        assert imputed_count == 1
        assert uncertain_count == 1


class TestMaskCodeWithSparseMatrix:
    """Test mask codes with sparse matrix storage."""

    def test_sparse_mask_with_various_codes(self):
        """Test sparse mask matrix with various codes."""
        X = sparse.csr_matrix(np.random.rand(10, 10))
        M_dense = np.zeros((10, 10), dtype=np.int8)
        M_dense[0, 0] = MaskCode.MBR
        M_dense[1, 1] = MaskCode.IMPUTED
        M_dense[2, 2] = MaskCode.LOD
        M_sparse = sparse.csr_matrix(M_dense)
        matrix = ScpMatrix(X=X, M=M_sparse)
        assert sparse.issparse(matrix.M)

    def test_sparse_mask_get_m_returns_sparse(self):
        """Test that get_m returns sparse matrix when input is sparse."""
        X = np.random.rand(5, 5)
        M_sparse = sparse.csr_matrix(np.zeros((5, 5), dtype=np.int8))
        matrix = ScpMatrix(X=X, M=M_sparse)
        result = matrix.get_m()
        assert sparse.issparse(result)
