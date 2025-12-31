"""
Utility functions for enhanced ScpMatrix operations with mask code support.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union, List, Optional
from .structures import ScpMatrix, MaskCode


class MatrixOps:
    """
    Enhanced matrix operations for ScpMatrix with full mask code support.
    """

    @staticmethod
    def get_valid_mask(matrix: ScpMatrix) -> np.ndarray:
        """Get mask for valid (VALID) data points."""
        M = matrix.get_m()
        return M == MaskCode.VALID

    @staticmethod
    def get_missing_mask(matrix: ScpMatrix) -> np.ndarray:
        """Get mask for any missing data (any non-VALID code)."""
        M = matrix.get_m()
        return M != MaskCode.VALID

    @staticmethod
    def get_missing_type_mask(matrix: ScpMatrix, mask_code: MaskCode) -> np.ndarray:
        """Get mask for specific missing type."""
        M = matrix.get_m()
        return M == mask_code

    @staticmethod
    def mark_values(
        matrix: ScpMatrix,
        indices: Tuple[np.ndarray, np.ndarray],
        mask_code: MaskCode
    ) -> ScpMatrix:
        """
        Mark specific values with given mask code.

        Args:
            matrix: Input ScpMatrix
            indices: Tuple of (row_indices, col_indices) to mark
            mask_code: MaskCode to assign

        Returns:
            New ScpMatrix with updated mask
        """
        new_matrix = matrix.copy()
        M = new_matrix.get_m()
        if isinstance(M, sp.spmatrix):
            M = M.toarray()

        M[indices] = mask_code.value
        new_matrix.M = M
        return new_matrix

    @staticmethod
    def mark_imputed(matrix: ScpMatrix, indices: Tuple[np.ndarray, np.ndarray]) -> ScpMatrix:
        """Mark values as imputed."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.IMPUTED)

    @staticmethod
    def mark_outliers(
        matrix: ScpMatrix,
        indices: Tuple[np.ndarray, np.ndarray]
    ) -> ScpMatrix:
        """Mark values as outliers."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.OUTLIER)

    @staticmethod
    def mark_lod(
        matrix: ScpMatrix,
        indices: Tuple[np.ndarray, np.ndarray]
    ) -> ScpMatrix:
        """Mark values as below limit of detection."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.LOD)

    @staticmethod
    def mark_uncertain(
        matrix: ScpMatrix,
        indices: Tuple[np.ndarray, np.ndarray]
    ) -> ScpMatrix:
        """Mark values as uncertain quality."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.UNCERTAIN)

    @staticmethod
    def combine_masks(
        masks: List[np.ndarray],
        operation: str = 'union'
    ) -> np.ndarray:
        """
        Combine multiple masks with specified operation.

        Args:
            masks: List of boolean masks to combine
            operation: 'union' (OR), 'intersection' (AND), or 'majority'

        Returns:
            Combined boolean mask
        """
        if not masks:
            raise ValueError("No masks provided")

        if len(masks) == 1:
            return masks[0].copy()

        masks_stacked = np.stack(masks, axis=0)

        if operation == 'union':
            return np.any(masks_stacked, axis=0)
        elif operation == 'intersection':
            return np.all(masks_stacked, axis=0)
        elif operation == 'majority':
            return np.sum(masks_stacked, axis=0) > (len(masks) // 2)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def get_mask_statistics(matrix: ScpMatrix) -> dict:
        """
        Get comprehensive statistics about mask codes in the matrix.

        Returns:
            Dictionary with statistics for each mask code
        """
        M = matrix.get_m()
        total_elements = M.size

        stats = {}
        for code in MaskCode:
            count = np.sum(M == code.value)
            percentage = (count / total_elements) * 100
            stats[code.name] = {
                'count': int(count),
                'percentage': float(percentage)
            }

        return stats

    @staticmethod
    def filter_by_mask(
        matrix: ScpMatrix,
        keep_codes: List[MaskCode]
    ) -> ScpMatrix:
        """
        Create new matrix with only specified mask codes.
        All other values are set to NaN and marked as FILTERED.

        Args:
            matrix: Input ScpMatrix
            keep_codes: List of MaskCode values to keep

        Returns:
            Filtered ScpMatrix
        """
        new_matrix = matrix.copy()
        M = new_matrix.get_m()
        X = new_matrix.X.copy()

        if isinstance(M, sp.spmatrix):
            M = M.toarray()
            if isinstance(X, sp.spmatrix):
                X = X.toarray()

        # Create mask of values to keep
        keep_mask = np.isin(M, [code.value for code in keep_codes])

        # Mark others as filtered and set to NaN
        filter_mask = ~keep_mask
        M[filter_mask] = MaskCode.FILTERED.value
        X[filter_mask] = np.nan

        new_matrix.M = M
        new_matrix.X = X
        return new_matrix

    @staticmethod
    def apply_mask_to_values(
        matrix: ScpMatrix,
        operation: str = 'zero'
    ) -> ScpMatrix:
        """
        Apply mask to values in X matrix.

        Args:
            matrix: Input ScpMatrix
            operation: 'zero', 'nan', or 'keep'

        Returns:
            ScpMatrix with mask applied to values
        """
        new_matrix = matrix.copy()
        X = new_matrix.X.copy()
        M = new_matrix.get_m()

        if isinstance(M, sp.spmatrix):
            M = M.toarray()
            if isinstance(X, sp.spmatrix):
                X = X.toarray()

        valid_mask = M == MaskCode.VALID

        if operation == 'zero':
            X[~valid_mask] = 0.0
        elif operation == 'nan':
            X[~valid_mask] = np.nan
        elif operation != 'keep':
            raise ValueError(f"Unknown operation: {operation}")

        new_matrix.X = X
        return new_matrix