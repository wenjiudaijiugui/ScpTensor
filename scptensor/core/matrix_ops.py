"""
Utility functions for enhanced ScpMatrix operations with mask code support.

Optimized for sparse matrix operations - preserves sparsity when possible.
"""

import numpy as np
import scipy.sparse as sp

from .structures import MaskCode, ScpMatrix


def sparse_filter_mask(M: sp.spmatrix, keep_values: list[int]) -> sp.spmatrix:
    """
    Efficiently create boolean mask for sparse matrix filtering.

    Parameters
    ----------
    M : sp.spmatrix
        Sparse mask matrix
    keep_values : List[int]
        Values to keep (marked as True)

    Returns
    -------
    sp.spmatrix
        Boolean sparse mask where True indicates values to keep
    """
    # Convert to COO for efficient element-wise operations
    M_coo = M.tocoo()

    # Create boolean mask
    keep_mask = np.isin(M_coo.data, keep_values)

    # Create sparse boolean matrix
    result = sp.coo_matrix((keep_mask, (M_coo.row, M_coo.col)), shape=M.shape, dtype=bool)

    return result.tocsr()


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
        return mask_code == M

    @staticmethod
    def mark_values(
        matrix: ScpMatrix, indices: tuple[np.ndarray, np.ndarray], mask_code: MaskCode
    ) -> ScpMatrix:
        """
        Mark specific values with given mask code.

        Optimized to preserve sparsity - avoids unnecessary densification.

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
            # Efficient sparse modification using LIL format for fast indexing
            M_lil = M.tolil()
            M_lil[indices] = mask_code.value
            # Convert back to efficient format (CSR for row operations, CSC for column)
            new_matrix.M = M_lil.tocsr()
        else:
            M[indices] = mask_code.value
            new_matrix.M = M

        return new_matrix

    @staticmethod
    def mark_imputed(matrix: ScpMatrix, indices: tuple[np.ndarray, np.ndarray]) -> ScpMatrix:
        """Mark values as imputed."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.IMPUTED)

    @staticmethod
    def mark_outliers(matrix: ScpMatrix, indices: tuple[np.ndarray, np.ndarray]) -> ScpMatrix:
        """Mark values as outliers."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.OUTLIER)

    @staticmethod
    def mark_lod(matrix: ScpMatrix, indices: tuple[np.ndarray, np.ndarray]) -> ScpMatrix:
        """Mark values as below limit of detection."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.LOD)

    @staticmethod
    def mark_uncertain(matrix: ScpMatrix, indices: tuple[np.ndarray, np.ndarray]) -> ScpMatrix:
        """Mark values as uncertain quality."""
        return MatrixOps.mark_values(matrix, indices, MaskCode.UNCERTAIN)

    @staticmethod
    def combine_masks(masks: list[np.ndarray], operation: str = "union") -> np.ndarray:
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

        if operation == "union":
            return np.any(masks_stacked, axis=0)
        elif operation == "intersection":
            return np.all(masks_stacked, axis=0)
        elif operation == "majority":
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
            count = np.sum(code.value == M)
            percentage = (count / total_elements) * 100
            stats[code.name] = {"count": int(count), "percentage": float(percentage)}

        return stats

    @staticmethod
    def filter_by_mask(matrix: ScpMatrix, keep_codes: list[MaskCode]) -> ScpMatrix:
        """
        Create new matrix with only specified mask codes.
        All other values are set to NaN and marked as FILTERED.

        Optimized to preserve sparsity - avoids unnecessary densification.

        Args:
            matrix: Input ScpMatrix
            keep_codes: List of MaskCode values to keep

        Returns:
            Filtered ScpMatrix
        """
        new_matrix = matrix.copy()
        M = new_matrix.get_m()
        X = new_matrix.X.copy()

        # Handle sparse matrices efficiently
        if isinstance(M, sp.spmatrix):
            # For sparse matrices, use efficient sparse operations
            keep_values = [code.value for code in keep_codes]
            keep_mask = sparse_filter_mask(M, keep_values)

            # Update M in sparse format using LIL for efficient indexing
            M_lil = M.tolil()
            X_lil = X.tolil() if isinstance(X, sp.spmatrix) else None

            # Get indices where we DON'T want to keep (filter these out)
            # For sparse boolean, we need to find indices where keep_mask is False
            filter_mask_csr = keep_mask.copy()
            filter_mask_csr.data = ~filter_mask_csr.data  # Invert boolean values
            rows, cols = filter_mask_csr.nonzero()

            for r, c in zip(rows, cols, strict=False):
                M_lil[r, c] = MaskCode.FILTERED.value
                if X_lil is not None:
                    X_lil[r, c] = np.nan
                else:
                    X[r, c] = np.nan

            new_matrix.M = M_lil.tocsr()
            new_matrix.X = X_lil.tocsr() if X_lil is not None else X
        else:
            # Dense matrix operations (original code)
            keep_mask = np.isin(M, [code.value for code in keep_codes])
            filter_mask = ~keep_mask
            M[filter_mask] = MaskCode.FILTERED.value
            X[filter_mask] = np.nan

            new_matrix.M = M
            new_matrix.X = X

        return new_matrix

    @staticmethod
    def apply_mask_to_values(matrix: ScpMatrix, operation: str = "zero") -> ScpMatrix:
        """
        Apply mask to values in X matrix.

        Optimized to preserve sparsity - avoids unnecessary densification.

        Args:
            matrix: Input ScpMatrix
            operation: 'zero', 'nan', or 'keep'

        Returns:
            ScpMatrix with mask applied to values
        """
        new_matrix = matrix.copy()
        X = new_matrix.X.copy()
        M = new_matrix.get_m()

        # Handle sparse matrices efficiently
        if isinstance(M, sp.spmatrix):
            # Use sparse operations (use != for efficiency with sparse)
            valid_mask = M.copy()
            valid_mask.data = valid_mask.data == MaskCode.VALID.value

            if operation == "zero":
                # Set invalid values to zero (efficient for sparse)
                if isinstance(X, sp.spmatrix):
                    # For sparse, we multiply by valid mask (zeros out invalid entries)
                    X = X.multiply(valid_mask.astype(float))
                else:
                    # Dense X, sparse M
                    X[~valid_mask.toarray()] = 0.0
            elif operation == "nan":
                # Set invalid values to NaN
                if isinstance(X, sp.spmatrix):
                    # Convert to COO for efficient modification
                    X_coo = X.tocoo()
                    M_coo = valid_mask.tocoo()

                    # Find invalid entries
                    invalid_mask = M_coo.data == 0
                    X_coo.data[invalid_mask] = np.nan

                    X = X_coo.tocsr()
                else:
                    X[~valid_mask.toarray()] = np.nan
            elif operation != "keep":
                raise ValueError(f"Unknown operation: {operation}")
        else:
            # Dense matrix operations (original code)
            valid_mask = M == MaskCode.VALID

            if operation == "zero":
                X[~valid_mask] = 0.0
            elif operation == "nan":
                X[~valid_mask] = np.nan
            elif operation != "keep":
                raise ValueError(f"Unknown operation: {operation}")

        new_matrix.X = X
        return new_matrix
