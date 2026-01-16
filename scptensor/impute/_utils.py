"""Utility functions for imputation modules."""

import numpy as np
import scipy.sparse as sp

from scptensor.core.structures import MaskCode


def _update_imputed_mask(
    M_original: np.ndarray | sp.spmatrix | None,
    missing_mask: np.ndarray,
) -> np.ndarray | sp.spmatrix | None:
    """Update mask matrix to mark imputed values.

    Parameters
    ----------
    M_original : np.ndarray, sp.spmatrix, or None
        Original mask matrix
    missing_mask : np.ndarray
        Boolean mask indicating which values were originally missing

    Returns
    -------
    np.ndarray, sp.spmatrix, or None
        Updated mask with IMPUTED code for previously missing entries.
        Returns None if M_original was None and no values were missing.
    """
    if not np.any(missing_mask):
        return M_original.copy() if M_original is not None else None

    if M_original is not None:
        new_M = M_original.copy()
        if sp.issparse(new_M):
            M_dense = new_M.toarray()  # type: ignore[union-attr]
            M_dense[missing_mask] = MaskCode.IMPUTED
            return sp.csr_matrix(M_dense, dtype=np.int8)
        else:
            new_M[missing_mask] = MaskCode.IMPUTED
            return new_M
    else:
        # Create new mask with IMPUTED values for previously missing entries,
        # VALID (0) for observed entries
        new_M = np.zeros(missing_mask.shape, dtype=np.int8)
        new_M[missing_mask] = MaskCode.IMPUTED
        return new_M
