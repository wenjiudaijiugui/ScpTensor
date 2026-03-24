"""Utility functions for imputation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse as sp

from scptensor.core._layer_processing import create_result_layer
from scptensor.core._structure_matrix import MaskCode, ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


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
        new_M[missing_mask] = MaskCode.IMPUTED
        return new_M
    # Create new mask with IMPUTED values for previously missing entries,
    # VALID (0) for observed entries
    new_M = np.zeros(missing_mask.shape, dtype=np.int8)
    new_M[missing_mask] = MaskCode.IMPUTED
    return new_M


def to_dense_float_copy(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Return a dense float64 copy for imputation algorithms."""
    if sp.issparse(x):
        return cast("sp.spmatrix", x).toarray().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64).copy()


def preserve_observed_values(
    x_imputed: np.ndarray,
    x_original: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    """Restore observed entries so only original NaN positions are changed."""
    if np.any(~missing_mask):
        x_imputed[~missing_mask] = x_original[~missing_mask]
    return x_imputed


def build_imputed_matrix(
    x: np.ndarray | sp.spmatrix,
    input_matrix: ScpMatrix,
    missing_mask: np.ndarray,
    *,
    source_assay_name: str | None = None,
    source_layer_name: str | None = None,
    action: str | None = None,
    output_layer_name: str | None = None,
) -> ScpMatrix:
    """Create a result matrix with the contract-preserving imputed mask."""
    result = create_result_layer(
        x,
        input_matrix,
        source_assay_name=source_assay_name,
        source_layer_name=source_layer_name,
        action=action,
        output_layer_name=output_layer_name,
    )
    result.M = _update_imputed_mask(input_matrix.M, missing_mask)
    return result


def add_imputed_layer(
    assay: Assay,
    layer_name: str,
    x: np.ndarray | sp.spmatrix,
    input_matrix: ScpMatrix,
    missing_mask: np.ndarray,
    *,
    source_assay_name: str | None = None,
    source_layer_name: str | None = None,
    action: str | None = None,
) -> ScpMatrix:
    """Write an imputed layer and return the stored matrix."""
    result = build_imputed_matrix(
        x,
        input_matrix,
        missing_mask,
        source_assay_name=source_assay_name,
        source_layer_name=source_layer_name,
        action=action,
        output_layer_name=layer_name,
    )
    assay.add_layer(layer_name, result)
    return result


def clone_layer_matrix(
    source_layer: ScpMatrix,
    *,
    source_assay_name: str | None = None,
    source_layer_name: str | None = None,
    action: str | None = None,
    output_layer_name: str | None = None,
) -> ScpMatrix:
    """Clone X and M for passthrough/no-op layer creation."""
    if sp.issparse(source_layer.X):
        x_copy: np.ndarray | sp.spmatrix = source_layer.X.copy()
    else:
        x_copy = np.array(source_layer.X, copy=True)

    result = create_result_layer(
        x_copy,
        source_layer,
        source_assay_name=source_assay_name,
        source_layer_name=source_layer_name,
        action=action,
        output_layer_name=output_layer_name,
    )
    result.M = source_layer.M.copy() if source_layer.M is not None else None
    return result


def log_imputation_operation(
    container: ScpContainer,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Append imputation provenance and return the same container."""
    container.log_operation(action=action, params=params, description=description)
    return container
