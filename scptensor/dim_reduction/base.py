"""Base utilities for dimensionality reduction modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from scptensor.core._layer_processing import ensure_dense_matrix, resolve_layer_context
from scptensor.core.exceptions import ValidationError

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def _validate_assay_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, np.ndarray | sp.spmatrix]:
    """Validate and get assay and layer data.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of assay.
    layer_name : str
        Name of layer.

    Returns
    -------
    tuple[Assay, np.ndarray | sp.spmatrix]
        Assay object and data matrix X.

    Raises
    ------
    AssayNotFoundError
        If assay not found.
    LayerNotFoundError
        If layer not found.

    """
    ctx = resolve_layer_context(container, assay_name, layer_name)
    return ctx.assay, ctx.layer.X


def _prepare_matrix(
    X: np.ndarray | sp.spmatrix,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Convert matrix to dense numpy array with optional dtype.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix (dense or sparse).
    dtype : np.dtype | None, optional
        Target dtype. If None, uses float64.

    Returns
    -------
    np.ndarray
        Dense numpy array.

    """
    X = ensure_dense_matrix(X)

    if dtype is not None:
        X = X.astype(dtype)
    elif X.dtype != np.float64:
        X = X.astype(np.float64)

    return np.asarray(X)


def _check_no_nan_inf(X: np.ndarray | sp.spmatrix) -> None:
    """Check that matrix contains no NaN or Inf values.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix.

    Raises
    ------
    ValidationError
        If NaN or Inf values found.

    """
    if sp.issparse(X):
        X_data = X.data
    else:
        X_data = X

    if np.any(np.isnan(X_data)):
        raise ValidationError(
            "Data contains NaN values. Please impute or remove missing values first.",
            field="X",
        )

    if np.any(np.isinf(X_data)):
        raise ValidationError(
            "Data contains Inf values. Please check your data for infinite values.",
            field="X",
        )


__all__ = [
    "_validate_assay_layer",
    "_prepare_matrix",
    "_check_no_nan_inf",
]
