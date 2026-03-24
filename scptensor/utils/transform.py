"""Data transformation utilities for DIA-based single-cell proteomics data.

This module provides common data transformations used in proteomics analysis,
including quantile normalization and robust scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from scptensor.core._rank_normalization import quantile_normalize_dense_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray


def quantile_normalize(
    X: NDArray[np.float64] | sp.spmatrix,
    axis: int = 0,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Apply quantile normalization to make distributions identical.

    Quantile normalization forces all samples (or features) to have the
    same distribution. This is commonly used in genomics and proteomics
    to remove technical variation between samples.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input data matrix of shape (n_samples, n_features).
    axis : int, default=0
        Axis along which to normalize.
        - 0: Normalize each column (all features share the same distribution)
        - 1: Normalize each row (all samples share the same distribution)
    copy : bool, default=True
        Whether to create a copy of the input data.

    Returns
    -------
    NDArray[np.float64]
        Normalized data. Sparse matrices are converted to dense.

    Raises
    ------
    ValueError
        If axis is not 0 or 1.

    Notes
    -----
    Quantile normalization works by:
    1. Sorting each column (or row)
    2. Computing the mean across rows for each rank
    3. Replacing each value with the rank-specific mean
    4. Unsorting to restore original order

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])
    >>> X_norm = quantile_normalize(X, axis=0)
    >>> # All columns should have the same distribution

    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    X_working = X.toarray() if sp.issparse(X) else np.asarray(X)  # type: ignore[union-attr]
    X_working = X_working.astype(np.float64, copy=copy)

    if axis == 0:
        return quantile_normalize_dense_rows(X_working.T).T.astype(np.float64, copy=False)

    return quantile_normalize_dense_rows(X_working).astype(np.float64, copy=False)


def robust_scale(
    X: NDArray[np.float64] | sp.spmatrix,
    center: float | None = None,
    scale: float | None = None,
    with_centering: bool = True,
    with_scaling: bool = True,
    axis: int = 0,
    copy: bool = True,
) -> NDArray[np.float64] | sp.spmatrix:
    """Apply robust scaling using median and IQR (interquartile range).

    Robust scaling is less sensitive to outliers than standard z-score
    normalization. It uses median instead of mean and IQR instead of
    standard deviation.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input data matrix of shape (n_samples, n_features).
    center : float, optional
        Pre-computed center value. If None, uses median of data.
    scale : float, optional
        Pre-computed scale value. If None, uses IQR of data.
    with_centering : bool, default=True
        Whether to center the data.
    with_scaling : bool, default=True
        Whether to scale the data.
    axis : int, default=0
        Axis along which to compute median and IQR.
    copy : bool, default=True
        Whether to create a copy of the input data.

    Returns
    -------
    Union[NDArray[np.float64], sp.spmatrix]
        Scaled data with same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 1000]])  # Outlier in column 2
    >>> X_scaled = robust_scale(X, axis=0)
    >>> X_scaled.shape
    (4, 2)

    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    # Get dense version for statistics
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)  # type: ignore[union-attr]
    X_dense = X_dense.astype(np.float64, copy=False)
    X_working = X_dense if not copy else X_dense.copy()

    # Compute statistics
    if axis == 0:
        if with_centering and center is None:
            center = np.median(X_dense, axis=0)
        if with_scaling and scale is None:
            q75, q25 = np.percentile(X_dense, [75, 25], axis=0)
            scale = q75 - q25
            scale[scale == 0] = 1.0  # type: ignore[index]
    else:
        if with_centering and center is None:
            center = np.median(X_dense, axis=1, keepdims=True)
        if with_scaling and scale is None:
            q75, q25 = np.percentile(X_dense, [75, 25], axis=1, keepdims=True)
            scale = q75 - q25
            scale[scale == 0] = 1.0  # type: ignore[index]

    # Apply transformations
    result = X_working
    if with_centering and center is not None:
        result = result - center
    if with_scaling and scale is not None:
        result = result / scale

    # Return sparse if input was sparse
    if sp.issparse(X) and copy:
        return sp.csr_matrix(result)

    return result.astype(np.float64)
