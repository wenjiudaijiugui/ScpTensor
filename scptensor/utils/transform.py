"""Data transformation utilities for single-cell proteomics data.

This module provides common data transformations used in proteomics analysis,
including inverse hyperbolic sine (asinh), logicle, and quantile normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Constants
_LOG_10 = np.log(10)
_DEFAULT_LOGICLE_T = 262144.0
_DEFAULT_LOGICLE_W = 0.5
_DEFAULT_LOGICLE_M = 4.5
_DEFAULT_ASINH_COFACTOR = 5.0


def asinh_transform(
    X: NDArray[np.float64] | sp.spmatrix,
    cofactor: float = _DEFAULT_ASINH_COFACTOR,
    copy: bool = True,
) -> NDArray[np.float64] | sp.spmatrix:
    """Apply inverse hyperbolic sine (asinh) transformation.

    The asinh transformation is commonly used in cytometry and proteomics
    to handle data with both negative and positive values while maintaining
    symmetry around zero. It behaves linearly near zero and logarithmically
    for large values.

    The transformation is: asinh(x / cofactor) * ln(10)

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input data matrix of shape (n_samples, n_features).
    cofactor : float, default=5.0
        The cofactor determines the transition point between linear and
        logarithmic behavior. Larger values make the transformation more
        linear near zero. Typical values are 5-150 for cytometry data.
    copy : bool, default=True
        Whether to create a copy of the input data. If False, modifies
        in-place (only for dense arrays).

    Returns
    -------
    Union[NDArray[np.float64], sp.spmatrix]
        Transformed data with same shape as input.

    Raises
    ------
    ValueError
        If cofactor is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 1, 10], [0, 5, 100]])
    >>> X_transformed = asinh_transform(X, cofactor=5.0)
    >>> X_transformed.shape
    (2, 3)
    """
    if cofactor <= 0:
        raise ValueError(f"cofactor must be positive, got {cofactor}")

    if sp.issparse(X):
        X_copy = X.copy()
        X_copy.data = np.arcsinh(X_copy.data / cofactor) * _LOG_10  # type: ignore[misc,operator]
        return X_copy

    X_working = X.copy() if copy else X
    np.arcsinh(X_working / cofactor, out=X_working)
    X_working *= _LOG_10
    return X_working


def logicle_transform(
    X: NDArray[np.float64] | sp.spmatrix,
    T: float = _DEFAULT_LOGICLE_T,
    W: float = _DEFAULT_LOGICLE_W,
    M: float = _DEFAULT_LOGICLE_M,
    A: float = 0.0,
    copy: bool = True,
) -> NDArray[np.float64] | sp.spmatrix:
    """Apply Logicle transformation for flow cytometry data.

    The Logicle transformation combines the benefits of logarithmic and
    linear scales, displaying both positive and negative values clearly.
    It is widely used in cytometry data analysis.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input data matrix of shape (n_samples, n_features).
    T : float, default=262144.0
        Maximum range value (typically 2^18 for 18-bit data).
    W : float, default=0.5
        Logicle width parameter (in log decades). Controls how much
        negative data is shown.
    M : float, default=4.5
        Logicle magnitude (in log decades). Controls the number of
        decades shown.
    A : float, default=0.0
        Additional negative range to display.
    copy : bool, default=True
        Whether to create a copy of the input data.

    Returns
    -------
    Union[NDArray[np.float64], sp.spmatrix]
        Transformed data with same shape as input.

    Raises
    ------
    ValueError
        If T, W, or M are not positive.

    Notes
    -----
    The Logicle transformation is defined as:
    - x = T * 10^(-M - W) for x <= 0
    - x = T * (10^(x/T * (M + W) - M) / (1 + 10^(x/T * (M + W) - M)) + W / (M + W)) for x > 0

    For computational efficiency, this implementation uses a simplified
    approximation based on the asinh transform with adjusted parameters.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-10, 0, 10, 100]])
    >>> X_transformed = logicle_transform(X)
    >>> X_transformed.shape
    (1, 4)
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if W <= 0:
        raise ValueError(f"W must be positive, got {W}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    # Use asinh-based approximation for efficiency
    cofactor = T / (10**M)
    result = asinh_transform(X, cofactor=cofactor, copy=copy)

    # Apply Logicle scaling
    if sp.issparse(result):
        result.data *= M  # type: ignore[misc,operator]
    else:
        result *= M

    return result


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
        - 0: Normalize each column (make samples identical)
        - 1: Normalize each row (make features identical)
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

    # Convert sparse to dense for sorting
    X_working = X.toarray() if sp.issparse(X) else X  # type: ignore[union-attr]
    if copy:
        X_working = X_working.astype(np.float64, copy=True)

    if axis == 0:
        # Normalize columns
        sorted_X = np.sort(X_working, axis=0)
        row_means = sorted_X.mean(axis=1, keepdims=True)

        X_normalized = np.empty_like(X_working)
        for col_idx in range(X_working.shape[1]):
            ranks = np.argsort(np.argsort(X_working[:, col_idx]))
            X_normalized[:, col_idx] = row_means[ranks, 0]
    else:
        # Normalize rows
        sorted_X = np.sort(X_working, axis=1)
        col_means = sorted_X.mean(axis=0, keepdims=True)

        X_normalized = np.empty_like(X_working)
        for row_idx in range(X_working.shape[0]):
            ranks = np.argsort(np.argsort(X_working[row_idx, :]))
            X_normalized[row_idx, :] = col_means[0, ranks]

    return X_normalized.astype(np.float64)


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
    # Get dense version for statistics
    X_dense = X.toarray() if sp.issparse(X) else X  # type: ignore[union-attr]
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
