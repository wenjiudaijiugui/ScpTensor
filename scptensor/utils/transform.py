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


if __name__ == "__main__":
    """Run tests to verify functionality."""
    print("Running tests for transform module...")

    # Test data
    np.random.seed(42)
    X_dense = np.abs(np.random.randn(20, 10)) * 10
    X_sparse = sp.csr_matrix(X_dense)
    X_with_negatives = np.random.randn(20, 10) * 5

    # Test 1: asinh_transform
    print("\n1. Testing asinh_transform...")
    X_asinh = asinh_transform(X_dense, cofactor=5.0)
    assert X_asinh.shape == X_dense.shape
    assert X_asinh is not X_dense
    assert np.all(np.argsort(X_dense.ravel()) == np.argsort(X_asinh.ravel()))
    print(f"   Input shape: {X_dense.shape}")
    print(f"   Output shape: {X_asinh.shape}")
    print(f"   Sample values: {X_asinh[0, :3]}")

    # Test 2: asinh_transform in-place
    print("\n2. Testing asinh_transform (no copy)...")
    X_copy = X_dense.copy()
    X_asinh_no_copy = asinh_transform(X_copy, cofactor=5.0, copy=False)
    assert X_asinh_no_copy is X_copy
    print(f"   Modified in place: {X_asinh_no_copy is X_copy}")

    # Test 3: asinh_transform with sparse
    print("\n3. Testing asinh_transform (sparse)...")
    X_asinh_sparse = asinh_transform(X_sparse, cofactor=5.0)
    assert sp.issparse(X_asinh_sparse)
    print(f"   Sparse format preserved: {sp.issparse(X_asinh_sparse)}")

    # Test 4: asinh_transform with negative values
    print("\n4. Testing asinh_transform (with negatives)...")
    X_asinh_neg = asinh_transform(X_with_negatives, cofactor=5.0)
    assert np.all(np.isfinite(X_asinh_neg))
    print(f"   Negative values handled: {np.min(X_asinh_neg):.4f} to {np.max(X_asinh_neg):.4f}")

    # Test 5: logicle_transform
    print("\n5. Testing logicle_transform...")
    X_logicle = logicle_transform(X_with_negatives)
    assert X_logicle.shape == X_with_negatives.shape
    print(f"   Shape: {X_logicle.shape}")

    # Test 6: quantile_normalize
    print("\n6. Testing quantile_normalize...")
    X_qn = quantile_normalize(X_dense[:10, :5], axis=0)
    assert X_qn.shape == (10, 5)
    X_qn_sorted = np.sort(X_qn, axis=0)
    col_means = X_qn_sorted.mean(axis=1)
    assert np.allclose(X_qn_sorted[:, 0], col_means, atol=1e-10)
    print(f"   Shape: {X_qn.shape}")
    print(f"   Columns normalized: {np.allclose(X_qn_sorted[:, 0], X_qn_sorted[:, 1])}")

    # Test 7: quantile_normalize along rows
    print("\n7. Testing quantile_normalize (axis=1)...")
    X_qn_rows = quantile_normalize(X_dense[:10, :5], axis=1)
    assert X_qn_rows.shape == (10, 5)
    print(f"   Shape: {X_qn_rows.shape}")

    # Test 8: robust_scale
    print("\n8. Testing robust_scale...")
    X_with_outlier = X_dense.copy()
    X_with_outlier[0, 0] = 1000
    X_robust = robust_scale(X_with_outlier, axis=0)
    assert X_robust.shape == X_with_outlier.shape
    print(f"   Shape: {X_robust.shape}")
    print(f"   Median-centered: {np.allclose(np.median(X_robust, axis=0), 0, atol=0.1)}")

    # Test 9: robust_scale with pre-computed statistics
    print("\n9. Testing robust_scale (pre-computed stats)...")
    center = np.median(X_dense, axis=0)
    q75, q25 = np.percentile(X_dense, [75, 25], axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    X_robust2 = robust_scale(X_dense, center=center, scale=scale)
    print("   Applied pre-computed statistics")

    # Test 10: Error handling
    print("\n10. Testing error handling...")
    try:
        asinh_transform(X_dense, cofactor=-1)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"   Correct error raised: {e}")

    try:
        quantile_normalize(X_dense, axis=2)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"   Correct error raised: {e}")

    print("\n All tests passed for transform module!")
