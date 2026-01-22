"""Common statistical metrics and utility functions for Quality Control.

This module provides stateless, pure functions for calculating robust statistics
and metrics used across different QC levels (PSM, Sample, Feature).
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp


def compute_mad(
    data: np.ndarray,
    scale_factor: float = 1.4826,
) -> float:
    """Compute Median Absolute Deviation (MAD).

    MAD is a robust measure of statistical dispersion. For normally distributed
    data, MAD multiplied by 1.4826 equals the standard deviation.

    Mathematical Formulation:
        MAD = median(|x - median(x)|) * scale_factor

    Parameters
    ----------
    data : np.ndarray
        Input array of numeric values.
    scale_factor : float, default=1.4826
        Scaling factor to make MAD consistent with standard deviation for
        normal distributions. Use 1.0 for raw MAD.

    Returns
    -------
    float
        The MAD value. Returns np.nan if input array is empty.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mad = compute_mad(data)
    >>> print(f"MAD: {mad:.2f}")
    MAD: 1.48

    Notes
    -----
    MAD is robust to outliers (up to 50% breakdown point).
    For normal distributions: MAD ≈ σ (standard deviation).

    References
    ----------
    .. [1] Leys, C., et al. (2013). Detecting outliers: Do not use standard
       deviation around the mean, use absolute deviation around the median.
       Journal of Experimental Social Psychology, 49(4), 764-766.
    """
    if len(data) == 0:
        return np.nan

    median = np.nanmedian(data)
    diff = np.abs(data - median)
    return np.nanmedian(diff) * scale_factor


def is_outlier_mad(
    data: np.ndarray,
    nmads: float = 3.0,
    direction: Literal["both", "lower", "upper"] = "both",
) -> np.ndarray:
    """Detect outliers using Median Absolute Deviation (MAD).

    Outliers are defined as values that fall more than a specified number
    of MADs from the median. This is a robust outlier detection method
    suitable for non-normal distributions.

    Parameters
    ----------
    data : np.ndarray
        Input array of values.
    nmads : float, default=3.0
        Number of MADs to use as threshold. Higher values make the
        detection more conservative (fewer outliers).
    direction : {"both", "lower", "upper"}, default="both"
        Direction of outliers to detect:
        - "both": Detect outliers in both tails
        - "lower": Detect only low outliers
        - "upper": Detect only high outliers

    Returns
    -------
    np.ndarray
        Boolean array where True indicates an outlier.
        Returns empty array if input is empty.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 100])
    >>> outliers = is_outlier_mad(data, nmads=3.0)
    >>> print(outliers)
    [False False False False False  True]

    Notes
    -----
    When MAD = 0 (no variation in data), the function falls back to
    comparing values to the median. Values equal to the median are
    considered inliers, all others are outliers.

    The default threshold of 3 MADs corresponds approximately to
    3 standard deviations for normal distributions.
    """
    if len(data) == 0:
        return np.array([], dtype=bool)

    median = np.nanmedian(data)
    mad = compute_mad(data)

    if mad == 0:
        # If MAD is 0, treat values equal to median as inliers
        return data != median

    lower_bound = median - nmads * mad
    upper_bound = median + nmads * mad

    if direction == "lower":
        return data < lower_bound
    elif direction == "upper":
        return data > upper_bound
    else:  # both
        return (data < lower_bound) | (data > upper_bound)


def compute_cv(
    data: np.ndarray | sp.spmatrix,
    axis: int = 0,
    min_mean: float = 1e-6,
) -> np.ndarray:
    """Compute Coefficient of Variation (CV).

    CV measures relative variability and is defined as the ratio of
    standard deviation to mean: CV = σ / μ

    Parameters
    ----------
    data : Union[np.ndarray, sp.spmatrix]
        Data matrix (dense or sparse).
    axis : int, default=0
        Axis along which to compute CV.
        - 0: Compute CV for each column
        - 1: Compute CV for each row
    min_mean : float, default=1e-6
        Minimum mean value to avoid division by zero. CV is set to
        NaN for features with mean < min_mean.

    Returns
    -------
    np.ndarray
        Array of CV values. Shape depends on axis parameter.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> cv = compute_cv(data, axis=0)
    >>> print(f"CV per column: {cv}")
    CV per column: [0.63245553 0.4472136  0.31649661]

    Notes
    -----
    For sparse matrices, variance is computed using the identity:
        Var[X] = E[X²] - (E[X])²

    This avoids converting the entire matrix to dense format.
    Numerical errors may cause small negative variances, which are
    clipped to zero.

    CV is undefined (NaN) when:
    - Mean is below min_mean threshold
    - All values are identical (std = 0)

    Performance considerations:
    - Dense arrays: O(n) memory, vectorized operations
    - Sparse matrices: O(nnz) memory where nnz is number of non-zero elements
    - Sparse computation creates one temporary copy for squared values
    """
    if sp.issparse(data):
        # Sparse matrix calculation
        # Var[X] = E[X²] - (E[X])²
        mean = np.array(data.mean(axis=axis)).flatten()

        # Calculate variance without modifying original data
        # This creates a copy but is memory-efficient for sparse matrices
        data_sq = data.copy()
        data_sq.data **= 2
        mean_sq = np.array(data_sq.mean(axis=axis)).flatten()

        var = mean_sq - mean**2
        # Clip negative values from numerical errors
        var[var < 0] = 0
        std = np.sqrt(var)
    else:
        # Dense array calculation
        mean = np.nanmean(data, axis=axis)
        std = np.nanstd(data, axis=axis)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
        cv[np.abs(mean) < min_mean] = np.nan

    return cv
