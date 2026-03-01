"""Quality metrics for automatic method selection.

This module provides functions to compute various quality metrics for evaluating
data processing effectiveness in single-cell proteomics analysis.

All metrics return values in the range [0, 1], where higher values indicate
better quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kurtosis, skew

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Numerical stability constant
_EPS = 1e-10


def cv_stability(X: NDArray[np.float64]) -> float:
    """Calculate coefficient of variation stability.

    Computes the stability of coefficient of variation (CV) across features.
    More stable CVs (lower standard deviation relative to mean) result in
    higher scores.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix of shape (n_samples, n_features).

    Returns
    -------
    float
        Stability score in range [0, 1]. Higher values indicate more stable CVs.
        Returns 0.0 for empty arrays, single values, all zeros, or all NaNs.

    Notes
    -----
    CV (coefficient of variation) is calculated as std/mean for each feature.
    Stability is measured as 1 - (std(CVs) / mean(CVs)).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 10) * 10 + 100
    >>> score = cv_stability(X)
    >>> 0.0 <= score <= 1.0
    True
    """
    if X.size == 0 or X.shape[0] < 2:
        return 0.0

    # Use nanmean/nanstd to handle NaN automatically
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=1)

    # Calculate CVs for valid features (non-zero means and non-NaN stds)
    valid = ~(np.isnan(means) | np.isnan(stds) | (means < _EPS))
    if not np.any(valid):
        return 0.0

    cvs = stds[valid] / means[valid]

    # Calculate stability: low std(CVs) / mean(CVs) = high stability
    cv_mean = np.mean(cvs)
    if cv_mean < _EPS:
        return 0.0

    stability = 1.0 - min(np.std(cvs, ddof=1) / cv_mean, 1.0)
    return float(np.clip(stability, 0.0, 1.0))


def skewness_improvement(X_before: NDArray[np.float64], X_after: NDArray[np.float64]) -> float:
    """Calculate skewness improvement after transformation.

    Measures how much the skewness of the distribution has improved (moved
    closer to 0, indicating more symmetric distribution).

    Parameters
    ----------
    X_before : NDArray[np.float64]
        Data before transformation.
    X_after : NDArray[np.float64]
        Data after transformation. Must have same shape as X_before.

    Returns
    -------
    float
        Improvement score in range [0, 1]. Higher values indicate greater
        improvement in skewness. Returns 0.0 for empty arrays or no improvement.

    Raises
    ------
    ValueError
        If X_before and X_after have different shapes.

    Notes
    -----
    Skewness measures asymmetry of the distribution. A value of 0 indicates
    perfect symmetry. The improvement is calculated as the reduction in
    absolute skewness.

    Examples
    --------
    >>> import numpy as np
    >>> X_before = np.exp(np.random.randn(100, 5))  # Right-skewed
    >>> X_after = np.log1p(X_before)  # More symmetric
    >>> score = skewness_improvement(X_before, X_after)
    >>> 0.0 <= score <= 1.0
    True
    """
    if X_before.shape != X_after.shape:
        raise ValueError(f"Shape mismatch: X_before {X_before.shape} vs X_after {X_after.shape}")

    if X_before.size == 0:
        return 0.0

    # scipy.stats skew already has nan_policy='omit'
    skew_before = np.abs(skew(X_before, axis=None, nan_policy="omit"))
    skew_after = np.abs(skew(X_after, axis=None, nan_policy="omit"))

    # Handle NaN skewness (e.g., constant data)
    if np.isnan(skew_before) or np.isnan(skew_after):
        return 0.0

    # If already perfect, no room for improvement
    if skew_before < _EPS:
        return 1.0 if skew_after < _EPS else 0.0

    improvement = (skew_before - skew_after) / skew_before
    return float(np.clip(improvement, 0.0, 1.0))


def kurtosis_improvement(X_before: NDArray[np.float64], X_after: NDArray[np.float64]) -> float:
    """Calculate kurtosis improvement after transformation.

    Measures how much the kurtosis of the distribution has improved (moved
    closer to 3, indicating normal-like tails).

    Parameters
    ----------
    X_before : NDArray[np.float64]
        Data before transformation.
    X_after : NDArray[np.float64]
        Data after transformation. Must have same shape as X_before.

    Returns
    -------
    float
        Improvement score in range [0, 1]. Higher values indicate greater
        improvement in kurtosis. Returns 0.0 for empty arrays or no improvement.

    Raises
    ------
    ValueError
        If X_before and X_after have different shapes.

    Notes
    -----
    Kurtosis measures the "tailedness" of the distribution. A value of 3
    indicates normal distribution tails (using Fisher's definition). Values
    > 3 indicate heavy tails, < 3 indicate light tails.

    Examples
    --------
    >>> import numpy as np
    >>> X_before = np.random.standard_t(df=3, size=(100, 5))  # Heavy tails
    >>> X_after = np.random.randn(100, 5)  # Normal tails
    >>> score = kurtosis_improvement(X_before, X_after)
    >>> 0.0 <= score <= 1.0
    True
    """
    if X_before.shape != X_after.shape:
        raise ValueError(f"Shape mismatch: X_before {X_before.shape} vs X_after {X_after.shape}")

    if X_before.size == 0:
        return 0.0

    # Calculate deviation from normal kurtosis (3)
    kurt_before = np.abs(kurtosis(X_before, axis=None, fisher=True, nan_policy="omit") + 3)
    kurt_after = np.abs(kurtosis(X_after, axis=None, fisher=True, nan_policy="omit") + 3)

    if np.isnan(kurt_before) or np.isnan(kurt_after):
        return 0.0

    if kurt_before < _EPS:
        return 1.0 if kurt_after < _EPS else 0.0

    improvement = (kurt_before - kurt_after) / kurt_before
    return float(np.clip(improvement, 0.0, 1.0))


def dynamic_range(X: NDArray[np.float64]) -> float:
    """Calculate dynamic range appropriateness.

    Evaluates whether the dynamic range (span of values) is appropriate
    for single-cell proteomics data. Ideal range is 2-10 orders of magnitude.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix. Negative values are handled by taking absolute values.

    Returns
    -------
    float
        Appropriateness score in range [0, 1]. Higher values indicate more
        appropriate dynamic range. Returns 0.0 for empty arrays or all zeros.

    Notes
    -----
    Dynamic range is calculated as log10(max/min). The ideal range is
    approximately 2-10 orders of magnitude. The scoring function uses a
    bell curve centered around 6 orders of magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.lognormal(mean=0, sigma=1, size=(100, 5))
    >>> score = dynamic_range(X)
    >>> 0.0 <= score <= 1.0
    True
    """
    if X.size == 0:
        return 0.0

    # Take absolute value and filter NaN + zeros
    X_nonzero = np.abs(X)[(np.abs(X) > _EPS) & ~np.isnan(X)]
    if X_nonzero.size == 0:
        return 0.0

    # Calculate dynamic range in orders of magnitude
    dyn_range = np.log10(np.max(X_nonzero) / np.min(X_nonzero))

    # Gaussian-like scoring centered at 6 orders of magnitude
    score = np.exp(-0.5 * ((dyn_range - 6.0) / 3.0) ** 2)
    return float(np.clip(score, 0.0, 1.0))


def outlier_ratio(X: NDArray[np.float64]) -> float:
    """Calculate outlier ratio score.

    Measures the proportion of non-outlier values using the IQR method.
    Higher scores indicate fewer outliers (better quality).

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data matrix.

    Returns
    -------
    float
        Outlier score in range [0, 1]. Higher values indicate fewer outliers.
        Returns 0.0 for empty arrays.

    Notes
    -----
    Outliers are detected using the IQR method:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Outliers: values < (Q1 - 1.5*IQR) or > (Q3 + 1.5*IQR)

    Score = 1 - (n_outliers / n_total)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> score = outlier_ratio(X)
    >>> 0.0 <= score <= 1.0
    True
    """
    if X.size == 0:
        return 0.0

    # Flatten and filter NaN
    X_clean = X.flatten()
    X_clean = X_clean[~np.isnan(X_clean)]

    if X_clean.size == 0:
        return 0.0

    # Calculate quartiles
    Q1, Q3 = np.percentile(X_clean, [25, 75])
    IQR = Q3 - Q1

    # No outliers if all values are identical
    if IQR < _EPS:
        return 1.0

    # Identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (X_clean < lower_bound) | (X_clean > upper_bound)

    score = 1.0 - np.sum(outliers) / len(X_clean)
    return float(np.clip(score, 0.0, 1.0))
