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
    # Handle edge cases
    if X.size == 0 or X.shape[0] < 2 or X.shape[1] < 1:
        return 0.0

    # Handle NaN values
    X_clean = X[~np.isnan(X).any(axis=1)] if X.ndim > 1 else X[~np.isnan(X)]
    if X_clean.size == 0 or X_clean.shape[0] < 2:
        return 0.0

    # Calculate CV for each feature (column)
    means = np.mean(X_clean, axis=0)
    stds = np.std(X_clean, axis=0, ddof=1)

    # Avoid division by zero
    valid_mask = means > _EPS
    if not np.any(valid_mask):
        return 0.0

    cvs = np.zeros_like(means)
    cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

    # Filter out invalid CVs (from zero or near-zero means)
    cvs_valid = cvs[valid_mask]

    if len(cvs_valid) < 2:
        return 0.0

    # Calculate stability: low std(CVs) / mean(CVs) = high stability
    cv_mean = np.mean(cvs_valid)
    cv_std = np.std(cvs_valid, ddof=1)

    if cv_mean < _EPS:
        return 0.0

    # Stability score: 1 - coefficient of variation of CVs
    stability = 1.0 - min(cv_std / cv_mean, 1.0)

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
    # Check shape match
    if X_before.shape != X_after.shape:
        raise ValueError(f"Shape mismatch: X_before {X_before.shape} vs X_after {X_after.shape}")

    # Handle edge cases
    if X_before.size == 0:
        return 0.0

    # Handle NaN values
    mask = ~np.isnan(X_before) & ~np.isnan(X_after)
    if not np.any(mask):
        return 0.0

    # Calculate absolute skewness before and after
    # Use Fisher's definition (normal => skewness = 0)
    skew_before = np.abs(skew(X_before[mask], axis=None, nan_policy="omit"))
    skew_after = np.abs(skew(X_after[mask], axis=None, nan_policy="omit"))

    # Handle NaN skewness (e.g., constant data)
    if np.isnan(skew_before) or np.isnan(skew_after):
        return 0.0

    # Calculate improvement
    # If skewness improved (decreased), score > 0
    # If skewness got worse (increased), score < 0 (will be clamped to 0)
    if skew_before < _EPS:
        # Already perfect, no room for improvement
        return 1.0 if skew_after < _EPS else 0.0

    improvement = (skew_before - skew_after) / skew_before

    # Clamp to [0, 1]
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
    # Check shape match
    if X_before.shape != X_after.shape:
        raise ValueError(f"Shape mismatch: X_before {X_before.shape} vs X_after {X_after.shape}")

    # Handle edge cases
    if X_before.size == 0:
        return 0.0

    # Handle NaN values
    mask = ~np.isnan(X_before) & ~np.isnan(X_after)
    if not np.any(mask):
        return 0.0

    # Calculate deviation from normal kurtosis (3) before and after
    # Use Fisher's definition (normal => kurtosis = 0, so we add 3)
    kurt_before = np.abs(kurtosis(X_before[mask], axis=None, fisher=True, nan_policy="omit") + 3)
    kurt_after = np.abs(kurtosis(X_after[mask], axis=None, fisher=True, nan_policy="omit") + 3)

    # Handle NaN kurtosis
    if np.isnan(kurt_before) or np.isnan(kurt_after):
        return 0.0

    # Calculate improvement (reduction in deviation from 3)
    if kurt_before < _EPS:
        # Already perfect
        return 1.0 if kurt_after < _EPS else 0.0

    improvement = (kurt_before - kurt_after) / kurt_before

    # Clamp to [0, 1]
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
    # Handle edge cases
    if X.size == 0:
        return 0.0

    # Take absolute value to handle negatives
    X_abs = np.abs(X)

    # Handle NaN values
    X_clean = X_abs[~np.isnan(X_abs)]
    if X_clean.size == 0:
        return 0.0

    # Find min and max (ignoring zeros for min)
    X_nonzero = X_clean[X_clean > _EPS]
    if X_nonzero.size == 0:
        return 0.0

    min_val = np.min(X_nonzero)
    max_val = np.max(X_clean)

    if min_val < _EPS or max_val < _EPS:
        return 0.0

    # Calculate dynamic range in orders of magnitude
    dyn_range = np.log10(max_val / min_val)

    # Score based on ideal range (2-10 orders of magnitude)
    # Use a bell curve centered at 6 with std of 3
    # Peak score at 6 orders of magnitude
    ideal_center = 6.0
    ideal_width = 3.0

    # Gaussian-like scoring
    score = np.exp(-0.5 * ((dyn_range - ideal_center) / ideal_width) ** 2)

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
    # Handle edge cases
    if X.size == 0:
        return 0.0

    # Flatten array for global outlier detection
    X_flat = X.flatten()

    # Handle NaN values
    X_clean = X_flat[~np.isnan(X_flat)]
    if X_clean.size == 0:
        return 0.0

    # Calculate quartiles
    Q1 = np.percentile(X_clean, 25)
    Q3 = np.percentile(X_clean, 75)
    IQR = Q3 - Q1

    # Handle case where IQR is 0 (all values same)
    if IQR < _EPS:
        # No outliers if all values are identical
        return 1.0

    # Identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (X_clean < lower_bound) | (X_clean > upper_bound)
    n_outliers = np.sum(outliers)
    n_total = len(X_clean)

    # Calculate score (fewer outliers = higher score)
    score = 1.0 - (n_outliers / n_total)

    return float(np.clip(score, 0.0, 1.0))
