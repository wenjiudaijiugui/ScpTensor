"""Quality metrics module for automatic method selection.

This module provides quality metrics for evaluating data processing
effectiveness in the automatic method selection system.

Available Metrics
-----------------
cv_stability
    Coefficient of variation stability across features
skewness_improvement
    Improvement in distribution skewness after transformation
kurtosis_improvement
    Improvement in distribution kurtosis after transformation
dynamic_range
    Appropriateness of data dynamic range
outlier_ratio
    Proportion of non-outlier values

All metrics return values in the range [0, 1], where higher values
indicate better quality.

Example
-------
>>> import numpy as np
>>> from scptensor.autoselect.metrics import cv_stability, outlier_ratio
>>> X = np.random.randn(100, 10)
>>> stability = cv_stability(X)
>>> outliers = outlier_ratio(X)
>>> print(f"Stability: {stability:.3f}, Outliers: {outliers:.3f}")
"""

from __future__ import annotations

from scptensor.autoselect.metrics.quality import (
    cv_stability,
    dynamic_range,
    kurtosis_improvement,
    outlier_ratio,
    skewness_improvement,
)

__all__ = [
    "cv_stability",
    "skewness_improvement",
    "kurtosis_improvement",
    "dynamic_range",
    "outlier_ratio",
]
