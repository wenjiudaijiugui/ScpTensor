"""Normalization methods for ScpTensor.

This module provides normalization and transformation functions for
single-cell proteomics data preprocessing.

Available Methods:
    log_transform: Logarithmic transformation (base configurable)
    norm_mean: Mean centering/scaling normalization
    norm_median: Median centering/scaling normalization (robust to outliers)
    norm_quantile: Quantile normalization (aligns distributions)

Common Workflow:
    >>> from scptensor.normalization import log_transform, norm_quantile
    >>> container = log_transform(container, base=2.0)
    >>> container = norm_quantile(container)

For advanced usage, see individual function documentation.
"""

from .base import (
    apply_normalization,
    create_result_layer,
    ensure_dense,
    get_layer_name,
    log_operation,
    validate_assay_and_layer,
)
from .log_transform import log_transform
from .mean_normalization import norm_mean
from .median_normalization import norm_median
from .quantile_normalization import norm_quantile

__all__ = [
    # Public API
    "log_transform",
    "norm_mean",
    "norm_median",
    "norm_quantile",
    # Base utilities (for internal use)
    "apply_normalization",
    "create_result_layer",
    "ensure_dense",
    "get_layer_name",
    "log_operation",
    "validate_assay_and_layer",
]
