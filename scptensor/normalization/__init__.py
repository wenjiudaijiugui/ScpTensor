"""Normalization methods for ScpTensor.

This module provides *normalization-only* functions for DIA-based single-cell proteomics
data preprocessing.

Available Methods:
    norm_none: Explicit no-op normalization (passthrough layer)
    norm_mean: Mean centering/scaling normalization
    norm_median: Median centering/scaling normalization (robust to outliers)
    norm_quantile: Quantile normalization (aligns sample distributions)
    norm_trqn: Tail-robust quantile normalization for protein-level matrices
    normalize: Unified normalization dispatcher

Note
----
Log transformation is categorized under :mod:`scptensor.transformation`.
"""

from .api import norm_none, normalize
from .base import (
    apply_normalization,
    create_result_layer,
    ensure_dense,
    get_layer_name,
    log_operation,
    validate_assay_and_layer,
)
from .mean_normalization import norm_mean
from .median_normalization import norm_median
from .quantile_normalization import norm_quantile
from .trqn_normalization import norm_trqn

__all__ = [
    # Public API
    "norm_none",
    "norm_mean",
    "norm_median",
    "norm_quantile",
    "norm_trqn",
    "normalize",
    # Base utilities (for internal use)
    "apply_normalization",
    "create_result_layer",
    "ensure_dense",
    "get_layer_name",
    "log_operation",
    "validate_assay_and_layer",
]
