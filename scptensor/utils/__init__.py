"""Utility functions for single-cell proteomics data analysis.

This module provides various utility functions including:
- Data generation for synthetic datasets
- Statistical operations (correlation, similarity)
- Data transformations (asinh, logicle, quantile normalization)
- Batch processing for large datasets
"""

from .batch import (
    BatchProcessor,
    apply_by_batch,
    batch_apply_along_axis,
    batch_iterator,
)
from .data_generator import ScpDataGenerator
from .stats import (
    correlation_matrix,
    cosine_similarity,
    partial_correlation,
    spearman_correlation,
)
from .transform import (
    asinh_transform,
    logicle_transform,
    quantile_normalize,
    robust_scale,
)

__all__ = [
    # Data generation
    "ScpDataGenerator",
    # Statistics
    "correlation_matrix",
    "partial_correlation",
    "spearman_correlation",
    "cosine_similarity",
    # Transformations
    "asinh_transform",
    "logicle_transform",
    "quantile_normalize",
    "robust_scale",
    # Batch processing
    "batch_iterator",
    "apply_by_batch",
    "batch_apply_along_axis",
    "BatchProcessor",
]
