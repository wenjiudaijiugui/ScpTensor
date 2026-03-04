"""Utility functions for DIA-based single-cell proteomics data analysis.

This module provides various utility functions including:
- Data generation for synthetic datasets
- Statistical operations (correlation, similarity)
- Data transformations (quantile normalization, robust scaling)
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
    "quantile_normalize",
    "robust_scale",
    # Batch processing
    "batch_iterator",
    "apply_by_batch",
    "batch_apply_along_axis",
    "BatchProcessor",
]
