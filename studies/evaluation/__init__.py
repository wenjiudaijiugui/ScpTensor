"""
Evaluation metrics for pipeline comparison.

This package provides comprehensive metrics for evaluating single-cell
proteomics analysis pipelines across four dimensions:
- Batch effect removal
- Computational performance
- Data distribution changes
- Data structure preservation
"""

from .batch_effects import (
    compute_kbet,
    compute_lisi,
    compute_mixing_entropy,
    compute_variance_ratio,
)
from .distribution import (
    compute_quantiles,
    compute_sparsity,
    compute_statistics,
    distribution_test,
)
from .metrics import PipelineEvaluator
from .performance import (
    compute_efficiency_score,
    estimate_complexity,
    monitor_performance,
)
from .structure import (
    compute_distance_preservation,
    compute_global_structure,
    compute_nn_consistency,
    compute_pca_variance,
)

__all__ = [
    # Main evaluator
    "PipelineEvaluator",
    # Batch effect metrics
    "compute_kbet",
    "compute_lisi",
    "compute_mixing_entropy",
    "compute_variance_ratio",
    # Performance metrics
    "monitor_performance",
    "compute_efficiency_score",
    "estimate_complexity",
    # Distribution metrics
    "compute_sparsity",
    "compute_statistics",
    "distribution_test",
    "compute_quantiles",
    # Structure metrics
    "compute_pca_variance",
    "compute_nn_consistency",
    "compute_distance_preservation",
    "compute_global_structure",
]
