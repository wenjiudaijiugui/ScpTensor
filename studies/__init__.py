"""Studies module - pipeline comparison utilities."""

from .evaluation.core import (
    compute_distribution_change,
    compute_pca_variance,
    compute_sparsity,
    compute_statistics,
    evaluate_pipeline,
    monitor_performance,
)
from .pipelines.executor import (
    PIPELINE_CONFIGS,
    get_available_pipelines,
    get_pipeline_description,
    run_pipeline,
)

__all__ = [
    # Pipeline
    "run_pipeline",
    "get_available_pipelines",
    "get_pipeline_description",
    "PIPELINE_CONFIGS",
    # Evaluation
    "compute_sparsity",
    "compute_statistics",
    "compute_distribution_change",
    "compute_pca_variance",
    "evaluate_pipeline",
    "monitor_performance",
]
