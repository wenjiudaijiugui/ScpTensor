"""Benchmark evaluators for computing quality metrics.

Evaluators compute metrics on processed data to assess the quality
of analysis methods such as normalization, imputation, or integration.
"""

from .accuracy import (
    AccuracyEvaluator,
    AccuracyResult,
    evaluate_accuracy,
    evaluate_classification_accuracy,
    evaluate_regression_accuracy,
)
from .biological import BaseEvaluator, BiologicalEvaluator, evaluate_biological
from .clustering_metrics import (
    ClusteringEvaluator,
    compare_pca_variance_explained,
    compare_umap_embedding_quality,
    compute_clustering_ari,
    compute_clustering_nmi,
    compute_clustering_silhouette,
)
from .parameter_sensitivity import (
    ParameterSensitivityEvaluator,
    SensitivityResult,
    evaluate_parameter_sensitivity,
    get_parameter_spec,
    get_supported_parameters,
)
from .performance import (
    PerformanceEvaluator,
    PerformanceResult,
    benchmark_scalability,
    evaluate_performance,
)

__all__ = [
    "BaseEvaluator",
    "BiologicalEvaluator",
    "evaluate_biological",
    "ParameterSensitivityEvaluator",
    "SensitivityResult",
    "evaluate_parameter_sensitivity",
    "get_supported_parameters",
    "get_parameter_spec",
    "PerformanceEvaluator",
    "PerformanceResult",
    "evaluate_performance",
    "benchmark_scalability",
    "AccuracyEvaluator",
    "AccuracyResult",
    "evaluate_accuracy",
    "evaluate_regression_accuracy",
    "evaluate_classification_accuracy",
    "ClusteringEvaluator",
    "compute_clustering_ari",
    "compute_clustering_nmi",
    "compute_clustering_silhouette",
    "compare_pca_variance_explained",
    "compare_umap_embedding_quality",
]
