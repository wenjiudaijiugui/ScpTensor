"""Benchmark evaluators for computing quality metrics.

Evaluators compute metrics on processed data to assess the quality
of analysis methods such as normalization, imputation, or integration.
"""

from .biological import BaseEvaluator, BiologicalEvaluator, evaluate_biological
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
]
