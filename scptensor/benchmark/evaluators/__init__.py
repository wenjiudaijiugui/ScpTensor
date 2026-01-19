"""Benchmark evaluators for computing quality metrics.

Evaluators compute metrics on processed data to assess the quality
of analysis methods such as normalization, imputation, or integration.
"""

from .biological import BaseEvaluator, BiologicalEvaluator, evaluate_biological

__all__ = ["BaseEvaluator", "BiologicalEvaluator", "evaluate_biological"]
