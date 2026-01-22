"""Core result dataclasses for the benchmark module.

This module provides structured dataclasses for storing and comparing
benchmark results between ScpTensor and Scanpy frameworks.

Main exports:
- MethodCategory, ComparisonLayer: Enums for method categorization
- MethodSpec: Method specification metadata
- PerformanceMetrics, AccuracyMetrics, BiologicalMetrics: Metric containers
- BenchmarkResult: Single method benchmark result
- ComparisonResult: Comparison between two methods
- BenchmarkResults: Container for all benchmark results
"""

from .result import (
    AccuracyMetrics,
    BenchmarkResult,
    BenchmarkResults,
    BiologicalMetrics,
    ComparisonLayer,
    ComparisonResult,
    MethodCategory,
    MethodSpec,
    PerformanceMetrics,
)

__all__ = [
    "MethodCategory",
    "ComparisonLayer",
    "MethodSpec",
    "PerformanceMetrics",
    "AccuracyMetrics",
    "BiologicalMetrics",
    "BenchmarkResult",
    "ComparisonResult",
    "BenchmarkResults",
]
