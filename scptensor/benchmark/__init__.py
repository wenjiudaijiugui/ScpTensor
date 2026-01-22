# flake8: noqa
# Benchmark module exports

# Core result classes
from .core import (
    AccuracyMetrics,
    BenchmarkResult,
    BenchmarkResults,
    MethodCategory,
    MethodSpec,
    PerformanceMetrics,
    ComparisonLayer,
    BiologicalMetrics,
    ComparisonResult,
)

# Data generation
from .synthetic_data import SyntheticDataset

# Competitor benchmark
from .competitor_benchmark import (
    COMPETITOR_REGISTRY,
    ScanpyStyleOps,
    get_competitor,
    get_competitors_by_operation,
    list_competitors,
)
from .competitor_suite import CompetitorBenchmarkSuite
from .competitor_viz import CompetitorResultVisualizer

# Display and styling
from .display.common import PlotStyle, configure_plots
from .display.report import BenchmarkReportGenerator as ReportGenerator

# Scanpy comparison framework
from .data_provider import ComparisonDataset, COMPARISON_DATASETS, DataProvider, get_provider
from .method_registry import (
    ComparisonLayer as RegistryComparisonLayer,
    MethodCategory as RegistryMethodCategory,
    MethodRegistry,
    get_registry,
)
from .comparison_engine import ComparisonEngine, MethodResult, get_engine

# Testing modules
from .modules import BaseModule, ClusteringTestModule, ModuleConfig, ModuleResult

# Evaluators
from .evaluators import (
    BaseEvaluator,
    BiologicalEvaluator,
    evaluate_biological,
    evaluate_performance,
    benchmark_scalability,
    ParameterSensitivityEvaluator,
    evaluate_parameter_sensitivity,
    get_parameter_spec,
    get_supported_parameters,
    PerformanceEvaluator,
    PerformanceResult,
)

__all__ = [
    # Core result classes
    "BenchmarkResults",
    "BenchmarkResult",
    "MethodSpec",
    "PerformanceMetrics",
    "AccuracyMetrics",
    "BiologicalMetrics",
    "MethodCategory",
    "ComparisonLayer",
    "ComparisonResult",
    # Data generation
    "SyntheticDataset",
    # Competitor benchmark
    "CompetitorBenchmarkSuite",
    "CompetitorResultVisualizer",
    "COMPETITOR_REGISTRY",
    "ScanpyStyleOps",
    "list_competitors",
    "get_competitor",
    "get_competitors_by_operation",
    # Scanpy comparison framework
    "DataProvider",
    "ComparisonDataset",
    "COMPARISON_DATASETS",
    "get_provider",
    "MethodRegistry",
    "RegistryMethodCategory",
    "RegistryComparisonLayer",
    "get_registry",
    "ComparisonEngine",
    "MethodResult",
    "get_engine",
    "PlotStyle",
    "configure_plots",
    "ReportGenerator",
    # Testing modules
    "BaseModule",
    "ClusteringTestModule",
    "ModuleConfig",
    "ModuleResult",
    # Evaluators
    "BaseEvaluator",
    "BiologicalEvaluator",
    "evaluate_biological",
    "evaluate_performance",
    "benchmark_scalability",
    "ParameterSensitivityEvaluator",
    "evaluate_parameter_sensitivity",
    "get_parameter_spec",
    "get_supported_parameters",
    "PerformanceEvaluator",
    "PerformanceResult",
]
