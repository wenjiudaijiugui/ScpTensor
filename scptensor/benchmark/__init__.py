# flake8: noqa
# Benchmark module exports

from .benchmark_suite import BenchmarkSuite
from .competitor_benchmark import (
    COMPETITOR_REGISTRY,
    ScanpyStyleOps,
    get_competitor,
    get_competitors_by_operation,
    list_competitors,
)
from .competitor_suite import CompetitorBenchmarkSuite, ComparisonResult
from .competitor_viz import CompetitorResultVisualizer
from .core import BenchmarkResults, MethodRunResult
from .metrics import (
    BiologicalMetrics,
    ComputationalMetrics,
    MetricsEngine,
    TechnicalMetrics,
)
from .parameter_grid import (
    MethodConfig,
    ParameterGrid,
    create_method_configs,
    create_normalization_parameter_grids,
)
from .scptensor_methods import (
    SCPTENSOR_METHODS,
    ScpTensorKNNImputer,
    ScpTensorKMeans,
    ScpTensorLogNormalize,
    ScpTensorPCA,
    ScpTensorSVDImputer,
    get_scptensor_method,
    list_scptensor_methods,
)
from .synthetic_data import SyntheticDataset
from .visualization import ResultsVisualizer

# Scanpy comparison
from .data_provider import DataProvider, ComparisonDataset, COMPARISON_DATASETS, get_provider
from .method_registry import MethodRegistry, MethodCategory, ComparisonLayer, get_registry
from .comparison_engine import ComparisonEngine, MethodResult, ComparisonResult, get_engine
from .comparison_viz import ComparisonVisualizer, PlotStyle, configure_plots, get_visualizer
from .report_generator import ReportGenerator, get_report_generator

# Configuration
from .config import (
    BenchmarkConfig,
    ChartConfig,
    ModuleConfigEntry,
    OutputConfig,
    get_default_config,
    load_charts_config,
    load_config,
    save_config,
)

# Modules
from .modules import BaseModule, ModuleConfig, ModuleResult, ClusteringTestModule
from .evaluators import (
    BaseEvaluator,
    BiologicalEvaluator,
    evaluate_biological,
    ParameterSensitivityEvaluator,
    evaluate_parameter_sensitivity,
    get_parameter_spec,
    get_supported_parameters,
    PerformanceEvaluator,
    PerformanceResult,
    evaluate_performance,
    benchmark_scalability,
)

__all__ = [
    # Core classes
    "BenchmarkSuite",
    "BenchmarkResults",
    "MethodRunResult",
    # Metrics
    "TechnicalMetrics",
    "BiologicalMetrics",
    "ComputationalMetrics",
    "MetricsEngine",
    # Parameter optimization
    "ParameterGrid",
    "MethodConfig",
    "create_method_configs",
    "create_normalization_parameter_grids",
    # Data generation
    "SyntheticDataset",
    # Visualization
    "ResultsVisualizer",
    # Competitor benchmarks
    "CompetitorBenchmarkSuite",
    "ComparisonResult",
    "CompetitorResultVisualizer",
    "COMPETITOR_REGISTRY",
    "ScanpyStyleOps",
    "list_competitors",
    "get_competitor",
    "get_competitors_by_operation",
    # ScpTensor methods for benchmarking
    "SCPTENSOR_METHODS",
    "ScpTensorLogNormalize",
    "ScpTensorKNNImputer",
    "ScpTensorSVDImputer",
    "ScpTensorPCA",
    "ScpTensorKMeans",
    "list_scptensor_methods",
    "get_scptensor_method",
    # Scanpy comparison
    "DataProvider",
    "ComparisonDataset",
    "COMPARISON_DATASETS",
    "get_provider",
    "MethodRegistry",
    "MethodCategory",
    "ComparisonLayer",
    "get_registry",
    "ComparisonEngine",
    "MethodResult",
    "get_engine",
    "ComparisonVisualizer",
    "PlotStyle",
    "configure_plots",
    "get_visualizer",
    "ReportGenerator",
    "get_report_generator",
    # Configuration
    "BenchmarkConfig",
    "ChartConfig",
    "ModuleConfigEntry",
    "OutputConfig",
    "load_config",
    "load_charts_config",
    "save_config",
    "get_default_config",
    # Modules
    "BaseModule",
    "ModuleConfig",
    "ModuleResult",
    "ClusteringTestModule",
    # Evaluators
    "BaseEvaluator",
    "BiologicalEvaluator",
    "evaluate_biological",
    "ParameterSensitivityEvaluator",
    "evaluate_parameter_sensitivity",
    "get_parameter_spec",
    "get_supported_parameters",
    "PerformanceEvaluator",
    "PerformanceResult",
    "evaluate_performance",
    "benchmark_scalability",
]
