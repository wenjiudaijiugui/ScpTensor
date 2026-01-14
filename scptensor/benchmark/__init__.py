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
]
