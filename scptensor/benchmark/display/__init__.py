"""Display module for benchmark visualization and reporting.

This module provides a comprehensive interface for visualizing and reporting
benchmark results from ScpTensor preprocessing method comparisons with
competing frameworks.

The display module is organized into specialized submodules for different
preprocessing categories, each providing publication-quality visualizations
using SciencePlots style.

Architecture
------------
The display module follows a hierarchical structure:

- base: Abstract base classes for display handlers
- config: Configuration classes and enums for visualization settings
- normalization: Log and z-score normalization comparison displays
- imputation: KNN and ScpTensor-exclusive imputation method displays
- integration: Batch correction and integration method displays
- dim_reduction: PCA and UMAP dimensionality reduction displays
- qc: Quality control dashboard and missing type analysis displays
- end_to_end: Complete preprocessing pipeline comparison displays
- report: Markdown report generation with embedded figures
- regression: Performance regression detection for CI/CD

Main Exports
------------
Display Classes:
    - LogNormalizeDisplay: Log normalization comparison visualizations
    - ZScoreDisplay: Z-score normalization verification visualizations
    - KNNImputeDisplay: KNN imputation comparison visualizations
    - ExclusiveImputeDisplay: ScpTensor-exclusive imputation method visualizations
    - IntegrationDisplay: Batch correction/integration method visualizations
    - PCADisplay: PCA comparison visualizations
    - UMAPDisplay: UMAP comparison visualizations
    - QCDashboardDisplay: Comprehensive QC metrics dashboard
    - MissingTypeDisplay: MBR vs LOD missing value analysis displays
    - QCBatchDisplay: Batch effect and CV comparison displays
    - EndToEndDisplay: End-to-end pipeline comparison visualizations
    - BenchmarkReportGenerator: Comprehensive Markdown report generation
    - RegressionChecker: Performance regression detection
    - TrendChartGenerator: Performance trend chart generation

Configuration:
    - ReportConfig: Configuration for report generation
    - RegressionThreshold: Thresholds for regression detection

Data Classes:
    - NormalizationComparisonResult: Normalization comparison data
    - ZScoreVerificationResult: Z-score verification data
    - ImputationComparisonResult: Imputation comparison data
    - ExclusiveImputationResults: Aggregated exclusive imputation results
    - IntegrationComparisonResult: Integration method comparison data
    - IntegrationMetricsSummary: Integration metrics summary
    - PCAResult: PCA dimensionality reduction result
    - UMAPResult: UMAP dimensionality reduction result
    - DimReductionComparisonResult: Dimensionality reduction comparison data
    - QCComparisonResult: QC metrics comparison data
    - MissingTypeReport: Missing value type analysis report
    - BatchCVReport: Batch coefficient of variation report
    - PipelineResult: Complete pipeline execution result
    - PipelineStep: Single pipeline step information
    - ClusteringMetrics: Clustering quality metrics
    - IntermediateResults: Intermediate pipeline results
    - RegressionReport: Regression detection report
    - TrendDataPoint: Single trend chart data point

Backward Compatibility:
    - ComparisonVisualizer: Main visualization class (removed)
    - PlotStyle: Enum for matplotlib style presets (legacy)
    - configure_plots: Plot configuration function (legacy)
    - BenchmarkResults: Result data class (legacy)
    - MethodSpec, BenchmarkResult: Method result data (legacy)
    - MethodCategory, ComparisonLayer: Method categorization enums (legacy)

Examples
--------
Generate normalization comparison visualization:

>>> from scptensor.benchmark.display import LogNormalizeDisplay
>>> display = LogNormalizeDisplay()
>>> result = NormalizationComparisonResult(...)
>>> fig_path = display.render_distribution_flow(result)

Generate imputation comparison heatmap:

>>> from scptensor.benchmark.display import ExclusiveImputeDisplay
>>> display = ExclusiveImputeDisplay()
>>> results = ExclusiveImputationResults(...)
>>> fig_path = display.render_mse_heatmap(results, metric="mse")

Generate a comprehensive benchmark report:

>>> from scptensor.benchmark.display import BenchmarkReportGenerator
>>> generator = BenchmarkReportGenerator(output_dir="results")
>>> report_path = generator.generate(benchmark_results)

Check for performance regressions:

>>> from scptensor.benchmark.display import RegressionChecker, RegressionThreshold
>>> thresholds = RegressionThreshold(runtime_increase_pct=10.0)
>>> checker = RegressionChecker(thresholds=thresholds)
>>> report = checker.check_regression(current_results, baseline)
"""

from __future__ import annotations

# =============================================================================
# Base and Configuration
# =============================================================================
from scptensor.benchmark.display.base import (
    ComparisonDisplay,
    DisplayBase,
)
from scptensor.benchmark.display.common import (
    DEFAULT_LAYOUT,
    DEFAULT_TYPOGRAPHY,
    DIM_REDUCTION_COLORS,
    END_TO_END_COLORS,
    FEATURE_SELECTION_COLORS,
    IMPUTATION_COLORS,
    INTEGRATION_COLORS,
    NORMALIZATION_COLORS,
    QC_COLORS,
    ColorPalette,
    LayoutConfig,
    ModuleType,
    TypographyConfig,
    apply_color_style,
    apply_layout_config,
    apply_typography_theme,
    get_compatible_color,
    get_module_colors,
    get_status_color,
)
from scptensor.benchmark.display.config import (
    CATEGORY_METRICS,
    ComparisonLayer,
    MethodCategory,
    PlotStyle,
    ReportConfig,
    get_category_metrics,
    get_style_string,
)

# =============================================================================
# Imputation Display
# =============================================================================
from scptensor.benchmark.display.imputation import (
    ExclusiveImputationResults,
    ExclusiveImputeDisplay,
    ImputationComparisonResult,
    KNNImputeDisplay,
)
from scptensor.benchmark.display.imputation import (
    setup_plot_style as setup_imputation_plot_style,
)

# =============================================================================
# Integration Display
# =============================================================================
from scptensor.benchmark.display.integration import (
    IntegrationComparisonResult,
    IntegrationDisplay,
    IntegrationMetricsSummary,
)

# =============================================================================
# Normalization Display
# =============================================================================
from scptensor.benchmark.display.normalization import (
    LogNormalizeDisplay,
    NormalizationComparisonResult,
    ZScoreDisplay,
    ZScoreVerificationResult,
)
from scptensor.benchmark.display.normalization import (
    setup_plot_style as setup_normalization_plot_style,
)


# Legacy setup_plot_style for backward compatibility
def setup_integration_plot_style(dpi: int = 300) -> None:
    from scptensor.benchmark.display.config import setup_plot_style

    setup_plot_style(dpi)


# =============================================================================
# Dimensionality Reduction Display
# =============================================================================
from scptensor.benchmark.display.dim_reduction import (
    DimReductionComparisonResult,
    PCADisplay,
    PCAResult,
    UMAPDisplay,
    UMAPResult,
)

# =============================================================================
# End-to-End Pipeline Display
# =============================================================================
from scptensor.benchmark.display.end_to_end import (
    ClusteringMetrics,
    EndToEndDisplay,
    IntermediateResults,
    PipelineResult,
    PipelineStep,
    compute_jaccard_index,
)

# =============================================================================
# Quality Control Display
# =============================================================================
from scptensor.benchmark.display.qc import (
    BatchCVReport,
    MissingTypeDisplay,
    MissingTypeReport,
    QCBatchDisplay,
    QCComparisonResult,
    QCDashboardDisplay,
)

# =============================================================================
# Regression Detection
# =============================================================================
from scptensor.benchmark.display.regression import (
    RegressionChecker,
    RegressionDetail,
    RegressionReport,
    RegressionThreshold,
    TrendChartGenerator,
    TrendDataPoint,
    format_regression_message,
    load_baseline,
    save_baseline,
)

# =============================================================================
# Report Generation
# =============================================================================
from scptensor.benchmark.display.report import (
    BenchmarkReportGenerator,
    ReportSection,
    format_duration,
    format_metric_value,
    get_figure_relative_path,
)
from scptensor.benchmark.display.report import (
    ReportConfig as ReportReportConfig,  # Alias to avoid conflict
)

# =============================================================================
# Backward Compatibility - Legacy Exports
# =============================================================================
# Import MethodResult and ComparisonResult from comparison_engine if available
try:
    from scptensor.benchmark.comparison_engine import (
        ComparisonResult,
        MethodResult,
    )
except ImportError:
    # Fallback if comparison_engine is not available
    ComparisonResult = None  # type: ignore
    MethodResult = None  # type: ignore

# Plot style configuration (now in common.py)
from scptensor.benchmark.core import BenchmarkResult, BenchmarkResults
from scptensor.benchmark.display.common import PlotStyle, configure_plots

# Type aliases for backward compatibility
if MethodResult is not None:
    MethodSpec = MethodResult
else:
    MethodSpec = BenchmarkResult

# Keep BenchmarkResult as is (already imported from core)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # =========================================================================
    # Base and Configuration
    # =========================================================================
    "DisplayBase",
    "ComparisonDisplay",
    "MethodCategory",
    "ComparisonLayer",
    "PlotStyle",
    "ReportConfig",
    "CATEGORY_METRICS",
    "get_style_string",
    "get_category_metrics",
    # Common visual styling utilities
    "ColorPalette",
    "TypographyConfig",
    "LayoutConfig",
    "ModuleType",
    "NORMALIZATION_COLORS",
    "IMPUTATION_COLORS",
    "INTEGRATION_COLORS",
    "DIM_REDUCTION_COLORS",
    "QC_COLORS",
    "END_TO_END_COLORS",
    "FEATURE_SELECTION_COLORS",
    "DEFAULT_TYPOGRAPHY",
    "DEFAULT_LAYOUT",
    "get_module_colors",
    "apply_typography_theme",
    "apply_color_style",
    "apply_layout_config",
    "get_compatible_color",
    "get_status_color",
    # =========================================================================
    # Normalization Display
    # =========================================================================
    "LogNormalizeDisplay",
    "ZScoreDisplay",
    "NormalizationComparisonResult",
    "ZScoreVerificationResult",
    "setup_normalization_plot_style",
    # =========================================================================
    # Imputation Display
    # =========================================================================
    "KNNImputeDisplay",
    "ExclusiveImputeDisplay",
    "ImputationComparisonResult",
    "ExclusiveImputationResults",
    "setup_imputation_plot_style",
    # =========================================================================
    # Integration Display
    # =========================================================================
    "IntegrationDisplay",
    "IntegrationComparisonResult",
    "IntegrationMetricsSummary",
    "setup_integration_plot_style",
    # =========================================================================
    # Dimensionality Reduction Display
    # =========================================================================
    "PCADisplay",
    "UMAPDisplay",
    "DimReductionComparisonResult",
    "PCAResult",
    "UMAPResult",
    "setup_dim_reduction_plot_style",
    # =========================================================================
    # Quality Control Display
    # =========================================================================
    "QCDashboardDisplay",
    "MissingTypeDisplay",
    "QCBatchDisplay",
    "QCComparisonResult",
    "MissingTypeReport",
    "BatchCVReport",
    "setup_qc_plot_style",
    # =========================================================================
    # End-to-End Pipeline Display
    # =========================================================================
    "EndToEndDisplay",
    "PipelineResult",
    "PipelineStep",
    "ClusteringMetrics",
    "IntermediateResults",
    "compute_jaccard_index",
    "setup_end_to_end_plot_style",
    # =========================================================================
    # Report Generation
    # =========================================================================
    "BenchmarkReportGenerator",
    "ReportSection",
    "ReportReportConfig",  # Alias to avoid conflict with config.ReportConfig
    "format_metric_value",
    "format_duration",
    "get_figure_relative_path",
    # =========================================================================
    # Regression Detection
    # =========================================================================
    "RegressionChecker",
    "RegressionThreshold",
    "RegressionDetail",
    "RegressionReport",
    "TrendChartGenerator",
    "TrendDataPoint",
    "load_baseline",
    "save_baseline",
    "format_regression_message",
    # =========================================================================
    # Backward Compatibility - Legacy Exports
    # =========================================================================
    "configure_plots",
    "BenchmarkResults",
    "BenchmarkResult",
    "MethodSpec",
    "ComparisonResult",
]


# Module version
__version__ = "0.1.0"
