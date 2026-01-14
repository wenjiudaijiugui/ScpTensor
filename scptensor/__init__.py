"""ScpTensor: Single-Cell Proteomics Analysis Framework.

A comprehensive Python library for analyzing single-cell proteomics (SCP) data
with an intuitive hierarchical data structure (ScpContainer -> Assay -> ScpMatrix)
and a complete analysis pipeline.

Key Features:
    - Hierarchical data structure: ScpContainer -> Assay -> ScpMatrix
    - Quality control: Basic QC, outlier detection, feature/sample filtering
    - Normalization: Log, TMM, median scaling, upper quartile, and more
    - Imputation: KNN, PPCA, SVD, MissForest
    - Batch correction: ComBat, Harmony, MNN, Scanorama
    - Dimensionality reduction: PCA, UMAP
    - Clustering: K-means, graph-based clustering
    - Visualization: Scatter plots, heatmaps, violin plots, and more
    - Differential expression: t-test, Mann-Whitney, ANOVA, Kruskal-Wallis
    - Benchmarking: Comprehensive performance evaluation tools

Quick Start:
    >>> from scptensor import ScpContainer, load_csv, log_normalize, pca, run_kmeans
    >>> container = load_csv("data.csv")
    >>> container = log_normalize(container, assay_name="proteins")
    >>> container = pca(container, assay_name="proteins")
    >>> container = run_kmeans(container, n_clusters=5)

Version: v0.1.0-alpha
Documentation: https://github.com/your-org/scptensor
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "ScpTensor Team"

# Core data structures and exceptions
from scptensor.core import (
    AggregationLink,
    Assay,
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    MaskCode,
    MaskCodeError,
    MatrixMetadata,
    MatrixOps,
    MissingDependencyError,
    NUMBA_AVAILABLE,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
    ScpTensorError,
    ScpValueError,
    StructureError,
    ValidationError,
)

# Core I/O and sparse utilities
from scptensor.core import (
    apply_mask_threshold,
    auto_convert_for_operation,
    cleanup_layers,
    compute_euclidean_distance,
    count_mask_codes,
    ensure_sparse_format,
    fill_missing_with_value,
    find_missing_indices,
    from_scanpy,
    get_format_recommendation,
    get_memory_usage,
    get_sparsity_ratio,
    is_sparse_matrix,
    load_csv,
    load_h5ad,
    load_npz,
    optimal_format_for_operation,
    read_h5ad,
    reader,
    save_csv,
    save_h5ad,
    save_npz,
    sparse_center_rows,
    sparse_col_operation,
    sparse_copy,
    sparse_multiply_colwise,
    sparse_multiply_rowwise,
    sparse_row_operation,
    sparse_safe_log1p,
    to_scanpy,
    to_sparse_if_beneficial,
    write_h5ad,
)

# Normalization
from scptensor.normalization import (
    global_median_normalization,
    log_normalize,
    sample_mean_normalization,
    sample_median_normalization,
    tmm_normalization,
    upper_quartile_normalization,
)

# Imputation
from scptensor.impute import knn, missforest, ppca, svd_impute

# Integration (Batch Correction)
from scptensor.integration import combat, harmony, mnn_correct, scanorama_integrate

# Quality Control
from scptensor.qc import (
    basic_qc,
    calculate_qc_metrics,
    detect_contaminant_proteins,
    detect_doublets,
    detect_outliers,
    filter_features_by_missing_rate,
    filter_features_by_prevalence,
    filter_features_by_variance,
    filter_samples_by_missing_rate,
    filter_samples_by_total_count,
)

# Dimensionality Reduction
from scptensor.dim_reduction import pca, umap

# Clustering
from scptensor.cluster import run_kmeans

# Visualization
from scptensor.viz import (
    embedding,
    heatmap,
    qc_completeness,
    qc_matrix_spy,
    scatter,
    violin,
    volcano,
)

# Differential Expression
from scptensor.diff_expr import (
    DiffExprResult,
    adjust_fdr,
    diff_expr_anova,
    diff_expr_kruskal,
    diff_expr_mannwhitney,
    diff_expr_ttest,
)

# Benchmarking
from scptensor.benchmark import (
    BenchmarkResults,
    BenchmarkSuite,
    BiologicalMetrics,
    COMPETITOR_REGISTRY,
    ComparisonResult,
    CompetitorBenchmarkSuite,
    CompetitorResultVisualizer,
    ComputationalMetrics,
    create_method_configs,
    create_normalization_parameter_grids,
    get_competitor,
    get_competitors_by_operation,
    list_competitors,
    MetricsEngine,
    MethodConfig,
    MethodRunResult,
    ParameterGrid,
    ResultsVisualizer,
    ScanpyStyleOps,
    SyntheticDataset,
)

# Datasets
from scptensor.datasets import (
    DatasetSize,
    DatasetType,
    load_example_with_clusters,
    load_simulated_scrnaseq_like,
    load_toy_example,
    REPRODUCIBILITY_NOTE,
)

# Utilities
from scptensor.utils import ScpDataGenerator

# Standardization (deprecated, re-exported for backward compatibility)
from scptensor.standardization import zscore


# Public API
__all__ = [
    "__version__",
    # Core structures
    "ScpContainer",
    "Assay",
    "ScpMatrix",
    "ProvenanceLog",
    "MatrixMetadata",
    "MaskCode",
    "AggregationLink",
    "MatrixOps",
    "reader",
    # Core exceptions
    "ScpTensorError",
    "StructureError",
    "ValidationError",
    "LayerNotFoundError",
    "AssayNotFoundError",
    "MissingDependencyError",
    "DimensionError",
    "MaskCodeError",
    "ScpValueError",
    # Core I/O
    "load_csv",
    "save_csv",
    "load_h5ad",
    "save_h5ad",
    "load_npz",
    "save_npz",
    "from_scanpy",
    "to_scanpy",
    "read_h5ad",
    "write_h5ad",
    # Sparse utilities
    "is_sparse_matrix",
    "get_sparsity_ratio",
    "to_sparse_if_beneficial",
    "ensure_sparse_format",
    "sparse_copy",
    "cleanup_layers",
    "get_memory_usage",
    "optimal_format_for_operation",
    "auto_convert_for_operation",
    "sparse_row_operation",
    "sparse_col_operation",
    "sparse_multiply_rowwise",
    "sparse_multiply_colwise",
    "sparse_center_rows",
    "sparse_safe_log1p",
    "get_format_recommendation",
    # JIT operations
    "NUMBA_AVAILABLE",
    "count_mask_codes",
    "find_missing_indices",
    "compute_euclidean_distance",
    "apply_mask_threshold",
    "fill_missing_with_value",
    # Normalization
    "log_normalize",
    "sample_median_normalization",
    "sample_mean_normalization",
    "global_median_normalization",
    "tmm_normalization",
    "upper_quartile_normalization",
    # Imputation
    "knn",
    "missforest",
    "ppca",
    "svd_impute",
    # Integration
    "combat",
    "harmony",
    "mnn_correct",
    "scanorama_integrate",
    # Quality Control
    "basic_qc",
    "detect_outliers",
    "calculate_qc_metrics",
    "filter_features_by_missing_rate",
    "filter_features_by_variance",
    "filter_features_by_prevalence",
    "filter_samples_by_total_count",
    "filter_samples_by_missing_rate",
    "detect_contaminant_proteins",
    "detect_doublets",
    # Dimensionality Reduction
    "pca",
    "umap",
    # Clustering
    "run_kmeans",
    # Visualization
    "scatter",
    "heatmap",
    "violin",
    "embedding",
    "qc_completeness",
    "qc_matrix_spy",
    "volcano",
    # Differential Expression
    "DiffExprResult",
    "diff_expr_ttest",
    "diff_expr_mannwhitney",
    "diff_expr_anova",
    "diff_expr_kruskal",
    "adjust_fdr",
    # Benchmarking
    "BenchmarkSuite",
    "BenchmarkResults",
    "MethodRunResult",
    "BiologicalMetrics",
    "ComputationalMetrics",
    "MetricsEngine",
    "ParameterGrid",
    "MethodConfig",
    "create_method_configs",
    "create_normalization_parameter_grids",
    "SyntheticDataset",
    "ResultsVisualizer",
    "CompetitorBenchmarkSuite",
    "ComparisonResult",
    "CompetitorResultVisualizer",
    "COMPETITOR_REGISTRY",
    "ScanpyStyleOps",
    "list_competitors",
    "get_competitor",
    "get_competitors_by_operation",
    # Datasets
    "load_toy_example",
    "load_simulated_scrnaseq_like",
    "load_example_with_clusters",
    "DatasetType",
    "DatasetSize",
    "REPRODUCIBILITY_NOTE",
    # Utilities
    "ScpDataGenerator",
    # Standardization (deprecated)
    "zscore",
]
