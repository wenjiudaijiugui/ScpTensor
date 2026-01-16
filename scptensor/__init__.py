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
    >>> from scptensor import ScpContainer, load_csv, norm_log, reduce_pca, cluster_kmeans
    >>> container = load_csv("data.csv")
    >>> container = norm_log(container, assay_name="proteins")
    >>> container = reduce_pca(container, assay_name="proteins")
    >>> container = cluster_kmeans(container, n_clusters=5)

Version: v0.1.0-beta
Documentation: https://github.com/your-org/scptensor
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "ScpTensor Team"

# Core data structures and exceptions
# Benchmarking
from scptensor.benchmark import (
    COMPETITOR_REGISTRY,
    BenchmarkResults,
    BenchmarkSuite,
    BiologicalMetrics,
    ComparisonResult,
    CompetitorBenchmarkSuite,
    CompetitorResultVisualizer,
    ComputationalMetrics,
    MethodConfig,
    MethodRunResult,
    MetricsEngine,
    ParameterGrid,
    ResultsVisualizer,
    ScanpyStyleOps,
    SyntheticDataset,
    create_method_configs,
    create_normalization_parameter_grids,
    get_competitor,
    get_competitors_by_operation,
    list_competitors,
)

# Clustering
from scptensor.cluster import (
    cluster_kmeans,
    cluster_kmeans_assay,
    cluster_leiden,
)

# Core I/O and sparse utilities
from scptensor.core import (
    NUMBA_AVAILABLE,
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
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
    ScpTensorError,
    ScpValueError,
    StructureError,
    ValidationError,
    VisualizationError,
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

# Datasets
from scptensor.datasets import (
    REPRODUCIBILITY_NOTE,
    DatasetSize,
    DatasetType,
    load_example_with_clusters,
    load_simulated_scrnaseq_like,
    load_toy_example,
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

# Dimensionality Reduction
from scptensor.dim_reduction import (
    get_solver_info,
    optimal_svd_solver,
    reduce_pca,
    reduce_umap,
)

# Imputation
from scptensor.impute import (
    impute_knn,
    impute_mf,
    impute_ppca,
    impute_svd,
)

# Integration (Batch Correction)
from scptensor.integration import (
    integrate_combat,
    integrate_harmony,
    integrate_mnn,
    integrate_scanorama,
)

# HDF5 I/O
from scptensor.io import IOFormatError, IOPasswordError, IOWriteError, load_hdf5, save_hdf5

# Normalization
from scptensor.normalization import (
    norm_global_median,
    norm_log,
    norm_median_center,
    norm_median_scale,
    norm_quartile,
    norm_sample_mean,
    norm_sample_median,
    norm_tmm,
    norm_zscore,
)

# Quality Control
from scptensor.qc import (
    calculate_qc_metrics,
    compute_batch_pca,
    compute_feature_missing_rate,
    compute_feature_variance,
    compute_pairwise_correlation,
    compute_sample_similarity_network,
    detect_batch_effects,
    detect_contaminants,
    detect_doublets,
    detect_outlier_samples,
    detect_outliers,
    filter_features_missing,
    filter_features_prevalence,
    filter_features_variance,
    filter_samples_count,
    filter_samples_missing,
    qc_basic,
    qc_batch_metrics,
    qc_score,
)

# Standardization (deprecated, re-exported for backward compatibility)
from scptensor.standardization import zscore

# Utilities
from scptensor.utils import ScpDataGenerator

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
    "VisualizationError",
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
    # HDF5 I/O
    "save_hdf5",
    "load_hdf5",
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
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
    "norm_log",
    "norm_zscore",
    "norm_median_center",
    "norm_median_scale",
    "norm_sample_median",
    "norm_sample_mean",
    "norm_global_median",
    "norm_tmm",
    "norm_quartile",
    # Imputation
    "impute_knn",
    "impute_ppca",
    "impute_svd",
    "impute_mf",
    # Integration
    "integrate_combat",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
    # Quality Control
    "qc_basic",
    "qc_score",
    "qc_batch_metrics",
    "filter_features_missing",
    "filter_features_variance",
    "filter_features_prevalence",
    "filter_samples_count",
    "filter_samples_missing",
    "detect_contaminants",
    "detect_outliers",
    "detect_doublets",
    "calculate_qc_metrics",
    "compute_feature_variance",
    "compute_feature_missing_rate",
    "compute_batch_pca",
    "compute_pairwise_correlation",
    "compute_sample_similarity_network",
    "detect_batch_effects",
    "detect_outlier_samples",
    # Dimensionality Reduction
    "reduce_pca",
    "reduce_umap",
    "get_solver_info",
    "optimal_svd_solver",
    # Clustering
    "cluster_kmeans",
    "cluster_leiden",
    "cluster_kmeans_assay",
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
