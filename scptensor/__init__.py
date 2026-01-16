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
    cluster_kmeans_assay,  # formerly run_kmeans
    cluster_leiden,
    run_kmeans,  # Deprecated: use cluster_kmeans or cluster_kmeans_assay
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
    pca,  # Deprecated: use reduce_pca
    reduce_pca,
    reduce_umap,
    umap,  # Deprecated: use reduce_umap
)

# Imputation
from scptensor.impute import (
    impute_knn,  # New API
    impute_mf,  # New API
    impute_ppca,  # New API
    impute_svd,  # New API
    knn,  # Deprecated: use impute_knn
    missforest,  # Deprecated: use impute_mf
    ppca,  # Deprecated: use impute_ppca
    svd_impute,  # Deprecated: use impute_svd
)

# Integration (Batch Correction)
from scptensor.integration import (
    combat,  # Deprecated: use integrate_combat
    harmony,  # Deprecated: use integrate_harmony
    integrate_combat,  # New API
    integrate_harmony,  # New API
    integrate_mnn,  # New API
    integrate_scanorama,  # New API
    mnn_correct,  # Deprecated: use integrate_mnn
    scanorama_integrate,  # Deprecated: use integrate_scanorama
)

# HDF5 I/O
from scptensor.io import IOFormatError, IOPasswordError, IOWriteError, load_hdf5, save_hdf5

# Normalization
from scptensor.normalization import (
    global_median_normalization,  # Deprecated: use norm_global_median
    log_normalize,  # Deprecated: use norm_log
    median_centering,  # Deprecated: use norm_median_center
    median_scaling,  # Deprecated: use norm_median_scale
    norm_global_median,  # New API
    norm_log,  # New API
    norm_median_center,  # New API
    norm_median_scale,  # New API
    norm_quartile,  # New API
    norm_sample_mean,  # New API
    norm_sample_median,  # New API
    norm_tmm,  # New API
    norm_zscore,  # New API
    sample_mean_normalization,  # Deprecated: use norm_sample_mean
    sample_median_normalization,  # Deprecated: use norm_sample_median
    tmm_normalization,  # Deprecated: use norm_tmm
    upper_quartile_normalization,  # Deprecated: use norm_quartile
)

# Quality Control
from scptensor.qc import (
    basic_qc,  # Deprecated: use qc_basic
    calculate_qc_metrics,
    compute_batch_metrics,  # Deprecated: use qc_batch_metrics
    compute_batch_pca,
    compute_feature_missing_rate,
    compute_feature_variance,
    compute_pairwise_correlation,
    compute_quality_score,  # Deprecated: use qc_score
    compute_sample_similarity_network,
    detect_batch_effects,
    detect_contaminant_proteins,  # Deprecated: use detect_contaminants
    detect_contaminants,  # New API
    detect_doublets,
    detect_outlier_samples,
    detect_outliers,
    filter_features_by_missing_rate,  # Deprecated: use filter_features_missing
    filter_features_by_prevalence,  # Deprecated: use filter_features_prevalence
    filter_features_by_variance,  # Deprecated: use filter_features_variance
    filter_features_missing,  # New API
    filter_features_prevalence,  # New API
    filter_features_variance,  # New API
    filter_samples_by_missing_rate,  # Deprecated: use filter_samples_missing
    filter_samples_by_total_count,  # Deprecated: use filter_samples_count
    filter_samples_count,  # New API
    filter_samples_missing,  # New API
    qc_basic,  # New API
    qc_batch_metrics,  # New API
    qc_score,  # New API
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
    # Normalization (new API: norm_*)
    "norm_log",
    "norm_zscore",
    "norm_median_center",
    "norm_median_scale",
    "norm_sample_median",
    "norm_sample_mean",
    "norm_global_median",
    "norm_tmm",
    "norm_quartile",
    # Normalization (deprecated, for backward compatibility)
    "log_normalize",
    "zscore",
    "median_centering",
    "median_scaling",
    "sample_median_normalization",
    "sample_mean_normalization",
    "global_median_normalization",
    "tmm_normalization",
    "upper_quartile_normalization",
    # Imputation (new API: impute_*)
    "impute_knn",
    "impute_ppca",
    "impute_svd",
    "impute_mf",
    # Imputation (deprecated, for backward compatibility)
    "knn",
    "missforest",
    "ppca",
    "svd_impute",
    # Integration (new API: integrate_*)
    "integrate_combat",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
    # Integration (deprecated, for backward compatibility)
    "combat",
    "harmony",
    "mnn_correct",
    "scanorama_integrate",
    # Quality Control (new API: qc_*, filter_*, detect_*)
    "qc_basic",
    "qc_score",
    "qc_batch_metrics",
    "filter_features_missing",
    "filter_features_variance",
    "filter_features_prevalence",
    "filter_samples_count",
    "filter_samples_missing",
    "detect_contaminants",
    # Quality Control (deprecated, for backward compatibility)
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
    # Quality Control (additional functions)
    "compute_feature_variance",
    "compute_feature_missing_rate",
    "compute_batch_metrics",
    "compute_batch_pca",
    "compute_pairwise_correlation",
    "compute_quality_score",
    "compute_sample_similarity_network",
    "detect_batch_effects",
    "detect_outlier_samples",
    # Dimensionality Reduction (new API: reduce_*)
    "reduce_pca",
    "reduce_umap",
    # Dimensionality Reduction (deprecated, for backward compatibility)
    "pca",
    "umap",
    "get_solver_info",
    # Clustering (new API: cluster_*)
    "cluster_kmeans",
    "cluster_leiden",
    "cluster_kmeans_assay",  # formerly run_kmeans
    "run_kmeans",  # Deprecated: use cluster_kmeans or cluster_kmeans_assay
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
