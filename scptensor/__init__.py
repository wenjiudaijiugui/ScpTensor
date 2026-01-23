"""ScpTensor: Single-Cell Proteomics Analysis Framework.

A comprehensive Python library for analyzing single-cell proteomics (SCP) data
with an intuitive hierarchical data structure (ScpContainer -> Assay -> ScpMatrix)
and a complete analysis pipeline.

Key Features:
    - Hierarchical data structure: ScpContainer -> Assay -> ScpMatrix
    - Quality control: Basic QC, outlier detection, feature/sample filtering
    - Data transformation: Log transformation for preprocessing
    - Normalization: TMM, median scaling, upper quartile, and more
    - Imputation: KNN, LLS, BPCA, MissForest, QRILC, MinProb
    - Batch correction: ComBat, Harmony, MNN, Scanorama
    - Dimensionality reduction: PCA, UMAP
    - Clustering: K-means, graph-based clustering
    - Visualization: Scatter plots, heatmaps, violin plots, and more
    - Differential expression: t-test, Mann-Whitney, ANOVA, Kruskal-Wallis
    - Benchmarking: Comprehensive performance evaluation tools

Quick Start:
    >>> from scptensor import ScpContainer, load_csv, log_transform, reduce_pca, cluster_kmeans
    >>> container = load_csv("data.csv")
    >>> container = log_transform(container, assay_name="proteins")
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
    BenchmarkResult,
    BenchmarkResults,
    BiologicalMetrics,
    ComparisonResult,
    CompetitorBenchmarkSuite,
    CompetitorResultVisualizer,
    ScanpyStyleOps,
    SyntheticDataset,
    get_competitor,
    get_competitors_by_operation,
    list_competitors,
)

# Clustering
from scptensor.cluster import (
    cluster_kmeans,
    cluster_leiden,
)

# Core data structures and exceptions
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
    count_mask_codes,
    ensure_sparse_format,
    fill_missing_with_value,
    find_missing_indices,
    get_format_recommendation,
    get_memory_usage,
    get_sparsity_ratio,
    is_sparse_matrix,
    optimal_format_for_operation,
    reader,
    sparse_center_rows,
    sparse_col_operation,
    sparse_copy,
    sparse_multiply_colwise,
    sparse_multiply_rowwise,
    sparse_row_operation,
    sparse_safe_log1p,
    to_sparse_if_beneficial,
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
    impute_bpca,
    impute_knn,
    impute_lls,
    impute_mf,
    impute_minprob,
    impute_qrilc,
)

# Integration (Batch Correction)
from scptensor.integration import (
    integrate_combat,
    integrate_harmony,
    integrate_mnn,
    integrate_scanorama,
)

# CSV, NPZ, and Scanpy I/O functions
# HDF5 I/O
from scptensor.io import (
    IOFormatError,
    IOPasswordError,
    IOWriteError,
    from_scanpy,
    load_csv,
    load_h5ad,
    load_hdf5,
    load_npz,
    read_h5ad,
    save_csv,
    save_h5ad,
    save_hdf5,
    save_npz,
    to_scanpy,
    write_h5ad,
)

# Normalization and transformation
from scptensor.normalization import (
    log_transform,
    norm_mean,
    norm_median,
    norm_quantile,
)

# Quality Control
from scptensor.qc import (
    assess_batch_effects,
    calculate_feature_qc_metrics,
    calculate_sample_qc_metrics,
    filter_contaminants,
    filter_doublets_mad,
    filter_features_by_cv,
    filter_features_by_missingness,
    filter_low_quality_samples,
    filter_psms_by_pif,
    qc_feature,
    qc_psm,
    qc_sample,
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
    "apply_mask_threshold",
    "fill_missing_with_value",
    # Normalization and transformation
    "log_transform",
    "norm_mean",
    "norm_median",
    "norm_quantile",
    # Imputation
    "impute_knn",
    "impute_lls",
    "impute_bpca",
    "impute_mf",
    "impute_qrilc",
    "impute_minprob",
    # Integration
    "integrate_combat",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
    # Quality Control
    "qc_feature",
    "qc_psm",
    "qc_sample",
    "filter_contaminants",
    "filter_psms_by_pif",
    "filter_features_by_cv",
    "filter_features_by_missingness",
    "filter_low_quality_samples",
    "filter_doublets_mad",
    "assess_batch_effects",
    "calculate_sample_qc_metrics",
    "calculate_feature_qc_metrics",
    # Dimensionality Reduction
    "reduce_pca",
    "reduce_umap",
    "get_solver_info",
    "optimal_svd_solver",
    # Clustering
    "cluster_kmeans",
    "cluster_leiden",
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
    "BenchmarkResult",
    "BenchmarkResults",
    "BiologicalMetrics",
    "ComparisonResult",
    "CompetitorBenchmarkSuite",
    "CompetitorResultVisualizer",
    "COMPETITOR_REGISTRY",
    "ScanpyStyleOps",
    "list_competitors",
    "get_competitor",
    "get_competitors_by_operation",
    "SyntheticDataset",
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
