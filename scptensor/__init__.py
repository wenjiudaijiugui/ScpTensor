"""ScpTensor: DIA-based single-cell proteomics preprocessing framework.

ScpTensor provides an end-to-end analysis stack built around a hierarchical
container model (ScpContainer -> Assay -> ScpMatrix), with strong support for
DIA quant-table workflows.

Current I/O scope focuses on DIA-NN and Spectronaut outputs at protein and
peptide/precursor levels (long and matrix table shapes).

Quick Start:
    >>> from scptensor import aggregate_to_protein, load_diann, log_transform
    >>> from scptensor import norm_median, reduce_pca
    >>> container = load_diann("report.tsv", level="peptide", table_format="long")
    >>> container = aggregate_to_protein(container, source_assay="peptides", target_assay="proteins")
    >>> container = log_transform(container, assay_name="proteins", source_layer="raw", new_layer_name="log2")
    >>> container = norm_median(container, assay_name="proteins", source_layer="log2", new_layer_name="norm")
    >>> container = reduce_pca(container, assay_name="proteins", base_layer="norm")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "ScpTensor Team"

# Aggregation
from scptensor.aggregation import aggregate_to_protein

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
    sparse_center_rows,
    sparse_col_operation,
    sparse_copy,
    sparse_multiply_colwise,
    sparse_multiply_rowwise,
    sparse_row_operation,
    sparse_safe_log1p,
    to_sparse_if_beneficial,
)

# Dimensionality Reduction
from scptensor.dim_reduction import (
    SolverType,
    reduce_pca,
    reduce_tsne,
    reduce_umap,
)

# Imputation
from scptensor.impute import (
    impute_bpca,
    impute_half_row_min,
    impute_iterative_svd,
    impute_knn,
    impute_lls,
    impute_mf,
    impute_minprob,
    impute_none,
    impute_qrilc,
    impute_row_mean,
    impute_row_median,
    impute_softimpute,
    impute_zero,
)

# Integration (Batch Correction)
from scptensor.integration import (
    integrate_combat,
    integrate_harmony,
    integrate_limma,
    integrate_mnn,
    integrate_none,
    integrate_scanorama,
)

# Mass-spec I/O functions
from scptensor.io import (
    IOFormatError,
    IOPasswordError,
    IOWriteError,
    load_diann,
    load_peptide_pivot,
    load_spectronaut,
)

# Normalization
from scptensor.normalization import (
    norm_mean,
    norm_median,
    norm_none,
    norm_quantile,
    norm_trqn,
    normalize,
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

# Transformation
from scptensor.transformation import log_transform

# Utilities
from scptensor.utils import ScpDataGenerator

# Visualization
from scptensor.viz import (
    embedding,
    heatmap,
    plot_data_overview,
    plot_embedding_panels,
    plot_missingness_reduction,
    plot_preprocessing_summary,
    plot_qc_filtering_summary,
    plot_recent_operations,
    plot_reduction_summary,
    plot_saved_artifact_sizes,
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
    "load_diann",
    "load_peptide_pivot",
    "load_spectronaut",
    # Aggregation
    "aggregate_to_protein",
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
    # Transformation
    "log_transform",
    # Normalization
    "norm_none",
    "norm_mean",
    "norm_median",
    "norm_quantile",
    "norm_trqn",
    "normalize",
    # Imputation
    "impute_none",
    "impute_zero",
    "impute_row_mean",
    "impute_row_median",
    "impute_half_row_min",
    "impute_knn",
    "impute_lls",
    "impute_iterative_svd",
    "impute_softimpute",
    "impute_bpca",
    "impute_mf",
    "impute_qrilc",
    "impute_minprob",
    # Integration
    "integrate_none",
    "integrate_combat",
    "integrate_limma",
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
    "reduce_tsne",
    "reduce_umap",
    "SolverType",
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
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
    # Utilities
    "ScpDataGenerator",
    # Standardization (deprecated)
    "zscore",
]
