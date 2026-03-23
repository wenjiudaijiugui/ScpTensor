"""ScpTensor: DIA-based single-cell proteomics preprocessing framework.

ScpTensor provides an end-to-end analysis stack built around a hierarchical
container model (ScpContainer -> Assay -> ScpMatrix), with strong support for
DIA quant-table workflows.

Current I/O scope focuses on DIA-NN and Spectronaut outputs at protein and
peptide/precursor levels (long and matrix table shapes).

Quick Start:
    >>> from scptensor import aggregate_to_protein, load_diann, log_transform
    >>> from scptensor import norm_median
    >>> container = load_diann("report.tsv", level="peptide", table_format="long")
    >>> container = aggregate_to_protein(container, source_assay="peptides", target_assay="proteins")
    >>> container = log_transform(container, assay_name="proteins", source_layer="raw", new_layer_name="log2")
    >>> container = norm_median(container, assay_name="proteins", source_layer="log2", new_layer_name="norm")
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "ScpTensor Team"

_CORE_EXPORTS = [
    "ScpContainer",
    "Assay",
    "ScpMatrix",
    "ProvenanceLog",
    "MatrixMetadata",
    "MaskCode",
    "AggregationLink",
    "MatrixOps",
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
    "NUMBA_AVAILABLE",
    "count_mask_codes",
    "find_missing_indices",
    "apply_mask_threshold",
    "fill_missing_with_value",
]

_AGGREGATION_EXPORTS = ["aggregate_to_protein"]

_IO_EXPORTS = [
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    "load_diann",
    "load_peptide_pivot",
    "load_spectronaut",
]

_TRANSFORMATION_EXPORTS = ["log_transform"]

_STANDARDIZATION_EXPORTS = ["zscore"]

_NORMALIZATION_EXPORTS = [
    "norm_none",
    "norm_mean",
    "norm_median",
    "norm_quantile",
    "norm_trqn",
    "normalize",
]

_IMPUTATION_EXPORTS = [
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
]

_INTEGRATION_EXPORTS = [
    "integrate_none",
    "integrate_combat",
    "integrate_limma",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
]

_QC_EXPORTS = [
    "qc_feature",
    "qc_sample",
    "filter_features_by_cv",
    "filter_features_by_missingness",
    "filter_low_quality_samples",
    "filter_doublets_mad",
    "assess_batch_effects",
    "calculate_sample_qc_metrics",
    "calculate_feature_qc_metrics",
]

_UTIL_EXPORTS = ["ScpDataGenerator"]

_VIZ_EXPORTS = [
    "scatter",
    "heatmap",
    "violin",
    "embedding",
    "qc_completeness",
    "qc_matrix_spy",
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
]

_EXPORT_GROUPS: dict[str, list[str]] = {
    "scptensor.core": _CORE_EXPORTS,
    "scptensor.aggregation": _AGGREGATION_EXPORTS,
    "scptensor.io": _IO_EXPORTS,
    "scptensor.transformation": _TRANSFORMATION_EXPORTS,
    "scptensor.standardization": _STANDARDIZATION_EXPORTS,
    "scptensor.normalization": _NORMALIZATION_EXPORTS,
    "scptensor.impute": _IMPUTATION_EXPORTS,
    "scptensor.integration": _INTEGRATION_EXPORTS,
    "scptensor.qc": _QC_EXPORTS,
    "scptensor.utils": _UTIL_EXPORTS,
    "scptensor.viz": _VIZ_EXPORTS,
}

_EXPORT_MAP = {
    symbol: (module_name, symbol)
    for module_name, symbols in _EXPORT_GROUPS.items()
    for symbol in symbols
}

__all__ = [
    "__version__",
    *_CORE_EXPORTS,
    *_IO_EXPORTS,
    *_AGGREGATION_EXPORTS,
    *_TRANSFORMATION_EXPORTS,
    *_STANDARDIZATION_EXPORTS,
    *_NORMALIZATION_EXPORTS,
    *_IMPUTATION_EXPORTS,
    *_INTEGRATION_EXPORTS,
    *_QC_EXPORTS,
    *_VIZ_EXPORTS,
    *_UTIL_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level reexports from their owning subpackages."""
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - stdlib-facing fallback
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy reexports through interactive discovery tools."""
    return sorted(set(globals()) | set(__all__))
