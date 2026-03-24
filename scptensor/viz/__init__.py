"""Public visualization API.

Top-level exports prefer canonical ``plot_*`` names plus a small set of base
primitives. Backward-compatible recipe aliases stay in explicit submodules and
are not re-exported here.
"""

from .base.heatmap import heatmap as _base_heatmap
from .base.violin import violin as _base_violin
from .recipes import (
    ReportTheme,
    generate_analysis_report,
    plot_aggregation_summary,
    plot_correlation_matrix,
    plot_data_overview,
    plot_dendrogram,
    plot_embedding,
    plot_embedding_panels,
    plot_embedding_pca,
    plot_embedding_scatter,
    plot_embedding_tsne,
    plot_embedding_umap,
    plot_feature_dotplot,
    plot_imputation_comparison,
    plot_imputation_metrics,
    plot_imputation_scatter,
    plot_integration_batch_summary,
    plot_matrix_heatmap,
    plot_matrixplot,
    plot_missing_pattern,
    plot_missingness_reduction,
    plot_normalization_summary,
    plot_preprocessing_summary,
    plot_qc_completeness,
    plot_qc_filtering_summary,
    plot_qc_matrix_spy,
    plot_qc_missing_value_patterns,
    plot_qc_pca_overview,
    plot_recent_operations,
    plot_reduction_summary,
    plot_saved_artifact_sizes,
    plot_tracksplot,
)

heatmap = _base_heatmap
violin = _base_violin

__all__ = [
    # Base primitives
    "heatmap",
    "violin",
    # Canonical plot_* names
    "plot_embedding_scatter",
    "plot_embedding_umap",
    "plot_embedding_pca",
    "plot_embedding_tsne",
    "plot_embedding",
    "plot_feature_dotplot",
    "plot_matrixplot",
    "plot_matrix_heatmap",
    "plot_tracksplot",
    "plot_qc_completeness",
    "plot_qc_matrix_spy",
    "plot_qc_pca_overview",
    "plot_qc_missing_value_patterns",
    "plot_correlation_matrix",
    "plot_dendrogram",
    "plot_imputation_comparison",
    "plot_imputation_scatter",
    "plot_imputation_metrics",
    "plot_missing_pattern",
    "plot_aggregation_summary",
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_normalization_summary",
    "plot_missingness_reduction",
    "plot_integration_batch_summary",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
    # Report
    "generate_analysis_report",
    "ReportTheme",
]
