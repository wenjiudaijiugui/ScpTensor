"""Recipe-level visualization APIs.

This module exports high-level visualization functions. Canonical public names
follow ``plot_*`` where practical; legacy names are retained as aliases for
backward compatibility.
"""

from .embedding import (
    embedding,
    pca,
    plot_embedding,
    plot_embedding_pca,
    plot_embedding_scatter,
    plot_embedding_tsne,
    plot_embedding_umap,
    scatter,
    tsne,
    umap,
)
from .feature import dotplot, plot_feature_dotplot
from .impute import (
    plot_imputation_comparison,
    plot_imputation_metrics,
    plot_imputation_scatter,
    plot_missing_pattern,
)
from .matrix import (
    heatmap,
    matrixplot,
    plot_matrix_heatmap,
    plot_matrixplot,
    plot_tracksplot,
    tracksplot,
)
from .qc import (
    missing_value_patterns,
    pca_overview,
    plot_qc_completeness,
    plot_qc_matrix_spy,
    plot_qc_missing_value_patterns,
    plot_qc_pca_overview,
    qc_completeness,
    qc_matrix_spy,
)
from .qc_advanced import (
    plot_cumulative_sensitivity,
    plot_cv_by_feature,
    plot_cv_comparison,
    plot_cv_distribution,
    plot_jaccard_heatmap,
    plot_missing_summary,
    plot_missing_type_heatmap,
    plot_sensitivity_summary,
)
from .report import ReportTheme, generate_analysis_report
from .statistics import (
    correlation_matrix,
    dendrogram,
    plot_correlation_matrix,
    plot_dendrogram,
)
from .workflow import (
    plot_aggregation_summary,
    plot_data_overview,
    plot_embedding_panels,
    plot_integration_batch_summary,
    plot_missingness_reduction,
    plot_normalization_summary,
    plot_preprocessing_summary,
    plot_qc_filtering_summary,
    plot_recent_operations,
    plot_reduction_summary,
    plot_saved_artifact_sizes,
)

__all__ = [
    # Canonical embedding names
    "plot_embedding_scatter",
    "plot_embedding_umap",
    "plot_embedding_pca",
    "plot_embedding_tsne",
    "plot_embedding",
    # Canonical feature/matrix/QC/statistics names
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
    # Workflow names
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
    # Existing advanced and imputation plots
    "plot_sensitivity_summary",
    "plot_cumulative_sensitivity",
    "plot_jaccard_heatmap",
    "plot_missing_type_heatmap",
    "plot_missing_summary",
    "plot_cv_distribution",
    "plot_cv_by_feature",
    "plot_cv_comparison",
    "plot_imputation_comparison",
    "plot_imputation_scatter",
    "plot_imputation_metrics",
    "plot_missing_pattern",
    # Report
    "generate_analysis_report",
    "ReportTheme",
    # Backward-compatible aliases
    "scatter",
    "umap",
    "pca",
    "tsne",
    "embedding",
    "dotplot",
    "heatmap",
    "matrixplot",
    "tracksplot",
    "qc_completeness",
    "qc_matrix_spy",
    "pca_overview",
    "missing_value_patterns",
    "correlation_matrix",
    "dendrogram",
]
