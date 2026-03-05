from .embedding import embedding, pca, scatter, tsne, umap
from .feature import dotplot
from .impute import (
    plot_imputation_comparison,
    plot_imputation_metrics,
    plot_imputation_scatter,
    plot_missing_pattern,
)
from .matrix import heatmap, matrixplot, tracksplot
from .qc import missing_value_patterns, pca_overview, qc_completeness, qc_matrix_spy
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
from .statistics import correlation_matrix, dendrogram
from .stats import volcano
from .workflow import (
    plot_data_overview,
    plot_embedding_panels,
    plot_missingness_reduction,
    plot_preprocessing_summary,
    plot_qc_filtering_summary,
    plot_recent_operations,
    plot_reduction_summary,
    plot_saved_artifact_sizes,
)

__all__ = [
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
    "volcano",
    # Advanced QC visualizations
    "plot_sensitivity_summary",
    "plot_cumulative_sensitivity",
    "plot_jaccard_heatmap",
    "plot_missing_type_heatmap",
    "plot_missing_summary",
    # CV visualizations (Phase 3)
    "plot_cv_distribution",
    "plot_cv_by_feature",
    "plot_cv_comparison",
    # Imputation visualizations
    "plot_imputation_comparison",
    "plot_imputation_scatter",
    "plot_imputation_metrics",
    "plot_missing_pattern",
    # Workflow visualizations
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
    "generate_analysis_report",
    "ReportTheme",
]
