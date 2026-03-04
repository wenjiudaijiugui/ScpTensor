from .base import heatmap, violin
from .base import scatter as base_scatter
from .recipes import (
    ReportTheme,
    embedding,
    generate_analysis_report,
    pca,
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
    tsne,
    umap,
    volcano,
)

__all__ = [
    # Base primitives
    "base_scatter",
    "heatmap",
    "violin",
    # Recipe plots
    "scatter",
    "umap",
    "pca",
    "tsne",
    "embedding",
    "qc_completeness",
    "qc_matrix_spy",
    "volcano",
    # Workflow recipes
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
    # Report
    "generate_analysis_report",
    "ReportTheme",
]
