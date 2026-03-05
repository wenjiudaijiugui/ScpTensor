"""Public visualization API."""

from .base.heatmap import heatmap as _base_heatmap
from .base.scatter import scatter as _base_scatter
from .base.violin import violin as _base_violin
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

base_scatter = _base_scatter
heatmap = _base_heatmap
violin = _base_violin

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
