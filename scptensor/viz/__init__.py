from .base import heatmap, violin
from .base import scatter as base_scatter
from .recipes import (
    ReportTheme,
    embedding,
    generate_analysis_report,
    pca,
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
    # Report
    "generate_analysis_report",
    "ReportTheme",
]
