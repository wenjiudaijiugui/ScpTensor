from .differential import (
    rank_genes_groups_dotplot,
    rank_genes_groups_stacked_violin,
)
from .differential import (
    volcano as volcano_enhanced,
)
from .embedding import embedding, pca, scatter, tsne, umap
from .feature import dotplot
from .matrix import heatmap, matrixplot, tracksplot
from .qc import (
    missing_value_patterns,
    pca_overview,
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
from .statistics import correlation_matrix, dendrogram
from .stats import volcano

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
    "volcano_enhanced",
    "rank_genes_groups_dotplot",
    "rank_genes_groups_stacked_violin",
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
    "generate_analysis_report",
    "ReportTheme",
]
