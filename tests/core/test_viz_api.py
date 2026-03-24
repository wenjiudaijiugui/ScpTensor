"""Tests for stable visualization namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.viz as stable_viz
from scptensor.viz import (
    ReportTheme,
    generate_analysis_report,
    heatmap,
    plot_data_overview,
    plot_embedding_panels,
    plot_embedding_scatter,
    plot_imputation_comparison,
    plot_matrix_heatmap,
    plot_qc_completeness,
    plot_qc_filtering_summary,
    plot_recent_operations,
    violin,
)
from scptensor.viz.base.heatmap import heatmap as heatmap_core
from scptensor.viz.base.violin import violin as violin_core
from scptensor.viz.recipes import (
    ReportTheme as ReportThemeCore,
)
from scptensor.viz.recipes import (
    generate_analysis_report as generate_analysis_report_core,
)
from scptensor.viz.recipes import (
    plot_data_overview as plot_data_overview_core,
)
from scptensor.viz.recipes import (
    plot_embedding_panels as plot_embedding_panels_core,
)
from scptensor.viz.recipes import (
    plot_embedding_scatter as plot_embedding_scatter_core,
)
from scptensor.viz.recipes import (
    plot_imputation_comparison as plot_imputation_comparison_core,
)
from scptensor.viz.recipes import (
    plot_matrix_heatmap as plot_matrix_heatmap_core,
)
from scptensor.viz.recipes import (
    plot_qc_completeness as plot_qc_completeness_core,
)
from scptensor.viz.recipes import (
    plot_qc_filtering_summary as plot_qc_filtering_summary_core,
)
from scptensor.viz.recipes import (
    plot_recent_operations as plot_recent_operations_core,
)


def test_stable_viz_namespace_reexports_canonical_plotting_surface() -> None:
    assert stable_viz.__all__ == [
        "heatmap",
        "violin",
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
        "generate_analysis_report",
        "ReportTheme",
    ]
    assert plot_embedding_scatter is plot_embedding_scatter_core
    assert plot_qc_completeness is plot_qc_completeness_core
    assert plot_imputation_comparison is plot_imputation_comparison_core
    assert plot_matrix_heatmap is plot_matrix_heatmap_core
    assert plot_data_overview is plot_data_overview_core
    assert plot_qc_filtering_summary is plot_qc_filtering_summary_core
    assert plot_embedding_panels is plot_embedding_panels_core
    assert plot_recent_operations is plot_recent_operations_core
    assert generate_analysis_report is generate_analysis_report_core
    assert ReportTheme is ReportThemeCore


def test_stable_viz_namespace_keeps_only_unambiguous_base_primitives() -> None:
    assert heatmap is heatmap_core
    assert violin is violin_core


def test_stable_viz_namespace_does_not_reexport_recipe_aliases() -> None:
    for name in (
        "base_scatter",
        "scatter",
        "embedding",
        "umap",
        "pca",
        "tsne",
        "qc_completeness",
        "qc_matrix_spy",
    ):
        assert name not in stable_viz.__all__
        assert not hasattr(stable_viz, name)


def test_root_package_does_not_reexport_visualization_surface() -> None:
    for name in (
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
    ):
        assert name not in scp.__all__
        assert not hasattr(scp, name)

    for name in (
        "plot_embedding_scatter",
        "plot_qc_completeness",
        "plot_imputation_comparison",
        "plot_matrix_heatmap",
        "generate_analysis_report",
        "ReportTheme",
    ):
        assert name not in scp.__all__
        assert not hasattr(scp, name)


def test_root_visualization_convenience_exports_are_removed() -> None:
    for name in (
        "scatter",
        "heatmap",
        "violin",
        "embedding",
        "qc_completeness",
        "qc_matrix_spy",
        "plot_data_overview",
        "plot_qc_filtering_summary",
        "plot_embedding_panels",
        "plot_recent_operations",
    ):
        assert not hasattr(scp, name)
