"""Tests for stable visualization namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.viz as stable_viz
from scptensor.viz import (
    ReportTheme,
    embedding,
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
    qc_completeness,
    qc_matrix_spy,
    scatter,
    violin,
)
from scptensor.viz.base.heatmap import heatmap as heatmap_core
from scptensor.viz.base.violin import violin as violin_core
from scptensor.viz.recipes import (
    ReportTheme as ReportThemeCore,
)
from scptensor.viz.recipes import (
    embedding as embedding_core,
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
from scptensor.viz.recipes import (
    qc_completeness as qc_completeness_core,
)
from scptensor.viz.recipes import (
    qc_matrix_spy as qc_matrix_spy_core,
)
from scptensor.viz.recipes import (
    scatter as recipe_scatter_core,
)


def test_stable_viz_namespace_reexports_canonical_plotting_surface() -> None:
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


def test_stable_viz_namespace_keeps_base_and_alias_exports() -> None:
    assert heatmap is heatmap_core
    assert violin is violin_core
    assert stable_viz.base_scatter is stable_viz._base_scatter
    assert scatter is recipe_scatter_core
    assert embedding is embedding_core
    assert qc_completeness is qc_completeness_core
    assert qc_matrix_spy is qc_matrix_spy_core


def test_root_package_only_reexports_visualization_subset() -> None:
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
        assert name in scp.__all__

    for name in (
        "plot_embedding_scatter",
        "plot_qc_completeness",
        "plot_imputation_comparison",
        "plot_matrix_heatmap",
        "generate_analysis_report",
        "ReportTheme",
    ):
        assert name not in scp.__all__


def test_root_visualization_reexports_match_current_subset_objects() -> None:
    assert scp.scatter is recipe_scatter_core
    assert scp.heatmap is heatmap_core
    assert scp.violin is violin_core
    assert scp.embedding is embedding_core
    assert scp.qc_completeness is qc_completeness_core
    assert scp.qc_matrix_spy is qc_matrix_spy_core
    assert scp.plot_data_overview is plot_data_overview_core
    assert scp.plot_qc_filtering_summary is plot_qc_filtering_summary_core
    assert scp.plot_embedding_panels is plot_embedding_panels_core
    assert scp.plot_recent_operations is plot_recent_operations_core
