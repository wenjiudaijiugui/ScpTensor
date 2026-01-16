"""Tests for report generation module."""

import matplotlib.pyplot as plt

from scptensor.viz.recipes.report import ReportTheme


def test_theme_default_values():
    """Test ReportTheme default values."""
    theme = ReportTheme()
    assert theme.figsize == (16, 12)
    assert theme.dpi == 300
    assert theme.primary_color == "#1f77b4"
    assert theme.title_fontsize == 14


def test_theme_dark_preset():
    """Test ReportTheme dark mode preset."""
    theme = ReportTheme.dark()
    assert theme.primary_color == "#4fc3f7"
    assert theme.cmap_missing == "Oranges"


def test_theme_colorblind_preset():
    """Test ReportTheme colorblind-friendly preset."""
    theme = ReportTheme.colorblind()
    assert theme.primary_color == "#0072B2"
    assert theme.cmap_cluster == "cividis"


def test_generate_report_basic(sample_container):
    """Test generate_analysis_report basic functionality."""
    from scptensor.viz.recipes.report import generate_analysis_report

    fig = generate_analysis_report(sample_container)
    assert fig is not None
    assert fig.get_size_inches()[0] >= 16
    plt.close(fig)


def test_render_overview_panel(sample_container):
    """Test overview panel rendering."""
    from scptensor.viz.recipes.report import _render_overview_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_overview_panel(ax, sample_container)

    # Check that table was rendered (table is a child of axes)
    tables = [child for child in ax.get_children() if hasattr(child, "get_celld")]
    assert len(tables) > 0
    plt.close(fig)


def test_render_qc_panel(sample_container):
    """Test QC panel rendering."""
    from scptensor.viz.recipes.report import _render_qc_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_qc_panel(ax, sample_container)

    # Check violin plot was created
    assert len(ax.collections) > 0
    plt.close(fig)


def test_render_missing_panel(sample_container):
    """Test missing panel rendering."""
    from scptensor.viz.recipes.report import _render_missing_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_missing_panel(ax, sample_container)

    # Check heatmap was created
    assert len(ax.images) > 0
    plt.close(fig)


def test_render_embedding_panel(sample_container):
    """Test embedding panel rendering."""
    from scptensor.viz.recipes.report import _render_embedding_panel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _render_embedding_panel(ax1, ax2, sample_container, "proteins", "group")

    # Check scatter plots were created
    assert len(ax1.collections) > 0
    assert len(ax2.collections) > 0
    plt.close(fig)


def test_render_feature_panel(sample_container):
    """Test feature panel rendering."""
    from scptensor.viz.recipes.report import _render_feature_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_feature_panel(ax, sample_container)

    assert len(ax.collections) > 0
    plt.close(fig)


def test_render_cluster_panel(sample_container):
    """Test cluster panel rendering."""
    from scptensor.viz.recipes.report import _render_cluster_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_cluster_panel(ax, sample_container)

    assert len(ax.images) > 0
    plt.close(fig)


def test_render_batch_panel(sample_container):
    """Test batch panel rendering."""
    from scptensor.viz.recipes.report import _render_batch_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_batch_panel(ax, sample_container)

    assert len(ax.collections) > 0
    plt.close(fig)


def test_render_diff_expr_panel(sample_container):
    """Test diff expr panel rendering."""
    from scptensor.viz.recipes.report import _render_diff_expr_panel

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _render_diff_expr_panel(ax, sample_container, group1="A", group2="B")

    assert len(ax.collections) > 0
    plt.close(fig)


def test_full_report_generation(sample_container):
    """Test complete report generation."""
    from scptensor.viz.recipes.report import generate_analysis_report

    fig = generate_analysis_report(sample_container)
    assert fig is not None
    # Note: colorbar adds extra axes, so we check for at least 8
    assert len(fig.axes) >= 8
    plt.close(fig)
