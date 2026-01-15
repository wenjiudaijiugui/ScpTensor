"""Tests for report generation module."""

import matplotlib.pyplot as plt
import pytest
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
