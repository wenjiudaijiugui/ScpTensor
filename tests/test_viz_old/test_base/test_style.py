"""Tests for PlotStyle in viz.base.style module."""

import pytest

from scptensor.viz.base.style import PlotStyle


def test_apply_default_style():
    """Test applying default science style."""
    PlotStyle.apply_style()
    import matplotlib.pyplot as plt

    assert not plt.rcParams["axes.unicode_minus"]
    # Check if science styles are available
    assert "science" in plt.style.available


def test_apply_custom_theme():
    """Test applying specific theme."""
    PlotStyle.apply_style(theme="ieee")
    import matplotlib.pyplot as plt

    assert "ieee" in plt.style.available


def test_get_colormap_default():
    """Test getting default colormap for purpose."""
    cmap = PlotStyle.get_colormap("expression")
    assert cmap == "viridis"


def test_get_colormap_custom():
    """Test custom colormap override."""
    cmap = PlotStyle.get_colormap("expression", custom="plasma")
    assert cmap == "plasma"


def test_get_colormap_missing():
    """Test missing value colormap."""
    cmap = PlotStyle.get_colormap("missing")
    assert cmap == "gray_r"


def test_invalid_purpose_raises():
    """Test invalid purpose raises error."""
    with pytest.raises(ValueError, match="Unknown purpose"):
        PlotStyle.get_colormap("invalid_purpose")


def test_invalid_theme_raises():
    """Test invalid theme raises error."""
    with pytest.raises(ValueError, match="Unknown theme"):
        PlotStyle.apply_style(theme="invalid_theme")


def test_all_colormap_purposes():
    """Test all predefined colormap purposes."""
    purposes = ["expression", "missing", "logfc", "significance", "clusters"]
    for purpose in purposes:
        cmap = PlotStyle.get_colormap(purpose)
        assert isinstance(cmap, str)
        assert len(cmap) > 0


def test_dpi_setting():
    """Test DPI is set correctly."""
    PlotStyle.apply_style(dpi=150)
    import matplotlib.pyplot as plt

    assert plt.rcParams["figure.dpi"] == 150
    assert plt.rcParams["savefig.dpi"] == 150
