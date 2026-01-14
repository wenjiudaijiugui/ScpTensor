"""Tests for multi-panel layout manager."""

import matplotlib.pyplot as plt
import numpy as np
from scptensor.viz.base.multi_panel import PanelLayout


def test_panel_layout_init():
    """Test PanelLayout initialization"""
    layout = PanelLayout(figsize=(10, 8))
    assert layout.figsize == (10, 8)
    assert layout.grid is None


def test_panel_layout_with_grid():
    """Test PanelLayout with explicit grid"""
    layout = PanelLayout(figsize=(12, 8), grid=(2, 2))
    assert layout.grid == (2, 2)


def test_add_panel():
    """Test adding a panel"""
    layout = PanelLayout(figsize=(10, 8), grid=(2, 2))
    ax = layout.add_panel(0, lambda ax: ax.plot([1, 2, 3]))
    assert ax is not None


def test_finalize_creates_figure():
    """Test finalize creates figure"""
    layout = PanelLayout(figsize=(10, 8), grid=(2, 2))
    layout.add_panel(0, lambda ax: ax.set_title("Panel 0"))
    fig = layout.finalize()
    assert isinstance(fig, plt.Figure)


def test_shared_colorbar():
    """Test shared colorbar"""
    layout = PanelLayout(figsize=(10, 8))
    im = np.random.rand(10, 10)
    layout.add_panel(0, lambda ax: ax.imshow(im, cmap="viridis"))
    layout.add_colorbar(position="right")
    layout.finalize()


def test_add_panel_with_tuple_position():
    """Test adding a panel with tuple position (row, col)"""
    layout = PanelLayout(figsize=(10, 8), grid=(2, 2))
    ax = layout.add_panel((0, 1), lambda ax: ax.set_title("Top Right"))
    assert ax is not None
    assert ax.get_title() == "Top Right"


def test_auto_grid_computation():
    """Test automatic grid computation when grid is None"""
    layout = PanelLayout(figsize=(10, 8))
    # Add 3 panels - should create 2x2 grid
    layout.add_panel(0, lambda ax: ax.plot([1, 2, 3]))
    layout.add_panel(1, lambda ax: ax.plot([1, 2, 3]))
    layout.add_panel(2, lambda ax: ax.plot([1, 2, 3]))
    assert layout.grid == (2, 2)


def test_add_legend():
    """Test adding legend"""
    layout = PanelLayout(figsize=(10, 8), grid=(2, 2))
    layout.add_panel(0, lambda ax: ax.plot([1, 2, 3], label="data"))
    layout.add_legend(labels=["data"])
    fig = layout.finalize()
    assert isinstance(fig, plt.Figure)


def test_figure_property():
    """Test figure property access"""
    layout = PanelLayout(figsize=(10, 8))
    fig = layout.figure
    assert isinstance(fig, plt.Figure)


def test_axes_property():
    """Test axes property access"""
    layout = PanelLayout(figsize=(10, 8), grid=(2, 2))
    layout.add_panel(0, lambda ax: ax.plot([1, 2, 3]))
    axes = layout.axes
    assert len(axes) == 4  # 2x2 grid
