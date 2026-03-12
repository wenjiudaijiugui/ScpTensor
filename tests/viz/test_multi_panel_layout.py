"""Tests for PanelLayout multi-panel behavior and edge cases."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scptensor.viz.base import PanelLayout


def test_auto_grid_preserves_scatter_when_expanding():
    """Auto-grid expansion should re-render existing non-line artists."""
    plt.close("all")
    layout = PanelLayout(figsize=(8, 4), grid=None)

    layout.add_panel(0, lambda ax: ax.scatter([0, 1], [1, 0], c=[0.1, 0.9]))
    layout.add_panel(1, lambda ax: ax.imshow(np.arange(4).reshape(2, 2), cmap="viridis"))

    assert len(layout.axes) == 2
    assert len(layout.axes[0].collections) > 0  # scatter preserved after recreate
    assert len(layout.axes[1].images) > 0
    plt.close("all")


def test_auto_grid_high_index_capacity_is_stable():
    """High integer panel index should allocate enough axes and finalize cleanly."""
    plt.close("all")
    layout = PanelLayout(grid=None)
    layout.add_panel(5, lambda ax: ax.plot([0, 1], [1, 0]))

    fig = layout.finalize()
    assert fig is not None
    assert len(layout.axes) == 6
    assert len(layout.axes[5].lines) == 1
    plt.close("all")


def test_add_panel_negative_index_raises():
    """Negative panel index should be rejected explicitly."""
    plt.close("all")
    layout = PanelLayout(grid=(1, 1))
    with pytest.raises(IndexError, match="non-negative"):
        layout.add_panel(-1, lambda ax: ax.plot([0], [0]))
    plt.close("all")


def test_tuple_position_requires_explicit_grid_in_auto_mode():
    """Tuple positions are only supported with explicit fixed grid."""
    plt.close("all")
    layout = PanelLayout(grid=None)
    with pytest.raises(ValueError, match="explicit grid"):
        layout.add_panel((0, 0), lambda ax: ax.plot([0], [0]))
    plt.close("all")


def test_add_colorbar_invalid_position_raises():
    """Unsupported colorbar position should raise ValueError."""
    plt.close("all")
    layout = PanelLayout(grid=(1, 1))
    layout.add_panel(0, lambda ax: ax.imshow(np.eye(3), cmap="magma"))

    with pytest.raises(ValueError, match="position must be one of"):
        layout.add_colorbar(position="center")
    plt.close("all")


def test_add_colorbar_adds_labeled_axis():
    """Colorbar should be added from first tracked mappable."""
    plt.close("all")
    layout = PanelLayout(grid=(1, 1))
    layout.add_panel(0, lambda ax: ax.imshow(np.eye(3), cmap="magma"))
    layout.add_colorbar(label="Intensity")
    fig = layout.finalize()

    assert fig is not None
    assert len(fig.axes) >= 2  # main axis + colorbar axis
    labels = [ax.get_ylabel() for ax in fig.axes] + [ax.get_xlabel() for ax in fig.axes]
    assert "Intensity" in labels
    plt.close("all")


def test_finalize_invalid_legend_position_raises():
    """Unsupported legend position should fail during finalize."""
    plt.close("all")
    layout = PanelLayout(grid=(1, 1))
    layout.add_panel(0, lambda ax: ax.plot([0, 1], [1, 0], label="line"))
    layout.add_legend(position="middle")

    with pytest.raises(ValueError, match="legend position must be one of"):
        layout.finalize()
    plt.close("all")
