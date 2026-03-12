from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import PlotStyle, setup_style


def heatmap(
    X: np.ndarray,
    m: np.ndarray | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    colorbar: bool = True,
    cbar_label: str | None = None,
    xtick_rotation: int = 60,
    ytick_rotation: int = 0,
    **kwargs: Any,
) -> plt.Axes:
    """Create a heatmap with optional mask hatching.

    Parameters
    ----------
    X : np.ndarray
        2D array of values to plot.
    m : np.ndarray | None
        Mask array (same shape as X). Non-zero values are hatched.
    xticklabels : Sequence[str] | None
        Labels for x-axis ticks.
    yticklabels : Sequence[str] | None
        Labels for y-axis ticks.
    ax : plt.Axes | None
        Matplotlib axes. If None, creates new figure.
    cmap : str | None
        Colormap name. If None, uses proteomics expression default.
    colorbar : bool
        Whether to draw colorbar.
    cbar_label : str | None
        Optional colorbar label.
    xtick_rotation : int
        Rotation angle (degrees) for x tick labels.
    ytick_rotation : int
        Rotation angle (degrees) for y tick labels.
    **kwargs : Any
        Passed to ``imshow``.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    """
    setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if cmap is None:
        cmap = PlotStyle.get_colormap("expression")
    if m is not None and m.shape != X.shape:
        raise ValueError(f"Mask shape {m.shape} does not match matrix shape {X.shape}.")

    title = kwargs.pop("title", None)
    im = ax.imshow(X, aspect="auto", cmap=cmap, interpolation="nearest", **kwargs)

    if title:
        ax.set_title(title)

    # Draw hatching over masked regions
    if m is not None:
        rows, cols = X.shape
        x = np.arange(cols + 1)
        y = np.arange(rows + 1)
        masked_data = np.ma.masked_where(m == 0, m)
        ax.pcolormesh(x, y, masked_data, hatch="////", alpha=0.0, shading="flat")

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        x_align = "right" if xtick_rotation else "center"
        ax.set_xticklabels(xticklabels, rotation=xtick_rotation, ha=x_align)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, rotation=ytick_rotation)

    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        if cbar_label:
            cbar.set_label(cbar_label)

    return ax
