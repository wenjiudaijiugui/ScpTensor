from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import setup_style


def heatmap(
    X: np.ndarray,
    m: np.ndarray | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
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
    cmap : str
        Colormap name.
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

    title = kwargs.pop("title", None)
    im = ax.imshow(X, aspect="auto", cmap=cmap, **kwargs)

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
        ax.set_xticklabels(xticklabels, rotation=90)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    plt.colorbar(im, ax=ax)

    return ax
