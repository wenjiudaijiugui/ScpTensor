from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import setup_style


def violin(
    data: Sequence[np.ndarray],
    labels: Sequence[str],
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> plt.Axes:
    """Create a violin plot.

    Parameters
    ----------
    data : Sequence[np.ndarray]
        Sequence of arrays, each containing the data for one violin.
    labels : Sequence[str]
        Labels for each violin (x-axis).
    ax : plt.Axes | None
        Matplotlib axes. If None, creates new figure.
    title : str | None
        Plot title.
    ylabel : str | None
        Y-axis label.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    """
    setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(list(data), showmeans=False, showmedians=True)

    # Style the violin bodies
    face_color = "#D43F3A"
    for pc in parts["bodies"]:  # type: ignore[attr-defined]
        pc.set_facecolor(face_color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax
