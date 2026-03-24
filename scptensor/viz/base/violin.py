from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .style import setup_style


def violin(
    data: Sequence[np.ndarray],
    labels: Sequence[str],
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.65,
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
    alpha : float
        Violin body transparency.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    """
    setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if len(data) != len(labels):
        raise ValueError(f"Data groups ({len(data)}) and labels ({len(labels)}) must match.")

    prepared: list[np.ndarray] = []
    for group in data:
        arr = np.asarray(group, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([-1e-9, 1e-9], dtype=np.float64)
        elif arr.size == 1:
            arr = np.array([arr[0] - 1e-9, arr[0] + 1e-9], dtype=np.float64)
        elif np.std(arr) == 0:
            arr = arr + np.linspace(-1e-9, 1e-9, arr.size)
        prepared.append(arr)

    parts = ax.violinplot(prepared, showmeans=False, showmedians=True)

    # Style violins with publication-friendly categorical colors.
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C72B0"])
    bodies = cast("list[PolyCollection]", parts["bodies"])
    for idx, body in enumerate(bodies):
        body.set_facecolor(palette[idx % len(palette)])
        body.set_edgecolor("#222222")
        body.set_linewidth(0.7)
        body.set_alpha(alpha)

    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.2)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#555555")
            parts[key].set_linewidth(0.8)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax
