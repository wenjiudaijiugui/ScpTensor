from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from .style import setup_style

MaskStyle = Literal["subtle", "explicit", "none"]


def scatter(
    X: np.ndarray,
    c: np.ndarray | None = None,
    m: np.ndarray | None = None,
    mask_style: MaskStyle = "subtle",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Create a scatter plot with mask-aware rendering.

    Parameters
    ----------
    X : np.ndarray
        Coordinates of shape (N, 2).
    c : np.ndarray | None
        Color values of shape (N,).
    m : np.ndarray | None
        Mask values (N,), where 0=Valid, non-zero=Missing/Imputed.
    mask_style : {"subtle", "explicit", "none"}
        How to render masked points:
        - "subtle": alpha coding (valid=opaque, invalid=transparent)
        - "explicit": shape coding (valid=circle, invalid=x)
        - "none": no distinction
    ax : plt.Axes | None
        Matplotlib axes. If None, creates new figure.
    title : str | None
        Plot title.
    xlabel : str | None
        X-axis label.
    ylabel : str | None
        Y-axis label.
    **kwargs : Any
        Passed to ``scatter``.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    """
    setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Default: all points valid
    if m is None:
        m = np.zeros(X.shape[0], dtype=np.int8)

    valid = m == 0
    invalid = ~valid

    base_kwargs = {"s": 20, "edgecolor": "none", **kwargs}

    if mask_style == "subtle":
        _scatter_subtle(ax, X, c, valid, invalid, base_kwargs)
    elif mask_style == "explicit":
        _scatter_explicit(ax, X, c, valid, invalid, base_kwargs)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=c, **base_kwargs)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def _scatter_subtle(
    ax: plt.Axes,
    X: np.ndarray,
    c: np.ndarray | None,
    valid: np.ndarray,
    invalid: np.ndarray,
    kwargs: dict,
) -> None:
    """Render scatter with alpha-based mask distinction."""
    if np.any(valid):
        c_valid = None if c is None else c[valid]
        ax.scatter(
            X[valid, 0], X[valid, 1],
            c=c_valid, alpha=1.0, zorder=10, label="Measured", **kwargs
        )

    if np.any(invalid):
        c_invalid = "gray" if c is None else c[invalid]
        ax.scatter(
            X[invalid, 0], X[invalid, 1],
            c=c_invalid, alpha=0.3, zorder=0, label="Imputed/Missing", **kwargs
        )


def _scatter_explicit(
    ax: plt.Axes,
    X: np.ndarray,
    c: np.ndarray | None,
    valid: np.ndarray,
    invalid: np.ndarray,
    kwargs: dict,
) -> None:
    """Render scatter with marker-based mask distinction."""
    if np.any(valid):
        c_valid = None if c is None else c[valid]
        ax.scatter(
            X[valid, 0], X[valid, 1],
            c=c_valid, marker="o", label="Measured", **kwargs
        )

    if np.any(invalid):
        c_invalid = "gray" if c is None else c[invalid]
        ax.scatter(
            X[invalid, 0], X[invalid, 1],
            c=c_invalid, marker="x", label="Imputed/Missing", **kwargs
        )
