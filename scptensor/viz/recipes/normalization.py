"""Normalization visualization recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def plot_normalization_comparison(
    container: "ScpContainer",
    assay_name: str = "protein",
    layers: list[str] | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot comparison of normalization methods.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Assay name.
    layers : list[str] | None, optional
        Layers to compare.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if layers is None:
        layers = ["raw", "log", "normalized"]

    # Box plot of values
    data = []
    labels = []
    for layer in layers:
        if assay_name in container.assays and layer in container.assays[assay_name].layers:
            X = container.assays[assay_name].layers[layer].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            data.append(X.flatten()[:1000])  # Sample for speed
            labels.append(layer)

    if data:
        axes[0].boxplot(data, labels=labels)
        axes[0].set_title("Value Distribution")
        axes[0].set_ylabel("Intensity")

    # Density plot
    for i, (d, label) in enumerate(zip(data, labels)):
        axes[1].hist(d, bins=50, alpha=0.5, label=label, density=True)
    axes[1].set_title("Value Density")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_normalization_diagnostics(
    container: "ScpContainer",
    assay_name: str = "protein",
    layer_name: str = "log",
    **kwargs,
) -> plt.Figure:
    """Plot normalization diagnostics.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Assay name.
    layer_name : str, default="log"
        Layer name.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    if assay_name not in container.assays:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        return fig

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        return fig

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Value distribution
    axes[0, 0].hist(X.flatten()[:10000], bins=100)
    axes[0, 0].set_title("Value Distribution")
    axes[0, 0].set_xlabel("Value")

    # Missing value pattern
    missing = np.isnan(X)
    axes[0, 1].imshow(missing[:50, :50], aspect="auto", cmap="RdBu")
    axes[0, 1].set_title("Missing Value Pattern (subset)")

    # Per-sample statistics
    sample_means = np.nanmean(X, axis=1)
    axes[1, 0].hist(sample_means, bins=50)
    axes[1, 0].set_title("Per-Sample Mean Distribution")
    axes[1, 0].set_xlabel("Mean")

    # Per-feature statistics
    feature_means = np.nanmean(X, axis=0)
    axes[1, 1].hist(feature_means, bins=50)
    axes[1, 1].set_title("Per-Feature Mean Distribution")
    axes[1, 1].set_xlabel("Mean")

    plt.tight_layout()
    return fig


def plot_normalization_effect(
    container: "ScpContainer",
    assay_name: str = "protein",
    before_layer: str = "raw",
    after_layer: str = "log",
    **kwargs,
) -> plt.Figure:
    """Compare before and after normalization.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Assay name.
    before_layer : str, default="raw"
        Layer before normalization.
    after_layer : str, default="log"
        Layer after normalization.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if assay_name not in container.assays:
        for ax in axes:
            ax.text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        return fig

    assay = container.assays[assay_name]

    # Before
    if before_layer in assay.layers:
        X_before = assay.layers[before_layer].X
        if hasattr(X_before, "toarray"):
            X_before = X_before.toarray()
        axes[0].hist(X_before.flatten()[:10000], bins=100)
        axes[0].set_title(f"Before ({before_layer})")
        axes[0].set_xlabel("Value")
    else:
        axes[0].text(0.5, 0.5, f"Layer '{before_layer}' not found", ha="center")

    # After
    if after_layer in assay.layers:
        X_after = assay.layers[after_layer].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()
        axes[1].hist(X_after.flatten()[:10000], bins=100)
        axes[1].set_title(f"After ({after_layer})")
        axes[1].set_xlabel("Value")
    else:
        axes[1].text(0.5, 0.5, f"Layer '{after_layer}' not found", ha="center")

    plt.tight_layout()
    return fig


__all__ = ["plot_normalization_comparison", "plot_normalization_diagnostics", "plot_normalization_effect"]
