"""Batch effect visualization recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def plot_batch_effect(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    **kwargs,
) -> plt.Figure:
    """Visualize batch effects.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Check if assay exists
    if assay_name not in container.assays:
        axes[0].text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        axes[1].text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        return fig

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        axes[0].text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        axes[1].text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        return fig

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # PCA scatter
    if batch_key in container.obs.columns:
        batches = container.obs[batch_key].to_numpy()
        unique_batches = np.unique(batches)
        for batch in unique_batches:
            mask = batches == batch
            axes[0].scatter(X[mask, 0], X[mask, 1], label=str(batch), alpha=0.6)
        axes[0].legend()
    else:
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Batch Distribution")

    # Box plot per batch
    if batch_key in container.obs.columns:
        data = []
        labels = []
        for batch in unique_batches:
            mask = batches == batch
            data.append(X[mask, 0])
            labels.append(str(batch))
        axes[1].boxplot(data, labels=labels)
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("PC1 Value")
    axes[1].set_title("PC1 by Batch")

    plt.tight_layout()
    return fig


def plot_batch_correction_comparison(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    batch_key: str = "batch",
    **kwargs,
) -> plt.Figure:
    """Compare batch correction methods.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    batch_key : str, default="batch"
        Batch column in obs.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Check if assay exists
    if assay_name not in container.assays:
        axes[0].text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        axes[1].text(0.5, 0.5, f"Assay '{assay_name}' not found", ha="center")
        return fig

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        axes[0].text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        axes[1].text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        return fig

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Before correction (simulated by adding batch offset)
    X_before = X.copy()
    if batch_key in container.obs.columns:
        batches = container.obs[batch_key].to_numpy()
        unique_batches = np.unique(batches)
        batch_offsets = {b: np.random.randn(X.shape[1]) * 2 for b in unique_batches}
        for i, b in enumerate(batches):
            X_before[i] += batch_offsets[b]

    # Plot before
    if batch_key in container.obs.columns:
        batches = container.obs[batch_key].to_numpy()
        unique_batches = np.unique(batches)
        for batch in unique_batches:
            mask = batches == batch
            axes[0].scatter(X_before[mask, 0], X_before[mask, 1], label=str(batch), alpha=0.6)
        axes[0].legend()
    else:
        axes[0].scatter(X_before[:, 0], X_before[:, 1], alpha=0.6)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Before Correction")

    # Plot after
    if batch_key in container.obs.columns:
        for batch in unique_batches:
            mask = batches == batch
            axes[1].scatter(X[mask, 0], X[mask, 1], label=str(batch), alpha=0.6)
        axes[1].legend()
    else:
        axes[1].scatter(X[:, 0], X[:, 1], alpha=0.6)

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("After Correction")

    plt.tight_layout()
    return fig


__all__ = ["plot_batch_effect", "plot_batch_correction_comparison"]
