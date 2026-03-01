"""Clustering visualization recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def plot_clustering(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    cluster_key: str = "leiden",
    **kwargs,
) -> plt.Figure:
    """Visualize clustering results.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    cluster_key : str, default="leiden"
        Cluster column in obs.

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

    # Cluster scatter
    if cluster_key in container.obs.columns:
        clusters = container.obs[cluster_key].to_numpy()
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            mask = clusters == cluster
            axes[0].scatter(X[mask, 0], X[mask, 1], label=str(cluster), alpha=0.6)
        axes[0].legend(title=cluster_key, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Clustering Results")

    # Cluster size bar plot
    if cluster_key in container.obs.columns:
        cluster_counts = container.obs[cluster_key].value_counts().sort_index()
        axes[1].bar(cluster_counts.to_numpy(), cluster_counts.to_list())
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Cluster Sizes")
    else:
        axes[1].text(0.5, 0.5, f"Cluster key '{cluster_key}' not found", ha="center")

    plt.tight_layout()
    return fig


def plot_clustering_optimization(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    cluster_keys: list[str] | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot clustering optimization across different parameters.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    cluster_keys : list[str] | None, optional
        List of cluster column names to compare.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if cluster_keys is None:
        cluster_keys = [k for k in container.obs.columns if "leiden" in k or "kmeans" in k]

    if not cluster_keys:
        axes[0].text(0.5, 0.5, "No cluster keys found", ha="center")
        axes[1].text(0.5, 0.5, "No cluster keys found", ha="center")
        return fig

    # Number of clusters per method
    n_clusters = []
    labels = []
    for key in cluster_keys:
        if key in container.obs.columns:
            n = container.obs[key].n_unique()
            n_clusters.append(n)
            labels.append(key)

    axes[0].bar(range(len(labels)), n_clusters)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("Number of Clusters")
    axes[0].set_title("Cluster Count Comparison")

    # Cluster size distribution for first key
    if cluster_keys[0] in container.obs.columns:
        sizes = container.obs[cluster_keys[0]].value_counts().sort_index()
        axes[1].bar(range(len(sizes)), sizes.to_list())
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Size")
        axes[1].set_title(f"Cluster Sizes ({cluster_keys[0]})")

    plt.tight_layout()
    return fig


def plot_clustering_quality(
    container: "ScpContainer",
    assay_name: str = "pca",
    layer_name: str = "X",
    cluster_key: str = "leiden",
    **kwargs,
) -> plt.Figure:
    """Plot clustering quality metrics.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Assay name.
    layer_name : str, default="X"
        Layer name.
    cluster_key : str, default="leiden"
        Cluster column in obs.

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
    if layer_name not in assay.layers:
        for ax in axes:
            ax.text(0.5, 0.5, f"Layer '{layer_name}' not found", ha="center")
        return fig

    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    if cluster_key not in container.obs.columns:
        for ax in axes:
            ax.text(0.5, 0.5, f"Cluster key '{cluster_key}' not found", ha="center")
        return fig

    clusters = container.obs[cluster_key].to_numpy()
    unique_clusters = np.unique(clusters)

    # Cluster sizes pie chart
    sizes = [np.sum(clusters == c) for c in unique_clusters]
    axes[0].pie(sizes, labels=unique_clusters, autopct="%1.1f%%")
    axes[0].set_title("Cluster Proportions")

    # Silhouette-like visualization (simplified)
    # Show distance to cluster center for each sample
    for i, cluster in enumerate(unique_clusters[:5]):  # Limit to 5 clusters
        mask = clusters == cluster
        if np.sum(mask) > 0:
            cluster_center = np.mean(X[mask], axis=0)
            distances = np.linalg.norm(X[mask] - cluster_center, axis=1)
            axes[1].scatter([i] * len(distances), distances, alpha=0.5, label=str(cluster))

    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Distance to Center")
    axes[1].set_title("Intra-cluster Spread")
    if unique_clusters[:5].size > 0:
        axes[1].legend()

    plt.tight_layout()
    return fig


__all__ = ["plot_clustering", "plot_clustering_optimization", "plot_clustering_quality"]
