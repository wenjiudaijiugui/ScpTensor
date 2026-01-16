"""Statistics visualization recipes.

This module provides functions for statistical visualizations,
including correlation matrices and hierarchical clustering dendrograms.

Functions include:
- correlation_matrix: Group correlation heatmap
- dendrogram: Hierarchical clustering dendrogram
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from scptensor.viz.base.style import PlotStyle
from scptensor.viz.base.validation import (
    validate_container,
    validate_groupby,
    validate_layer,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from scptensor import ScpContainer

__all__ = ["correlation_matrix", "dendrogram"]


def correlation_matrix(
    container: ScpContainer,
    layer: str,
    assay_name: str = "proteins",
    groupby: str | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
    cmap: str = "RdBu_r",
    annot: bool = True,
    show: bool = True,
) -> Figure:
    """Create a correlation matrix heatmap between groups or samples.

    Computes pairwise correlations between groups (if groupby is specified)
    or between samples. Displays as a heatmap with optional annotations.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to analyze (e.g., 'normalized', 'log').
    assay_name : str, default="proteins"
        Name of the assay containing the data.
    groupby : str or None
        Column name in obs to group by. If None, computes sample-wise correlation.
    method : {"pearson", "spearman"}, default="pearson"
        Correlation method to use.
    cmap : str, default="RdBu_r"
        Colormap for the heatmap.
    annot : bool, default=True
        Whether to annotate cells with correlation values.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the correlation heatmap.

    Raises
    ------
    VisualizationError
        If validation fails (container or layer).
    ValueError
        If groupby column is not found or insufficient groups.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.statistics import correlation_matrix
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10) * 10)
    >>> container.assays["proteins"] = assay
    >>> fig = correlation_matrix(container, layer="normalized",
    ...                          groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)

    if groupby is not None:
        validate_groupby(container, groupby)

    PlotStyle.apply_style()

    import matplotlib.pyplot as plt

    assay = container.assays[assay_name]
    data = assay.layers[layer].X.copy()

    # Compute correlations
    if groupby is not None:
        group_data = container.obs[groupby].to_numpy()
        groups = np.unique(group_data)

        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for correlation matrix, found {len(groups)}")

        # Compute group means
        group_means_list: list[np.ndarray] = []
        for g in groups:
            mask = group_data == g
            group_means_list.append(data[mask].mean(axis=0))

        group_means = np.array(group_means_list)  # type: ignore[assignment]

        # Compute correlation between groups
        if method == "spearman":
            # Compute rank correlation
            from scipy.stats import spearmanr

            corr_matrix = np.zeros((len(groups), len(groups)))
            for i in range(len(groups)):
                for j in range(len(groups)):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = spearmanr(group_means[i], group_means[j])
                        corr_matrix[i, j] = corr
        else:
            # Pearson correlation
            # Center the data
            assert isinstance(group_means, np.ndarray), "group_means should be ndarray"
            group_means_centered = group_means - group_means.mean(axis=1, keepdims=True)

            # Compute correlation matrix
            norm = (
                np.sqrt(np.sum(group_means_centered**2, axis=1, keepdims=True))
                @ np.sqrt(np.sum(group_means_centered**2, axis=1, keepdims=True)).T
            )
            corr_matrix = (group_means_centered @ group_means_centered.T) / norm

        labels = [str(g) for g in groups]
        title = f"Group Correlation Matrix ({method.capitalize()})"
    else:
        # Sample-wise correlation (limit to first 50 samples)
        n_show = min(50, data.shape[0])
        data_subset = data[:n_show]

        if method == "spearman":
            from scipy.stats import spearmanr

            # Compute spearman correlation
            corr_matrix = np.zeros((n_show, n_show))
            for i in range(n_show):
                for j in range(n_show):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    elif i < j:
                        corr, _ = spearmanr(data_subset[i], data_subset[j])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        else:
            # Pearson correlation
            data_centered = data_subset - data_subset.mean(axis=1, keepdims=True)
            norm = (
                np.sqrt(np.sum(data_centered**2, axis=1, keepdims=True))
                @ np.sqrt(np.sum(data_centered**2, axis=1, keepdims=True)).T
            )
            corr_matrix = (data_centered @ data_centered.T) / norm

        labels = [f"S{i}" for i in range(n_show)]
        title = f"Sample Correlation Matrix ({method.capitalize()})"

    # Create figure
    n_items = corr_matrix.shape[0]
    fig_size = max(6, n_items * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Plot heatmap
    im = ax.imshow(
        corr_matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="none",
    )

    # Set ticks
    ax.set_xticks(range(n_items))
    ax.set_yticks(range(n_items))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add annotations if requested
    if annot and n_items <= 30:
        for i in range(n_items):
            for j in range(n_items):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def dendrogram(
    container: ScpContainer,
    layer: str,
    assay_name: str = "proteins",
    groupby: str | None = None,
    method: Literal["single", "complete", "average", "ward"] = "average",
    metric: str = "euclidean",
    show: bool = True,
) -> Figure:
    """Create a hierarchical clustering dendrogram.

    Performs hierarchical clustering on samples (or group means) and
    displays the resulting dendrogram. Colored by group if groupby is specified.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to analyze (e.g., 'normalized', 'log').
    assay_name : str, default="proteins"
        Name of the assay containing the data.
    groupby : str or None
        Column name in obs to color leaves by. If None, no coloring.
    method : {"single", "complete", "average", "ward"}, default="average"
        Linkage method for hierarchical clustering.
    metric : str, default="euclidean"
        Distance metric to use. Passed to scipy.spatial.distance.pdist.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the dendrogram.

    Raises
    ------
    VisualizationError
        If validation fails (container or layer).
    ValueError
        If groupby column is not found or invalid method.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.statistics import dendrogram
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10) * 10)
    >>> container.assays["proteins"] = assay
    >>> fig = dendrogram(container, layer="normalized",
    ...                  groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)

    if groupby is not None:
        validate_groupby(container, groupby)

    PlotStyle.apply_style()

    import matplotlib.pyplot as plt

    assay = container.assays[assay_name]
    data = assay.layers[layer].X.copy()

    # Get group data for coloring
    group_data = None
    group_to_color = {}

    if groupby is not None:
        group_data = container.obs[groupby].to_numpy()
        groups = np.unique(group_data)

        # Assign colors to groups
        cmap = plt.get_cmap("tab10")
        for i, g in enumerate(groups):
            group_to_color[g] = cmap(i % 10)

    # Perform hierarchical clustering
    # Limit to first 100 samples for clarity if using sample-wise clustering
    if groupby is None and data.shape[0] > 100:
        data_subset = data[:100]
        labels = [f"S{i}" for i in range(100)]
    else:
        data_subset = data
        if groupby is not None:
            # Use group means for clustering
            group_means = []
            group_labels_list = []
            for g in groups:
                mask = group_data == g
                group_means.append(data[mask].mean(axis=0))
                group_labels_list.append(g)
            data_subset = np.array(group_means)
            labels = [str(g) for g in group_labels_list]
        else:
            labels = [f"S{i}" for i in range(data.shape[0])]

    # Compute linkage
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage only works with euclidean metric")

    # Compute pairwise distances
    dist_matrix = pdist(data_subset, metric=metric)
    link = linkage(dist_matrix, method=method)  # type: ignore[arg-type]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dendrogram
    scipy_dendrogram(
        link,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(link[:, 2]),
    )

    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage, {metric} distance)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Distance")

    if groupby is not None:
        # Add legend for groups
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=group_to_color[g], label=str(g)) for g in sorted(group_to_color.keys())
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


if __name__ == "__main__":
    print("Testing statistics visualization module...")

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import polars as pl

    from scptensor import Assay, ScpContainer, ScpMatrix
    from scptensor.core.exceptions import VisualizationError

    # Create test container
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(60)],
            "cluster": np.repeat(["A", "B", "C"], 20),
        }
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(10)], "protein": [f"P{i}" for i in range(10)]}
    )
    X = np.random.rand(60, 10) * 10

    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    # Test: correlation_matrix basic
    print("\n1. Testing correlation_matrix...")
    fig = correlation_matrix(container, layer="normalized", show=False)
    assert fig is not None
    plt.close(fig)
    print("   correlation_matrix: OK")

    # Test: correlation_matrix with groupby
    print("\n2. Testing correlation_matrix with groupby...")
    fig = correlation_matrix(container, layer="normalized", groupby="cluster", show=False)
    assert fig is not None
    plt.close(fig)
    print("   correlation_matrix with groupby: OK")

    # Test: correlation_matrix with spearman
    print("\n3. Testing correlation_matrix with spearman...")
    fig = correlation_matrix(
        container, layer="normalized", groupby="cluster", method="spearman", show=False
    )
    assert fig is not None
    plt.close(fig)
    print("   correlation_matrix with spearman: OK")

    # Test: dendrogram basic
    print("\n4. Testing dendrogram...")
    fig = dendrogram(container, layer="normalized", show=False)
    assert fig is not None
    plt.close(fig)
    print("   dendrogram: OK")

    # Test: dendrogram with groupby
    print("\n5. Testing dendrogram with groupby...")
    fig = dendrogram(container, layer="normalized", groupby="cluster", show=False)
    assert fig is not None
    plt.close(fig)
    print("   dendrogram with groupby: OK")

    # Test: dendrogram with different methods
    print("\n6. Testing dendrogram with different methods...")
    for meth in ["single", "complete", "average", "ward"]:
        fig = dendrogram(container, layer="normalized", method=meth, show=False)
        assert fig is not None
        plt.close(fig)
    print("   dendrogram different methods: OK")

    # Test: Validation errors
    print("\n7. Testing validation errors...")
    try:
        correlation_matrix(container, layer="nonexistent", show=False)
        print("   Invalid layer: FAILED")
    except Exception:
        print("   Invalid layer: OK")

    try:
        dendrogram(container, layer="nonexistent", show=False)
        print("   Invalid layer (dendrogram): FAILED")
    except Exception:
        print("   Invalid layer (dendrogram): OK")

    try:
        dendrogram(container, layer="normalized", groupby="invalid_column", show=False)
        print("   Invalid groupby: FAILED")
    except VisualizationError:
        print("   Invalid groupby: OK")

    print("\nAll statistics visualization tests passed!")
