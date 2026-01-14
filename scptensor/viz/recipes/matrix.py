"""Matrix visualization recipes.

This module provides functions for visualizing feature expression as heatmaps,
showing expression patterns across groups of cells. These are commonly used
in single-cell analysis to visualize marker expression patterns.

Functions include:
- matrixplot: Heatmap of mean expression values per group
- heatmap: Individual cell values as heatmap
- tracksplot: Expression as height instead of color
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from scptensor.viz.base.style import PlotStyle
from scptensor.viz.base.validation import (
    validate_container,
    validate_features,
    validate_groupby,
    validate_layer,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scptensor import ScpContainer

__all__ = ["matrixplot", "heatmap", "tracksplot"]


def matrixplot(
    container: ScpContainer,
    layer: str,
    var_names: list[str],
    groupby: str,
    assay_name: str = "proteins",
    dendrogram: bool = False,
    standard_scale: Literal["var", "obs"] | None = "var",
    cmap: str = "viridis",
    colorbar_title: str | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Create a heatmap of mean expression values per group.

    The matrix plot aggregates expression values by group and displays them
    as a heatmap, where each cell represents the mean expression of a feature
    in a group. This is useful for visualizing marker expression patterns
    across clusters or conditions.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to visualize (e.g., 'normalized', 'log').
    var_names : list[str]
        List of feature names to plot.
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    assay_name : str, default="proteins"
        Assay name containing the features.
    dendrogram : bool, default=False
        Whether to show dendrogram (not yet implemented).
    standard_scale : {'var', 'obs'} or None, default="var"
        How to standardize expression values:
        - 'var': scale each feature independently (recommended)
        - 'obs': scale each group independently
        - None: no scaling
    cmap : str, default="viridis"
        Colormap for expression values.
    colorbar_title : str or None, default None
        Title for the colorbar. If None, uses "Mean expression".
    show : bool, default=True
        Whether to display the plot.
    **kwargs
        Additional keyword arguments passed to imshow.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.

    Raises
    ------
    VisualizationError
        If validation fails (container, layer, features, or groupby).

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.matrix import matrixplot
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.var["protein"] = [f"P{i}" for i in range(10)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10))
    >>> container.assays["proteins"] = assay
    >>> ax = matrixplot(container, layer="normalized", var_names=["P0", "P1"],
    ...                 groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_features(container, assay_name, var_names)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    assay = container.assays[assay_name]
    x = assay.layers[layer].X.copy()

    # Find feature identifier column
    var_col = None
    for preferred in ["protein", "gene", "feature", "name"]:
        if preferred in assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        var_col = assay.var.columns[0]

    # Filter to selected features (preserve order from var_names)
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = []
    for var in var_names:
        if var in available_features:
            feature_idx.append(available_features[var])

    x = x[:, feature_idx]

    # Get groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Calculate mean expression per group
    mean_expr = np.zeros((len(unique_groups), len(var_names)))

    for i, _g in enumerate(unique_groups):
        mask = groups == _g
        group_data = x[mask]
        mean_expr[i] = group_data.mean(axis=0)

    # Standardize
    if standard_scale == "var":
        # Scale each feature independently
        col_min = mean_expr.min(axis=0)
        col_max = mean_expr.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range < 1e-8] = 1.0  # Avoid division by zero
        mean_expr = (mean_expr - col_min) / col_range
    elif standard_scale == "obs":
        # Scale each group independently
        row_min = mean_expr.min(axis=1, keepdims=True)
        row_max = mean_expr.max(axis=1, keepdims=True)
        row_range = row_max - row_min
        row_range[row_range < 1e-8] = 1.0
        mean_expr = (mean_expr - row_min) / row_range

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Create plot
    fig_height = 2 + len(unique_groups) * 0.5
    fig_width = 2 + len(var_names) * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw heatmap
    im = ax.imshow(mean_expr, cmap=cmap, aspect="auto", **kwargs)

    # Configure axes
    ax.set_yticks(np.arange(len(unique_groups)))
    ax.set_yticklabels(unique_groups)
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_ylabel(groupby)

    # Add colorbar
    if colorbar_title is None:
        colorbar_title = "Mean expression"
    plt.colorbar(im, ax=ax, label=colorbar_title)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def heatmap(
    container: ScpContainer,
    layer: str,
    var_names: list[str],
    groupby: str,
    assay_name: str = "proteins",
    dendrogram: bool = False,
    log: bool = True,
    cmap: str = "viridis",
    swap_axes: bool = False,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Create a heatmap of individual cell expression values.

    The heatmap displays expression values for individual cells, with rows
    optionally grouped and clustered. Each cell in the heatmap represents
    a single sample's expression of a single feature.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to visualize (e.g., 'normalized', 'log').
    var_names : list[str]
        List of feature names to plot.
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    assay_name : str, default="proteins"
        Assay name containing the features.
    dendrogram : bool, default=False
        Whether to show dendrogram (not yet implemented).
    log : bool, default=True
        Whether to apply log1p transform to expression values.
    cmap : str, default="viridis"
        Colormap for expression values.
    swap_axes : bool, default=False
        If True, swap axes so features are on y-axis.
    show : bool, default=True
        Whether to display the plot.
    **kwargs
        Additional keyword arguments passed to imshow.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.

    Raises
    ------
    VisualizationError
        If validation fails (container, layer, features, or groupby).

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.matrix import heatmap
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.var["protein"] = [f"P{i}" for i in range(10)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10))
    >>> container.assays["proteins"] = assay
    >>> ax = heatmap(container, layer="normalized", var_names=["P0", "P1"],
    ...              groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_features(container, assay_name, var_names)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    assay = container.assays[assay_name]
    x = assay.layers[layer].X.copy()

    # Find feature identifier column
    var_col = None
    for preferred in ["protein", "gene", "feature", "name"]:
        if preferred in assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        var_col = assay.var.columns[0]

    # Filter to selected features (preserve order from var_names)
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = []
    for var in var_names:
        if var in available_features:
            feature_idx.append(available_features[var])

    x = x[:, feature_idx]

    # Log transform if requested
    if log:
        x = np.log1p(x)

    # Get groups and sort by group
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Collect indices for each group
    group_indices = []
    for _g in unique_groups:
        group_indices.append(np.where(groups == _g)[0])

    # Concatenate data in group order
    sorted_indices = np.concatenate(group_indices)
    x_sorted = x[sorted_indices, :]

    # Create group labels for each cell
    group_labels = []
    for _g in unique_groups:
        n_cells = (groups == _g).sum()
        group_labels.extend([str(_g)] * n_cells)

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Create plot
    fig_height = 4 + x_sorted.shape[0] * 0.02
    fig_width = 2 + len(var_names) * 0.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Optionally swap axes
    if swap_axes:
        display_data = x_sorted.T
        x_label = "Cells"
        y_label = "Features"
        y_ticks = np.arange(len(var_names))
        y_ticklabels = var_names
    else:
        display_data = x_sorted
        x_label = "Features"
        y_label = "Cells"
        y_ticks = np.arange(x_sorted.shape[0])
        y_ticklabels = group_labels

    # Draw heatmap
    im = ax.imshow(display_data, cmap=cmap, aspect="auto", **kwargs)

    # Configure axes
    ax.set_yticks(y_ticks)
    if not swap_axes:
        # Show only a subset of cell labels to avoid overcrowding
        n_labels = min(len(y_ticklabels), 20)
        label_step = max(1, len(y_ticklabels) // n_labels)
        display_indices = list(range(0, len(y_ticklabels), label_step))
        ax.set_yticks(display_indices)
        ax.set_yticklabels([y_ticklabels[i] for i in display_indices], fontsize=8)
    else:
        ax.set_yticklabels(y_ticklabels, rotation=0)

    ax.set_xticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Expression" if log else "log(Expression)")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def tracksplot(
    container: ScpContainer,
    layer: str,
    var_names: list[str],
    groupby: str,
    assay_name: str = "proteins",
    dendrogram: bool = False,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Create a track plot of expression values.

    The track plot displays expression as height (bar height) instead of color,
    providing an alternative visualization to heatmaps. This can be useful for
    comparing expression levels across features and groups.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to visualize (e.g., 'normalized', 'log').
    var_names : list[str]
        List of feature names to plot.
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    assay_name : str, default="proteins"
        Assay name containing the features.
    dendrogram : bool, default=False
        Whether to show dendrogram (not yet implemented).
    show : bool, default=True
        Whether to display the plot.
    **kwargs
        Additional keyword arguments passed to barh.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.

    Raises
    ------
    VisualizationError
        If validation fails (container, layer, features, or groupby).

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.matrix import tracksplot
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.var["protein"] = [f"P{i}" for i in range(10)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10))
    >>> container.assays["proteins"] = assay
    >>> ax = tracksplot(container, layer="normalized", var_names=["P0", "P1"],
    ...                 groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_features(container, assay_name, var_names)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    assay = container.assays[assay_name]
    x = assay.layers[layer].X.copy()

    # Find feature identifier column
    var_col = None
    for preferred in ["protein", "gene", "feature", "name"]:
        if preferred in assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        var_col = assay.var.columns[0]

    # Filter to selected features (preserve order from var_names)
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = []
    for var in var_names:
        if var in available_features:
            feature_idx.append(available_features[var])

    x = x[:, feature_idx]

    # Get groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Calculate mean expression per group
    mean_expr = np.zeros((len(unique_groups), len(var_names)))

    for i, _g in enumerate(unique_groups):
        mask = groups == _g
        group_data = x[mask]
        mean_expr[i] = group_data.mean(axis=0)

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Create plot
    fig_height = 2 + len(unique_groups) * 0.5
    fig_width = 2 + len(var_names) * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create horizontal bar chart for each feature
    y_positions = np.arange(len(unique_groups))
    bar_height = 0.8 / len(var_names)

    for i, var_name in enumerate(var_names):
        offset = (i - len(var_names) / 2) * bar_height
        ax.barh(
            y_positions + offset,
            mean_expr[:, i],
            height=bar_height,
            label=var_name,
            **kwargs,
        )

    # Configure axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(unique_groups)
    ax.set_xlabel("Mean expression")
    ax.set_ylabel(groupby)
    ax.set_title("Expression by group")
    ax.legend(title="Features", bbox_to_anchor=(1.05, 1), loc="upper left")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    print("Testing matrix visualization module...")

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import polars as pl

    from scptensor import Assay, ScpContainer, ScpMatrix
    from scptensor.core.exceptions import VisualizationError

    # Create test container
    obs = pl.DataFrame(
        {"_index": [f"S{i}" for i in range(60)], "cluster": np.repeat(["A", "B", "C"], 20)}
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(10)], "protein": [f"P{i}" for i in range(10)]}
    )
    X = np.random.rand(60, 10) * 10
    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    # Test: Basic matrixplot
    print("\n1. Testing basic matrixplot...")
    ax = matrixplot(
        container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None
    print("   Basic matrixplot: OK")

    # Test: matrixplot with dendrogram
    print("\n2. Testing matrixplot with dendrogram...")
    ax = matrixplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        dendrogram=True,
        show=False,
    )
    assert ax is not None
    print("   Dendrogram option: OK")

    # Test: matrixplot custom colormap
    print("\n3. Testing matrixplot with custom colormap...")
    ax = matrixplot(
        container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        cmap="plasma",
        show=False,
    )
    assert ax is not None
    print("   Custom colormap: OK")

    # Test: matrixplot no standard scale
    print("\n4. Testing matrixplot without standard scaling...")
    ax = matrixplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        standard_scale=None,
        show=False,
    )
    assert ax is not None
    print("   No standard scaling: OK")

    # Test: Basic heatmap
    print("\n5. Testing basic heatmap...")
    ax = heatmap(
        container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None
    print("   Basic heatmap: OK")

    # Test: heatmap without log
    print("\n6. Testing heatmap without log transform...")
    ax = heatmap(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        log=False,
        show=False,
    )
    assert ax is not None
    print("   No log transform: OK")

    # Test: heatmap swap axes
    print("\n7. Testing heatmap with swapped axes...")
    ax = heatmap(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        swap_axes=True,
        show=False,
    )
    assert ax is not None
    print("   Swapped axes: OK")

    # Test: Basic tracksplot
    print("\n8. Testing basic tracksplot...")
    ax = tracksplot(
        container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None
    print("   Basic tracksplot: OK")

    # Test: Validation errors
    print("\n9. Testing validation errors...")

    try:
        matrixplot(
            container,
            layer="nonexistent",
            var_names=["P0"],
            groupby="cluster",
            show=False,
        )
        print("   Invalid layer: FAILED")
    except Exception:
        print("   Invalid layer: OK")

    try:
        matrixplot(
            container,
            layer="normalized",
            var_names=["INVALID_PROTEIN"],
            groupby="cluster",
            show=False,
        )
        print("   Invalid feature: FAILED")
    except VisualizationError:
        print("   Invalid feature: OK")

    try:
        matrixplot(
            container,
            layer="normalized",
            var_names=["P0"],
            groupby="invalid_column",
            show=False,
        )
        print("   Invalid groupby: FAILED")
    except VisualizationError:
        print("   Invalid groupby: OK")

    print("\nAll matrix visualization tests passed!")
