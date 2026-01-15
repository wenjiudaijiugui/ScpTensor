"""Feature visualization recipes.

This module provides functions for visualizing feature (protein/gene) expression
patterns across groups of cells. These are commonly used in single-cell analysis
to identify marker features and visualize expression patterns.

Functions include:
- dotplot: Dot size shows fraction of cells expressing, color shows mean expression
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
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from scptensor import ScpContainer

__all__ = ["dotplot"]


def dotplot(
    container: ScpContainer,
    layer: str,
    var_names: list[str],
    groupby: str,
    assay_name: str = "proteins",
    dendrogram: bool = False,
    log: bool = True,
    cmap: str = "viridis",
    dot_size: float = 5.0,
    standard_scale: Literal["var", "obs"] | None = "var",
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs,
) -> Axes:
    """Create a dot plot for feature expression visualization.

    The dot plot combines two visual encodings:
    - Dot size: fraction of cells expressing the feature (expression percentage)
    - Dot color: mean expression level across cells in the group

    This is particularly useful for identifying marker features and visualizing
    expression patterns across clusters or conditions.

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
        Colormap for mean expression.
    dot_size : float, default=5.0
        Base size multiplier for dots.
    standard_scale : {'var', 'obs'} or None, default="var"
        How to standardize expression values:
        - 'var': scale each feature independently (recommended)
        - 'obs': scale each group independently
        - None: no scaling
    show : bool, default=True
        Whether to display the plot.
    ax : plt.Axes or None
        Pre-existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to scatter.

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
    >>> from scptensor.viz.recipes.feature import dotplot
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> assay.var["protein"] = [f"P{i}" for i in range(10)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 10))
    >>> container.assays["proteins"] = assay
    >>> ax = dotplot(container, layer="normalized", var_names=["P0", "P1"],
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

    # Get groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Calculate mean expression per group
    mean_expr = np.zeros((len(unique_groups), len(var_names)))
    pct_expr = np.zeros_like(mean_expr)

    for i, _g in enumerate(unique_groups):
        mask = groups == _g
        group_data = x[mask]
        mean_expr[i] = group_data.mean(axis=0)
        pct_expr[i] = (group_data > 0).mean(axis=0)

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
    if ax is None:
        fig_height = 2 + len(unique_groups) * 0.5
        fig_width = 2 + len(var_names) * 0.5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw dots
    for i, _g in enumerate(unique_groups):
        for j in range(len(var_names)):
            size = pct_expr[i, j] * dot_size * 20
            color = mean_expr[i, j]
            ax.scatter(
                j + 0.5,
                i + 0.5,
                s=size,
                c=[[color]],
                cmap=cmap,
                vmin=0,
                vmax=1,
                edgecolors="black",
                linewidth=0.5,
                **kwargs,
            )

    # Configure axes
    ax.set_yticks(np.arange(len(unique_groups)) + 0.5)
    ax.set_yticklabels(unique_groups)
    ax.set_xticks(np.arange(len(var_names)) + 0.5)
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_xlim(0, len(var_names))
    ax.set_ylim(0, len(unique_groups))
    ax.set_ylabel(groupby)
    ax.invert_yaxis()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean expression")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    print("Testing feature visualization module...")

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
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

    # Test: Basic dotplot
    print("\n1. Testing basic dotplot...")
    ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1", "P2"],
        groupby="cluster",
        show=False,
    )
    assert ax is not None
    print("   Basic dotplot: OK")

    # Test: No log transform
    print("\n2. Testing dotplot without log transform...")
    ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        log=False,
        show=False,
    )
    assert ax is not None
    print("   No log transform: OK")

    # Test: Custom colormap
    print("\n3. Testing dotplot with custom colormap...")
    ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        cmap="plasma",
        show=False,
    )
    assert ax is not None
    print("   Custom colormap: OK")

    # Test: No standard scale
    print("\n4. Testing dotplot without standard scaling...")
    ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        standard_scale=None,
        show=False,
    )
    assert ax is not None
    print("   No standard scaling: OK")

    # Test: Obs standard scale
    print("\n5. Testing dotplot with obs standard scaling...")
    ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        standard_scale="obs",
        show=False,
    )
    assert ax is not None
    print("   Obs standard scaling: OK")

    # Test: With custom axes
    print("\n6. Testing dotplot with custom axes...")
    fig, ax = plt.subplots()
    result_ax = dotplot(
        container,
        layer="normalized",
        var_names=["P0", "P1"],
        groupby="cluster",
        show=False,
        ax=ax,
    )
    assert result_ax is ax
    print("   Custom axes: OK")

    # Test: Validation errors
    print("\n7. Testing validation errors...")

    try:
        dotplot(
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
        dotplot(
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
        dotplot(
            container,
            layer="normalized",
            var_names=["P0"],
            groupby="invalid_column",
            show=False,
        )
        print("   Invalid groupby: FAILED")
    except VisualizationError:
        print("   Invalid groupby: OK")

    print("\nAll feature visualization tests passed!")
