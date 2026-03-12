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
from scipy import sparse

from scptensor.core.exceptions import VisualizationError
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

__all__ = [
    "plot_feature_dotplot",
    "dotplot",
]


def _to_dense_array(matrix: np.ndarray) -> np.ndarray:
    """Convert dense/sparse matrix-like input to a 2D NumPy array."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique values preserving first appearance order."""
    _, first_idx = np.unique(values, return_index=True)
    return values[np.sort(first_idx)]


def _resolve_feature_column(assay) -> str:
    """Resolve feature name column from assay metadata."""
    preferred_cols = [
        assay.feature_id_col,
        "protein",
        "gene",
        "feature",
        "name",
        "_index",
    ]
    for col in preferred_cols:
        if col in assay.var.columns:
            return col
    return assay.var.columns[0]


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
        Whether to show dendrogram. Currently not supported and will raise an
        explicit error when set to True.
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
    if len(var_names) == 0:
        raise VisualizationError("var_names must contain at least one feature")
    if dendrogram:
        raise VisualizationError(
            "dendrogram=True is not supported in plot_feature_dotplot/dotplot yet"
        )
    if standard_scale not in {"var", "obs", None}:
        raise VisualizationError(
            f"standard_scale must be one of ['var', 'obs', None], got: {standard_scale}"
        )

    PlotStyle.apply_style()

    assay = container.assays[assay_name]
    x = _to_dense_array(assay.layers[layer].X.copy())

    # Find feature identifier column
    var_col = _resolve_feature_column(assay)

    # Filter to selected features (preserve order from var_names)
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = [available_features[var] for var in var_names]

    x = x[:, feature_idx]

    # Log transform if requested
    if log:
        x = np.log1p(np.clip(x, a_min=0.0, a_max=None))

    # Get groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = _ordered_unique(groups)

    # Calculate mean expression per group
    mean_expr = np.zeros((len(unique_groups), len(var_names)))
    pct_expr = np.zeros_like(mean_expr)

    for i, _g in enumerate(unique_groups):
        mask = groups == _g
        group_data = x[mask]
        mean_expr[i] = np.asarray(group_data.mean(axis=0)).ravel()
        pct_expr[i] = (group_data > 0).mean(axis=0)

    color_vmin = float(np.min(mean_expr))
    color_vmax = float(np.max(mean_expr))

    # Standardize
    if standard_scale == "var":
        # Scale each feature independently
        col_min = mean_expr.min(axis=0)
        col_max = mean_expr.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range < 1e-8] = 1.0  # Avoid division by zero
        mean_expr = (mean_expr - col_min) / col_range
        color_vmin, color_vmax = 0.0, 1.0
    elif standard_scale == "obs":
        # Scale each group independently
        row_min = mean_expr.min(axis=1, keepdims=True)
        row_max = mean_expr.max(axis=1, keepdims=True)
        row_range = row_max - row_min
        row_range[row_range < 1e-8] = 1.0
        mean_expr = (mean_expr - row_min) / row_range
        color_vmin, color_vmax = 0.0, 1.0
    elif abs(color_vmax - color_vmin) < 1e-8:
        color_vmin = color_vmin - 0.5
        color_vmax = color_vmax + 0.5

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
                vmin=color_vmin,
                vmax=color_vmax,
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_vmin, vmax=color_vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean expression")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_feature_dotplot(*args, **kwargs):
    """Canonical alias of :func:`dotplot`."""
    return dotplot(*args, **kwargs)
