"""Embedding visualization functions.

This module provides generic scatter plot functions for visualizing
dimensionality reduction embeddings (PCA, UMAP, t-SNE) with
SciencePlots styling and missing value visualization support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.missing_value import MissingValueHandler
from scptensor.viz.base.style import PlotStyle
from scptensor.viz.base.validation import validate_container, validate_layer

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


__all__ = ["scatter", "umap", "pca", "tsne"]


def scatter(
    container: ScpContainer,
    layer: str,
    basis: str = "umap",
    color: str | list[str] | None = None,
    groupby: str | None = None,
    palette: str | list[str] | None = None,
    size: float = 5.0,
    alpha: float = 0.8,
    use_raw: bool = False,
    show_missing_values: bool = True,
    legend_loc: str = "right margin",
    frameon: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Generic scatter plot for any embedding coordinates.

    Creates a scatter plot of dimensionality reduction results (PCA/UMAP/t-SNE)
    with support for coloring by metadata columns or feature expression values.
    Automatically handles missing values with distinct visualization.

    Parameters
    ----------
    container : ScpContainer
        Input data container containing embedding coordinates in obs.
    layer : str
        Layer name in the assay to use for feature-based coloring.
    basis : str, default "umap"
        Embedding basis name. Expects {basis}_1 and {basis}_2 columns in obs.
        Common values: "umap", "pca", "tsne".
    color : str | list[str] | None, default None
        Color specification:
        - None: Use default color
        - str: Column name in obs or feature name in var
        - list[str]: Multiple color specifications (uses first)
    groupby : str | None, default None
        Column name in obs to group points by (alternative to color).
    palette : str | list[str] | None, default None
        Color palette name or list of colors.
    size : float, default 5.0
        Point size for scatter plot.
    alpha : float, default 0.8
        Transparency (0=transparent, 1=opaque).
    use_raw : bool, default False
        If True, use 'raw' layer instead of specified layer for feature coloring.
    show_missing_values : bool, default True
        If True, display missing values with distinct markers/colors.
    legend_loc : str, default "right margin"
        Legend location. Matplotlib legend location codes.
    frameon : bool, default True
        If True, display axis frame.
    title : str | None, default None
        Plot title. If None, generates from basis and color.
    ax : plt.Axes | None, default None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to scatter.

    Returns
    -------
    plt.Axes
        The axes containing the scatter plot.

    Raises
    ------
    ValueError
        If embedding columns not found in obs.
        If assay or layer not found in container.
    VisualizationError
        If container validation fails.

    Examples
    --------
    >>> from scptensor import ScpContainer
    >>> from scptensor.viz.recipes.embedding import scatter
    >>> # Basic UMAP plot
    >>> ax = scatter(container, layer="normalized", basis="umap")
    >>> # Color by metadata column
    >>> ax = scatter(container, layer="normalized", basis="umap", color="cluster")
    >>> # Color by feature expression
    >>> ax = scatter(container, layer="normalized", basis="umap", color="CD3D")
    """
    # Validate inputs
    validate_container(container)

    # Default assay name
    assay_name = "proteins"
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found in container")

    validate_layer(container, assay_name, layer)

    # Apply SciencePlots style
    PlotStyle.apply_style()

    # Create figure if no axes provided
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

    # Get embedding coordinates from obs
    basis_1 = f"{basis}_1"
    basis_2 = f"{basis}_2"

    if basis_1 not in container.obs.columns or basis_2 not in container.obs.columns:
        available = container.obs.columns
        raise ValueError(
            f"Embedding columns '{basis_1}' and '{basis_2}' not found in obs. "
            f"Available columns: {list(available)}"
        )

    x = container.obs[basis_1].to_numpy()
    y = container.obs[basis_2].to_numpy()

    # Resolve color values and mask
    color_values, mask = _resolve_color_and_mask(container, assay_name, layer, color, use_raw)

    # Determine if color is categorical for colormap selection
    is_categorical = False
    if isinstance(color_values, np.ndarray):
        if (
            color_values.dtype.kind in {"U", "S", "O"} or color_values.dtype == pl.String
        ):  # String types
            is_categorical = True

    # Create scatter plot
    if show_missing_values and mask is not None and mask.sum() > 0:
        # Use missing value handler for layered scatter
        _plot_with_missing_values(
            ax, x, y, color_values, mask, size, alpha, is_categorical, **kwargs
        )
    else:
        # Simple scatter plot
        _plot_simple(ax, x, y, color_values, size, alpha, is_categorical, **kwargs)

    # Set labels and styling
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")

    if not frameon:
        ax.set_frame_on(False)

    # Set title
    if title is None:
        title = f"{basis.upper()}"
        if color:
            title += f" colored by {color}"
    ax.set_title(title)

    # Add legend for categorical colors
    if color and isinstance(color, str) and color in container.obs.columns:
        _add_categorical_legend(ax, color, container.obs[color].to_numpy(), legend_loc)

    return ax


def _resolve_color_and_mask(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    color: str | list[str] | None,
    use_raw: bool,
) -> tuple[np.ndarray | str | None, np.ndarray | None]:
    """Resolve color values and mask from metadata or feature expression.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of assay containing features.
    layer : str
        Layer name for feature-based coloring.
    color : str | list[str] | None
        Color specification.
    use_raw : bool
        If True, use 'raw' layer for feature coloring.

    Returns
    -------
    color_values : ndarray | str | None
        Color values for each sample.
    mask : ndarray | None
        Mask values (0=valid, >0=missing).
    """
    n_samples = container.n_samples

    if color is None:
        # Default color
        return "#1f77b4", np.zeros(n_samples, dtype=np.int8)

    # Handle list of colors (use first)
    if isinstance(color, list):
        color = color[0] if color else None
        if color is None:
            return "#1f77b4", np.zeros(n_samples, dtype=np.int8)

    # Check if color is in obs (metadata column)
    if color in container.obs.columns:
        color_values = container.obs[color].to_numpy()
        # Metadata columns have no missing values
        mask = np.zeros(n_samples, dtype=np.int8)
        return color_values, mask

    # Check if color is a feature in var
    assay = container.assays[assay_name]
    actual_layer = "raw" if use_raw else layer

    if actual_layer not in assay.layers:
        raise ValueError(f"Layer '{actual_layer}' not found in assay '{assay_name}'")

    scpmatrix = assay.layers[actual_layer]

    # Find feature index
    feature_idx = _find_feature_index(assay, color)
    if feature_idx is None:
        raise ValueError(f"Color '{color}' not found in obs or as feature in assay '{assay_name}'")

    # Extract color values from matrix
    if scpmatrix.X.ndim > 1:
        color_values = scpmatrix.X[:, feature_idx]
    else:
        color_values = scpmatrix.X

    # Extract mask
    mask = scpmatrix.get_m()
    if mask.ndim > 1:
        mask = mask[:, feature_idx]
    else:
        mask = mask

    return color_values, mask


def _find_feature_index(assay, feature_name: str) -> int | None:
    """Find feature index in assay var.

    Parameters
    ----------
    assay : Assay
        Assay containing var metadata.
    feature_name : str
        Feature name to find.

    Returns
    -------
    int | None
        Feature index if found, None otherwise.
    """
    # Try common column names
    for col_name in ["protein", "gene", "feature", "name", "_index"]:
        if col_name in assay.var.columns:
            feature_list = assay.var[col_name].to_list()
            try:
                return feature_list.index(feature_name)
            except ValueError:
                continue

    return None


def _plot_with_missing_values(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | str,
    mask: np.ndarray,
    size: float,
    alpha: float,
    is_categorical: bool,
    **kwargs,
) -> None:
    """Create scatter plot with missing value overlay.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    x, y : ndarray
        Coordinates.
    color_values : ndarray | str
        Color values or single color.
    mask : ndarray
        Mask values (0=valid, >0=missing).
    size : float
        Point size.
    alpha : float
        Transparency.
    is_categorical : bool
        Whether color is categorical.
    **kwargs
        Additional scatter arguments.
    """
    import matplotlib.pyplot as plt

    valid_mask = mask == 0

    # Plot valid values
    if valid_mask.sum() > 0:
        c_valid = color_values[valid_mask] if not isinstance(color_values, str) else color_values
        if is_categorical and not isinstance(c_valid, str):
            # Handle categorical data with colormap
            unique_vals = np.unique(color_values[valid_mask])
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(len(unique_vals))]
            color_map = {val: colors[i] for i, val in enumerate(unique_vals)}
            c_valid = [color_map[v] for v in c_valid]

        ax.scatter(
            x[valid_mask],
            y[valid_mask],
            s=size,
            alpha=alpha,
            c=c_valid,
            label="Detected",
            **kwargs,
        )

    # Plot missing values by type
    for m_type, m_color in MissingValueHandler.MISSING_COLORS.items():
        if isinstance(m_type, str):
            continue
        type_mask = mask == m_type
        if type_mask.sum() > 0:
            ax.scatter(
                x[type_mask],
                y[type_mask],
                s=size,
                alpha=alpha * 0.7,
                color=m_color,
                label=f"Missing ({m_type})",
                edgecolors="none",
            )


def _plot_simple(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | str | None,
    size: float,
    alpha: float,
    is_categorical: bool,
    **kwargs,
) -> None:
    """Create simple scatter plot without missing value handling.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    x, y : ndarray
        Coordinates.
    color_values : ndarray | str | None
        Color values or single color.
    size : float
        Point size.
    alpha : float
        Transparency.
    is_categorical : bool
        Whether color is categorical.
    **kwargs
        Additional scatter arguments.
    """
    import matplotlib.pyplot as plt

    c = color_values if color_values is not None else "#1f77b4"

    if is_categorical and isinstance(c, np.ndarray):
        # Convert categorical to numeric for colormap
        unique_vals = np.unique(c)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(unique_vals))]
        color_map = {val: colors[i] for i, val in enumerate(unique_vals)}
        c = [color_map[v] for v in c]

    ax.scatter(x, y, s=size, alpha=alpha, c=c, **kwargs)


def _add_categorical_legend(
    ax: plt.Axes,
    color: str,
    values: np.ndarray,
    legend_loc: str,
) -> None:
    """Add legend for categorical coloring.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    color : str
        Column name used for coloring.
    values : ndarray
        Unique values in the column.
    legend_loc : str
        Legend location.
    """
    unique_values = np.unique(values)

    # Clean legend location string
    loc = legend_loc.replace(" margin", "")

    # Map categorical values to colors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(unique_values))]

    legend_elements = [
        Patch(facecolor=colors[i], label=str(val)) for i, val in enumerate(unique_values)
    ]

    ax.legend(handles=legend_elements, title=color, loc=loc)


def umap(
    container: ScpContainer,
    layer: str = "normalized",
    **kwargs,
) -> plt.Axes:
    """UMAP scatter plot.

    Convenience function for creating UMAP embedding scatter plots.
    Expects 'umap_1' and 'umap_2' columns in container.obs.

    Parameters
    ----------
    container : ScpContainer
        Input data container with UMAP coordinates in obs.
    layer : str, default "normalized"
        Layer name for feature-based coloring.
    **kwargs
        Additional arguments passed to :func:`scatter`.

    Returns
    -------
    plt.Axes
        The axes containing the UMAP scatter plot.

    Examples
    --------
    >>> from scptensor.viz.recipes.embedding import umap
    >>> ax = umap(container, layer="normalized", color="cluster")
    """
    return scatter(container, layer=layer, basis="umap", **kwargs)


def pca(
    container: ScpContainer,
    layer: str = "normalized",
    **kwargs,
) -> plt.Axes:
    """PCA scatter plot.

    Convenience function for creating PCA embedding scatter plots.
    Expects 'pca_1' and 'pca_2' columns in container.obs.

    Parameters
    ----------
    container : ScpContainer
        Input data container with PCA coordinates in obs.
    layer : str, default "normalized"
        Layer name for feature-based coloring.
    **kwargs
        Additional arguments passed to :func:`scatter`.

    Returns
    -------
    plt.Axes
        The axes containing the PCA scatter plot.

    Examples
    --------
    >>> from scptensor.viz.recipes.embedding import pca
    >>> ax = pca(container, layer="normalized", color="batch")
    """
    return scatter(container, layer=layer, basis="pca", **kwargs)


def tsne(
    container: ScpContainer,
    layer: str = "normalized",
    **kwargs,
) -> plt.Axes:
    """t-SNE scatter plot.

    Convenience function for creating t-SNE embedding scatter plots.
    Expects 'tsne_1' and 'tsne_2' columns in container.obs.

    Parameters
    ----------
    container : ScpContainer
        Input data container with t-SNE coordinates in obs.
    layer : str, default "normalized"
        Layer name for feature-based coloring.
    **kwargs
        Additional arguments passed to :func:`scatter`.

    Returns
    -------
    plt.Axes
        The axes containing the t-SNE scatter plot.

    Examples
    --------
    >>> from scptensor.viz.recipes.embedding import tsne
    >>> ax = tsne(container, layer="normalized", color="condition")
    """
    return scatter(container, layer=layer, basis="tsne", **kwargs)


if __name__ == "__main__":
    # Simple test
    import matplotlib.pyplot as plt
    import polars as pl

    from scptensor import Assay, ScpMatrix

    # Create test container
    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(100)],
            "cluster": np.random.choice(["A", "B", "C"], 100),
            "umap_1": np.random.randn(100),
            "umap_2": np.random.randn(100),
            "pca_1": np.random.randn(100),
            "pca_2": np.random.randn(100),
        }
    )

    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(50)],
            "protein": [f"protein_{i}" for i in range(50)],
        }
    )

    X = np.random.rand(100, 50)
    M = np.zeros_like(X, dtype=np.int8)
    M[X < 0.1] = 1  # Some missing values

    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X, M=M)})
    container.assays["proteins"] = assay

    # Test basic scatter
    print("Testing scatter function...")
    ax = scatter(container, layer="normalized", basis="umap")
    print("Basic scatter: OK")
    plt.close(ax.figure)

    # Test with color
    ax = scatter(container, layer="normalized", basis="umap", color="cluster")
    print("Scatter with color: OK")
    plt.close(ax.figure)

    # Test convenience functions
    ax = umap(container)
    print("UMAP function: OK")
    plt.close(ax.figure)

    ax = pca(container)
    print("PCA function: OK")
    plt.close(ax.figure)

    print("\nAll tests passed!")
