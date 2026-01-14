import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.style import setup_style
from scptensor.viz.base.violin import violin


def qc_completeness(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    group_by: str = "batch",
    figsize: tuple[int, int] = (6, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualize data completeness per sample, grouped by metadata.

    Creates a violin plot showing the distribution of measured (valid)
    feature counts for each group in the specified metadata column.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "protein"
        Name of the assay to analyze.
    layer : str, default "raw"
        Layer name within the assay.
    group_by : str, default "batch"
        Column name in ``obs`` used for grouping.
    figsize : tuple[int, int], default (6, 4)
        Figure size (deprecated; use ax with pre-sized figure).
    ax : plt.Axes | None
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If assay_name or layer is not found.

    """
    setup_style()

    assay = container.assays.get(assay_name)
    if assay is None:
        raise ValueError(f"Assay '{assay_name}' not found.")

    matrix = assay.layers.get(layer)
    if matrix is None:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")

    # Valid counts: M == 0
    valid_counts = np.sum(matrix.M == 0, axis=1)

    # Get grouping labels
    if group_by not in container.obs.columns:
        groups = np.full(container.n_samples, "All")
    else:
        groups = container.obs[group_by].to_numpy()

    unique_groups = np.unique(groups)
    data_by_group = [valid_counts[groups == g] for g in unique_groups]
    labels = [str(g) for g in unique_groups]

    return violin(
        data=data_by_group,
        labels=labels,
        ax=ax,
        title=f"Data Completeness by {group_by}",
        ylabel="Number of Measured Features",
    )


def qc_matrix_spy(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualize missing value distribution as a spy plot.

    Creates a binary heatmap showing measured vs missing values
    across the entire matrix.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "protein"
        Name of the assay to visualize.
    layer : str, default "raw"
        Layer name within the assay.
    figsize : tuple[int, int], default (8, 6)
        Figure size (deprecated; use ax with pre-sized figure).
    ax : plt.Axes | None
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If assay_name or layer is not found.

    """
    setup_style()

    assay = container.assays.get(assay_name)
    if assay is None:
        raise ValueError(f"Assay '{assay_name}' not found.")

    matrix = assay.layers.get(layer)
    if matrix is None:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Binary spy: 0=Measured, 1=Missing
    spy_data = (matrix.M > 0).astype(np.uint8)

    # SciencePlots-style colors
    cmap = mcolors.ListedColormap(["#0C5DA5", "#E5E5E5"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    im = ax.imshow(spy_data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Measured", "Missing"])

    ax.set_title(f"Matrix Spy Plot ({assay_name}/{layer})")
    ax.set_xlabel("Features")
    ax.set_ylabel("Samples")

    return ax
