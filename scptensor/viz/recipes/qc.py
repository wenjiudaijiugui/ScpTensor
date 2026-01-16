"""QC visualization recipes.

This module provides functions for quality control visualizations,
including completeness, spy plots, PCA overview, and missing value patterns.

Functions include:
- qc_completeness: Data completeness per sample
- qc_matrix_spy: Missing value spy plot
- pca_overview: Multi-panel PCA visualization
- missing_value_patterns: Missing value pattern analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.multi_panel import PanelLayout
from scptensor.viz.base.style import PlotStyle, setup_style
from scptensor.viz.base.validation import (
    validate_container,
    validate_groupby,
    validate_layer,
)
from scptensor.viz.base.violin import violin

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def qc_completeness(
    container: ScpContainer,
    assay_name: str = "proteins",
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
    assay_name: str = "proteins",
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
    M = matrix.M if matrix.M is not None else np.zeros_like(matrix.X, dtype=np.int8)
    spy_data = (M > 0).astype(np.uint8)

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


def pca_overview(
    container: ScpContainer,
    layer: str,
    assay_name: str = "proteins",
    pca_assay_name: str = "pca",
    n_pcs: int = 3,
    color: str | None = None,
    show: bool = True,
) -> Figure:
    """Create a multi-panel PCA overview visualization.

    Displays comprehensive PCA analysis with four panels:
    1. PC1 vs PC2 scatter plot (colored by group if specified)
    2. PC2 vs PC3 scatter plot
    3. Explained variance ratio bar plot
    4. PC loadings heatmap

    Parameters
    ----------
    container : ScpContainer
        Container containing the data. Must have PCA results already computed.
    layer : str
        Layer name in PCA assay (typically 'scores').
    assay_name : str, default="proteins"
        Name of the original assay (for loading information).
    pca_assay_name : str, default="pca"
        Name of the PCA assay containing the results.
    n_pcs : int, default=3
        Number of principal components to visualize.
    color : str or None
        Column name in obs to color points by. If None, no coloring.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing all panels.

    Raises
    ------
    VisualizationError
        If validation fails (container, layer, or groupby).

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.dim_reduction import pca
    >>> from scptensor.viz.recipes.qc import pca_overview
    >>> container = ScpContainer(n_samples=100)
    >>> container.obs["cluster"] = np.random.choice(["A", "B", "C"], 100)
    >>> assay = Assay(n_features=20)
    >>> assay.layers["imputed"] = ScpMatrix(X=np.random.rand(100, 20) * 10)
    >>> container.assays["proteins"] = assay
    >>> container = pca(container, "proteins", "imputed", n_components=5)
    >>> fig = pca_overview(container, layer="scores", color="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, pca_assay_name, layer)

    if color is not None:
        validate_groupby(container, color)

    PlotStyle.apply_style()

    # Get PCA scores
    pca_assay = container.assays[pca_assay_name]
    scores = pca_assay.layers[layer].X

    # Get variance info
    var_col = None
    for preferred in ["explained_variance_ratio", "explained_inertia_ratio"]:
        if preferred in pca_assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        raise ValueError(
            f"PCA assay '{pca_assay_name}' does not contain "
            "explained_variance_ratio column. "
            "Please run PCA first."
        )

    variance_ratio = pca_assay.var[var_col].to_numpy()[:n_pcs]

    # Get loadings from original assay
    loadings = None
    if assay_name in container.assays:
        original_var = container.assays[assay_name].var
        loading_cols = [
            c
            for c in original_var.columns
            if c.startswith(f"{pca_assay_name}_PC") and "_loading" in c
        ]
        if loading_cols:
            loading_data = np.column_stack(
                [original_var[c].to_numpy() for c in loading_cols[:n_pcs]]
            )
            loadings = loading_data

    # Get color data
    color_data = None
    color_labels = None
    if color is not None:
        color_data = container.obs[color].to_numpy()
        color_labels = np.unique(color_data)

    # Create layout
    layout = PanelLayout(figsize=(14, 10), grid=(2, 2))

    # Panel 1: PC1 vs PC2
    def _plot_pc1_pc2(ax):
        if color_data is not None:
            for label in color_labels:
                mask = color_data == label
                ax.scatter(
                    scores[mask, 0],
                    scores[mask, 1],
                    alpha=0.6,
                    s=30,
                    label=str(label),
                    edgecolors="none",
                )
            ax.legend(loc="best", fontsize=8)
        else:
            ax.scatter(
                scores[:, 0],
                scores[:, 1],
                alpha=0.6,
                s=30,
                edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({variance_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance_ratio[1] * 100:.1f}%)")
        ax.set_title("PC1 vs PC2")

    layout.add_panel((0, 0), _plot_pc1_pc2)

    # Panel 2: PC2 vs PC3 (if available)
    def _plot_pc2_pc3(ax):
        if n_pcs >= 3:
            if color_data is not None:
                for label in color_labels:
                    mask = color_data == label
                    ax.scatter(
                        scores[mask, 1],
                        scores[mask, 2],
                        alpha=0.6,
                        s=30,
                        label=str(label),
                        edgecolors="none",
                    )
                ax.legend(loc="best", fontsize=8)
            else:
                ax.scatter(
                    scores[:, 1],
                    scores[:, 2],
                    alpha=0.6,
                    s=30,
                    edgecolors="none",
                )
            ax.set_xlabel(f"PC2 ({variance_ratio[1] * 100:.1f}%)")
            ax.set_ylabel(f"PC3 ({variance_ratio[2] * 100:.1f}%)")
        else:
            ax.text(
                0.5,
                0.5,
                f"Need at least 3 PCs for this plot.\nCurrent: {n_pcs}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title("PC2 vs PC3")

    layout.add_panel((0, 1), _plot_pc2_pc3)

    # Panel 3: Variance ratio
    def _plot_variance(ax):
        pc_names = [f"PC{i + 1}" for i in range(n_pcs)]
        bars = ax.bar(pc_names, variance_ratio * 100, edgecolor="black", alpha=0.7)
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title("Explained Variance Ratio")
        ax.set_ylim(0, max(variance_ratio * 100) * 1.2)

        # Add value labels on bars
        for _i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    layout.add_panel((1, 0), _plot_variance)

    # Panel 4: Loadings heatmap
    def _plot_loadings(ax):
        if loadings is not None:
            # Limit to top 20 features for clarity
            n_show = min(20, loadings.shape[0])
            im = ax.imshow(
                loadings[:n_show],
                aspect="auto",
                cmap="RdBu_r",
                interpolation="none",
            )
            ax.set_yticks(range(n_show))
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Feature")
            ax.set_title("PC Loadings (Top 20 Features)")
            ax.set_xticks(range(n_pcs))
            ax.set_xticklabels([f"PC{i + 1}" for i in range(n_pcs)])

            # Add colorbar
            plt.colorbar(im, ax=ax, label="Loading")
        else:
            ax.text(
                0.5,
                0.5,
                f"Loadings not found in assay '{assay_name}'.var.\n"
                f"Expected columns: {pca_assay_name}_PC1_loading, etc.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )

    layout.add_panel((1, 1), _plot_loadings)

    # Finalize
    fig = layout.finalize(tight=True)
    assert fig is not None, "finalize() should never return None"

    if show:
        plt.show()

    return fig


def missing_value_patterns(
    container: ScpContainer,
    layer: str,
    assay_name: str = "proteins",
    groupby: str | None = None,
    show: bool = True,
) -> Figure:
    """Create a multi-panel missing value pattern analysis visualization.

    Displays comprehensive missing value analysis with four panels:
    1. Sample missing rate bar plot
    2. Feature missing rate bar plot
    3. Missing value pattern heatmap
    4. Missing value type distribution

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to analyze (e.g., 'normalized', 'log').
    assay_name : str, default="proteins"
        Name of the assay containing the data.
    groupby : str or None
        Column name in obs to group by. If None, no grouping.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing all panels.

    Raises
    ------
    VisualizationError
        If validation fails (container or layer).

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.qc import missing_value_patterns
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["cluster"] = np.repeat(["A", "B", "C"], 20)
    >>> assay = Assay(n_features=10)
    >>> X = np.random.rand(60, 10) * 10
    >>> X[X < 3] = 0  # Simulate missing values
    >>> assay.layers["normalized"] = ScpMatrix(X=X)
    >>> container.assays["proteins"] = assay
    >>> fig = missing_value_patterns(container, layer="normalized",
    ...                             groupby="cluster", show=False)
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)

    if groupby is not None:
        validate_groupby(container, groupby)

    PlotStyle.apply_style()

    assay = container.assays[assay_name]
    data_x = assay.layers[layer].X
    mask_m = assay.layers[layer].M

    n_samples, n_features = data_x.shape

    # Calculate missing status
    # M values: 0=valid, 1=MBR, 2=LOD, 3=FILTERED, 5=IMPUTED
    if mask_m is None:
        is_missing = np.zeros((n_samples, n_features), dtype=bool)
    else:
        is_missing = mask_m != 0

    # Sample missing rate
    sample_missing_rate = is_missing.sum(axis=1) / n_features  # type: ignore[union-attr]

    # Feature missing rate
    feature_missing_rate = is_missing.sum(axis=0) / n_samples  # type: ignore[union-attr]

    # Get feature names
    var_col = None
    for preferred in ["protein", "gene", "feature", "name"]:
        if preferred in assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        var_col = assay.var.columns[0]

    feature_names = assay.var[var_col].to_list()

    # Get grouping info
    group_data = None
    group_labels = None
    if groupby is not None:
        group_data = container.obs[groupby].to_numpy()
        group_labels = np.unique(group_data)

    # Create layout
    layout = PanelLayout(figsize=(14, 10), grid=(2, 2))

    # Panel 1: Sample missing rate
    def _plot_sample_missing(ax):
        if group_data is not None:
            # Group by and show as box plot style
            data_by_group = []
            for _label in group_labels:
                mask = group_data == _label
                group_rates = sample_missing_rate[mask]
                data_by_group.append(group_rates)

            parts = ax.boxplot(
                data_by_group,
                labels=[str(lbl) for lbl in group_labels],
                patch_artist=True,
            )
            for patch in parts["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
        else:
            ax.bar(range(n_samples), sample_missing_rate, edgecolor="black")
            ax.set_xlabel("Sample")
            ax.set_xticks([])

        ax.set_ylabel("Missing Rate")
        ax.set_title("Sample Missing Rate")
        ax.set_ylim(0, 1)

    layout.add_panel((0, 0), _plot_sample_missing)

    # Panel 2: Feature missing rate (top 20)
    def _plot_feature_missing(ax):
        n_show = min(20, n_features)
        sorted_idx = np.argsort(feature_missing_rate)[-n_show:][::-1]

        ax.barh(range(n_show), feature_missing_rate[sorted_idx], edgecolor="black")
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Missing Rate")
        ax.set_title("Feature Missing Rate (Top 20)")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

    layout.add_panel((0, 1), _plot_feature_missing)

    # Panel 3: Missing pattern heatmap
    def _plot_pattern_heatmap(ax):
        # Show missing status for first 50 samples and 20 features
        n_show_samples = min(50, n_samples)
        n_show_features = min(20, n_features)

        # Subset by sample order (or group if specified)
        if group_data is not None:
            sample_idx = []
            for label in group_labels:
                mask = group_data == label
                sample_idx.extend(np.where(mask)[0][: n_show_samples // len(group_labels)])
            sample_idx = np.array(sample_idx[:n_show_samples])
        else:
            sample_idx = np.arange(n_show_samples)

        # Sort features by missing rate
        feature_idx = np.argsort(feature_missing_rate)[-n_show_features:][::-1]

        missing_subset = is_missing[np.ix_(sample_idx, feature_idx)].astype(int)

        ax.imshow(
            missing_subset.T,
            aspect="auto",
            cmap="gray_r",
            interpolation="none",
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Feature")
        ax.set_title("Missing Value Pattern")

        # Set ticks
        ax.set_yticks(range(n_show_features))
        ax.set_yticklabels([feature_names[i] for i in feature_idx])

        if group_data is not None:
            # Show group labels on x-axis
            ax.set_xticks([])
        else:
            ax.set_xticks([])

    layout.add_panel((1, 0), _plot_pattern_heatmap)

    # Panel 4: Missing type distribution
    def _plot_type_distribution(ax):
        # Count missing types
        # M values: 0=valid, 1=MBR, 2=LOD, 3=FILTERED, 5=IMPUTED
        type_counts = {}
        type_labels = {
            1: "MBR",
            2: "LOD",
            3: "FILTERED",
            5: "IMPUTED",
        }

        unique, counts = np.unique(mask_m, return_counts=True)
        total_missing = 0

        for val, count in zip(unique, counts, strict=False):
            if val in type_labels:
                type_counts[type_labels[val]] = count
                total_missing += count

        if type_counts:
            bars = ax.bar(
                type_counts.keys(),
                type_counts.values(),
                edgecolor="black",
                alpha=0.7,
            )
            ax.set_ylabel("Count")
            ax.set_title("Missing Value Type Distribution")

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Add total
            ax.text(
                0.98,
                0.95,
                f"Total: {total_missing}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No missing values found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    layout.add_panel((1, 1), _plot_type_distribution)

    # Finalize
    fig = layout.finalize(tight=True)
    assert fig is not None, "finalize() should never return None"

    if show:
        plt.show()

    return fig
