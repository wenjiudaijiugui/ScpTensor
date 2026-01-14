"""Differential expression visualization recipes.

This module provides functions for visualizing differential expression (DE) results,
including dot plots, stacked violin plots, and volcano plots.

Functions include:
- rank_genes_groups_dotplot: DE results as dot plot with logFC as color
- rank_genes_groups_stacked_violin: DE results as stacked violin plots
- volcano: Enhanced volcano plot with multi-panel display
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from scptensor.diff_expr import diff_expr_ttest
from scptensor.viz.base.multi_panel import PanelLayout
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
    from matplotlib.figure import Figure

    from scptensor import ScpContainer

__all__ = [
    "rank_genes_groups_dotplot",
    "rank_genes_groups_stacked_violin",
    "volcano",
]


def rank_genes_groups_dotplot(
    container: ScpContainer,
    layer: str,
    groupby: str,
    n_genes: int = 10,
    group1: str | None = None,
    group2: str | None = None,
    assay_name: str = "proteins",
    dendrogram: bool = False,
    values_to_plot: Literal["logfc", "pval", "mean"] = "logfc",
    cmap: str = "RdBu_r",
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs,
) -> Axes:
    """Create a dot plot for ranked gene groups from differential expression.

    The dot plot combines two visual encodings:
    - Dot size: fraction of cells expressing the feature (expression percentage)
    - Dot color: selected value (logFC, p-value, or mean expression)

    This function automatically computes differential expression between two groups
    and displays the top ranked genes.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to visualize (e.g., 'normalized', 'log').
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    n_genes : int, default=10
        Number of top DE genes to display.
    group1 : str or None
        First group name for comparison. If None, uses first unique group.
    group2 : str or None
        Second group name for comparison. If None, uses second unique group.
    assay_name : str, default="proteins"
        Assay name containing the features.
    dendrogram : bool, default=False
        Whether to show dendrogram (not yet implemented, reserved for future).
    values_to_plot : {'logfc', 'pval', 'mean'}, default="logfc"
        What value to use for dot color:
        - 'logfc': log2 fold change (group1 vs group2)
        - 'pval': -log10 p-value
        - 'mean': mean expression in group1
    cmap : str, default="RdBu_r"
        Colormap for the color values.
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
    >>> from scptensor.viz.recipes.differential import rank_genes_groups_dotplot
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["condition"] = np.repeat(["Control", "Treatment"], 30)
    >>> assay = Assay(n_features=20)
    >>> assay.var["protein"] = [f"P{i}" for i in range(20)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 20) * 10)
    >>> container.assays["proteins"] = assay
    >>> ax = rank_genes_groups_dotplot(
    ...     container, layer="normalized", groupby="condition",
    ...     group1="Treatment", group2="Control", n_genes=5, show=False
    ... )
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    # Get unique groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Determine groups to compare
    if group1 is None:
        group1 = unique_groups[0]
    if group2 is None:
        if len(unique_groups) < 2:
            raise ValueError(
                f"Need at least 2 groups for comparison, found {len(unique_groups)}"
            )
        group2 = unique_groups[1]

    # Verify groups exist
    if group1 not in unique_groups:
        raise ValueError(f"Group '{group1}' not found in {groupby}")
    if group2 not in unique_groups:
        raise ValueError(f"Group '{group2}' not found in {groupby}")

    # Compute differential expression
    de_result = diff_expr_ttest(
        container=container,
        assay_name=assay_name,
        group_col=groupby,
        group1=group1,
        group2=group2,
        layer_name=layer,
    )

    # Convert to DataFrame and get top genes
    de_df = de_result.to_dataframe()
    # Filter out NaN p-values and sort
    de_df = de_df.filter(pl.col("p_value").is_not_nan())
    de_df = de_df.sort("p_value").head(n_genes)

    top_genes = de_df["feature_id"].to_list()
    n_top = len(top_genes)

    if n_top == 0:
        raise ValueError("No significant genes found for visualization")

    # Validate features exist
    validate_features(container, assay_name, top_genes)

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

    # Filter to selected features (preserve order from top_genes)
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = []
    for var in top_genes:
        if var in available_features:
            feature_idx.append(available_features[var])

    x = x[:, feature_idx]

    # Get group indices
    g1_mask = groups == group1
    g2_mask = groups == group2

    # Calculate expression percentage for group1
    pct_expr = np.zeros(n_top)
    for i, idx in enumerate(feature_idx):
        g1_vals = x[g1_mask, idx]
        pct_expr[i] = (g1_vals > 0).mean()

    # Get color values based on values_to_plot
    if values_to_plot == "logfc":
        # Get logFC from DE result
        logfc_map = dict(zip(de_df["feature_id"].to_list(), de_df["log2_fc"].to_list(), strict=False))
        color_values = np.array([logfc_map.get(g, 0) for g in top_genes])
        color_label = "log2 Fold Change"
        # Symmetric color scale
        vlim = max(abs(color_values).min(0), 0.1)  # At least some range
        vmin, vmax = -vlim, vlim
    elif values_to_plot == "pval":
        pval_map = dict(zip(de_df["feature_id"].to_list(), de_df["p_value"].to_list(), strict=False))
        pvals = np.array([pval_map.get(g, 1.0) for g in top_genes])
        color_values = -np.log10(np.clip(pvals, 1e-300, None))
        color_label = "-log10 P-value"
        vmin, vmax = 0, color_values.max()
    else:  # mean
        mean_expr = np.zeros(n_top)
        for i, idx in enumerate(feature_idx):
            g1_vals = x[g1_mask, idx]
            mean_expr[i] = g1_vals.mean()
        color_values = mean_expr
        color_label = "Mean Expression"
        vmin, vmax = color_values.min(), color_values.max()

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Create plot
    if ax is None:
        fig_height = 2 + n_top * 0.4
        fig_width = 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Determine color scale
    if values_to_plot == "logfc":
        # Use symmetric scale for logFC
        vmax = max(abs(color_values))
        if vmax == 0:
            vmax = 1
        vmin, vmax = -vmax, vmax
    elif values_to_plot == "pval":
        vmin, vmax = 0, max(color_values)
    else:  # mean
        vmin, vmax = color_values.min(), color_values.max()

    # Draw dots
    for i in range(n_top):
        size = pct_expr[i] * 100
        ax.scatter(
            color_values[i],
            i + 0.5,
            s=size,
            c=[[color_values[i]]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidth=0.5,
            **kwargs,
        )

    # Configure axes
    ax.set_yticks(np.arange(n_top) + 0.5)
    ax.set_yticklabels(top_genes)
    ax.set_ylim(0, n_top)
    ax.set_ylabel("")
    ax.set_xlabel(color_label)
    ax.set_title(f"Top {n_top} DE Genes: {group1} vs {group2}")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=color_label)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def rank_genes_groups_stacked_violin(
    container: ScpContainer,
    layer: str,
    groupby: str,
    n_genes: int = 10,
    group1: str | None = None,
    group2: str | None = None,
    assay_name: str = "proteins",
    cmap: str = "viridis",
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs,
) -> Axes:
    """Create a stacked violin plot for ranked gene groups from DE results.

    Shows expression distribution of top DE genes across two comparison groups
    using side-by-side violin plots.

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to visualize (e.g., 'normalized', 'log').
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    n_genes : int, default=10
        Number of top DE genes to display.
    group1 : str or None
        First group name for comparison. If None, uses first unique group.
    group2 : str or None
        Second group name for comparison. If None, uses second unique group.
    assay_name : str, default="proteins"
        Assay name containing the features.
    cmap : str, default="viridis"
        Colormap for violin colors (gene progression).
    show : bool, default=True
        Whether to display the plot.
    ax : plt.Axes or None
        Pre-existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to violinplot.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.

    Raises
    ------
    VisualizationError
        If validation fails (container, layer, features, or groupby).
    ValueError
        If insufficient groups or no significant genes found.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.differential import rank_genes_groups_stacked_violin
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["condition"] = np.repeat(["Control", "Treatment"], 30)
    >>> assay = Assay(n_features=20)
    >>> assay.var["protein"] = [f"P{i}" for i in range(20)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 20) * 10)
    >>> container.assays["proteins"] = assay
    >>> ax = rank_genes_groups_stacked_violin(
    ...     container, layer="normalized", groupby="condition",
    ...     group1="Treatment", group2="Control", n_genes=5, show=False
    ... )
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    # Get unique groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Determine groups to compare
    if group1 is None:
        group1 = unique_groups[0]
    if group2 is None:
        if len(unique_groups) < 2:
            raise ValueError(
                f"Need at least 2 groups for comparison, found {len(unique_groups)}"
            )
        group2 = unique_groups[1]

    # Verify groups exist
    if group1 not in unique_groups:
        raise ValueError(f"Group '{group1}' not found in {groupby}")
    if group2 not in unique_groups:
        raise ValueError(f"Group '{group2}' not found in {groupby}")

    # Compute differential expression
    de_result = diff_expr_ttest(
        container=container,
        assay_name=assay_name,
        group_col=groupby,
        group1=group1,
        group2=group2,
        layer_name=layer,
    )

    # Convert to DataFrame and get top genes
    de_df = de_result.to_dataframe()
    # Filter out NaN p-values and sort
    de_df = de_df.filter(pl.col("p_value").is_not_nan())
    de_df = de_df.sort("p_value").head(n_genes)

    top_genes = de_df["feature_id"].to_list()
    n_top = len(top_genes)

    if n_top == 0:
        raise ValueError("No significant genes found for visualization")

    # Validate features exist
    validate_features(container, assay_name, top_genes)

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

    # Filter to selected features
    available_features = dict(
        zip(assay.var[var_col].to_list(), range(len(assay.var)), strict=False)
    )
    feature_idx = []
    for var in top_genes:
        if var in available_features:
            feature_idx.append(available_features[var])

    x = x[:, feature_idx]

    # Get group indices
    g1_mask = groups == group1
    g2_mask = groups == group2

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Create plot
    if ax is None:
        fig_height = 3 + n_top * 0.5
        fig_width = 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create stacked violin plot
    # Each gene gets a row with two side-by-side violins
    positions_g1 = np.arange(n_top) * 2 + 1
    positions_g2 = np.arange(n_top) * 2 + 1.5

    violin_data_g1 = []
    violin_data_g2 = []

    for i, idx in enumerate(feature_idx):
        g1_vals = x[g1_mask, idx]
        g2_vals = x[g2_mask, idx]
        violin_data_g1.append(g1_vals)
        violin_data_g2.append(g2_vals)

    # Plot violins
    parts = ax.violinplot(
        violin_data_g1 + violin_data_g2,
        positions=list(positions_g1) + list(positions_g2),
        showmeans=False,
        showmedians=True,
        **kwargs,
    )

    # Color violins by gene rank
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0, 1, n_top))
    for i in range(n_top):
        # Color group1 violin
        if i < len(parts["bodies"]) // 2:
            parts["bodies"][i].set_facecolor(colors[i])
            parts["bodies"][i].set_edgecolor("black")
            parts["bodies"][i].set_alpha(0.7)

        # Color group2 violin
        j = i + n_top
        if j < len(parts["bodies"]):
            parts["bodies"][j].set_facecolor(colors[i])
            parts["bodies"][j].set_edgecolor("black")
            parts["bodies"][j].set_alpha(0.7)

    # Configure axes
    ax.set_yticks(np.arange(n_top) * 2 + 1.25)
    ax.set_yticklabels(top_genes)
    ax.set_xlim(0, n_top * 2)
    ax.set_ylabel("")
    ax.set_xlabel("Expression Level")
    ax.set_title(f"Top {n_top} DE Genes: {group1} vs {group2}")

    # Add legend for groups
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, label=group1),
        Patch(facecolor="white", edgecolor="black", alpha=0.7, label=group2),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def volcano(
    container: ScpContainer,
    layer: str,
    groupby: str,
    group1: str | None = None,
    group2: str | None = None,
    assay_name: str = "proteins",
    pval_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    colors: tuple[str, str, str] = ("gray", "red", "blue"),
    show_labels: bool = True,
    n_labels: int = 5,
    show: bool = True,
) -> Figure:
    """Create an enhanced volcano plot with multi-panel display.

    Displays a comprehensive differential expression visualization with:
    - Main volcano plot (log2FC vs -log10 p-value)
    - Top upregulated genes table
    - Top downregulated genes table
    - Statistics summary

    Parameters
    ----------
    container : ScpContainer
        Container containing the data.
    layer : str
        Layer name to use for DE analysis (e.g., 'normalized').
    groupby : str
        Column name in obs to group by (e.g., 'cluster', 'condition').
    group1 : str or None
        First group name for comparison. If None, uses first unique group.
    group2 : str or None
        Second group name for comparison. If None, uses second unique group.
    assay_name : str, default="proteins"
        Assay name containing the features.
    pval_threshold : float, default=0.05
        P-value threshold for significance.
    logfc_threshold : float, default=1.0
        Log2 fold change threshold for significance.
    colors : tuple of str, default=("gray", "red", "blue")
        Colors for (not significant, upregulated, downregulated).
    show_labels : bool, default=True
        Whether to show labels for top genes in volcano plot.
    n_labels : int, default=5
        Number of top genes to label in each direction.
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
    ValueError
        If insufficient groups or no significant genes found.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor import ScpContainer, Assay, ScpMatrix
    >>> from scptensor.viz.recipes.differential import volcano
    >>> container = ScpContainer(n_samples=60)
    >>> container.obs["condition"] = np.repeat(["Control", "Treatment"], 30)
    >>> assay = Assay(n_features=50)
    >>> assay.var["protein"] = [f"P{i}" for i in range(50)]
    >>> assay.layers["normalized"] = ScpMatrix(X=np.random.rand(60, 50) * 10)
    >>> container.assays["proteins"] = assay
    >>> fig = volcano(
    ...     container, layer="normalized", groupby="condition",
    ...     group1="Treatment", group2="Control", show=False
    ... )
    """
    validate_container(container)
    validate_layer(container, assay_name, layer)
    validate_groupby(container, groupby)

    PlotStyle.apply_style()

    # Get unique groups
    groups = container.obs[groupby].to_numpy()
    unique_groups = np.unique(groups)

    # Determine groups to compare
    if group1 is None:
        group1 = unique_groups[0]
    if group2 is None:
        if len(unique_groups) < 2:
            raise ValueError(
                f"Need at least 2 groups for comparison, found {len(unique_groups)}"
            )
        group2 = unique_groups[1]

    # Verify groups exist
    if group1 not in unique_groups:
        raise ValueError(f"Group '{group1}' not found in {groupby}")
    if group2 not in unique_groups:
        raise ValueError(f"Group '{group2}' not found in {groupby}")

    # Compute differential expression
    de_result = diff_expr_ttest(
        container=container,
        assay_name=assay_name,
        group_col=groupby,
        group1=group1,
        group2=group2,
        layer_name=layer,
    )

    # Convert to DataFrame
    de_df = de_result.to_dataframe()
    # Filter out NaN values
    de_df = de_df.filter(
        pl.col("p_value").is_not_nan() & pl.col("log2_fc").is_not_nan()
    )

    n_total = len(de_df)
    if n_total == 0:
        raise ValueError("No valid DE results found")

    # Prepare data for plotting
    logfc = de_df["log2_fc"].to_numpy()
    pvals = de_df["p_value"].to_numpy()
    neg_log_pvals = -np.log10(np.clip(pvals, 1e-300, None))
    gene_names = de_df["feature_id"].to_numpy()

    # Determine significance
    is_sig = pvals < pval_threshold
    is_up = is_sig & (logfc > logfc_threshold)
    is_down = is_sig & (logfc < -logfc_threshold)

    n_up = is_up.sum()
    n_down = is_down.sum()
    n_sig = is_sig.sum()

    # Create multi-panel layout
    layout = PanelLayout(figsize=(14, 8), grid=(2, 3))

    # Panel 1: Main volcano plot (spanning 2 columns on left)
    def _plot_volcano(ax):
        # Color points
        point_colors = np.full(n_total, colors[0], dtype=object)
        point_colors[is_up] = colors[1]
        point_colors[is_down] = colors[2]

        ax.scatter(logfc, neg_log_pvals, c=point_colors, alpha=0.6, s=20, edgecolors="none")

        # Threshold lines
        ax.axhline(-np.log10(pval_threshold), linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(logfc_threshold, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(-logfc_threshold, linestyle="--", color="black", linewidth=0.5, alpha=0.5)

        # Labels for top genes
        if show_labels:
            # Top upregulated
            up_idx = np.where(is_up)[0]
            if len(up_idx) > 0:
                top_up_idx = up_idx[np.argsort(neg_log_pvals[up_idx])[-n_labels:]]
                for i in top_up_idx:
                    ax.annotate(
                        gene_names[i],
                        xy=(logfc[i], neg_log_pvals[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

            # Top downregulated
            down_idx = np.where(is_down)[0]
            if len(down_idx) > 0:
                top_down_idx = down_idx[np.argsort(neg_log_pvals[down_idx])[-n_labels:]]
                for i in top_down_idx:
                    ax.annotate(
                        gene_names[i],
                        xy=(logfc[i], neg_log_pvals[i]),
                        xytext=(-5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                        ha="right",
                    )

        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10 P-value")
        ax.set_title(f"Volcano Plot: {group1} vs {group2}")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label=f"Not significant ({n_total - n_sig})"),
            Patch(facecolor=colors[1], label=f"Upregulated ({n_up})"),
            Patch(facecolor=colors[2], label=f"Downregulated ({n_down})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    layout.add_panel((0, 0), _plot_volcano)
    layout.add_panel((0, 1), _plot_volcano)  # Span across

    # Panel 3: Statistics summary
    def _plot_stats(ax):
        ax.axis("off")
        stats_text = f"""
        DE Statistics: {group1} vs {group2}

        Total features: {n_total}
        Significant (p < {pval_threshold}): {n_sig}
        Upregulated (log2FC > {logfc_threshold}): {n_up}
        Downregulated (log2FC < -{logfc_threshold}): {n_down}

        Median log2FC: {np.median(logfc):.3f}
        Median -log10(p): {np.median(neg_log_pvals):.3f}
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="center", family="monospace")

    layout.add_panel((0, 2), _plot_stats)

    # Panel 4: Top upregulated genes table
    def _plot_up_table(ax):
        ax.axis("off")
        up_df = de_df.filter(is_up).sort("p_value").head(10)
        if len(up_df) > 0:
            table_data = []
            for row in up_df.iter_rows(named=True):
                table_data.append([
                    row["feature_id"],
                    f"{row['log2_fc']:.2f}",
                    f"{row['p_value']:.2e}",
                ])

            table = ax.table(
                cellText=table_data,
                colLabels=["Gene", "log2FC", "P-value"],
                cellLoc="left",
                loc="center",
                colWidths=[0.5, 0.25, 0.25],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            # Color header
            for i in range(3):
                table[(0, i)].set_facecolor("#4CAF50")
                table[(0, i)].set_text_props(weight="bold", color="white")
            ax.set_title("Top Upregulated", fontsize=10, pad=10)
        else:
            ax.text(0.5, 0.5, "No upregulated genes", ha="center", va="center")

    layout.add_panel((1, 0), _plot_up_table)

    # Panel 5: Top downregulated genes table
    def _plot_down_table(ax):
        ax.axis("off")
        down_df = de_df.filter(is_down).sort("p_value").head(10)
        if len(down_df) > 0:
            table_data = []
            for row in down_df.iter_rows(named=True):
                table_data.append([
                    row["feature_id"],
                    f"{row['log2_fc']:.2f}",
                    f"{row['p_value']:.2e}",
                ])

            table = ax.table(
                cellText=table_data,
                colLabels=["Gene", "log2FC", "P-value"],
                cellLoc="left",
                loc="center",
                colWidths=[0.5, 0.25, 0.25],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            # Color header
            for i in range(3):
                table[(0, i)].set_facecolor("#F44336")
                table[(0, i)].set_text_props(weight="bold", color="white")
            ax.set_title("Top Downregulated", fontsize=10, pad=10)
        else:
            ax.text(0.5, 0.5, "No downregulated genes", ha="center", va="center")

    layout.add_panel((1, 1), _plot_down_table)

    # Panel 6: Distribution plot
    def _plot_distribution(ax):
        ax.hist(logfc, bins=30, color="gray", alpha=0.7, edgecolor="black")
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.axvline(logfc_threshold, linestyle="--", color=colors[1], linewidth=1)
        ax.axvline(-logfc_threshold, linestyle="--", color=colors[2], linewidth=1)
        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("Count")
        ax.set_title("log2FC Distribution")

    layout.add_panel((1, 2), _plot_distribution)

    # Finalize
    fig = layout.finalize(tight=True)
    fig.suptitle(f"Differential Expression: {group1} vs {group2}", fontsize=14, y=0.98)

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    print("Testing differential visualization module...")

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import polars as pl

    from scptensor import Assay, ScpContainer, ScpMatrix
    from scptensor.core.exceptions import VisualizationError

    # Create test container
    obs = pl.DataFrame(
        {"_index": [f"S{i}" for i in range(60)], "condition": np.repeat(["Control", "Treatment"], 30)}
    )
    container = ScpContainer(obs=obs)

    var = pl.DataFrame(
        {"_index": [f"P{i}" for i in range(20)], "protein": [f"P{i}" for i in range(20)]}
    )
    # Create data with some differential expression
    np.random.seed(42)
    X = np.random.rand(60, 20) * 10
    # Make some proteins higher in Treatment
    X[30:, :5] += 5  # Upregulated in treatment
    X[:30, 5:10] += 3  # Upregulated in control

    assay = Assay(var=var, layers={"normalized": ScpMatrix(X=X)})
    container.assays["proteins"] = assay

    # Test: rank_genes_groups_dotplot basic
    print("\n1. Testing rank_genes_groups_dotplot...")
    ax = rank_genes_groups_dotplot(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
    )
    assert ax is not None
    print("   rank_genes_groups_dotplot: OK")

    # Test: rank_genes_groups_stacked_violin basic
    print("\n2. Testing rank_genes_groups_stacked_violin...")
    ax = rank_genes_groups_stacked_violin(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        show=False,
    )
    assert ax is not None
    print("   rank_genes_groups_stacked_violin: OK")

    # Test: volcano basic
    print("\n3. Testing volcano...")
    fig = volcano(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        show=False,
    )
    assert fig is not None
    plt.close(fig)
    print("   volcano: OK")

    # Test: rank_genes_groups_dotplot with pval values
    print("\n4. Testing rank_genes_groups_dotplot with pval values...")
    ax = rank_genes_groups_dotplot(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        values_to_plot="pval",
        show=False,
    )
    assert ax is not None
    print("   rank_genes_groups_dotplot pval: OK")

    # Test: rank_genes_groups_dotplot with mean values
    print("\n5. Testing rank_genes_groups_dotplot with mean values...")
    ax = rank_genes_groups_dotplot(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        n_genes=5,
        values_to_plot="mean",
        show=False,
    )
    assert ax is not None
    print("   rank_genes_groups_dotplot mean: OK")

    # Test: volcano with custom thresholds
    print("\n6. Testing volcano with custom thresholds...")
    fig = volcano(
        container,
        layer="normalized",
        groupby="condition",
        group1="Treatment",
        group2="Control",
        pval_threshold=0.01,
        logfc_threshold=0.5,
        show=False,
    )
    assert fig is not None
    plt.close(fig)
    print("   volcano custom thresholds: OK")

    # Test: Validation error - invalid layer
    print("\n7. Testing validation errors...")
    try:
        rank_genes_groups_dotplot(
            container,
            layer="nonexistent",
            groupby="condition",
            show=False,
        )
        print("   Invalid layer: FAILED")
    except Exception:
        print("   Invalid layer: OK")

    # Test: Validation error - invalid groupby
    try:
        rank_genes_groups_dotplot(
            container,
            layer="normalized",
            groupby="invalid_column",
            show=False,
        )
        print("   Invalid groupby: FAILED")
    except VisualizationError:
        print("   Invalid groupby: OK")

    print("\nAll differential visualization tests passed!")
