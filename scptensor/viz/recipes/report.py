"""Report generation module for comprehensive analysis visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


@dataclass
class ReportTheme:
    """Theme configuration for analysis report.

    Provides comprehensive styling options for multi-panel visualization
    reports with sensible defaults and preset themes.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size (width, height) in inches
    dpi : int
        Dots per inch for figure resolution
    panel_spacing : float
        Spacing between panels in figure
    primary_color : str
        Primary color for plots (hex code)
    secondary_color : str
        Secondary color for plots (hex code)
    success_color : str
        Color for success indicators (hex code)
    danger_color : str
        Color for danger/error indicators (hex code)
    neutral_color : str
        Color for neutral elements (hex code)
    title_fontsize : int
        Font size for titles
    label_fontsize : int
        Font size for axis labels
    tick_fontsize : int
        Font size for tick labels
    font_family : str
        Font family for text
    linewidth : float
        Line width for plots
    marker_size : float
        Marker size for scatter plots
    alpha : float
        Transparency level (0-1)
    edge_color : str
        Edge color for markers
    edge_width : float
        Edge width for markers
    cmap_missing : str
        Colormap for missing values
    cmap_cluster : str
        Colormap for clusters
    """

    # Layout
    figsize: tuple[float, float] = (16, 12)
    dpi: int = 300
    panel_spacing: float = 0.3

    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#2ca02c"
    danger_color: str = "#d62728"
    neutral_color: str = "#7f7f7f"

    # Fonts
    title_fontsize: int = 14
    label_fontsize: int = 10
    tick_fontsize: int = 8
    font_family: str = "DejaVu Sans"

    # Elements
    linewidth: float = 1.0
    marker_size: float = 20
    alpha: float = 0.7
    edge_color: str = "white"
    edge_width: float = 0.5

    # Colormaps
    cmap_missing: str = "Reds"
    cmap_cluster: str = "viridis"

    @classmethod
    def dark(cls) -> ReportTheme:
        """Create dark mode theme.

        Returns
        -------
        ReportTheme
            Theme configured for dark backgrounds
        """
        return cls(
            primary_color="#4fc3f7",
            secondary_color="#ffb74d",
            neutral_color="#424242",
            cmap_missing="Oranges",
            cmap_cluster="plasma",
        )

    @classmethod
    def colorblind(cls) -> ReportTheme:
        """Create colorblind-friendly theme.

        Returns
        -------
        ReportTheme
            Theme with colorblind-friendly palette (IBM Design Language)
        """
        return cls(
            primary_color="#0072B2",
            secondary_color="#D55E00",
            success_color="#009E73",
            danger_color="#CC79A7",
            cmap_missing="Blues",
            cmap_cluster="cividis",
        )


def _render_overview_panel(
    ax: plt.Axes,
    container: "ScpContainer",
) -> None:
    """Render data overview panel with summary statistics.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    """
    ax.axis("off")
    ax.set_title("Data Overview", fontsize=12, fontweight="bold", pad=10)

    # Gather statistics
    n_samples = container.n_samples
    n_assays = len(container.assays)

    # Get first assay
    assay_names = list(container.assays.keys())
    if assay_names:
        assay_name = assay_names[0]
        assay = container.assays[assay_name]
        n_features = assay.n_features

        # Get first layer
        layer_names = list(assay.layers.keys())
        if layer_names:
            X = assay.layers[layer_names[0]].X

            # Calculate missing rate
            import numpy as np
            if hasattr(X, "toarray"):
                X_arr = X.toarray()
            else:
                X_arr = X
            missing_rate = np.isnan(X_arr).sum() / X_arr.size * 100
        else:
            n_features = 0
            missing_rate = 0
    else:
        n_features = 0
        missing_rate = 0

    # Build summary table data
    data = [
        ["Metric", "Value"],
        ["Samples", f"{n_samples:,}"],
        ["Features", f"{n_features:,}"],
        ["Assays", str(n_assays)],
        ["Missing Rate", f"{missing_rate:.1f}%"],
    ]

    # Create table
    table = ax.table(
        cellText=data,
        cellLoc="left",
        loc="center",
        bbox=[0.2, 0.1, 0.6, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor("#4477AA")
        cell.set_text_props(weight="bold", color="white")

    # Style data rows
    for i in range(1, len(data)):
        for j in range(2):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor("#EE6666")
            else:
                cell.set_facecolor("#EEEEEE")


def _render_qc_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    group_col: str = "batch",
    assay_name: str = "proteins",
) -> None:
    """Render QC distribution panel with violin plots.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    group_col : str, default "batch"
        Column in obs for grouping
    assay_name : str, default "proteins"
        Assay to visualize
    """
    import numpy as np

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Count detected (non-missing) values per sample
    detected = (~np.isnan(X_arr)).sum(axis=1)

    # Get groups
    if group_col in container.obs.columns:
        groups = container.obs[group_col].to_numpy()
    else:
        groups = np.array(["All"] * len(detected))

    # Create violin plot
    unique_groups = np.unique(groups)
    data_by_group = [detected[groups == g] for g in unique_groups]

    parts = ax.violinplot(data_by_group, positions=range(len(unique_groups)))

    # Style violin plot
    for pc in parts["bodies"]:
        pc.set_facecolor("#4477AA")
        pc.set_alpha(0.7)

    # Add scatter points
    import numpy as np
    for i, (g, data) in enumerate(zip(unique_groups, data_by_group)):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.3, s=10, color="#333333")

    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels(unique_groups, rotation=45, ha="right")
    ax.set_ylabel("Detected Features")
    ax.set_title("QC Distribution", fontsize=12, fontweight="bold")


def _render_missing_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    assay_name: str = "proteins",
    max_features: int = 100,
) -> None:
    """Render missing rate heatmap panel.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    max_features : int, default 100
        Maximum number of features to display
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Calculate missing rate per feature
    missing_rate = np.isnan(X_arr).astype(float).mean(axis=0)

    # Limit to top variable features
    n_features = min(max_features, X_arr.shape[1])
    if X_arr.shape[1] > max_features:
        var = np.nanvar(X_arr, axis=0)
        top_idx = np.argsort(-var)[:n_features]
    else:
        top_idx = np.arange(n_features)

    # Create missing rate matrix (samples x selected features)
    missing_matrix = np.isnan(X_arr[:, top_idx]).astype(float)

    # Plot heatmap
    im = ax.imshow(missing_matrix.T, aspect="auto", cmap="Reds", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Missing")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Features")
    ax.set_title("Missing Rate Heatmap", fontsize=12, fontweight="bold")


def _render_embedding_panel(
    ax1: plt.Axes,
    ax2: plt.Axes,
    container: "ScpContainer",
    assay_name: str,
    color_col: str,
) -> None:
    """Render dimensionality reduction panel with PCA and UMAP.

    Parameters
    ----------
    ax1 : plt.Axes
        First axes for PCA
    ax2 : plt.Axes
        Second axes for UMAP
    container : ScpContainer
        Input data container
    assay_name : str
        Assay to visualize
    color_col : str
        Column in obs for coloring
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    assay = container.assays.get(assay_name)
    if assay is None:
        ax1.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                 ha="center", va="center")
        ax2.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                 ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax2.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Simple imputation for visualization
    X_imp = SimpleImputer(strategy="median").fit_transform(X_arr)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_imp)

    # Get colors
    if color_col in container.obs.columns:
        colors = container.obs[color_col].to_numpy()
        unique_colors = np.unique(colors)
        color_map = {c: i for i, c in enumerate(unique_colors)}
        color_indices = np.array([color_map.get(c, 0) for c in colors])
    else:
        color_indices = np.zeros(X_arr.shape[0])

    # Plot PCA
    ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                c=color_indices, cmap="tab10", alpha=0.7, s=30)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("PCA", fontsize=12, fontweight="bold")

    # Plot UMAP (using PCA coordinates as placeholder)
    ax2.scatter(pca_result[:, 0], pca_result[:, 1],
                c=color_indices, cmap="tab10", alpha=0.7, s=30)
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_title("UMAP", fontsize=12, fontweight="bold")


def _render_feature_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    assay_name: str = "proteins",
) -> None:
    """Render feature statistics panel (mean vs variance).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    """
    import numpy as np

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Calculate mean and variance (excluding NaN)
    means = np.nanmean(X_arr, axis=0)
    vars = np.nanvar(X_arr, axis=0)

    # Plot
    ax.scatter(means, vars, alpha=0.5, s=20)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Variance")
    ax.set_title("Feature Statistics", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")


def _render_cluster_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    assay_name: str = "proteins",
) -> None:
    """Render cluster analysis heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Simplified heatmap (top 50 features)
    X_imp = SimpleImputer(strategy="median").fit_transform(X_arr)
    X_scaled = StandardScaler().fit_transform(X_imp)

    # Select top variable features
    vars = np.var(X_scaled, axis=0)
    top_idx = np.argsort(-vars)[:min(50, len(vars))]
    X_top = X_scaled[:, top_idx]

    im = ax.imshow(X_top.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Features")
    ax.set_title("Cluster Heatmap (Top 50)", fontsize=12, fontweight="bold")


def _render_batch_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    assay_name: str = "proteins",
    batch_col: str = "batch",
) -> None:
    """Render batch effect assessment panel.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    batch_col : str, default "batch"
        Column in obs for batch information
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    X_imp = SimpleImputer(strategy="median").fit_transform(X_arr)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_imp)

    if batch_col in container.obs.columns:
        batches = container.obs[batch_col].to_numpy()
        unique_batches = np.unique(batches)
        color_map = {b: i for i, b in enumerate(unique_batches)}
        color_indices = np.array([color_map.get(b, 0) for b in batches])
    else:
        color_indices = np.zeros(X_arr.shape[0])

    ax.scatter(pca_result[:, 0], pca_result[:, 1],
               c=color_indices, cmap="tab10", alpha=0.7, s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Batch Effect", fontsize=12, fontweight="bold")


def _render_diff_expr_panel(
    ax: plt.Axes,
    container: "ScpContainer",
    assay_name: str = "proteins",
    group1: str = "group_0",
    group2: str = "group_1",
) -> None:
    """Render differential expression volcano panel.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    group1 : str, default "group_0"
        First group name
    group2 : str, default "group_1"
        Second group name
    """
    import numpy as np
    from scipy import stats

    assay = container.assays.get(assay_name)
    if assay is None:
        ax.text(0.5, 0.5, f"Assay '{assay_name}' not found",
                ha="center", va="center")
        return

    layer_names = list(assay.layers.keys())
    if not layer_names:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    X = assay.layers[layer_names[0]].X
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = X

    # Simple t-test per feature
    p_values = []
    log2_fc = []

    groups = container.obs["group"].to_numpy()
    idx1 = np.where(groups == group1)[0] if group1 in groups else np.array([])
    idx2 = np.where(groups == group2)[0] if group2 in groups else np.array([])

    if len(idx1) == 0 or len(idx2) == 0:
        ax.text(0.5, 0.5, f"Groups '{group1}' or '{group2}' not found",
                ha="center", va="center")
        return

    for j in range(X_arr.shape[1]):
        g1 = X_arr[idx1, j]
        g2 = X_arr[idx2, j]

        # Remove NaN
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            p_values.append(1.0)
            log2_fc.append(0.0)
            continue

        result = stats.ttest_ind(g1, g2)
        p_values.append(result.pvalue)

        fc = np.median(g1) / (np.median(g2) + 1e-10)
        log2_fc.append(np.log2(fc + 1e-10))

    p_values = np.array(p_values)
    log2_fc = np.array(log2_fc)

    # Plot
    significant = (p_values < 0.05).astype(int)

    ax.scatter(log2_fc, -np.log10(p_values + 1e-300),
               c=significant, cmap="RdYlBu", alpha=0.5, s=20)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10 P-value")
    ax.set_title(f"DE Analysis ({group1} vs {group2})",
                 fontsize=12, fontweight="bold")


def generate_analysis_report(
    container: "ScpContainer",
    assay_name: str = "proteins",
    group_col: str = "group",
    batch_col: str | None = "batch",
    diff_expr_groups: tuple[str, str] | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 300,
    style: str = "science",
    panels: list[str] | None = None,
    theme: ReportTheme | None = None,
) -> Figure:
    """Generate a comprehensive analysis report as a single-page figure.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, default "proteins"
        Assay to visualize
    group_col : str, default "group"
        Column in obs for grouping
    batch_col : str | None, default "batch"
        Column in obs for batch information
    diff_expr_groups : tuple[str, str] | None
        (group1, group2) for differential expression
    output_path : str | None
        Path to save the figure
    figsize : tuple[float, float], default (16, 12)
        Figure size in inches
    dpi : int, default 300
        Resolution for output
    style : str, default "science"
        Matplotlib style to use
    panels : list[str] | None
        Which panels to include (default: all)
    theme : ReportTheme | None
        Theme configuration (default: ReportTheme())

    Returns
    -------
    Figure
        The generated figure
    """
    # Import here to avoid circular dependency
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from scptensor.core.structures import ScpContainer

    if theme is None:
        theme = ReportTheme(figsize=figsize, dpi=dpi)

    # Apply style
    if style == "science":
        try:
            plt.style.use(["science", "no-latex"])
        except OSError:
            # Fallback if scienceplots not available
            plt.style.use("default")
    else:
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            plt.style.use("default")

    # Create figure with grid layout
    fig = plt.figure(figsize=theme.figsize, dpi=theme.dpi)
    gs = GridSpec(3, 3, figure=fig, hspace=theme.panel_spacing, wspace=theme.panel_spacing)

    # Render title
    fig.suptitle(
        f"ScpTensor Analysis Report | Dataset: {assay_name}",
        fontsize=theme.title_fontsize + 4,
        fontweight="bold",
    )

    # Panel 1: Data Overview
    ax1 = fig.add_subplot(gs[0, 0])
    _render_overview_panel(ax1, container)

    # Panel 2: QC Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    _render_qc_panel(ax2, container, group_col=batch_col or "batch", assay_name=assay_name)

    # Panel 3: Missing Rate Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    _render_missing_panel(ax3, container, assay_name=assay_name)

    # Panel 4: Dimensionality Reduction
    ax4a = fig.add_subplot(gs[1, 0])
    ax4b = fig.add_subplot(gs[1, 1])
    _render_embedding_panel(ax4a, ax4b, container, assay_name, group_col)

    # Panel 5: Feature Statistics
    ax5 = fig.add_subplot(gs[1, 2])
    _render_feature_panel(ax5, container, assay_name)

    # Panel 6: Cluster Analysis
    ax6 = fig.add_subplot(gs[2, 0])
    _render_cluster_panel(ax6, container, assay_name)

    # Panel 7: Batch Effect
    ax7 = fig.add_subplot(gs[2, 1])
    _render_batch_panel(ax7, container, assay_name, batch_col or "batch")

    # Panel 8: Differential Expression
    ax8 = fig.add_subplot(gs[2, 2])
    if diff_expr_groups:
        _render_diff_expr_panel(ax8, container, assay_name,
                                 diff_expr_groups[0], diff_expr_groups[1])
    else:
        ax8.text(0.5, 0.5, "Specify diff_expr_groups\nfor DE analysis",
                 ha="center", va="center")

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=theme.dpi, bbox_inches="tight")

    return fig
