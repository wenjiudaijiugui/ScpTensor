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

    # Placeholder panels (will be implemented in later tasks)
    for i in range(9):
        ax = fig.add_subplot(gs[i])
        ax.text(0.5, 0.5, f"Panel {i+1}", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=theme.dpi, bbox_inches="tight")

    return fig
