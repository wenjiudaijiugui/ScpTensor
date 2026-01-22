"""Common visual styling utilities for benchmark display modules.

This module provides a centralized configuration system for visual styling
across all benchmark display modules, ensuring consistent color palettes,
typography, and layout throughout the benchmarking visualization suite.

Components
----------
- ColorPalette: Dataclass for defining color schemes
- TypographyConfig: Dataclass for font size configurations
- LayoutConfig: Dataclass for spacing and padding settings
- get_module_colors: Retrieve color palette for a specific module
- apply_typography_theme: Apply standardized font settings to figures
- apply_color_style: Apply module-specific colors to axes

Examples
--------
Apply module-specific colors to a plot:

>>> import matplotlib.pyplot as plt
>>> from scptensor.benchmark.display.common import get_module_colors, apply_color_style
>>> fig, ax = plt.subplots()
>>> colors = get_module_colors("normalization")
>>> ax.plot([1, 2, 3], [1, 4, 2], color=colors.primary, label="Method A")
>>> apply_color_style(ax, "normalization")

Apply typography theme to a figure:

>>> from scptensor.benchmark.display.common import apply_typography_theme
>>> fig, ax = plt.subplots()
>>> apply_typography_theme(fig)
>>> ax.set_title("My Plot")  # Will use standardized title font size
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, assert_never

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =============================================================================
# Plot Style Configuration (from comparison_viz.py)
# =============================================================================


class PlotStyle(Enum):
    """Available plot styles for publication-quality figures.

    Options
    -------
    SCIENCE : str
        Science magazine style (clean, minimal)
    IEEE : str
        IEEE journal style (compact, technical)
    NATURE : str
        Nature journal style (elegant, colorful)
    DEFAULT : str
        Default matplotlib style
    """

    SCIENCE = "science"
    IEEE = "ieee"
    NATURE = "nature"
    DEFAULT = "default"


def configure_plots(
    style: PlotStyle = PlotStyle.SCIENCE,
    dpi: int = 300,
) -> None:
    """Configure matplotlib for publication-quality plots.

    This function sets up matplotlib with SciencePlots styles for
    professional, publication-ready figures. If SciencePlots is not
    available, falls back to default matplotlib style.

    Parameters
    ----------
    style : PlotStyle, default=PlotStyle.SCIENCE
        Plot style to use. Options: SCIENCE, IEEE, NATURE, DEFAULT
    dpi : int, default=300
        DPI for saved figures. Higher DPI = crisper output.

    Examples
    --------
    >>> from scptensor.benchmark.display.common import configure_plots, PlotStyle
    >>> configure_plots(style=PlotStyle.SCIENCE, dpi=300)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1, 2, 3], [1, 4, 2])
    >>> plt.savefig('figure.png')

    Notes
    -----
    Requires scienceplots package for SCIENCE/IEEE/NATURE styles.
    Install with: pip install scienceplots
    """
    # Try to import scienceplots
    try:
        import scienceplots  # noqa: F401 (tested for availability)

        scienceplots_available = True
    except ImportError:
        scienceplots_available = False

    # Apply style if available
    if scienceplots_available and style != PlotStyle.DEFAULT:
        style_map = {
            PlotStyle.SCIENCE: ["science", "no-latex"],
            PlotStyle.IEEE: ["ieee", "no-latex"],
            PlotStyle.NATURE: ["nature", "no-latex"],
        }
        plt.style.use(style_map.get(style, ["science", "no-latex"]))
    else:
        plt.style.use("default")

    # Configure matplotlib parameters
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
        }
    )


__all__ = [
    "ColorPalette",
    "TypographyConfig",
    "LayoutConfig",
    "get_module_colors",
    "apply_typography_theme",
    "apply_color_style",
    "ModuleType",
    "PlotStyle",
    "configure_plots",
]


ModuleType = Literal[
    "normalization",
    "imputation",
    "integration",
    "dim_reduction",
    "qc",
    "end_to_end",
    "feature_selection",
]


@dataclass(frozen=True)
class ColorPalette:
    """Color scheme definition for a benchmark module.

    Parameters
    ----------
    name : str
        Identifier for the color palette.
    primary : str
        Primary color for main data series.
    secondary : str
        Secondary color for supporting data series.
    accent : str
        Accent color for highlights and emphasis.
    neutral : str
        Neutral color for backgrounds and borders.
    success : str
        Color for positive indicators (e.g., high performance).
    warning : str
        Color for caution indicators.
    error : str
        Color for negative indicators (e.g., errors, low performance).
    """

    name: str
    primary: str
    secondary: str
    accent: str
    neutral: str
    success: str
    warning: str
    error: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return core colors as a tuple.

        Returns
        -------
        tuple[str, str, str, str]
            (primary, secondary, accent, neutral) colors.
        """
        return (self.primary, self.secondary, self.accent, self.neutral)


@dataclass(frozen=True)
class TypographyConfig:
    """Typography settings for benchmark figures.

    Standardizes font sizes across all visualization types to ensure
    consistent readability and professional appearance.

    Parameters
    ----------
    title_size : float, default 14.0
        Font size for figure and subplot titles.
    label_size : float, default 11.0
        Font size for axis labels.
    tick_size : float, default 10.0
        Font size for axis tick labels.
    legend_size : float, default 10.0
        Font size for legend text.
    annotation_size : float, default 9.0
        Font size for annotations and text notes.
    font_family : str, default "DejaVu Sans"
        Font family for all text elements.
    """

    title_size: float = 14.0
    label_size: float = 11.0
    tick_size: float = 10.0
    legend_size: float = 10.0
    annotation_size: float = 9.0
    font_family: str = "DejaVu Sans"


@dataclass(frozen=True)
class LayoutConfig:
    """Layout and spacing settings for benchmark figures.

    Standardizes margins, padding, and spacing to ensure consistent
    figure composition across different plot types.

    Parameters
    ----------
    figure_left : float, default 0.12
        Left margin as fraction of figure width.
    figure_right : float, default 0.95
        Right margin as fraction of figure width.
    figure_bottom : float, default 0.15
        Bottom margin as fraction of figure height.
    figure_top : float, default 0.92
        Top margin as fraction of figure height.
    figure_wspace : float, default 0.25
        Horizontal spacing between subplots.
    figure_hspace : float, default 0.30
        Vertical spacing between subplots.
    """

    figure_left: float = 0.12
    figure_right: float = 0.95
    figure_bottom: float = 0.15
    figure_top: float = 0.92
    figure_wspace: float = 0.25
    figure_hspace: float = 0.30


# Color palettes for each benchmark module
# Normalization: Blue/Green (transformation focus)
NORMALIZATION_COLORS = ColorPalette(
    name="normalization",
    primary="#1f77b4",  # Blue - primary data
    secondary="#2ca02c",  # Green - secondary data
    accent="#17becf",  # Cyan - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#ff7f0e",  # Orange - caution
    error="#d62728",  # Red - negative
)

# Imputation: Red/Purple (filling missing data)
IMPUTATION_COLORS = ColorPalette(
    name="imputation",
    primary="#d62728",  # Red - primary data
    secondary="#9467bd",  # Purple - secondary data
    accent="#e377c2",  # Pink - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#ff7f0e",  # Orange - caution
    error="#d62728",  # Red - negative
)

# Integration: Orange/Teal (batch correction)
INTEGRATION_COLORS = ColorPalette(
    name="integration",
    primary="#ff7f0e",  # Orange - primary data
    secondary="#00ced1",  # Dark Turquoise - secondary data
    accent="#20b2aa",  # Light Sea Green - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#d62728",  # Red - caution
    error="#8b0000",  # Dark Red - negative
)

# Dimensionality Reduction: Purple/Pink
DIM_REDUCTION_COLORS = ColorPalette(
    name="dim_reduction",
    primary="#9467bd",  # Purple - primary data
    secondary="#e377c2",  # Pink - secondary data
    accent="#db79db",  # Medium Purple - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#ff7f0e",  # Orange - caution
    error="#d62728",  # Red - negative
)

# Quality Control: Gray/Brown (quality assessment)
QC_COLORS = ColorPalette(
    name="qc",
    primary="#8c564b",  # Brown - primary data
    secondary="#7f7f7f",  # Gray - secondary data
    accent="#bcbd22",  # Olive - highlights
    neutral="#d3d3d3",  # Light Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#ff7f0e",  # Orange - caution
    error="#d62728",  # Red - negative
)

# End-to-End: Cyan/Magenta (comprehensive)
END_TO_END_COLORS = ColorPalette(
    name="end_to_end",
    primary="#00bcd4",  # Cyan - primary data
    secondary="#ff00ff",  # Magenta - secondary data
    accent="#1f77b4",  # Blue - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#00ff00",  # Lime - positive
    warning="#ffa500",  # Orange - caution
    error="#dc143c",  # Crimson - negative
)

# Feature Selection: Gold/Navy (feature importance)
FEATURE_SELECTION_COLORS = ColorPalette(
    name="feature_selection",
    primary="#b8860b",  # Dark Goldenrod - primary data
    secondary="#000080",  # Navy - secondary data
    accent="#daa520",  # Goldenrod - highlights
    neutral="#7f7f7f",  # Gray - neutral
    success="#2ca02c",  # Green - positive
    warning="#ff7f0e",  # Orange - caution
    error="#d62728",  # Red - negative
)

# Module-specific color palette mapping
_MODULE_COLORS: dict[ModuleType, ColorPalette] = {
    "normalization": NORMALIZATION_COLORS,
    "imputation": IMPUTATION_COLORS,
    "integration": INTEGRATION_COLORS,
    "dim_reduction": DIM_REDUCTION_COLORS,
    "qc": QC_COLORS,
    "end_to_end": END_TO_END_COLORS,
    "feature_selection": FEATURE_SELECTION_COLORS,
}

# Global configuration instances
DEFAULT_TYPOGRAPHY = TypographyConfig()
DEFAULT_LAYOUT = LayoutConfig()


def get_module_colors(module_name: ModuleType | str) -> ColorPalette:
    """Get the color palette for a specific benchmark module.

    Parameters
    ----------
    module_name : ModuleType | str
        Name of the benchmark module. Valid values are:
        "normalization", "imputation", "integration", "dim_reduction",
        "qc", "end_to_end", "feature_selection".

    Returns
    -------
    ColorPalette
        Color palette configured for the specified module.

    Raises
    ------
    ValueError
        If module_name is not a recognized module type.

    Examples
    --------
    >>> colors = get_module_colors("normalization")
    >>> print(colors.primary)
    '#1f77b4'

    >>> colors = get_module_colors("imputation")
    >>> print(colors.name)
    'imputation'
    """
    module: ModuleType
    if isinstance(module_name, str):
        # Validate and convert string to ModuleType
        valid_modules = {
            "normalization",
            "imputation",
            "integration",
            "dim_reduction",
            "qc",
            "end_to_end",
            "feature_selection",
        }
        if module_name not in valid_modules:
            raise ValueError(
                f"Unknown module '{module_name}'. "
                f"Valid modules are: {', '.join(sorted(valid_modules))}"
            )
        module = module_name  # type: ignore[assignment]
    else:
        module = module_name

    return _MODULE_COLORS[module]


def apply_typography_theme(
    fig: Figure,
    config: TypographyConfig | None = None,
) -> None:
    """Apply standardized typography settings to a matplotlib figure.

    This function updates the font sizes for titles, labels, ticks, and
    legends across all axes in the figure to ensure consistent styling.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to apply typography settings to.
    config : TypographyConfig, optional
        Typography configuration to apply. If None, uses DEFAULT_TYPOGRAPHY.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scptensor.benchmark.display.common import apply_typography_theme
    >>> fig, ax = plt.subplots()
    >>> ax.set_title("My Plot")
    >>> ax.set_xlabel("X axis")
    >>> apply_typography_theme(fig)
    """
    if config is None:
        config = DEFAULT_TYPOGRAPHY

    for ax in fig.axes:
        # Update title font size
        if ax.get_title() != "":
            ax.set_title(ax.get_title(), fontsize=config.title_size, fontfamily=config.font_family)

        # Update axis labels
        if ax.get_xlabel() != "":
            ax.set_xlabel(
                ax.get_xlabel(), fontsize=config.label_size, fontfamily=config.font_family
            )
        if ax.get_ylabel() != "":
            ax.set_ylabel(
                ax.get_ylabel(), fontsize=config.label_size, fontfamily=config.font_family
            )

        # Update tick labels
        ax.tick_params(
            axis="x",
            labelsize=config.tick_size,
        )
        ax.tick_params(
            axis="y",
            labelsize=config.tick_size,
        )

        # Update legend if present
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(config.legend_size)
                text.set_fontfamily(config.font_family)


def apply_color_style(
    ax: Axes,
    module_name: ModuleType | str,
    palette: ColorPalette | None = None,
) -> None:
    """Apply module-specific color styling to a matplotlib axes.

    This function applies the color palette for a specific module to
    various plot elements including spines, grid, and face color.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to apply color styling to.
    module_name : ModuleType | str
        Name of the benchmark module for color palette selection.
    palette : ColorPalette, optional
        Custom color palette to use. If None, retrieves palette for module_name.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scptensor.benchmark.display.common import apply_color_style
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 2])
    >>> apply_color_style(ax, "normalization")
    """
    if palette is None:
        palette = get_module_colors(module_name)

    # Apply neutral color to spines
    for spine in ax.spines.values():
        spine.set_edgecolor(palette.neutral)
        spine.set_linewidth(0.8)

    # Apply subtle grid using neutral color
    ax.grid(True, color=palette.neutral, alpha=0.3, linewidth=0.5)

    # Set face color to very light gray/white
    ax.set_facecolor("#fafafa")


def apply_layout_config(
    fig: Figure,
    config: LayoutConfig | None = None,
) -> None:
    """Apply standardized layout settings to a matplotlib figure.

    This function configures subplot margins and spacing to ensure
    consistent figure composition.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to apply layout settings to.
    config : LayoutConfig, optional
        Layout configuration to apply. If None, uses DEFAULT_LAYOUT.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scptensor.benchmark.display.common import apply_layout_config
    >>> fig, axes = plt.subplots(1, 2)
    >>> apply_layout_config(fig)
    """
    if config is None:
        config = DEFAULT_LAYOUT

    fig.subplots_adjust(
        left=config.figure_left,
        right=config.figure_right,
        bottom=config.figure_bottom,
        top=config.figure_top,
        wspace=config.figure_wspace,
        hspace=config.figure_hspace,
    )


def get_compatible_color(
    module_name: ModuleType | str,
    color_index: int,
) -> str:
    """Get a color from a module's palette by index.

    Provides sequential access to colors for multi-series plots.
    Returns colors in order: primary, secondary, accent, neutral,
    success, warning, error.

    Parameters
    ----------
    module_name : ModuleType | str
        Name of the benchmark module.
    color_index : int
        Index of the color to retrieve (0-6).

    Returns
    -------
    str
        Hex color code for the requested index.

    Raises
    ------
    IndexError
        If color_index is outside the range 0-6.

    Examples
    --------
    >>> get_compatible_color("normalization", 0)
    '#1f77b4'
    >>> get_compatible_color("imputation", 1)
    '#9467bd'
    """
    palette = get_module_colors(module_name)
    colors = [
        palette.primary,
        palette.secondary,
        palette.accent,
        palette.neutral,
        palette.success,
        palette.warning,
        palette.error,
    ]

    if color_index < 0 or color_index >= len(colors):
        raise IndexError(f"color_index must be between 0 and {len(colors) - 1}, got {color_index}")

    return colors[color_index]


def get_status_color(
    module_name: ModuleType | str,
    status: Literal["success", "warning", "error", "neutral"],
) -> str:
    """Get a status-indicator color for a specific module.

    Parameters
    ----------
    module_name : ModuleType | str
        Name of the benchmark module.
    status : {"success", "warning", "error", "neutral"}
        Status type for color selection.

    Returns
    -------
    str
        Hex color code for the status color.

    Examples
    --------
    >>> get_status_color("normalization", "success")
    '#2ca02c'
    >>> get_status_color("imputation", "error")
    '#d62728'
    """
    palette = get_module_colors(module_name)

    if status == "success":
        return palette.success
    elif status == "warning":
        return palette.warning
    elif status == "error":
        return palette.error
    elif status == "neutral":
        return palette.neutral
    else:
        assert_never(status)
