"""Extended style manager with SciencePlots integration.

This module provides unified style management for ScpTensor visualizations,
including SciencePlots integration and proteomics-specific color schemes.
"""

import importlib
from typing import Literal

from cycler import cycler

# Predefined themes
THEMES: dict[str, list[str]] = {
    "science": ["science", "no-latex"],
    "science_grid": ["science", "grid", "no-latex"],
    "ieee": ["ieee", "no-latex"],
    "nature": ["nature", "no-latex"],
}

# Proteomics color schemes (distinct from RNA-seq)
CMAP_PROTEOMICS: dict[str, str] = {
    "expression": "viridis",
    "missing": "gray_r",
    "logfc": "RdBu_r",
    "significance": "plasma",
    "clusters": "tab20",
}

CBLIND_FRIENDLY_CYCLE: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#F0E442",
]

_VALID_PURPOSES = set(CMAP_PROTEOMICS.keys())
_DEFAULT_THEME: Literal["science", "science_grid", "ieee", "nature"] = "science"
_DEFAULT_SAVE_DPI = 300


def _publication_rc_params(save_dpi: int) -> dict[str, object]:
    """Return publication-oriented rcParams.

    Defaults follow common journal figure constraints:
    readable sans-serif fonts, restrained line widths, high export DPI,
    and colorblind-friendly categorical palettes.
    """
    figure_dpi = min(max(120, save_dpi), 180)
    return {
        "figure.dpi": figure_dpi,
        "savefig.dpi": save_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.borderaxespad": 0.2,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "axes.prop_cycle": cycler(color=CBLIND_FRIENDLY_CYCLE),
        "image.cmap": CMAP_PROTEOMICS["expression"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


class PlotStyle:
    """Unified style management with SciencePlots integration.

    Provides methods for applying publication-ready plotting styles
    and retrieving appropriate color schemes for different visualization
    purposes in DIA-based single-cell proteomics analysis.
    """

    @staticmethod
    def apply_style(
        theme: Literal["science", "science_grid", "ieee", "nature"] = "science",
        dpi: int = 300,
    ) -> None:
        """Apply plotting style.

        Applies a predefined theme to matplotlib with SciencePlots integration.
        Sets DPI for figures and configures Unicode minus handling.

        Parameters
        ----------
        theme : {"science", "science_grid", "ieee", "nature"}
            Theme name from THEMES. Default is "science".
        dpi : int
            DPI for saved figures. Default is 300.

        Raises
        ------
        ValueError
            If the theme name is not recognized.

        """
        # Import scienceplots first to register its styles with matplotlib.
        scienceplots_available = importlib.util.find_spec("scienceplots") is not None
        if scienceplots_available:
            try:
                importlib.import_module("scienceplots")
            except ImportError:
                scienceplots_available = False

        import matplotlib.pyplot as plt

        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Choose from {list(THEMES.keys())}")

        styles = THEMES[theme]
        try:
            plt.style.use(styles)
        except Exception:
            # Fallback when SciencePlots styles are unavailable.
            plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(_publication_rc_params(save_dpi=dpi))

    @staticmethod
    def get_colormap(purpose: str, custom: str | None = None) -> str:
        """Get color scheme for specific purpose.

        Returns a colormap name appropriate for the visualization purpose,
        using proteomics-specific defaults. Custom colormaps can override
        the defaults.

        Parameters
        ----------
        purpose : str
            One of: expression, missing, logfc, significance, clusters
        custom : str or None
            Custom colormap name (overrides purpose).

        Returns
        -------
        str
            Colormap name.

        Raises
        ------
        ValueError
            If the purpose is not recognized.

        """
        if custom is not None:
            return custom
        if purpose not in _VALID_PURPOSES:
            raise ValueError(f"Unknown purpose: {purpose}. Choose from {_VALID_PURPOSES}")
        return CMAP_PROTEOMICS[purpose]


def setup_style() -> None:
    """Apply SciencePlots style with fallback.

    Uses the SciencePlots 'science' style without LaTeX for clean
    publication-ready figures. Falls back to seaborn-whitegrid if
    SciencePlots is unavailable.

    .. deprecated::
        Use :meth:`PlotStyle.apply_style` instead for more control.
    """
    try:
        PlotStyle.apply_style(theme=_DEFAULT_THEME, dpi=_DEFAULT_SAVE_DPI)
    except Exception:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(_publication_rc_params(save_dpi=_DEFAULT_SAVE_DPI))
