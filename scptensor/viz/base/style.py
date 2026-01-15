"""Extended style manager with SciencePlots integration.

This module provides unified style management for ScpTensor visualizations,
including SciencePlots integration and proteomics-specific color schemes.
"""

from typing import Literal

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

_VALID_PURPOSES = set(CMAP_PROTEOMICS.keys())


class PlotStyle:
    """Unified style management with SciencePlots integration.

    Provides methods for applying publication-ready plotting styles
    and retrieving appropriate color schemes for different visualization
    purposes in single-cell proteomics analysis.
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
        # Import scienceplots first to register its styles with matplotlib
        try:
            import scienceplots  # noqa: F401

        except ImportError:
            pass

        import matplotlib.pyplot as plt

        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Choose from {list(THEMES.keys())}")

        styles = THEMES[theme]
        plt.style.use(styles)
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams["savefig.dpi"] = dpi
        plt.rcParams["axes.unicode_minus"] = False

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
    import matplotlib.pyplot as plt

    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
