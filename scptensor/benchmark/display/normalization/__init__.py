"""Normalization display module for ScpTensor benchmark.

Provides visualization capabilities for comparing normalization results between
ScpTensor and competing frameworks (e.g., Scanpy). This module generates
publication-quality figures for log normalization and z-score normalization
comparisons.

Classes
-------
LogNormalizeDisplay
    Visualizes log normalization comparison results between ScpTensor and Scanpy.
ZScoreDisplay
    Visualizes z-score normalization verification and comparison results.

Functions
---------
setup_plot_style
    Configures matplotlib with SciencePlots style for publication-quality figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scptensor.benchmark.display.base import DisplayBase
from scptensor.benchmark.display.common import (
    ColorPalette,
    LayoutConfig,
    TypographyConfig,
    apply_color_style,
    apply_typography_theme,
    get_compatible_color,
    get_module_colors,
)

__all__ = [
    "LogNormalizeDisplay",
    "ZScoreDisplay",
    "setup_plot_style",
    "NormalizationComparisonResult",
    "ZScoreVerificationResult",
]


def setup_plot_style(dpi: int = 300) -> tuple[ColorPalette, TypographyConfig, LayoutConfig]:
    """Configure matplotlib with SciencePlots style for publication-quality figures.

    Parameters
    ----------
    dpi : int, default=300
        Resolution in dots per inch for saved figures.

    Returns
    -------
    tuple[ColorPalette, TypographyConfig, LayoutConfig]
        Color palette, typography configuration, and layout configuration
        for the normalization module.

    Notes
    -----
    Applies the SciencePlots 'science' style with 'no-latex' option
    for clean publication-ready figures. Falls back to seaborn-v0_8-whitegrid
    if SciencePlots is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.unicode_minus"] = False

    colors = get_module_colors("normalization")
    typo_config = TypographyConfig()
    layout_config = LayoutConfig()

    return colors, typo_config, layout_config


@dataclass(frozen=True)
class NormalizationComparisonResult:
    """Result of normalization comparison between ScpTensor and Scanpy.

    Attributes
    ----------
    raw_data : npt.NDArray[np.float64]
        Original data before normalization.
    scptensor_normalized : npt.NDArray[np.float64]
        Data normalized using ScpTensor.
    scanpy_normalized : npt.NDArray[np.float64]
        Data normalized using Scanpy.
    method_name : str
        Name of the normalization method (e.g., "log_normalize", "z_score").
    base : float | None
        Log base used for log normalization, if applicable.
    offset : float | None
        Offset added before log transformation, if applicable.
    """

    raw_data: npt.NDArray[np.float64]
    scptensor_normalized: npt.NDArray[np.float64]
    scanpy_normalized: npt.NDArray[np.float64]
    method_name: str
    base: float | None = None
    offset: float | None = None


@dataclass(frozen=True)
class ZScoreVerificationResult:
    """Result of z-score normalization verification.

    Attributes
    ----------
    before_data : npt.NDArray[np.float64]
        Data before z-score normalization.
    after_data : npt.NDArray[np.float64]
        Data after z-score normalization.
    has_missing : bool
        Whether the original data contained missing values.
    axis : int
        Axis along which z-score was computed (0=features, 1=samples).
    ddof : int
        Delta degrees of freedom used for std calculation.
    mean : float
        Mean of the normalized data (should be close to 0).
    std : float
        Standard deviation of the normalized data (should be close to 1).
    """

    before_data: npt.NDArray[np.float64]
    after_data: npt.NDArray[np.float64]
    has_missing: bool
    axis: int
    ddof: int
    mean: float
    std: float


class LogNormalizeDisplay(DisplayBase):
    """Display class for log normalization comparison visualizations.

    Generates publication-quality figures comparing ScpTensor and Scanpy
    log normalization results, including distribution flow plots and
    agreement scatter plots.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.normalization import LogNormalizeDisplay
    >>> display = LogNormalizeDisplay()
    >>> raw = np.random.lognormal(mean=2, sigma=0.5, size=(100, 50))
    >>> scptensor_result = np.log2(raw + 1)
    >>> scanpy_result = np.log2(raw + 1)
    >>> result = NormalizationComparisonResult(
    ...     raw_data=raw,
    ...     scptensor_normalized=scptensor_result,
    ...     scanpy_normalized=scanpy_result,
    ...     method_name="log_normalize",
    ...     base=2.0,
    ...     offset=1.0
    ... )
    >>> fig_path = display.render_distribution_flow(result)
    """

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "normalization"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_distribution_flow or render_agreement instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "log_normalize_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_distribution_flow or render_agreement",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_distribution_flow(self, result: NormalizationComparisonResult) -> Path:
        """Generate a figure showing raw to log transformed distribution flow.

        Creates a two-panel figure showing the distribution before and after
        log transformation, with histograms overlayed to visualize the
        transformation effect.

        Parameters
        ----------
        result : NormalizationComparisonResult
            Comparison result containing raw and normalized data.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'log_normalize_distribution.png' in the
        normalization figures subdirectory.
        """
        import matplotlib.pyplot as plt

        colors, typo_config, layout_config = setup_plot_style()

        raw_flat = result.raw_data.ravel()
        scptensor_flat = result.scptensor_normalized.ravel()

        # Remove NaN and infinite values for plotting
        raw_valid = raw_flat[~(np.isnan(raw_flat) | np.isinf(raw_flat))]
        scptensor_valid = scptensor_flat[~(np.isnan(scptensor_flat) | np.isinf(scptensor_flat))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Raw data histogram
        ax1.hist(raw_valid, bins=50, alpha=0.7, color=colors.primary, edgecolor="black")
        ax1.set_xlabel("Raw Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Before Log Transformation")
        apply_color_style(ax1, "normalization", colors)

        # Log-transformed data histogram
        base_label = f"log{result.base}" if result.base else "log"
        offset_label = f" + {result.offset}" if result.offset and result.offset != 1 else ""
        ax2.hist(scptensor_valid, bins=50, alpha=0.7, color=colors.secondary, edgecolor="black")
        ax2.set_xlabel(f"{base_label}(Intensity{offset_label})")
        ax2.set_ylabel("Frequency")
        ax2.set_title("After Log Transformation")
        apply_color_style(ax2, "normalization", colors)

        apply_typography_theme(fig, typo_config)

        output_path = self._figures_dir / "log_normalize_distribution.png"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_agreement(self, result: NormalizationComparisonResult) -> Path:
        """Generate a scatter plot comparing ScpTensor vs Scanpy results.

        Creates a scatter plot with identity line to visualize agreement
        between ScpTensor and Scanpy log normalization implementations.

        Parameters
        ----------
        result : NormalizationComparisonResult
            Comparison result containing both normalized datasets.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'log_normalize_agreement.png' in the
        normalization figures subdirectory.

        The scatter plot includes:
        - Identity line (y=x) for reference
        - Correlation coefficient in the title
        - Point density visualization using alpha blending
        """
        import matplotlib.pyplot as plt

        colors, typo_config, layout_config = setup_plot_style()

        scptensor_flat = result.scptensor_normalized.ravel()
        scanpy_flat = result.scanpy_normalized.ravel()

        # Remove NaN and infinite values
        valid_mask = ~(
            np.isnan(scptensor_flat)
            | np.isinf(scptensor_flat)
            | np.isnan(scanpy_flat)
            | np.isinf(scanpy_flat)
        )
        scptensor_valid = scptensor_flat[valid_mask]
        scanpy_valid = scanpy_flat[valid_mask]

        # Compute correlation
        from scipy.stats import pearsonr

        corr, _ = pearsonr(scptensor_valid, scanpy_valid)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot with transparency for density visualization
        ax.scatter(
            scanpy_valid, scptensor_valid, alpha=0.3, s=10, c=colors.primary, edgecolors="none"
        )

        # Identity line
        min_val = min(np.min(scptensor_valid), np.min(scanpy_valid))
        max_val = max(np.max(scptensor_valid), np.max(scanpy_valid))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color=colors.error,
            linestyle="--",
            linewidth=2,
            label="Identity Line",
        )

        ax.set_xlabel("Scanpy Normalized")
        ax.set_ylabel("ScpTensor Normalized")
        ax.set_title(f"Log Normalization Agreement\nCorrelation: r = {corr:.6f}")
        ax.legend()
        apply_color_style(ax, "normalization", colors)

        # Set equal aspect ratio for proper identity line visualization
        ax.set_aspect("equal", adjustable="box")

        apply_typography_theme(fig, typo_config)

        output_path = self._figures_dir / "log_normalize_agreement.png"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path


class ZScoreDisplay(DisplayBase):
    """Display class for z-score normalization verification visualizations.

    Generates before/after plots for z-score normalization, handling both
    complete data and data with missing values (after imputation).

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.normalization import ZScoreDisplay
    >>> display = ZScoreDisplay()
    >>> before = np.random.normal(loc=10, scale=3, size=(100, 50))
    >>> after = (before - before.mean(axis=0, keepdims=True)) / before.std(axis=0, ddof=1, keepdims=True)
    >>> result = ZScoreVerificationResult(
    ...     before_data=before,
    ...     after_data=after,
    ...     has_missing=False,
    ...     axis=0,
    ...     ddof=1,
    ...     mean=float(after.mean()),
    ...     std=float(after.std())
    ... )
    >>> fig_path = display.render_verification(result)
    """

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "normalization"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_verification instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "z_score_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "Use render_verification", ha="center", va="center", transform=ax.transAxes
        )
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_verification(self, result: ZScoreVerificationResult) -> Path:
        """Generate before/after plots for z-score normalization verification.

        Creates a two-panel figure showing data distribution before and after
        z-score normalization, with statistics indicating successful
        standardization (mean ~ 0, std ~ 1).

        Parameters
        ----------
        result : ZScoreVerificationResult
            Verification result containing before/after data and statistics.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'z_score_verification.png' in the
        normalization figures subdirectory.

        For data with missing values, the visualization handles NaN values
        appropriately by excluding them from the histograms.
        """
        import matplotlib.pyplot as plt

        colors, typo_config, layout_config = setup_plot_style()

        before_flat = result.before_data.ravel()
        after_flat = result.after_data.ravel()

        # Remove NaN values for plotting
        before_valid = before_flat[~np.isnan(before_flat)]
        after_valid = after_flat[~np.isnan(after_flat)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Before normalization histogram
        ax1.hist(before_valid, bins=50, alpha=0.7, color=colors.primary, edgecolor="black")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title(
            f"Before Z-Score\nMean: {np.mean(before_valid):.2f}, Std: {np.std(before_valid):.2f}"
        )
        apply_color_style(ax1, "normalization", colors)

        # After normalization histogram
        ax2.hist(after_valid, bins=50, alpha=0.7, color=colors.secondary, edgecolor="black")
        ax2.set_xlabel("Z-Score")
        ax2.set_ylabel("Frequency")

        missing_text = " (with missing values)" if result.has_missing else ""
        ax2.set_title(
            f"After Z-Score{missing_text}\nMean: {result.mean:.6f}, Std: {result.std:.6f}"
        )
        apply_color_style(ax2, "normalization", colors)

        # Add reference lines for expected z-score distribution
        ax2.axvline(
            x=0, color=colors.error, linestyle="--", alpha=0.7, linewidth=2, label="Expected Mean"
        )
        ax2.axvline(
            x=1, color=colors.success, linestyle=":", alpha=0.7, linewidth=1.5, label="+1 Std"
        )
        ax2.axvline(
            x=-1, color=colors.success, linestyle=":", alpha=0.7, linewidth=1.5, label="-1 Std"
        )
        ax2.legend(fontsize="small")

        apply_typography_theme(fig, typo_config)

        output_path = self._figures_dir / "z_score_verification.png"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_comparison(
        self,
        scptensor_result: ZScoreVerificationResult,
        scanpy_result: ZScoreVerificationResult,
    ) -> Path:
        """Generate a scatter plot comparing ScpTensor vs Scanpy z-score results.

        Creates a scatter plot with identity line to visualize agreement
        between ScpTensor and Scanpy z-score normalization implementations.

        Parameters
        ----------
        scptensor_result : ZScoreVerificationResult
            ScpTensor z-score verification result.
        scanpy_result : ZScoreVerificationResult
            Scanpy z-score verification result.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'z_score_comparison.png' in the
        normalization figures subdirectory.
        """
        import matplotlib.pyplot as plt

        colors, typo_config, layout_config = setup_plot_style()

        scptensor_flat = scptensor_result.after_data.ravel()
        scanpy_flat = scanpy_result.after_data.ravel()

        # Remove NaN values
        valid_mask = ~(np.isnan(scptensor_flat) | np.isnan(scanpy_flat))
        scptensor_valid = scptensor_flat[valid_mask]
        scanpy_valid = scanpy_flat[valid_mask]

        # Compute correlation
        from scipy.stats import pearsonr

        corr, _ = pearsonr(scptensor_valid, scanpy_valid)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(
            scanpy_valid, scptensor_valid, alpha=0.3, s=10, c=colors.primary, edgecolors="none"
        )

        # Identity line
        min_val = min(np.min(scptensor_valid), np.min(scanpy_valid))
        max_val = max(np.max(scptensor_valid), np.max(scanpy_valid))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color=colors.error,
            linestyle="--",
            linewidth=2,
            label="Identity Line",
        )

        ax.set_xlabel("Scanpy Z-Score")
        ax.set_ylabel("ScpTensor Z-Score")
        ax.set_title(f"Z-Score Normalization Comparison\nCorrelation: r = {corr:.6f}")
        ax.legend()
        apply_color_style(ax, "normalization", colors)
        ax.set_aspect("equal", adjustable="box")

        apply_typography_theme(fig, typo_config)

        output_path = self._figures_dir / "z_score_comparison.png"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path
