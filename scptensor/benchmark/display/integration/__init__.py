"""Integration display module for ScpTensor benchmark.

Provides visualization capabilities for comparing batch correction/integration methods
between ScpTensor and competing frameworks (e.g., Scanpy). This module generates
publication-quality figures for integration method comparisons including ComBat,
Harmony, MNN, and Scanorama.

Classes
-------
IntegrationDisplay
    Visualizes integration method comparison results between ScpTensor and Scanpy.
IntegrationComparisonResult
    Dataclass for storing integration comparison data.
IntegrationMetricsSummary
    Dataclass for storing integration metrics summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scptensor.benchmark.display.base import DisplayBase
from scptensor.benchmark.display.common import (
    DEFAULT_TYPOGRAPHY,
    ColorPalette,
    TypographyConfig,
    apply_color_style,
    apply_layout_config,
    apply_typography_theme,
    get_module_colors,
)


def setup_plot_style(dpi: int = 300) -> None:
    """Configure matplotlib with SciencePlots style for publication-quality figures.

    Parameters
    ----------
    dpi : int, default=300
        Resolution in dots per inch for saved figures.

    Notes
    -----
    Applies the SciencePlots 'science' style with 'no-latex' option
    for clean publication-ready figures. Falls back to seaborn-v0_8-whitegrid
    if SciencePlots is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import scienceplots

        plt.style.use(["science", "no-latex"])
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.unicode_minus"] = False


__all__ = [
    "IntegrationDisplay",
    "IntegrationComparisonResult",
    "IntegrationMetricsSummary",
    "setup_plot_style",
]


@dataclass(frozen=True)
class IntegrationComparisonResult:
    """Result of integration method comparison between frameworks.

    Attributes
    ----------
    method_name : str
        Name of the integration method (e.g., "combat", "harmony", "mnn", "scanorama").
    framework : str
        Name of the framework used (e.g., "scptensor", "scanpy").
    batch_labels : npt.NDArray[np.int_]
        Batch assignment for each sample, shape (n_samples,).
    umap_before : npt.NDArray[np.float64]
        UMAP coordinates before batch correction, shape (n_samples, 2).
    umap_after : npt.NDArray[np.float64]
        UMAP coordinates after batch correction, shape (n_samples, 2).
    kbet_score : float | None
        kBET score measuring batch mixing (higher is better), if computed.
    ilisi_score : float | None
        iLISI score measuring local batch mixing (higher is better), if computed.
    clisi_score : float | None
        cLISI score measuring biological cluster preservation (higher is better), if computed.
    asw_score : float | None
        Average silhouette width for batch separation (lower is better after correction), if computed.
    runtime_seconds : float | None
        Runtime of the integration method in seconds, if measured.
    n_batches : int | None
        Number of batches in the dataset, if available.
    n_clusters : int | None
        Number of biological clusters, if available.
    """

    method_name: str
    framework: str
    batch_labels: npt.NDArray[np.int_]
    umap_before: npt.NDArray[np.float64]
    umap_after: npt.NDArray[np.float64]
    kbet_score: float | None = None
    ilisi_score: float | None = None
    clisi_score: float | None = None
    asw_score: float | None = None
    runtime_seconds: float | None = None
    n_batches: int | None = None
    n_clusters: int | None = None


@dataclass(frozen=True)
class IntegrationMetricsSummary:
    """Summary of integration metrics across multiple methods.

    Attributes
    ----------
    methods : list[str]
        Names of the integration methods compared.
    frameworks : list[str]
        Frameworks for each method (e.g., ["scptensor", "scanpy"]).
    kbet_scores : npt.NDArray[np.float64]
        kBET scores for each method, shape (n_methods,).
    ilisi_scores : npt.NDArray[np.float64]
        iLISI scores for each method, shape (n_methods,).
    clisi_scores : npt.NDArray[np.float64]
        cLISI scores for each method, shape (n_methods,).
    asw_scores : npt.NDArray[np.float64]
        ASW scores for each method, shape (n_methods,).
    runtimes : npt.NDArray[np.float64]
        Runtime in seconds for each method, shape (n_methods,).
    """

    methods: list[str]
    frameworks: list[str]
    kbet_scores: npt.NDArray[np.float64]
    ilisi_scores: npt.NDArray[np.float64]
    clisi_scores: npt.NDArray[np.float64]
    asw_scores: npt.NDArray[np.float64]
    runtimes: npt.NDArray[np.float64]


class IntegrationDisplay(DisplayBase):
    """Display class for batch correction/integration comparison visualizations.

    Generates publication-quality figures comparing integration methods between
    ScpTensor and competing frameworks, including UMAP visualizations, method
    comparison grids, biological metrics radar charts, and conservatism analysis.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.integration import IntegrationDisplay
    >>> display = IntegrationDisplay()
    >>> batch_labels = np.array([0]*50 + [1]*50)
    >>> umap_before = np.random.randn(100, 2)
    >>> umap_after = np.random.randn(100, 2)
    >>> result = IntegrationComparisonResult(
    ...     method_name="combat",
    ...     framework="scptensor",
    ...     batch_labels=batch_labels,
    ...     umap_before=umap_before,
    ...     umap_after=umap_after,
    ...     kbet_score=0.85,
    ...     ilisi_score=0.72,
    ...     clisi_score=0.88,
    ...     asw_score=0.15,
    ...     runtime_seconds=2.5
    ... )
    >>> fig_path = display.render_batch_effect_removal(result)
    """

    SHARED_METHODS: tuple[str, ...] = ("combat", "harmony", "mnn", "scanorama")

    METHOD_DISPLAY_NAMES: dict[str, str] = {
        "combat": "ComBat",
        "harmony": "Harmony",
        "mnn": "MNN",
        "scanorama": "Scanorama",
    }

    FRAMEWORK_DISPLAY_NAMES: dict[str, str] = {
        "scptensor": "ScpTensor",
        "scanpy": "Scanpy",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "integration"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("integration")

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_batch_effect_removal, render_method_comparison,
        render_biological_metrics_radar, or render_conservatism_analysis instead.
        """
        from matplotlib.pyplot import close, figure, savefig, subplots

        output_path = self._figures_dir / "integration_summary.png"
        fig, ax = subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_batch_effect_removal, render_method_comparison,\n"
            "render_biological_metrics_radar, or render_conservatism_analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        apply_typography_theme(fig)
        apply_color_style(ax, "integration")
        savefig(output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False)
        close(fig)
        return output_path

    def render_batch_effect_removal(
        self, result: IntegrationComparisonResult, format: str = "png"
    ) -> Path:
        """Generate before/after UMAP plots colored by batch.

        Parameters
        ----------
        result : IntegrationComparisonResult
            Integration result containing UMAP coordinates and batch labels.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.
        """
        from matplotlib.pyplot import close, savefig, subplots

        colors = self._colors
        unique_batches = np.unique(result.batch_labels)
        batch_colors = [
            colors.primary if i % 2 == 0 else colors.secondary for i in range(len(unique_batches))
        ]

        fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 5))

        for i, batch_id in enumerate(unique_batches):
            mask = result.batch_labels == batch_id
            ax1.scatter(
                result.umap_before[mask, 0],
                result.umap_before[mask, 1],
                c=batch_colors[i],
                label=f"Batch {batch_id}",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
        ax1.set_title("Before Batch Correction")
        ax1.legend(fontsize=DEFAULT_TYPOGRAPHY.legend_size, loc="best")

        for i, batch_id in enumerate(unique_batches):
            mask = result.batch_labels == batch_id
            ax2.scatter(
                result.umap_after[mask, 0],
                result.umap_after[mask, 1],
                c=batch_colors[i],
                label=f"Batch {batch_id}",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        method_display = self.METHOD_DISPLAY_NAMES.get(
            result.method_name, result.method_name.title()
        )
        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.set_title(f"After {method_display}")
        ax2.legend(fontsize=DEFAULT_TYPOGRAPHY.legend_size, loc="best")

        for ax in (ax1, ax2):
            apply_color_style(ax, "integration")

        fig.suptitle(
            f"Batch Effect Removal: {method_display} ({framework_name})",
            fontsize=DEFAULT_TYPOGRAPHY.title_size,
            fontweight="bold",
        )

        apply_typography_theme(fig)
        apply_layout_config(fig)
        savefig(
            output_path := self._figures_dir
            / f"{result.method_name}_batch_effect_removal.{format}",
            dpi=300.0,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        close(fig)

        return output_path

    def render_method_comparison(
        self,
        results: list[IntegrationComparisonResult],
        methods: tuple[str, ...] = SHARED_METHODS,
        format: str = "png",
    ) -> Path:
        """Generate a 4x2 grid comparing integration methods across frameworks.

        Parameters
        ----------
        results : list[IntegrationComparisonResult]
            List of integration results for different method-framework combinations.
        methods : tuple[str, ...], default=SHARED_METHODS
            Methods to include in the comparison grid.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.
        """
        from matplotlib.pyplot import close, savefig, subplots

        colors = self._colors

        results_dict: dict[str, dict[str, IntegrationComparisonResult]] = {
            method: {} for method in methods
        }

        for result in results:
            if result.method_name in methods:
                results_dict[result.method_name][result.framework] = result

        fig, axes = subplots(len(methods), 2, figsize=(10, 3.5 * len(methods)))

        if len(methods) == 1:
            axes = axes.reshape(1, -1)

        for row_idx, method in enumerate(methods):
            for col_idx, framework in enumerate(["scptensor", "scanpy"]):
                ax = axes[row_idx, col_idx]

                if method not in results_dict or framework not in results_dict[method]:
                    ax.text(
                        0.5,
                        0.5,
                        f"{method.upper()}\n{framework.title()}\nNo data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=DEFAULT_TYPOGRAPHY.annotation_size,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                result = results_dict[method][framework]
                unique_batches = np.unique(result.batch_labels)

                for i, batch_id in enumerate(unique_batches):
                    mask = result.batch_labels == batch_id
                    batch_color = colors.primary if i % 2 == 0 else colors.secondary
                    ax.scatter(
                        result.umap_after[mask, 0],
                        result.umap_after[mask, 1],
                        c=batch_color,
                        label=f"B{batch_id}",
                        alpha=0.6,
                        s=15,
                        edgecolors="none",
                    )

                framework_display = self.FRAMEWORK_DISPLAY_NAMES.get(framework, framework.title())
                method_display = self.METHOD_DISPLAY_NAMES.get(method, method.title())

                metrics_text = []
                if result.kbet_score is not None:
                    metrics_text.append(f"kBET: {result.kbet_score:.3f}")
                if result.ilisi_score is not None:
                    metrics_text.append(f"iLISI: {result.ilisi_score:.3f}")

                if metrics_text:
                    ax.text(
                        0.02,
                        0.98,
                        "\n".join(metrics_text),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                        fontsize=DEFAULT_TYPOGRAPHY.annotation_size,
                    )

                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_title(f"{method_display} ({framework_display})", fontweight="bold")
                ax.legend(fontsize=DEFAULT_TYPOGRAPHY.legend_size - 1, loc="upper right")
                apply_color_style(ax, "integration")

        apply_typography_theme(fig)
        apply_layout_config(fig)
        savefig(
            output_path := self._figures_dir / f"method_comparison_grid.{format}",
            dpi=300.0,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        close(fig)

        return output_path

    def render_biological_metrics_radar(
        self,
        summary: IntegrationMetricsSummary,
        format: str = "png",
    ) -> Path:
        """Generate a radar chart comparing biological integration metrics.

        Parameters
        ----------
        summary : IntegrationMetricsSummary
            Summary of integration metrics across methods.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.
        """
        from matplotlib.pyplot import close, savefig, subplots

        colors = self._colors
        n_methods = len(summary.methods)
        labels = ["kBET", "iLISI", "cLISI", "ASW*"]
        n_metrics = len(labels)

        kbet_norm = np.clip(summary.kbet_scores, 0, 1)
        ilisi_norm = np.clip(summary.ilisi_scores, 0, 1)
        clisi_norm = np.clip(summary.clisi_scores, 0, 1)
        asw_norm = 1.0 - np.clip(summary.asw_scores, 0, 1)

        scores = np.vstack([kbet_norm, ilisi_norm, clisi_norm, asw_norm]).T

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        method_colors = [
            colors.primary if i % 2 == 0 else colors.secondary for i in range(n_methods)
        ]

        for i, method in enumerate(summary.methods):
            method_scores = scores[i].tolist()
            method_scores += method_scores[:1]

            framework = summary.frameworks[i] if i < len(summary.frameworks) else ""
            framework_display = self.FRAMEWORK_DISPLAY_NAMES.get(framework, framework.title())
            method_display = self.METHOD_DISPLAY_NAMES.get(method, method.title())

            ax.plot(
                angles,
                method_scores,
                "o-",
                linewidth=2,
                label=f"{method_display} ({framework_display})",
                color=method_colors[i],
            )
            ax.fill(angles, method_scores, alpha=0.15, color=method_colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(
            bbox_to_anchor=(1.3, 1.0),
            loc="upper left",
            fontsize=DEFAULT_TYPOGRAPHY.legend_size - 1,
        )

        ax.set_title(
            "Integration Method Comparison: Biological Metrics\n(ASW* = 1 - ASW, higher is better)",
            fontsize=DEFAULT_TYPOGRAPHY.title_size,
            fontweight="bold",
            pad=20,
        )

        apply_typography_theme(fig)
        apply_layout_config(fig)
        savefig(
            output_path := self._figures_dir / f"biological_metrics_radar.{format}",
            dpi=300.0,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        close(fig)

        return output_path

    def render_conservatism_analysis(
        self,
        results: list[IntegrationComparisonResult],
        baseline_umap: npt.NDArray[np.float64] | None = None,
        format: str = "png",
    ) -> Path:
        """Generate an analysis evaluating whether over-correction occurred.

        Parameters
        ----------
        results : list[IntegrationComparisonResult]
            List of integration results to analyze for over-correction.
        baseline_umap : npt.NDArray[np.float64] | None, default=None
            UMAP coordinates of uncorrected data for reference.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.
        """
        from matplotlib.patches import Patch
        from matplotlib.pyplot import close, savefig, subplots

        colors = self._colors

        methods = []
        clisi_scores = []
        ilisi_scores = []
        kbet_scores = []
        runtimes = []

        for result in results:
            method_display = self.METHOD_DISPLAY_NAMES.get(
                result.method_name, result.method_name.title()
            )
            framework_display = self.FRAMEWORK_DISPLAY_NAMES.get(
                result.framework, result.framework.title()
            )
            label = f"{method_display}\n({framework_display})"

            methods.append(label)
            clisi_scores.append(result.clisi_score if result.clisi_score is not None else 0)
            ilisi_scores.append(result.ilisi_score if result.ilisi_score is not None else 0)
            kbet_scores.append(result.kbet_score if result.kbet_score is not None else 0)
            runtimes.append(result.runtime_seconds if result.runtime_seconds is not None else 0)

        fig, (ax1, ax2) = subplots(1, 2, figsize=(14, 6))

        clisi_threshold = 0.5
        kbet_threshold = 0.7

        scatter_colors = []
        for clisi, kbet in zip(clisi_scores, kbet_scores, strict=True):
            if clisi < clisi_threshold:
                scatter_colors.append(colors.error)
            elif kbet < kbet_threshold:
                scatter_colors.append(colors.warning)
            else:
                scatter_colors.append(colors.success)

        ax1.scatter(
            kbet_scores,
            clisi_scores,
            c=scatter_colors,
            s=150,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )

        for i, method in enumerate(methods):
            ax1.annotate(
                method,
                xy=(kbet_scores[i], clisi_scores[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=DEFAULT_TYPOGRAPHY.annotation_size,
                alpha=0.8,
            )

        ax1.axhline(y=clisi_threshold, color=colors.error, linestyle="--", alpha=0.5, linewidth=2)
        ax1.axvline(x=kbet_threshold, color=colors.warning, linestyle="--", alpha=0.5, linewidth=2)

        ax1.axhspan(0, clisi_threshold, color=colors.error, alpha=0.1)
        ax1.text(
            0.02,
            clisi_threshold / 2,
            "Over-correction\n(Low cLISI)",
            fontsize=DEFAULT_TYPOGRAPHY.annotation_size,
            color=colors.error,
            style="italic",
        )

        ax1.set_xlabel("kBET Score (Batch Mixing)")
        ax1.set_ylabel("cLISI Score (Biological Preservation)")
        ax1.set_title("Conservatism Analysis: Batch Mixing vs Biological Signal")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        apply_color_style(ax1, "integration")

        x_pos = np.arange(len(methods))
        bars = ax2.bar(
            x_pos, runtimes, width=0.6, color=scatter_colors, edgecolor="black", alpha=0.8
        )

        for bar, runtime in zip(bars, runtimes, strict=True):
            if runtime > 0:
                ax2.annotate(
                    f"{runtime:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=DEFAULT_TYPOGRAPHY.annotation_size,
                )

        ax2.set_xlabel("Integration Method")
        ax2.set_ylabel("Runtime (seconds)")
        ax2.set_title("Integration Method Runtime")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods, fontsize=DEFAULT_TYPOGRAPHY.tick_size)
        ax2.grid(True, alpha=0.3, axis="y")

        apply_color_style(ax2, "integration")

        legend_elements = [
            Patch(facecolor=colors.success, edgecolor="black", label="Good Balance"),
            Patch(facecolor=colors.warning, edgecolor="black", label="Under-correction"),
            Patch(facecolor=colors.error, edgecolor="black", label="Over-correction Risk"),
        ]
        ax2.legend(
            handles=legend_elements, loc="upper right", fontsize=DEFAULT_TYPOGRAPHY.legend_size - 1
        )

        apply_typography_theme(fig)
        apply_layout_config(fig)
        savefig(
            output_path := self._figures_dir / f"conservatism_analysis.{format}",
            dpi=300.0,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        close(fig)

        return output_path
