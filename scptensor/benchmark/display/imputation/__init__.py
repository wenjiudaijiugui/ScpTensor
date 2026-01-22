"""Imputation display module for ScpTensor benchmark.

Provides visualization capabilities for comparing imputation methods between
ScpTensor and competing frameworks (e.g., Scanpy). This module generates
publication-quality figures for KNN imputation (shared method) comparisons
and ScpTensor-exclusive imputation methods.

Classes
-------
KNNImputeDisplay
    Visualizes KNN imputation comparison results between ScpTensor and Scanpy.
ExclusiveImputeDisplay
    Visualizes ScpTensor-exclusive imputation methods (PPCA, SVD, MissForest, etc.).
ImputationComparisonResult
    Dataclass for storing imputation comparison data.
setup_plot_style
    Configures matplotlib with SciencePlots style for publication-quality figures.

Functions
---------
setup_plot_style
    Configures matplotlib with SciencePlots style for publication-quality figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scptensor.benchmark.display.base import DisplayBase
from scptensor.benchmark.display.common import (
    TypographyConfig,
    apply_typography_theme,
    get_compatible_color,
    get_module_colors,
)
from scptensor.benchmark.display.config import PlotStyle, get_style_string

__all__ = [
    "KNNImputeDisplay",
    "ExclusiveImputeDisplay",
    "ImputationComparisonResult",
    "setup_plot_style",
]


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


@dataclass(frozen=True)
class ImputationComparisonResult:
    """Result of imputation method comparison.

    Attributes
    ----------
    method_name : str
        Name of the imputation method (e.g., "knn", "ppca", "svd").
    missing_rate : float
        Proportion of missing values in the test data (0.0 to 1.0).
    ground_truth : npt.NDArray[np.float64]
        Original complete data before masking.
    scptensor_imputed : npt.NDArray[np.float64]
        Data imputed using ScpTensor.
    competitor_imputed : npt.NDArray[np.float64] | None
        Data imputed using competitor framework, if available.
    framework : str
        Name of the competitor framework (e.g., "scanpy", "scikit-learn").
    mse : float | None
        Mean squared error of ScpTensor imputation, if computed.
    mae : float | None
        Mean absolute error of ScpTensor imputation, if computed.
    correlation : float | None
        Pearson correlation coefficient of ScpTensor imputation, if computed.
    runtime_seconds : float | None
        Runtime of the imputation in seconds, if measured.
    competitor_mse : float | None
        MSE of competitor imputation, if available.
    competitor_mae : float | None
        MAE of competitor imputation, if available.
    competitor_correlation : float | None
        Correlation of competitor imputation, if available.
    competitor_runtime_seconds : float | None
        Runtime of competitor imputation in seconds, if measured.
    """

    method_name: str
    missing_rate: float
    ground_truth: npt.NDArray[np.float64]
    scptensor_imputed: npt.NDArray[np.float64]
    framework: str = "scanpy"
    competitor_imputed: npt.NDArray[np.float64] | None = None
    mse: float | None = None
    mae: float | None = None
    correlation: float | None = None
    runtime_seconds: float | None = None
    competitor_mse: float | None = None
    competitor_mae: float | None = None
    competitor_correlation: float | None = None
    competitor_runtime_seconds: float | None = None


@dataclass(frozen=True)
class ExclusiveImputationResults:
    """Aggregated results for ScpTensor-exclusive imputation methods.

    Attributes
    ----------
    methods : list[str]
        Names of the imputation methods compared.
    missing_rates : list[float]
        Missing rate values tested (0.0 to 1.0).
    mse_matrix : npt.NDArray[np.float64]
        MSE values matrix of shape (n_methods, n_missing_rates).
    mae_matrix : npt.NDArray[np.float64]
        MAE values matrix of shape (n_methods, n_missing_rates).
    correlation_matrix : npt.NDArray[np.float64]
        Correlation values matrix of shape (n_methods, n_missing_rates).
    runtime_matrix : npt.NDArray[np.float64]
        Runtime values matrix in seconds of shape (n_methods, n_missing_rates).
    baseline_method : str
        Name of the baseline method for comparison (e.g., "knn").
    baseline_mse : list[float]
        MSE values for the baseline method across missing rates.
    """

    methods: list[str]
    missing_rates: list[float]
    mse_matrix: npt.NDArray[np.float64]
    mae_matrix: npt.NDArray[np.float64]
    correlation_matrix: npt.NDArray[np.float64]
    runtime_matrix: npt.NDArray[np.float64]
    baseline_method: str = "knn"
    baseline_mse: list[float] | None = None


class KNNImputeDisplay(DisplayBase):
    """Display class for KNN imputation comparison visualizations.

    KNN (k-Nearest Neighbors) imputation is available in both ScpTensor and
    competing frameworks, making it ideal for direct comparison. This class
    generates publication-quality figures comparing accuracy and performance.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.imputation import KNNImputeDisplay
    >>> display = KNNImputeDisplay()
    >>> ground_truth = np.random.randn(100, 50)
    >>> scptensor_imputed = ground_truth + np.random.randn(100, 50) * 0.1
    >>> scanpy_imputed = ground_truth + np.random.randn(100, 50) * 0.12
    >>> result = ImputationComparisonResult(
    ...     method_name="knn",
    ...     missing_rate=0.2,
    ...     ground_truth=ground_truth,
    ...     scptensor_imputed=scptensor_imputed,
    ...     competitor_imputed=scanpy_imputed,
    ...     framework="scanpy",
    ...     mse=0.01,
    ...     mae=0.05,
    ...     correlation=0.99,
    ...     runtime_seconds=1.5,
    ...     competitor_mse=0.015,
    ...     competitor_mae=0.06,
    ...     competitor_correlation=0.98,
    ...     competitor_runtime_seconds=2.0
    ... )
    >>> fig_path = display.render_accuracy_table([result])
    """

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "imputation"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._tables_dir = self.output_dir / "tables" / "imputation"
        self._tables_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("imputation")
        self._typography = TypographyConfig()

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_accuracy_table or render_performance_comparison instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "knn_impute_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_accuracy_table or render_performance_comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_accuracy_table(
        self, results: list[ImputationComparisonResult], format: str = "png"
    ) -> Path:
        """Generate a table comparing accuracy metrics across missing rates.

        Creates a formatted table showing MSE, MAE, and correlation metrics
        for both ScpTensor and the competitor framework across different
        missing rate conditions.

        Parameters
        ----------
        results : list[ImputationComparisonResult]
            List of imputation comparison results, typically across
            different missing rates.
        format : str, default="png"
            Output format for the table. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The table is saved as 'knn_accuracy_table.{format}' in the
        imputation figures subdirectory.

        The table includes:
        - Missing rate column
        - ScpTensor metrics (MSE, MAE, Correlation)
        - Competitor metrics (MSE, MAE, Correlation)
        - Improvement indicators
        """
        import matplotlib.pyplot as plt

        setup_plot_style()

        sorted_results = sorted(results, key=lambda r: r.missing_rate)

        row_labels = [f"{r.missing_rate:.1%}" for r in sorted_results]
        n_rows = len(sorted_results)

        table_data = []
        for r in sorted_results:
            row = [
                f"{r.missing_rate:.1%}",
                f"{r.mse:.4f}" if r.mse is not None else "N/A",
                f"{r.mae:.4f}" if r.mae is not None else "N/A",
                f"{r.correlation:.4f}" if r.correlation is not None else "N/A",
                f"{r.competitor_mse:.4f}" if r.competitor_mse is not None else "N/A",
                f"{r.competitor_mae:.4f}" if r.competitor_mae is not None else "N/A",
                f"{r.competitor_correlation:.4f}"
                if r.competitor_correlation is not None
                else "N/A",
            ]
            table_data.append(row)

        col_labels = [
            "Missing\nRate",
            "ScpTensor\nMSE",
            "ScpTensor\nMAE",
            "ScpTensor\nCorr",
            f"{r.framework.capitalize()}\nMSE" if sorted_results else "Competitor\nMSE",
            f"{sorted_results[0].framework.capitalize()}\nMAE"
            if sorted_results
            else "Competitor\nMAE",
            f"{sorted_results[0].framework.capitalize()}\nCorr"
            if sorted_results
            else "Competitor\nCorr",
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, n_rows * 0.5)))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor(self._colors.primary)
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        for i in range(1, n_rows + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#F2F2F2")

        for i, r in enumerate(sorted_results, start=1):
            if r.mse is not None and r.competitor_mse is not None:
                if r.mse < r.competitor_mse:
                    table[(i, 1)].set_facecolor(self._colors.success)
                else:
                    table[(i, 4)].set_facecolor(self._colors.error)

            if r.mae is not None and r.competitor_mae is not None:
                if r.mae < r.competitor_mae:
                    table[(i, 2)].set_facecolor(self._colors.success)
                else:
                    table[(i, 5)].set_facecolor(self._colors.error)

            if r.correlation is not None and r.competitor_correlation is not None:
                if r.correlation > r.competitor_correlation:
                    table[(i, 3)].set_facecolor(self._colors.success)
                else:
                    table[(i, 6)].set_facecolor(self._colors.error)

        plt.title(
            f"KNN Imputation Accuracy Comparison: ScpTensor vs {sorted_results[0].framework.capitalize() if sorted_results else 'Competitor'}",
            fontsize=self._typography.title_size,
            fontweight="bold",
            pad=20,
        )

        output_path = self._figures_dir / f"knn_accuracy_table.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_performance_comparison(
        self, results: list[ImputationComparisonResult], format: str = "png"
    ) -> Path:
        """Generate a bar chart comparing runtime performance.

        Creates a grouped bar chart showing runtime comparison between
        ScpTensor and the competitor framework across different missing rates.

        Parameters
        ----------
        results : list[ImputationComparisonResult]
            List of imputation comparison results with runtime data.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'knn_performance_comparison.{format}' in the
        imputation figures subdirectory.
        """
        import matplotlib.pyplot as plt

        setup_plot_style()

        sorted_results = sorted(results, key=lambda r: r.missing_rate)
        n_results = len(sorted_results)

        missing_rates = [r.missing_rate for r in sorted_results]
        scptensor_runtimes = [
            r.runtime_seconds if r.runtime_seconds is not None else 0 for r in sorted_results
        ]
        competitor_runtimes = [
            r.competitor_runtime_seconds if r.competitor_runtime_seconds is not None else 0
            for r in sorted_results
        ]

        x = np.arange(n_results)
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(
            x - width / 2,
            scptensor_runtimes,
            width,
            label="ScpTensor",
            color=self._colors.primary,
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            competitor_runtimes,
            width,
            label=sorted_results[0].framework.capitalize() if sorted_results else "Competitor",
            color=self._colors.secondary,
            edgecolor="black",
        )

        ax.set_xlabel("Missing Rate", fontsize=self._typography.label_size)
        ax.set_ylabel("Runtime (seconds)", fontsize=self._typography.label_size)
        ax.set_title(
            "KNN Imputation Runtime Comparison",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{mr:.1%}" for mr in missing_rates])
        ax.legend(fontsize=self._typography.legend_size)
        ax.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=self._typography.annotation_size,
                    )

        apply_typography_theme(fig, self._typography)
        plt.tight_layout()

        output_path = self._figures_dir / f"knn_performance_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path


class ExclusiveImputeDisplay(DisplayBase):
    """Display class for ScpTensor-exclusive imputation method visualizations.

    This class generates visualizations for imputation methods that are
    exclusive to ScpTensor, such as PPCA, SVD, MissForest, BPCA, LLS,
    NMF, and MNAR methods (MinProb, MinDet, QRILC).

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.imputation import ExclusiveImputeDisplay
    >>> display = ExclusiveImputeDisplay()
    >>> methods = ["ppca", "svd", "bpca", "mf"]
    >>> missing_rates = [0.1, 0.2, 0.3, 0.4]
    >>> mse_matrix = np.random.rand(4, 4) * 0.1
    >>> mae_matrix = np.random.rand(4, 4) * 0.2
    >>> corr_matrix = 1 - np.random.rand(4, 4) * 0.1
    >>> runtime_matrix = np.random.rand(4, 4) * 10 + 1
    >>> results = ExclusiveImputationResults(
    ...     methods=methods,
    ...     missing_rates=missing_rates,
    ...     mse_matrix=mse_matrix,
    ...     mae_matrix=mae_matrix,
    ...     correlation_matrix=corr_matrix,
    ...     runtime_matrix=runtime_matrix,
    ...     baseline_method="knn",
    ...     baseline_mse=[0.02, 0.03, 0.04, 0.05]
    ... )
    >>> fig_path = display.render_mse_heatmap(results)
    """

    EXCLUSIVE_METHODS: tuple[str, ...] = (
        "ppca",
        "svd",
        "bpca",
        "mf",
        "lls",
        "nmf",
        "minprob",
        "mindet",
        "qrilc",
    )

    METHOD_DISPLAY_NAMES: dict[str, str] = {
        "ppca": "PPCA",
        "svd": "SVD",
        "bpca": "BPCA",
        "mf": "MissForest",
        "lls": "LLS",
        "nmf": "NMF",
        "minprob": "MinProb",
        "mindet": "MinDet",
        "qrilc": "QRILC",
        "knn": "KNN",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "imputation"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("imputation")
        self._typography = TypographyConfig()

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_mse_heatmap, render_performance_advantage,
        or render_missing_rate_response instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "exclusive_impute_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_mse_heatmap, render_performance_advantage,\n"
            "or render_missing_rate_response",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_mse_heatmap(
        self, results: ExclusiveImputationResults, metric: str = "mse", format: str = "png"
    ) -> Path:
        """Generate a heatmap showing MSE comparison across methods and missing rates.

        Creates a heatmap visualization where each cell represents the MSE value
        for a specific method at a specific missing rate. Lower values (better)
        are shown in cooler colors.

        Parameters
        ----------
        results : ExclusiveImputationResults
            Aggregated imputation results across methods and missing rates.
        metric : str, default="mse"
            Metric to visualize. Options: "mse", "mae", "correlation".
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If metric is not one of "mse", "mae", or "correlation".

        Notes
        -----
        The figure is saved as 'exclusive_{metric}_heatmap.{format}' in the
        imputation figures subdirectory.
        """
        import matplotlib.pyplot as plt

        setup_plot_style()

        metric_matrices = {
            "mse": results.mse_matrix,
            "mae": results.mae_matrix,
            "correlation": results.correlation_matrix,
        }
        if metric not in metric_matrices:
            raise ValueError(
                f"metric must be one of {list(metric_matrices.keys())}, got '{metric}'"
            )

        data_matrix = metric_matrices[metric]

        method_names = [self.METHOD_DISPLAY_NAMES.get(m, m.upper()) for m in results.methods]
        rate_labels = [f"{mr:.1%}" for mr in results.missing_rates]

        cmap = "RdYlGn_r" if metric in ("mse", "mae") else "RdYlGn"

        fig, ax = plt.subplots(
            figsize=(max(6, len(results.missing_rates) * 0.8), max(4, len(results.methods) * 0.5))
        )

        im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(len(results.missing_rates)))
        ax.set_yticks(np.arange(len(results.methods)))
        ax.set_xticklabels(rate_labels)
        ax.set_yticklabels(method_names)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(results.methods)):
            for j in range(len(results.missing_rates)):
                value = data_matrix[i, j]
                text_color = (
                    "white"
                    if (metric in ("mse", "mae") and value > np.median(data_matrix))
                    or (metric == "correlation" and value < np.median(data_matrix))
                    else "black"
                )
                ax.text(
                    j,
                    i,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=self._typography.annotation_size,
                )

        ax.set_xlabel("Missing Rate", fontsize=self._typography.label_size)
        ax.set_ylabel("Imputation Method", fontsize=self._typography.label_size)

        metric_titles = {
            "mse": "Mean Squared Error (MSE)",
            "mae": "Mean Absolute Error (MAE)",
            "correlation": "Pearson Correlation",
        }
        ax.set_title(
            f"ScpTensor Imputation Methods: {metric_titles[metric]}\nvs Missing Rate",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            metric_titles[metric].split("(")[0].strip(), fontsize=self._typography.label_size
        )

        apply_typography_theme(fig, self._typography)
        plt.tight_layout()

        output_path = self._figures_dir / f"exclusive_{metric}_heatmap.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_performance_advantage(
        self,
        results: ExclusiveImputationResults,
        baseline_mse: list[float] | None = None,
        format: str = "png",
    ) -> Path:
        """Generate a bar chart showing MSE reduction percentage vs KNN baseline.

        Creates a grouped bar chart showing the percentage improvement in MSE
        for each ScpTensor-exclusive method compared to KNN baseline.

        Parameters
        ----------
        results : ExclusiveImputationResults
            Aggregated imputation results across methods and missing rates.
        baseline_mse : list[float] | None, default=None
            MSE values for baseline method across missing rates. If None,
            uses results.baseline_mse.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'exclusive_performance_advantage.{format}' in the
        imputation figures subdirectory.

        MSE reduction is calculated as: (baseline_mse - method_mse) / baseline_mse * 100
        Positive values indicate better performance than baseline.
        """
        import matplotlib.pyplot as plt

        setup_plot_style()

        if baseline_mse is None:
            baseline_mse = results.baseline_mse
        if baseline_mse is None:
            baseline_mse = [float(np.mean(results.mse_matrix))] * len(results.missing_rates)

        mse_reduction = np.zeros_like(results.mse_matrix)
        for i, method_mse in enumerate(results.mse_matrix.T):
            for j, (method_val, base_val) in enumerate(zip(method_mse, baseline_mse, strict=True)):
                if base_val > 0:
                    mse_reduction[j, i] = (base_val - method_val) / base_val * 100

        method_names = [self.METHOD_DISPLAY_NAMES.get(m, m.upper()) for m in results.methods]
        rate_labels = [f"{mr:.1%}" for mr in results.missing_rates]
        n_rates = len(results.missing_rates)
        n_methods = len(results.methods)

        x = np.arange(n_rates)
        bar_width = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, method in enumerate(results.methods):
            offset = (i - n_methods / 2 + 0.5) * bar_width
            color = get_compatible_color("imputation", i % 7)
            bars = ax.bar(
                x + offset,
                mse_reduction[i, :],
                bar_width,
                label=method_names[i],
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

            for bar, val in zip(bars, mse_reduction[i, :], strict=True):
                height = bar.get_height()
                ax.annotate(
                    f"{val:+.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=self._typography.annotation_size,
                )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Missing Rate", fontsize=self._typography.label_size)
        ax.set_ylabel("MSE Reduction vs KNN (%)", fontsize=self._typography.label_size)
        ax.set_title(
            "ScpTensor Imputation Methods: Performance Advantage vs KNN Baseline",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rate_labels)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=self._typography.legend_size)
        ax.grid(True, alpha=0.3, axis="y")

        apply_typography_theme(fig, self._typography)
        plt.tight_layout()

        output_path = self._figures_dir / f"exclusive_performance_advantage.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_missing_rate_response(
        self, results: ExclusiveImputationResults, metric: str = "mse", format: str = "png"
    ) -> Path:
        """Generate a line plot showing how accuracy metrics vary with missing rate.

        Creates a multi-line plot showing the response of different imputation
        methods to increasing missing rates. This helps identify which methods
        are most robust to high missingness.

        Parameters
        ----------
        results : ExclusiveImputationResults
            Aggregated imputation results across methods and missing rates.
        metric : str, default="mse"
            Metric to visualize. Options: "mse", "mae", "correlation".
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If metric is not one of "mse", "mae", or "correlation".

        Notes
        -----
        The figure is saved as 'exclusive_missing_rate_response.{format}' in the
        imputation figures subdirectory.

        Methods that maintain lower MSE/MAE or higher correlation as missing
        rate increases are considered more robust.
        """
        import matplotlib.pyplot as plt

        setup_plot_style()

        metric_matrices = {
            "mse": results.mse_matrix,
            "mae": results.mae_matrix,
            "correlation": results.correlation_matrix,
        }
        if metric not in metric_matrices:
            raise ValueError(
                f"metric must be one of {list(metric_matrices.keys())}, got '{metric}'"
            )

        data_matrix = metric_matrices[metric]

        method_names = [self.METHOD_DISPLAY_NAMES.get(m, m.upper()) for m in results.methods]

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, method in enumerate(results.methods):
            color = get_compatible_color("imputation", i % 7)
            ax.plot(
                results.missing_rates,
                data_matrix[i, :],
                marker="o",
                label=method_names[i],
                color=color,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Missing Rate", fontsize=self._typography.label_size)
        ax.set_ylabel(metric.upper(), fontsize=self._typography.label_size)

        metric_titles = {
            "mse": "Mean Squared Error (MSE)",
            "mae": "Mean Absolute Error (MAE)",
            "correlation": "Pearson Correlation Coefficient",
        }
        ax.set_title(
            f"Imputation Accuracy Response to Missing Rate: {metric_titles[metric]}",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        ax.legend(fontsize=self._typography.legend_size, loc="best")
        ax.grid(True, alpha=0.3)

        ax.set_xticks(results.missing_rates)
        ax.set_xticklabels([f"{mr:.1%}" for mr in results.missing_rates])

        apply_typography_theme(fig, self._typography)
        plt.tight_layout()

        output_path = self._figures_dir / f"exclusive_missing_rate_response_{metric}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path
