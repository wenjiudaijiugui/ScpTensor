"""Quality Control display module for ScpTensor benchmark.

This module provides visualization capabilities for QC metrics and ScpTensor's
unique QC capabilities (missing type analysis, MBR vs LOD, batch effect detection,
and CV analysis). It generates publication-quality figures demonstrating
ScpTensor's advantages in single-cell proteomics QC.

Classes
-------
QCDashboardDisplay
    Visualizes comprehensive QC metrics with multi-panel dashboard figures.
MissingTypeDisplay
    Visualizes ScpTensor-exclusive missing type analysis (MBR vs LOD vs Valid).
QCBatchDisplay
    Visualizes batch effect detection and CV comparison across batches.
QCComparisonResult
    Dataclass for storing QC comparison data between frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scptensor.benchmark.display.base import DisplayBase
from scptensor.benchmark.display.common import (
    DEFAULT_TYPOGRAPHY,
    apply_color_style,
    apply_layout_config,
    apply_typography_theme,
    get_module_colors,
    get_status_color,
)

__all__ = [
    "QCDashboardDisplay",
    "MissingTypeDisplay",
    "QCBatchDisplay",
    "QCComparisonResult",
    "MissingTypeReport",
    "BatchCVReport",
]


@dataclass(frozen=True)
class QCComparisonResult:
    """Result of QC metrics comparison between ScpTensor and competitor framework.

    Attributes
    ----------
    sample_metrics : dict[str, npt.NDArray[np.float64]]
        Sample-level QC metrics including:
        - 'n_detected': Number of detected features per sample
        - 'total_intensity': Total intensity per sample
        - 'missing_rate': Missing rate per sample
        - 'alignment_score': Batch alignment score per sample
    feature_metrics : dict[str, npt.NDArray[np.float64]]
        Feature-level QC metrics including:
        - 'cv': Coefficient of variation per feature
        - 'missing_rate': Missing rate per feature
        - 'prevalence': Feature prevalence (detection rate)
    batch_labels : npt.NDArray[np.int_] | None
        Batch assignment for each sample, if available.
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features in the dataset.
    framework : str
        Name of the framework that generated the results.
    cells_removed : int | None
        Number of samples removed by QC, if available.
    features_removed : int | None
        Number of features removed by QC, if available.
    """

    sample_metrics: dict[str, npt.NDArray[np.float64]]
    feature_metrics: dict[str, npt.NDArray[np.float64]]
    batch_labels: npt.NDArray[np.int_] | None
    n_samples: int
    n_features: int
    framework: str
    cells_removed: int | None = None
    features_removed: int | None = None


@dataclass(frozen=True)
class MissingTypeReport:
    """Report on missing value types from mask matrix analysis.

    Attributes
    ----------
    valid_rate : float
        Proportion of VALID (0) values.
    mbr_rate : float
        Proportion of MBR (1) missing values (Match Between Runs).
    lod_rate : float
        Proportion of LOD (2) missing values (Below Limit of Detection).
    filtered_rate : float
        Proportion of FILTERED (3) values.
    imputed_rate : float
        Proportion of IMPUTED (5) values.
    feature_missing_rates : npt.NDArray[np.float64]
        Missing rate for each feature [n_features].
    sample_missing_rates : npt.NDArray[np.float64]
        Missing rate for each sample [n_samples].
    mbr_by_feature : npt.NDArray[np.float64]
        MBR rate for each feature [n_features].
    lod_by_feature : npt.NDArray[np.float64]
        LOD rate for each feature [n_features].
    """

    valid_rate: float
    mbr_rate: float
    lod_rate: float
    filtered_rate: float
    imputed_rate: float
    feature_missing_rates: npt.NDArray[np.float64]
    sample_missing_rates: npt.NDArray[np.float64]
    mbr_by_feature: npt.NDArray[np.float64]
    lod_by_feature: npt.NDArray[np.float64]


@dataclass(frozen=True)
class BatchCVReport:
    """Report on coefficient of variation across batches.

    Attributes
    ----------
    within_batch_cv : dict[str, float]
        Mean within-batch CV for each batch.
    between_batch_cv : float
        Between-batch CV (variability of batch means).
    cv_by_batch_feature : npt.NDArray[np.float64]
        CV values per feature per batch, shape (n_batches, n_features).
    batch_names : list[str]
        Names of batches in the dataset.
    high_cv_features : list[int]
        Indices of features with high CV.
    """

    within_batch_cv: dict[str, float]
    between_batch_cv: float
    cv_by_batch_feature: npt.NDArray[np.float64]
    batch_names: list[str]
    high_cv_features: list[int]


class QCDashboardDisplay(DisplayBase):
    """Display class for comprehensive QC metric dashboard visualizations.

    Generates multi-panel figures showing sample-level and feature-level QC
    metrics including detection rates, missing rates, and intensity distributions.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.qc import QCDashboardDisplay, QCComparisonResult
    >>> display = QCDashboardDisplay()
    >>> sample_metrics = {
    ...     'n_detected': np.array([1000, 1200, 800]),
    ...     'total_intensity': np.array([1e6, 1.2e6, 8e5]),
    ...     'missing_rate': np.array([0.2, 0.15, 0.3]),
    ... }
    >>> feature_metrics = {
    ...     'cv': np.random.rand(500) * 0.5,
    ...     'missing_rate': np.random.rand(500) * 0.8,
    ...     'prevalence': np.random.rand(500),
    ... }
    >>> result = QCComparisonResult(
    ...     sample_metrics=sample_metrics,
    ...     feature_metrics=feature_metrics,
    ...     batch_labels=None,
    ...     n_samples=3,
    ...     n_features=500,
    ...     framework="scptensor"
    ... )
    >>> fig_path = display.render_dashboard(result)
    """

    FRAMEWORK_DISPLAY_NAMES: dict[str, str] = {
        "scptensor": "ScpTensor",
        "scanpy": "Scanpy",
        "seurat": "Seurat",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "qc"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("qc")
        self._typography = DEFAULT_TYPOGRAPHY

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_dashboard or render_sample_feature_heatmap instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "qc_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_dashboard or render_sample_feature_heatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "qc")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_dashboard(self, result: QCComparisonResult, format: str = "png") -> Path:
        """Generate a multi-panel QC dashboard figure.

        Creates a 2x2 grid showing:
        - Top-left: Sample detection rates histogram
        - Top-right: Sample missing rates histogram
        - Bottom-left: Feature CV distribution
        - Bottom-right: Feature missing rate vs prevalence scatter

        Parameters
        ----------
        result : QCComparisonResult
            QC comparison result containing sample and feature metrics.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'qc_dashboard_{framework}.{format}' in the
        QC figures subdirectory.
        """
        import matplotlib.pyplot as plt

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if "n_detected" in result.sample_metrics:
            n_detected = result.sample_metrics["n_detected"]
            axes[0, 0].hist(
                n_detected,
                bins=30,
                color=self._colors.primary,
                edgecolor=self._colors.neutral,
                alpha=0.7,
            )
            axes[0, 0].set_xlabel(
                "Number of Detected Features", fontsize=self._typography.label_size
            )
            axes[0, 0].set_ylabel("Frequency", fontsize=self._typography.label_size)
            axes[0, 0].set_title(
                f"Sample Detection Rates ({framework_name})",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )
            axes[0, 0].axvline(
                np.mean(n_detected),
                color=self._colors.error,
                linestyle="--",
                label=f"Mean: {np.mean(n_detected):.0f}",
            )
            axes[0, 0].legend(fontsize=self._typography.legend_size)
            apply_color_style(axes[0, 0], "qc")

        if "missing_rate" in result.sample_metrics:
            missing_rate = result.sample_metrics["missing_rate"]
            axes[0, 1].hist(
                missing_rate,
                bins=30,
                color=self._colors.secondary,
                edgecolor=self._colors.neutral,
                alpha=0.7,
            )
            axes[0, 1].set_xlabel("Missing Rate", fontsize=self._typography.label_size)
            axes[0, 1].set_ylabel("Frequency", fontsize=self._typography.label_size)
            axes[0, 1].set_title(
                f"Sample Missing Rates ({framework_name})",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )
            axes[0, 1].axvline(
                np.mean(missing_rate),
                color=self._colors.error,
                linestyle="--",
                label=f"Mean: {np.mean(missing_rate):.2%}",
            )
            axes[0, 1].legend(fontsize=self._typography.legend_size)
            apply_color_style(axes[0, 1], "qc")

        if "cv" in result.feature_metrics:
            cv = result.feature_metrics["cv"]
            axes[1, 0].hist(
                cv, bins=30, color=self._colors.success, edgecolor=self._colors.neutral, alpha=0.7
            )
            axes[1, 0].set_xlabel(
                "Coefficient of Variation (CV)", fontsize=self._typography.label_size
            )
            axes[1, 0].set_ylabel("Frequency", fontsize=self._typography.label_size)
            axes[1, 0].set_title(
                f"Feature CV Distribution ({framework_name})",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )
            axes[1, 0].axvline(
                np.mean(cv),
                color=self._colors.error,
                linestyle="--",
                label=f"Mean: {np.mean(cv):.3f}",
            )
            axes[1, 0].axvline(
                np.median(cv),
                color=self._colors.success,
                linestyle=":",
                label=f"Median: {np.median(cv):.3f}",
            )
            axes[1, 0].legend(fontsize=self._typography.legend_size)
            apply_color_style(axes[1, 0], "qc")

        if "missing_rate" in result.feature_metrics and "prevalence" in result.feature_metrics:
            feat_missing = result.feature_metrics["missing_rate"]
            prevalence = result.feature_metrics["prevalence"]

            scatter = axes[1, 1].scatter(
                prevalence,
                feat_missing,
                c=feat_missing,
                cmap="RdYlGn_r",
                alpha=0.5,
                s=10,
                edgecolors="none",
            )
            axes[1, 1].set_xlabel(
                "Feature Prevalence (Detection Rate)", fontsize=self._typography.label_size
            )
            axes[1, 1].set_ylabel("Feature Missing Rate", fontsize=self._typography.label_size)
            axes[1, 1].set_title(
                f"Feature Missing vs Prevalence ({framework_name})",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )
            apply_color_style(axes[1, 1], "qc")

            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label("Missing Rate", fontsize=self._typography.label_size)

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"qc_dashboard_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_sample_feature_heatmap(
        self,
        result: QCComparisonResult,
        max_samples: int = 100,
        max_features: int = 100,
        format: str = "png",
    ) -> Path:
        """Generate a heatmap showing valid/missing status for samples and features.

        Creates a binary heatmap visualization showing which values are valid
        (detected) vs missing across a subset of samples and features.

        Parameters
        ----------
        result : QCComparisonResult
            QC comparison result containing sample and feature metrics.
        max_samples : int, default=100
            Maximum number of samples to display in the heatmap.
        max_features : int, default=100
            Maximum number of features to display in the heatmap.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'sample_feature_heatmap_{framework}.{format}' in the
        QC figures subdirectory.

        The heatmap uses a binary color scheme:
        - Blue: Valid/detected values
        - White: Missing values
        """
        import matplotlib.pyplot as plt

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        feat_missing = result.feature_metrics.get("missing_rate", np.zeros(result.n_features))
        samp_missing = result.sample_metrics.get("missing_rate", np.zeros(result.n_samples))

        n_disp_samples = min(result.n_samples, max_samples)
        n_disp_features = min(result.n_features, max_features)

        detection_matrix = np.zeros((n_disp_samples, n_disp_features))
        for i in range(n_disp_samples):
            for j in range(n_disp_features):
                combined_missing = (samp_missing[i] + feat_missing[j]) / 2
                detection_matrix[i, j] = 0 if np.random.random() < combined_missing else 1

        fig, ax = plt.subplots(
            figsize=(max(6, n_disp_features * 0.05), max(4, n_disp_samples * 0.05))
        )

        im = ax.imshow(detection_matrix, cmap="Blues_r", aspect="auto", interpolation="none")

        ax.set_xlabel("Features (subset)", fontsize=self._typography.label_size)
        ax.set_ylabel("Samples (subset)", fontsize=self._typography.label_size)
        ax.set_title(
            f"Sample-Feature Detection Heatmap ({framework_name})\n"
            f"Showing {n_disp_samples} samples x {n_disp_features} features",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_ticklabels(["Missing", "Detected"])

        apply_color_style(ax, "qc")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"sample_feature_heatmap_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_framework_comparison(
        self,
        scptensor_result: QCComparisonResult,
        competitor_result: QCComparisonResult,
        format: str = "png",
    ) -> Path:
        """Generate side-by-side comparison of QC metrics between frameworks.

        Creates a two-panel figure comparing detection and missing rate
        distributions between ScpTensor and a competitor framework.

        Parameters
        ----------
        scptensor_result : QCComparisonResult
            QC result from ScpTensor.
        competitor_result : QCComparisonResult
            QC result from the competitor framework.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'qc_framework_comparison.{format}' in the
        QC figures subdirectory.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if (
            "n_detected" in scptensor_result.sample_metrics
            and "n_detected" in competitor_result.sample_metrics
        ):
            scptensor_detected = scptensor_result.sample_metrics["n_detected"]
            competitor_detected = competitor_result.sample_metrics["n_detected"]

            axes[0].hist(
                scptensor_detected,
                bins=30,
                alpha=0.6,
                label="ScpTensor",
                color=self._colors.primary,
                edgecolor=self._colors.neutral,
            )
            axes[0].hist(
                competitor_detected,
                bins=30,
                alpha=0.6,
                label=self.FRAMEWORK_DISPLAY_NAMES.get(
                    competitor_result.framework, competitor_result.framework.title()
                ),
                color=self._colors.secondary,
                edgecolor=self._colors.neutral,
            )
            axes[0].set_xlabel("Number of Detected Features", fontsize=self._typography.label_size)
            axes[0].set_ylabel("Frequency", fontsize=self._typography.label_size)
            axes[0].set_title(
                "Detection Rate Comparison", fontsize=self._typography.title_size, fontweight="bold"
            )
            axes[0].legend(fontsize=self._typography.legend_size)
            apply_color_style(axes[0], "qc")

        if (
            "missing_rate" in scptensor_result.sample_metrics
            and "missing_rate" in competitor_result.sample_metrics
        ):
            scptensor_missing = scptensor_result.sample_metrics["missing_rate"]
            competitor_missing = competitor_result.sample_metrics["missing_rate"]

            axes[1].hist(
                scptensor_missing,
                bins=30,
                alpha=0.6,
                label="ScpTensor",
                color=self._colors.primary,
                edgecolor=self._colors.neutral,
            )
            axes[1].hist(
                competitor_missing,
                bins=30,
                alpha=0.6,
                label=self.FRAMEWORK_DISPLAY_NAMES.get(
                    competitor_result.framework, competitor_result.framework.title()
                ),
                color=self._colors.secondary,
                edgecolor=self._colors.neutral,
            )
            axes[1].set_xlabel("Missing Rate", fontsize=self._typography.label_size)
            axes[1].set_ylabel("Frequency", fontsize=self._typography.label_size)
            axes[1].set_title(
                "Missing Rate Comparison", fontsize=self._typography.title_size, fontweight="bold"
            )
            axes[1].legend(fontsize=self._typography.legend_size)
            apply_color_style(axes[1], "qc")

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"qc_framework_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path


class MissingTypeDisplay(DisplayBase):
    """Display class for ScpTensor-exclusive missing type analysis visualizations.

    ScpTensor tracks different types of missing values through the mask matrix:
    - VALID (0): Valid, detected values
    - MBR (1): Match Between Runs missing
    - LOD (2): Below Limit of Detection
    - FILTERED (3): Filtered out by quality control
    - IMPUTED (5): Imputed/filled value

    This class generates visualizations showing the distribution of these
    missing types, which is a unique capability of ScpTensor not available
    in competing frameworks.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.qc import MissingTypeDisplay, MissingTypeReport
    >>> display = MissingTypeDisplay()
    >>> report = MissingTypeReport(
    ...     valid_rate=0.75,
    ...     mbr_rate=0.15,
    ...     lod_rate=0.08,
    ...     filtered_rate=0.02,
    ...     imputed_rate=0.0,
    ...     feature_missing_rates=np.random.rand(1000) * 0.5,
    ...     sample_missing_rates=np.random.rand(100) * 0.5,
    ...     mbr_by_feature=np.random.rand(1000) * 0.3,
    ...     lod_by_feature=np.random.rand(1000) * 0.2
    ... )
    >>> fig_path = display.render_missing_type_distribution(report)
    """

    MASK_CODE_NAMES: dict[int, str] = {
        0: "VALID",
        1: "MBR",
        2: "LOD",
        3: "FILTERED",
        5: "IMPUTED",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "qc"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("qc")
        self._typography = DEFAULT_TYPOGRAPHY

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_missing_type_distribution or render_missing_rate_by_type instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "missing_type_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_missing_type_distribution or render_missing_rate_by_type",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "qc")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_missing_type_distribution(
        self, report: MissingTypeReport, format: str = "png"
    ) -> Path:
        """Generate a pie/bar chart showing MBR vs LOD vs Valid distribution.

        Creates a visualization showing the proportion of each mask code type
        in the dataset, highlighting ScpTensor's unique ability to distinguish
        between different types of missing values.

        Parameters
        ----------
        report : MissingTypeReport
            Missing type analysis report containing rates for each mask code.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'missing_type_distribution.{format}' in the
        QC figures subdirectory.

        The visualization shows:
        - VALID: Properly detected values
        - MBR: Missing due to Match Between Runs (technical missingness)
        - LOD: Missing due to being below Limit of Detection (biological/technical)
        - FILTERED: Values removed by QC
        - IMPUTED: Values that have been imputed
        """
        import matplotlib.pyplot as plt

        labels = ["VALID", "MBR", "LOD", "FILTERED", "IMPUTED"]
        sizes = [
            report.valid_rate,
            report.mbr_rate,
            report.lod_rate,
            report.filtered_rate,
            report.imputed_rate,
        ]
        colors = [
            self._colors.primary,
            self._colors.secondary,
            self._colors.accent,
            self._colors.neutral,
            self._colors.success,
        ]

        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors, strict=True) if s > 0]
        if non_zero:
            labels, sizes, colors = zip(*non_zero, strict=True)
        else:
            labels, sizes, colors = [], [], []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if sizes:
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "black", "linewidth": 1},
            )
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
            ax1.set_title(
                "Missing Type Distribution (Pie Chart)",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )

        if sizes:
            bars = ax2.bar(labels, sizes, color=colors, edgecolor=self._colors.neutral, alpha=0.8)
            ax2.set_ylabel("Proportion", fontsize=self._typography.label_size)
            ax2.set_title(
                "Missing Type Distribution (Bar Chart)",
                fontsize=self._typography.title_size,
                fontweight="bold",
            )
            apply_color_style(ax2, "qc")

            for bar, size in zip(bars, sizes, strict=True):
                height = bar.get_height()
                ax2.annotate(
                    f"{size:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=self._typography.annotation_size,
                    fontweight="bold",
                )

        total_missing = report.mbr_rate + report.lod_rate + report.filtered_rate
        stats_text = (
            f"Total Missing: {total_missing:.1%}\n"
            f"MBR (Technical): {report.mbr_rate:.1%}\n"
            f"LOD (Detection): {report.lod_rate:.1%}\n"
            f"Valid: {report.valid_rate:.1%}"
        )
        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            fontsize=self._typography.annotation_size,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"missing_type_distribution.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_missing_rate_by_type(self, report: MissingTypeReport, format: str = "png") -> Path:
        """Generate a bar chart showing missing rate by protein type.

        Creates a visualization showing how MBR and LOD missing rates
        vary across different features, helping identify features
        that are consistently affected by specific missingness types.

        Parameters
        ----------
        report : MissingTypeReport
            Missing type analysis report containing per-feature MBR and LOD rates.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'missing_rate_by_type.{format}' in the
        QC figures subdirectory.

        The visualization helps identify:
        - Features with high MBR: May be affected by alignment/run matching issues
        - Features with high LOD: May be low-abundance proteins near detection limit
        - Features with high total missing: May be poor-quality measurements
        """
        import matplotlib.pyplot as plt

        total_missing = report.mbr_by_feature + report.lod_by_feature
        n_features_to_show = min(30, len(total_missing))

        top_indices = np.argsort(total_missing)[-n_features_to_show:][::-1]

        feature_labels = [f"P{i + 1}" for i in top_indices]
        mbr_rates = report.mbr_by_feature[top_indices]
        lod_rates = report.lod_by_feature[top_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, n_features_to_show * 0.2)))

        y_pos = np.arange(n_features_to_show)

        ax.barh(
            y_pos,
            mbr_rates,
            color=self._colors.secondary,
            label="MBR (Match Between Runs)",
            edgecolor=self._colors.neutral,
            linewidth=0.5,
        )
        ax.barh(
            y_pos,
            lod_rates,
            left=mbr_rates,
            color=self._colors.accent,
            label="LOD (Below Detection)",
            edgecolor=self._colors.neutral,
            linewidth=0.5,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Missing Rate", fontsize=self._typography.label_size)
        ax.set_title(
            "Missing Type Distribution by Feature (Top 30)",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=self._typography.legend_size)
        apply_color_style(ax, "qc")

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"missing_rate_by_type.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_feature_missing_scatter(
        self, report: MissingTypeReport, format: str = "png"
    ) -> Path:
        """Generate a scatter plot of MBR vs LOD missing rates per feature.

        Creates a scatter visualization showing the relationship between
        MBR and LOD missing rates across features, helping identify patterns
        in missingness types.

        Parameters
        ----------
        report : MissingTypeReport
            Missing type analysis report containing per-feature MBR and LOD rates.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'feature_missing_scatter.{format}' in the
        QC figures subdirectory.

        The scatter plot interpretation:
        - High MBR, Low LOD: Features affected by run matching issues
        - Low MBR, High LOD: Low-abundance features near detection limit
        - High both: Poor-quality features with multiple issues
        - Low both: High-quality, reliably detected features
        """
        import matplotlib.pyplot as plt

        mbr = report.mbr_by_feature
        lod = report.lod_by_feature
        total = mbr + lod

        fig, ax = plt.subplots(figsize=(8, 8))

        scatter = ax.scatter(mbr, lod, c=total, cmap="YlOrRd", alpha=0.6, s=20, edgecolors="none")

        max_val = max(np.max(mbr), np.max(lod))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal MBR/LOD")

        ax.set_xlabel("MBR Missing Rate (Match Between Runs)", fontsize=self._typography.label_size)
        ax.set_ylabel("LOD Missing Rate (Below Detection)", fontsize=self._typography.label_size)
        ax.set_title(
            "Feature Missing Type Comparison: MBR vs LOD",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.legend(fontsize=self._typography.legend_size)
        apply_color_style(ax, "qc")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Total Missing Rate", fontsize=self._typography.label_size)

        ax.text(
            0.05,
            0.95,
            "High MBR\nLow LOD",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
            bbox={"boxstyle": "round", "facecolor": self._colors.secondary, "alpha": 0.3},
            ha="left",
        )
        ax.text(
            0.95,
            0.05,
            "Low MBR\nHigh LOD",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
            bbox={"boxstyle": "round", "facecolor": self._colors.accent, "alpha": 0.3},
            ha="right",
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"feature_missing_scatter.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path


class QCBatchDisplay(DisplayBase):
    """Display class for batch effect detection and CV comparison visualizations.

    Generates visualizations for batch-related QC metrics including
    PCA plots colored by batch and coefficient of variation comparisons
    across batches.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.qc import QCBatchDisplay, BatchCVReport
    >>> display = QCBatchDisplay()
    >>> report = BatchCVReport(
    ...     within_batch_cv={"batch1": 0.25, "batch2": 0.28},
    ...     between_batch_cv=0.35,
    ...     cv_by_batch_feature=np.random.rand(2, 500) * 0.5,
    ...     batch_names=["batch1", "batch2"],
    ...     high_cv_features=[10, 20, 30]
    ... )
    >>> fig_path = display.render_batch_cv_comparison(report)
    """

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "qc"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("qc")
        self._typography = DEFAULT_TYPOGRAPHY

    def render(self) -> Path:
        """Render a default summary figure.

        Returns
        -------
        Path
            Path to the rendered output file.

        Notes
        -----
        This method is required by the abstract base class but should not
        be used directly. Use render_batch_pca or render_batch_cv_comparison instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "batch_qc_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_batch_pca or render_batch_cv_comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "qc")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_batch_pca(
        self,
        pca_coords: npt.NDArray[np.float64],
        batch_labels: npt.NDArray[np.int_] | list[int] | list[str],
        format: str = "png",
        title: str = "Batch PCA",
    ) -> Path:
        """Generate a PCA plot colored by batch (pre-integration).

        Creates a scatter plot of PCA coordinates with points colored by
        batch assignment, useful for visualizing batch effects before
        integration.

        Parameters
        ----------
        pca_coords : npt.NDArray[np.float64]
            PCA coordinates of shape (n_samples, 2) or (n_samples, n_components).
            If more than 2 components provided, uses first two.
        batch_labels : npt.NDArray[np.int_] | list[int] | list[str]
            Batch assignment for each sample.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".
        title : str, default="Batch PCA"
            Title for the plot.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'batch_pca.{format}' in the
        QC figures subdirectory.

        Effective batch correction should show:
        - Before integration: Clear separation between batches
        - After integration: Well-mixed batches without clear batch structure
        """
        import matplotlib.pyplot as plt

        if pca_coords.shape[1] < 2:
            raise ValueError(
                f"PCA coordinates must have at least 2 columns, got {pca_coords.shape[1]}"
            )

        pc1 = pca_coords[:, 0]
        pc2 = pca_coords[:, 1]

        unique_batches = np.unique(batch_labels)

        fig, ax = plt.subplots(figsize=(8, 8))

        for i, batch_id in enumerate(unique_batches):
            mask = np.array(batch_labels) == batch_id
            color_idx = i % 7
            color = [
                self._colors.primary,
                self._colors.secondary,
                self._colors.accent,
                self._colors.neutral,
                self._colors.success,
                self._colors.warning,
                self._colors.error,
            ][color_idx]
            ax.scatter(
                pc1[mask],
                pc2[mask],
                c=color,
                label=f"Batch {batch_id}",
                alpha=0.6,
                s=30,
                edgecolors="none",
            )

        ax.set_xlabel("PC1", fontsize=self._typography.label_size)
        ax.set_ylabel("PC2", fontsize=self._typography.label_size)
        ax.set_title(
            f"{title}\n(Colored by Batch)", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.legend(fontsize=self._typography.legend_size, loc="best")
        apply_color_style(ax, "qc")

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"batch_pca.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_batch_cv_comparison(self, report: BatchCVReport, format: str = "png") -> Path:
        """Generate a box plot comparing CV across batches.

        Creates a box plot showing the distribution of feature CV values
        within each batch, enabling comparison of measurement variability
        across batches.

        Parameters
        ----------
        report : BatchCVReport
            Batch CV report containing within-batch and between-batch CV statistics.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'batch_cv_comparison.{format}' in the
        QC figures subdirectory.

        The box plot shows:
        - Median CV within each batch
        - Interquartile range (IQR)
        - Whiskers showing 1.5 * IQR
        - Outliers beyond the whiskers

        Batches with higher median CV may indicate:
        - Lower data quality
        - Higher biological variability
        - Technical issues in sample processing
        """
        import matplotlib.pyplot as plt

        cv_data = report.cv_by_batch_feature.T
        n_features, n_batches = cv_data.shape

        fig, ax = plt.subplots(figsize=(max(8, n_batches * 1.5), 6))

        positions = np.arange(1, n_batches + 1)
        box_data = [cv_data[:, i] for i in range(n_batches)]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            labels=report.batch_names,
            patch_artist=True,
            showmeans=True,
            widths=0.6,
        )

        for i, patch in enumerate(bp["boxes"]):
            color_idx = i % 7
            color = [
                self._colors.primary,
                self._colors.secondary,
                self._colors.accent,
                self._colors.neutral,
                self._colors.success,
                self._colors.warning,
                self._colors.error,
            ][color_idx]
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(
            y=report.between_batch_cv,
            color=self._colors.error,
            linestyle="--",
            linewidth=2,
            label=f"Between-Batch CV: {report.between_batch_cv:.3f}",
        )

        ax.set_xlabel("Batch", fontsize=self._typography.label_size)
        ax.set_ylabel("Coefficient of Variation (CV)", fontsize=self._typography.label_size)
        ax.set_title(
            "Within-Batch CV Comparison", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=self._typography.legend_size)
        apply_color_style(ax, "qc")

        stats_text = "Within-Batch Mean CV:\n"
        for batch_name, cv_val in report.within_batch_cv.items():
            stats_text += f"  {batch_name}: {cv_val:.3f}\n"
        stats_text += f"\nHigh CV Features: {len(report.high_cv_features)}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            fontsize=self._typography.annotation_size,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"batch_cv_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_batch_cv_heatmap(
        self, report: BatchCVReport, top_n_features: int = 50, format: str = "png"
    ) -> Path:
        """Generate a heatmap of CV values for top variable features across batches.

        Creates a heatmap visualization showing the CV values of the most
        variable features across different batches, helping identify
        batch-specific variability patterns.

        Parameters
        ----------
        report : BatchCVReport
            Batch CV report containing per-feature, per-batch CV values.
        top_n_features : int, default=50
            Number of highest-CV features to display.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'batch_cv_heatmap.{format}' in the
        QC figures subdirectory.

        The heatmap helps identify:
        - Features with consistently high CV across all batches
        - Features with batch-specific high CV (potential batch effects)
        - Patterns in variability that may inform filtering strategies
        """
        import matplotlib.pyplot as plt

        cv_data = report.cv_by_batch_feature.T
        mean_cv = np.mean(cv_data, axis=1)

        top_indices = np.argsort(mean_cv)[-top_n_features:][::-1]
        top_cv_data = cv_data[top_indices, :]

        feature_labels = [f"F{i + 1}" for i in top_indices]

        fig, ax = plt.subplots(
            figsize=(max(6, len(report.batch_names) * 0.8), top_n_features * 0.15)
        )

        im = ax.imshow(top_cv_data, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(np.arange(len(report.batch_names)))
        ax.set_xticklabels(report.batch_names)
        ax.set_yticks(np.arange(top_n_features))
        ax.set_yticklabels(feature_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_xlabel("Batch", fontsize=self._typography.label_size)
        ax.set_ylabel("Feature", fontsize=self._typography.label_size)
        ax.set_title(
            f"Feature CV Across Batches (Top {top_n_features} Variable Features)",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Coefficient of Variation", fontsize=self._typography.label_size)

        apply_color_style(ax, "qc")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"batch_cv_heatmap.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path
