"""Visualization module for ScpTensor vs Scanpy comparison benchmarks.

This module generates publication-quality comparison visualizations using
SciencePlots style.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots

    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False

from scptensor.benchmark.comparison_engine import ComparisonResult, ComparisonEngine, compute_accuracy_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Configuration
# =============================================================================


class PlotStyle(Enum):
    """Available plot styles."""

    SCIENCE = "science"
    IEEE = "ieee"
    NATURE = "nature"
    DEFAULT = "default"


def configure_plots(style: PlotStyle = PlotStyle.SCIENCE, dpi: int = 300) -> None:
    """Configure matplotlib for publication-quality plots.

    Parameters
    ----------
    style : PlotStyle
        Plot style to use.
    dpi : int
        DPI for saved figures.
    """
    if SCIENCEPLOTS_AVAILABLE and style != PlotStyle.DEFAULT:
        if style == PlotStyle.SCIENCE:
            plt.style.use(["science", "no-latex"])
        elif style == PlotStyle.IEEE:
            plt.style.use(["ieee", "no-latex"])
        elif style == PlotStyle.NATURE:
            plt.style.use(["nature", "no-latex"])
    else:
        plt.style.use("default")

    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
    })


# =============================================================================
# Comparison Visualizations
# =============================================================================


class ComparisonVisualizer:
    """Visualizer for ScpTensor vs Scanpy comparisons."""

    def __init__(
        self,
        output_dir: str | Path = "benchmark_results/figures",
        style: PlotStyle = PlotStyle.SCIENCE,
    ) -> None:
        """Initialize the visualizer.

        Parameters
        ----------
        output_dir : str | Path
            Directory for output figures.
        style : PlotStyle
            Plot style to use.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._style = style

        configure_plots(style)

        # Subdirectories for different plot types
        self._dirs = {
            "performance": self._output_dir / "01_performance",
            "normalization": self._output_dir / "02_normalization",
            "imputation": self._output_dir / "03_imputation",
            "dim_reduction": self._output_dir / "04_dim_reduction",
            "clustering": self._output_dir / "05_clustering",
            "feature_selection": self._output_dir / "06_feature_selection",
            "summary": self._output_dir / "summary",
        }

        for d in self._dirs.values():
            d.mkdir(exist_ok=True, parents=True)

    def plot_runtime_comparison(
        self,
        results: dict[str, list[ComparisonResult]],
        filename: str = "runtime_comparison.png",
    ) -> Path:
        """Plot runtime comparison between ScpTensor and Scanpy.

        Parameters
        ----------
        results : dict[str, list[ComparisonResult]]
            Comparison results by method name.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        methods = []
        scptensor_times = []
        scanpy_times = []

        for method_name, result_list in results.items():
            if not result_list:
                continue

            # Average times across datasets
            scptensor_avg = np.mean([
                r.scptensor_result.runtime_seconds
                for r in result_list
                if r.scptensor_result and r.scptensor_result.success
            ])

            scanpy_avg = np.mean([
                r.scanpy_result.runtime_seconds
                for r in result_list
                if r.scanpy_result and r.scanpy_result.success
            ])

            if scptensor_avg > 0 or scanpy_avg > 0:
                methods.append(method_name.replace("_", " ").title())
                scptensor_times.append(scptensor_avg)
                scanpy_times.append(scanpy_avg)

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width / 2, scptensor_times, width, label="ScpTensor", color="#2E86AB")
        bars2 = ax.bar(x + width / 2, scanpy_times, width, label="Scanpy", color="#A23B72")

        ax.set_xlabel("Method")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime Comparison: ScpTensor vs Scanpy")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        output_path = self._dirs["performance"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_speedup_heatmap(
        self,
        results: dict[str, list[ComparisonResult]],
        filename: str = "speedup_heatmap.png",
    ) -> Path:
        """Plot speedup heatmap.

        Parameters
        ----------
        results : dict[str, list[ComparisonResult]]
            Comparison results.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        methods = list(results.keys())
        datasets = set()

        for result_list in results.values():
            for r in result_list:
                datasets.add(r.dataset_name)

        datasets = sorted(datasets)

        # Create speedup matrix
        speedup_matrix = np.full((len(methods), len(datasets)), np.nan)

        for i, method in enumerate(methods):
            for j, dataset in enumerate(datasets):
                for r in results[method]:
                    if r.dataset_name == dataset and "speedup" in r.comparison_metrics:
                        speedup_matrix[i, j] = r.comparison_metrics["speedup"]
                        break

        fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), max(4, len(methods) * 0.8)))

        im = ax.imshow(
            speedup_matrix,
            cmap="RdYlGn",
            aspect="auto",
            vmin=0.5,
            vmax=2.0,
        )

        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels([d.replace("_", " ").title() for d in datasets], rotation=45, ha="right")
        ax.set_yticklabels([m.replace("_", " ").title() for m in methods])

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(datasets)):
                value = speedup_matrix[i, j]
                if not np.isnan(value):
                    text_color = "white" if value < 1.0 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}x",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                    )

        ax.set_title("Speedup: ScpTensor vs Scanpy (>1 means ScpTensor faster)")
        fig.colorbar(im, ax=ax, label="Speedup")

        plt.tight_layout()

        output_path = self._dirs["performance"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_accuracy_scatter(
        self,
        scptensor_output: np.ndarray,
        scanpy_output: np.ndarray,
        method_name: str,
        filename: str = None,
    ) -> Path:
        """Plot scatter comparison of outputs.

        Parameters
        ----------
        scptensor_output : np.ndarray
            ScpTensor output.
        scanpy_output : np.ndarray
            Scanpy output.
        method_name : str
            Method name for title.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        # Flatten arrays
        scptensor_flat = scptensor_output.ravel()
        scanpy_flat = scanpy_output.ravel()

        # Limit to 10000 points for performance
        if len(scptensor_flat) > 10000:
            idx = np.random.choice(len(scptensor_flat), 10000, replace=False)
            scptensor_flat = scptensor_flat[idx]
            scanpy_flat = scanpy_flat[idx]

        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot
        ax.scatter(scanpy_flat, scptensor_flat, alpha=0.3, s=1, color="#2E86AB")

        # Diagonal line
        min_val = min(np.min(scptensor_flat), np.min(scanpy_flat))
        max_val = max(np.max(scptensor_flat), np.max(scanpy_flat))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="Perfect agreement")

        # Calculate correlation
        if np.std(scptensor_flat) > 0 and np.std(scanpy_flat) > 0:
            corr = np.corrcoef(scptensor_flat, scanpy_flat)[0, 1]
        else:
            corr = 0.0

        ax.set_xlabel("Scanpy Output")
        ax.set_ylabel("ScpTensor Output")
        ax.set_title(f"{method_name.replace('_', ' ').title()}: Output Comparison\nCorrelation: {corr:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Determine output directory based on method
        if "normalize" in method_name:
            output_dir = self._dirs["normalization"]
        elif "impute" in method_name:
            output_dir = self._dirs["imputation"]
        elif "pca" in method_name or "umap" in method_name:
            output_dir = self._dirs["dim_reduction"]
        elif "kmeans" in method_name or "cluster" in method_name:
            output_dir = self._dirs["clustering"]
        elif "hvg" in method_name:
            output_dir = self._dirs["feature_selection"]
        else:
            output_dir = self._dirs["summary"]

        if filename is None:
            filename = f"{method_name}_comparison.png"

        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_pca_variance(
        self,
        scptensor_variance: np.ndarray,
        scanpy_variance: np.ndarray,
        filename: str = "pca_variance.png",
    ) -> Path:
        """Plot PCA variance comparison.

        Parameters
        ----------
        scptensor_variance : np.ndarray
            ScpTensor explained variance ratio.
        scanpy_variance : np.ndarray
            Scanpy explained variance ratio.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        n_components = min(len(scptensor_variance), len(scanpy_variance))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        x = np.arange(1, n_components + 1)

        # Individual variance
        ax1.bar(x - 0.2, scptensor_variance[:n_components], 0.4, label="ScpTensor", color="#2E86AB", alpha=0.8)
        ax1.bar(x + 0.2, scanpy_variance[:n_components], 0.4, label="Scanpy", color="#A23B72", alpha=0.8)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("PCA: Explained Variance per Component")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Cumulative variance
        cum_scptensor = np.cumsum(scptensor_variance[:n_components])
        cum_scanpy = np.cumsum(scanpy_variance[:n_components])

        ax2.plot(x, cum_scptensor, "o-", label="ScpTensor", color="#2E86AB", markersize=4)
        ax2.plot(x, cum_scanpy, "s-", label="Scanpy", color="#A23B72", markersize=4)
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("PCA: Cumulative Explained Variance")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        output_path = self._dirs["dim_reduction"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_umap_comparison(
        self,
        scptensor_umap: np.ndarray,
        scanpy_umap: np.ndarray,
        labels: np.ndarray | None = None,
        filename: str = "umap_comparison.png",
    ) -> Path:
        """Plot UMAP embedding comparison.

        Parameters
        ----------
        scptensor_umap : np.ndarray
            ScpTensor UMAP embedding (n_samples, 2).
        scanpy_umap : np.ndarray
            Scanpy UMAP embedding (n_samples, 2).
        labels : np.ndarray | None
            Cluster/group labels for coloring.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Color by labels if provided
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            color_map = {l: colors[i] for i, l in enumerate(unique_labels)}
            point_colors = [color_map[l] for l in labels]
        else:
            point_colors = "#2E86AB"

        # ScpTensor UMAP
        axes[0].scatter(
            scptensor_umap[:, 0],
            scptensor_umap[:, 1],
            c=point_colors,
            s=5,
            alpha=0.6,
        )
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        axes[0].set_title("ScpTensor UMAP")

        # Scanpy UMAP
        axes[1].scatter(
            scanpy_umap[:, 0],
            scanpy_umap[:, 1],
            c=point_colors,
            s=5,
            alpha=0.6,
        )
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("UMAP 2")
        axes[1].set_title("Scanpy UMAP")

        plt.tight_layout()

        output_path = self._dirs["dim_reduction"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_clustering_consistency(
        self,
        scptensor_labels: np.ndarray,
        scanpy_labels: np.ndarray,
        ari: float,
        nmi: float,
        filename: str = "kmeans_consistency.png",
    ) -> Path:
        """Plot clustering consistency visualization.

        Parameters
        ----------
        scptensor_labels : np.ndarray
            ScpTensor cluster labels.
        scanpy_labels : np.ndarray
            Scanpy cluster labels.
        ari : float
            Adjusted Rand Index.
        nmi : float
            Normalized Mutual Information.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        # Create confusion matrix
        scptensor_unique = np.unique(scptensor_labels)
        scanpy_unique = np.unique(scanpy_labels)

        matrix = np.zeros((len(scptensor_unique), len(scanpy_unique)))

        for i, s_label in enumerate(scptensor_unique):
            for j, p_label in enumerate(scanpy_unique):
                matrix[i, j] = np.sum((scptensor_labels == s_label) & (scanpy_labels == p_label))

        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_norm = matrix / row_sums

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(matrix_norm, cmap="Blues", aspect="auto")

        ax.set_xticks(np.arange(len(scanpy_unique)))
        ax.set_yticks(np.arange(len(scptensor_unique)))
        ax.set_xticklabels([f"Cluster {i}" for i in scanpy_unique])
        ax.set_yticklabels([f"Cluster {i}" for i in scptensor_unique])

        # Add text annotations
        for i in range(len(scptensor_unique)):
            for j in range(len(scanpy_unique)):
                value = int(matrix[i, j])
                text_color = "white" if matrix_norm[i, j] > 0.5 else "black"
                ax.text(j, i, value, ha="center", va="center", color=text_color)

        ax.set_xlabel("Scanpy Clusters")
        ax.set_ylabel("ScpTensor Clusters")
        ax.set_title(f"Clustering Consistency (ARI={ari:.3f}, NMI={nmi:.3f})")

        fig.colorbar(im, ax=ax, label="Normalized count")

        plt.tight_layout()

        output_path = self._dirs["clustering"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_summary_radar(
        self,
        summary_metrics: dict[str, float],
        filename: str = "summary_radar.png",
    ) -> Path:
        """Plot summary radar chart.

        Parameters
        ----------
        summary_metrics : dict[str, float]
            Summary metrics for different categories.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved figure.
        """
        categories = list(summary_metrics.keys())
        values = list(summary_metrics.values())

        # Normalize to 0-1
        max_val = max(values) if values else 1.0
        if max_val == 0:
            max_val = 1.0
        normalized = [v / max_val for v in values]

        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized += normalized[:1]
        angles += angles[:1]

        ax.plot(angles, normalized, "o-", linewidth=2, color="#2E86AB")
        ax.fill(angles, normalized, alpha=0.25, color="#2E86AB")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.grid(True)

        plt.tight_layout()

        output_path = self._dirs["summary"] / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path


def get_visualizer(
    output_dir: str | Path = "benchmark_results/figures",
    style: PlotStyle = PlotStyle.SCIENCE,
) -> ComparisonVisualizer:
    """Get a comparison visualizer instance.

    Parameters
    ----------
    output_dir : str | Path
        Output directory.
    style : PlotStyle
        Plot style.

    Returns
    -------
    ComparisonVisualizer
        Visualizer instance.
    """
    return ComparisonVisualizer(output_dir=output_dir, style=style)
