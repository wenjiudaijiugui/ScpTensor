"""Dimensionality reduction display module for ScpTensor benchmark.

Provides visualization capabilities for comparing dimensionality reduction results
between ScpTensor and competing frameworks (e.g., Scanpy). This module generates
publication-quality figures for PCA and UMAP comparisons, including variance
explained analysis, component loadings, and parameter sensitivity visualization.

Classes
-------
PCADisplay
    Visualizes PCA comparison results between ScpTensor and Scanpy.
UMAPDisplay
    Visualizes UMAP comparison results between ScpTensor and Scanpy.
DimReductionComparisonResult
    Dataclass for storing dimensionality reduction comparison data.
PCAResult
    Dataclass for storing PCA-specific results.
UMAPResult
    Dataclass for storing UMAP-specific results.
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
)

__all__ = [
    "PCADisplay",
    "UMAPDisplay",
    "DimReductionComparisonResult",
    "PCAResult",
    "UMAPResult",
]


@dataclass(frozen=True)
class PCAResult:
    """Result of PCA dimensionality reduction.

    Attributes
    ----------
    framework : str
        Name of the framework used (e.g., "scptensor", "scanpy").
    components : npt.NDArray[np.float64]
        Principal components matrix of shape (n_samples, n_components).
    explained_variance : npt.NDArray[np.float64]
        Explained variance for each component, shape (n_components,).
    explained_variance_ratio : npt.NDArray[np.float64]
        Ratio of explained variance for each component, shape (n_components,).
    loadings : npt.NDArray[np.float64] | None
        Feature loadings matrix of shape (n_features, n_components), if computed.
    feature_names : list[str] | None
        Names of features used for PCA, if available.
    n_components : int
        Number of principal components computed.
    runtime_seconds : float | None
        Runtime of the PCA computation in seconds, if measured.
    """

    framework: str
    components: npt.NDArray[np.float64]
    explained_variance: npt.NDArray[np.float64]
    explained_variance_ratio: npt.NDArray[np.float64]
    n_components: int
    loadings: npt.NDArray[np.float64] | None = None
    feature_names: list[str] | None = None
    runtime_seconds: float | None = None


@dataclass(frozen=True)
class UMAPResult:
    """Result of UMAP dimensionality reduction.

    Attributes
    ----------
    framework : str
        Name of the framework used (e.g., "scptensor", "scanpy").
    embedding : npt.NDArray[np.float64]
        UMAP embedding coordinates of shape (n_samples, 2).
    n_neighbors : int
        Number of neighbors parameter used for UMAP.
    min_dist : float
        Minimum distance parameter used for UMAP.
    metric : str
        Distance metric used for UMAP.
    runtime_seconds : float | None
        Runtime of the UMAP computation in seconds, if measured.
    """

    framework: str
    embedding: npt.NDArray[np.float64]
    n_neighbors: int
    min_dist: float
    metric: str
    runtime_seconds: float | None = None


@dataclass(frozen=True)
class DimReductionComparisonResult:
    """Result of dimensionality reduction comparison between frameworks.

    Attributes
    ----------
    scptensor_pca : PCAResult | None
        PCA result from ScpTensor, if available.
    competitor_pca : PCAResult | None
        PCA result from competitor framework, if available.
    scptensor_umap : UMAPResult | None
        UMAP result from ScpTensor, if available.
    competitor_umap : UMAPResult | None
        UMAP result from competitor framework, if available.
    competitor_name : str
        Name of the competitor framework (e.g., "scanpy").
    method_name : str
        Name of the dimensionality reduction method ("pca", "umap", or "both").
    dataset_name : str
        Name of the dataset used for comparison.
    """

    scptensor_pca: PCAResult | None = None
    competitor_pca: PCAResult | None = None
    scptensor_umap: UMAPResult | None = None
    competitor_umap: UMAPResult | None = None
    competitor_name: str = "scanpy"
    method_name: str = "both"
    dataset_name: str = "unknown"


class PCADisplay(DisplayBase):
    """Display class for PCA comparison visualizations.

    Generates publication-quality figures comparing PCA results between
    ScpTensor and competing frameworks, including cumulative explained
    variance plots, scree plots, and loading heatmaps.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.dim_reduction import PCADisplay, PCAResult
    >>> display = PCADisplay()
    >>> components = np.random.randn(100, 10)
    >>> explained_var = np.array([10, 5, 3, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05])
    >>> explained_var_ratio = explained_var / explained_var.sum()
    >>> pca_result = PCAResult(
    ...     framework="scptensor",
    ...     components=components,
    ...     explained_variance=explained_var,
    ...     explained_variance_ratio=explained_var_ratio,
    ...     loadings=np.random.randn(50, 10),
    ...     n_components=10,
    ...     runtime_seconds=0.5
    ... )
    >>> fig_path = display.render_variance_explained(pca_result)
    """

    FRAMEWORK_DISPLAY_NAMES: dict[str, str] = {
        "scptensor": "ScpTensor",
        "scanpy": "Scanpy",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "dim_reduction"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("dim_reduction")
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
        be used directly. Use render_variance_explained, render_per_component_variance,
        or render_loadings_heatmap instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "pca_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_variance_explained, render_per_component_variance,\n"
            "or render_loadings_heatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_variance_explained(self, result: PCAResult, format: str = "png") -> Path:
        """Generate a cumulative explained variance plot.

        Creates a line plot showing the cumulative proportion of variance
        explained by increasing numbers of principal components. This helps
        determine the optimal number of components to retain.

        Parameters
        ----------
        result : PCAResult
            PCA result containing explained variance ratio data.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'pca_variance_explained.{format}' in the
        dimensionality reduction figures subdirectory.

        The plot includes:
        - Cumulative variance curve
        - Reference lines at common thresholds (50%, 80%, 90%, 95%)
        - Annotation of components needed for each threshold
        """
        import matplotlib.pyplot as plt

        cumulative_variance = np.cumsum(result.explained_variance_ratio)
        n_components = len(result.explained_variance_ratio)

        fig, ax = plt.subplots(figsize=(8, 6))

        x_values = np.arange(1, n_components + 1)
        ax.plot(
            x_values,
            cumulative_variance,
            "o-",
            color=self._colors.primary,
            linewidth=2,
            markersize=5,
            label="Cumulative Variance",
        )

        thresholds = [0.5, 0.8, 0.9, 0.95]
        threshold_colors = [
            self._colors.error,
            self._colors.warning,
            self._colors.accent,
            self._colors.secondary,
        ]
        labels = ["50%", "80%", "90%", "95%"]

        for threshold, color, label in zip(thresholds, threshold_colors, labels, strict=True):
            idx = np.argmax(cumulative_variance >= threshold)
            if cumulative_variance[idx] >= threshold:
                ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.6, linewidth=1.5)
                ax.axvline(x=idx + 1, color=color, linestyle="--", alpha=0.6, linewidth=1.5)
                ax.annotate(
                    f"{label} at PC{idx + 1}",
                    xy=(idx + 1, threshold),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=self._typography.annotation_size,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                )

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        ax.set_xlabel("Number of Principal Components", fontsize=self._typography.label_size)
        ax.set_ylabel(
            "Cumulative Proportion of Variance Explained", fontsize=self._typography.label_size
        )
        ax.set_title(
            f"PCA Cumulative Explained Variance ({framework_name})",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.legend(fontsize=self._typography.legend_size)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(1, n_components + 1, max(1, n_components // 10)))

        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"pca_variance_explained_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_per_component_variance(
        self, result: PCAResult, max_components: int = 30, format: str = "png"
    ) -> Path:
        """Generate a scree plot showing individual component variance.

        Creates a bar chart showing the proportion of variance explained
        by each principal component (scree plot). This helps identify
        the "elbow" point where additional components provide diminishing
        returns.

        Parameters
        ----------
        result : PCAResult
            PCA result containing explained variance ratio data.
        max_components : int, default=30
            Maximum number of components to display. If n_components exceeds
            this value, only the first max_components are shown.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'pca_scree_plot.{format}' in the
        dimensionality reduction figures subdirectory.

        The scree plot helps visualize the "elbow" in the variance curve,
        indicating the optimal number of components to retain.
        """
        import matplotlib.pyplot as plt

        n_display = min(len(result.explained_variance_ratio), max_components)
        variance_to_show = result.explained_variance_ratio[:n_display]
        x_values = np.arange(1, n_display + 1)

        fig, ax = plt.subplots(figsize=(max(8, n_display * 0.3), 5))

        bars = ax.bar(
            x_values,
            variance_to_show,
            color=self._colors.primary,
            edgecolor=self._colors.neutral,
            alpha=0.8,
        )

        for bar, val in zip(bars, variance_to_show, strict=True):
            if val > 0.01:
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=self._typography.annotation_size,
                )

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        ax.set_xlabel("Principal Component", fontsize=self._typography.label_size)
        ax.set_ylabel("Proportion of Variance Explained", fontsize=self._typography.label_size)

        title = f"PCA Scree Plot ({framework_name})"
        if len(result.explained_variance_ratio) > max_components:
            title += f"\n(Showing first {max_components} of {len(result.explained_variance_ratio)} components)"
        ax.set_title(title, fontsize=self._typography.title_size, fontweight="bold")

        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x_values)

        ax2 = ax.twinx()
        cumulative = np.cumsum(variance_to_show)
        ax2.plot(
            x_values,
            cumulative,
            "o-",
            color=self._colors.secondary,
            linewidth=2,
            markersize=4,
            label="Cumulative",
        )
        ax2.set_ylabel("Cumulative Proportion", fontsize=self._typography.label_size)
        ax2.tick_params(axis="y", labelcolor=self._colors.secondary)
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            fontsize=self._typography.legend_size,
        )

        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"pca_scree_plot_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_loadings_heatmap(
        self,
        result: PCAResult,
        top_n_features: int = 20,
        top_n_components: int = 10,
        format: str = "png",
    ) -> Path:
        """Generate a heatmap of top feature loadings for top PCs.

        Creates a heatmap visualization showing the loadings (contributions)
        of the most variable features across the top principal components.
        This helps interpret which features drive each component.

        Parameters
        ----------
        result : PCAResult
            PCA result containing loadings matrix.
        top_n_features : int, default=20
            Number of top features to display based on variance.
        top_n_components : int, default=10
            Number of top principal components to display.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If loadings matrix is not available in the result.

        Notes
        -----
        The figure is saved as 'pca_loadings_heatmap.{format}' in the
        dimensionality reduction figures subdirectory.

        Features are selected based on their total variance across all
        components. The heatmap uses a diverging colormap (red-blue) to
        show positive (blue) and negative (red) loadings.
        """
        import matplotlib.pyplot as plt

        if result.loadings is None:
            raise ValueError("Loadings matrix is not available in the PCA result.")

        loadings = result.loadings
        feature_variance = np.var(loadings, axis=1)

        top_feature_indices = np.argsort(feature_variance)[-top_n_features:][::-1]
        top_loadings = loadings[top_feature_indices, :top_n_components]

        if result.feature_names is not None:
            feature_labels = [result.feature_names[i] for i in top_feature_indices]
        else:
            feature_labels = [f"Feature {i}" for i in top_feature_indices]

        feature_labels = [
            (label[:20] + "...") if len(label) > 20 else label for label in feature_labels
        ]

        fig, ax = plt.subplots(figsize=(max(6, top_n_components * 0.6), top_n_features * 0.3))

        im = ax.imshow(
            top_loadings,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-np.max(np.abs(top_loadings)),
            vmax=np.max(np.abs(top_loadings)),
        )

        ax.set_xticks(np.arange(top_n_components))
        ax.set_xticklabels([f"PC{i + 1}" for i in range(top_n_components)])
        ax.set_yticks(np.arange(top_n_features))
        ax.set_yticklabels(feature_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(top_n_features):
            for j in range(top_n_components):
                value = top_loadings[i, j]
                text_color = "white" if abs(value) > 0.5 * np.max(np.abs(top_loadings)) else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=self._typography.annotation_size,
                )

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        ax.set_xlabel("Principal Component", fontsize=self._typography.label_size)
        ax.set_ylabel("Feature", fontsize=self._typography.label_size)
        ax.set_title(
            f"PCA Loadings Heatmap ({framework_name})\nTop {top_n_features} Features vs Top {top_n_components} Components",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Loading Value", fontsize=self._typography.label_size)

        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"pca_loadings_heatmap_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_framework_comparison(
        self, scptensor_result: PCAResult, competitor_result: PCAResult, format: str = "png"
    ) -> Path:
        """Generate a side-by-side comparison of PCA variance between frameworks.

        Creates a two-panel figure comparing the explained variance between
        ScpTensor and the competitor framework.

        Parameters
        ----------
        scptensor_result : PCAResult
            PCA result from ScpTensor.
        competitor_result : PCAResult
            PCA result from the competitor framework.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'pca_framework_comparison.{format}' in the
        dimensionality reduction figures subdirectory.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        framework_names = [
            self.FRAMEWORK_DISPLAY_NAMES.get(
                scptensor_result.framework, scptensor_result.framework.title()
            ),
            self.FRAMEWORK_DISPLAY_NAMES.get(
                competitor_result.framework, competitor_result.framework.title()
            ),
        ]
        results = [scptensor_result, competitor_result]

        for ax, result, fw_name in zip([ax1, ax2], results, framework_names, strict=True):
            cumulative_variance = np.cumsum(result.explained_variance_ratio)
            n_components = len(result.explained_variance_ratio)

            x_values = np.arange(1, n_components + 1)
            ax.plot(
                x_values,
                cumulative_variance,
                "o-",
                color=self._colors.primary,
                linewidth=2,
                markersize=4,
                label=fw_name,
            )

            ax.axhline(y=0.9, color=self._colors.error, linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Number of Principal Components", fontsize=self._typography.label_size)
            ax.set_ylabel("Cumulative Variance Explained", fontsize=self._typography.label_size)
            ax.set_title(fw_name, fontsize=self._typography.title_size, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=self._typography.legend_size)
            apply_color_style(ax, "dim_reduction")

        fig.suptitle(
            "PCA Framework Comparison: Cumulative Explained Variance",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"pca_framework_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path


class UMAPDisplay(DisplayBase):
    """Display class for UMAP comparison visualizations.

    Generates publication-quality figures comparing UMAP embeddings between
    ScpTensor and competing frameworks, including side-by-side scatter plots
    and parameter sensitivity analysis.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.dim_reduction import UMAPDisplay, UMAPResult
    >>> display = UMAPDisplay()
    >>> embedding = np.random.randn(100, 2)
    >>> umap_result = UMAPResult(
    ...     framework="scptensor",
    ...     embedding=embedding,
    ...     n_neighbors=15,
    ...     min_dist=0.1,
    ...     metric="euclidean",
    ...     runtime_seconds=1.5
    ... )
    >>> fig_path = display.render_embedding(umap_result, cluster_labels=np.random.randint(0, 3, 100))
    """

    FRAMEWORK_DISPLAY_NAMES: dict[str, str] = {
        "scptensor": "ScpTensor",
        "scanpy": "Scanpy",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "dim_reduction"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("dim_reduction")
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
        be used directly. Use render_embedding or render_embedding_comparison instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "umap_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_embedding or render_embedding_comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_embedding(
        self,
        result: UMAPResult,
        cluster_labels: npt.NDArray[np.int_] | None = None,
        color_values: npt.NDArray[np.float64] | None = None,
        color_label: str = "Value",
        format: str = "png",
    ) -> Path:
        """Generate a UMAP embedding scatter plot.

        Creates a scatter plot of the UMAP embedding, optionally colored by
        cluster labels or continuous values.

        Parameters
        ----------
        result : UMAPResult
            UMAP result containing embedding coordinates.
        cluster_labels : npt.NDArray[np.int_] | None, default=None
            Cluster labels for coloring points. If provided, points are colored
            by discrete cluster assignment.
        color_values : npt.NDArray[np.float64] | None, default=None
            Continuous values for coloring points (e.g., expression level).
            Used if cluster_labels is None.
        color_label : str, default="Value"
            Label for the color bar when using continuous values.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If embedding dimensions are not 2D.

        Notes
        -----
        The figure is saved as 'umap_embedding_{framework}.{format}' in the
        dimensionality reduction figures subdirectory.
        """
        import matplotlib.pyplot as plt

        if result.embedding.shape[1] != 2:
            raise ValueError("UMAP embedding must be 2-dimensional for visualization.")

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            result.framework, result.framework.title()
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels)
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_labels == cluster_id
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
                    result.embedding[mask, 0],
                    result.embedding[mask, 1],
                    c=color,
                    label=f"Cluster {cluster_id}",
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )
            ax.legend(fontsize=self._typography.legend_size, loc="best")
        elif color_values is not None:
            scatter = ax.scatter(
                result.embedding[:, 0],
                result.embedding[:, 1],
                c=color_values,
                cmap="viridis",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_label, fontsize=self._typography.label_size)
        else:
            ax.scatter(
                result.embedding[:, 0],
                result.embedding[:, 1],
                c=self._colors.primary,
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_xlabel("UMAP 1", fontsize=self._typography.label_size)
        ax.set_ylabel("UMAP 2", fontsize=self._typography.label_size)

        title = f"UMAP Embedding ({framework_name})"
        param_text = f"n_neighbors={result.n_neighbors}, min_dist={result.min_dist}"
        ax.set_title(
            f"{title}\n{param_text}", fontsize=self._typography.title_size, fontweight="bold"
        )

        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"umap_embedding_{result.framework}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_embedding_comparison(
        self,
        scptensor_result: UMAPResult,
        competitor_result: UMAPResult,
        cluster_labels: npt.NDArray[np.int_] | None = None,
        format: str = "png",
    ) -> Path:
        """Generate side-by-side UMAP plots comparing ScpTensor vs competitor.

        Creates a two-panel figure showing UMAP embeddings from both frameworks
        for visual comparison of the resulting low-dimensional representations.

        Parameters
        ----------
        scptensor_result : UMAPResult
            UMAP result from ScpTensor.
        competitor_result : UMAPResult
            UMAP result from the competitor framework.
        cluster_labels : npt.NDArray[np.int_] | None, default=None
            Cluster labels for coloring points in both plots.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If embeddings from both frameworks have different shapes.

        Notes
        -----
        The figure is saved as 'umap_comparison.{format}' in the
        dimensionality reduction figures subdirectory.

        The side-by-side comparison allows visual assessment of:
        - Global structure preservation
        - Local neighborhood relationships
        - Cluster separation and cohesion
        """
        import matplotlib.pyplot as plt

        if scptensor_result.embedding.shape != competitor_result.embedding.shape:
            raise ValueError(
                f"Embedding shapes must match: "
                f"{scptensor_result.embedding.shape} vs {competitor_result.embedding.shape}"
            )

        framework_names = [
            self.FRAMEWORK_DISPLAY_NAMES.get(
                scptensor_result.framework, scptensor_result.framework.title()
            ),
            self.FRAMEWORK_DISPLAY_NAMES.get(
                competitor_result.framework, competitor_result.framework.title()
            ),
        ]
        results = [scptensor_result, competitor_result]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, result, fw_name in zip(axes, results, framework_names, strict=True):
            if cluster_labels is not None:
                unique_clusters = np.unique(cluster_labels)
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_labels == cluster_id
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
                        result.embedding[mask, 0],
                        result.embedding[mask, 1],
                        c=color,
                        label=f"C{cluster_id}",
                        alpha=0.6,
                        s=20,
                        edgecolors="none",
                    )
                ax.legend(fontsize=self._typography.legend_size, loc="best")
            else:
                ax.scatter(
                    result.embedding[:, 0],
                    result.embedding[:, 1],
                    c=self._colors.primary,
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

            ax.set_xlabel("UMAP 1", fontsize=self._typography.label_size)
            ax.set_ylabel("UMAP 2", fontsize=self._typography.label_size)
            ax.set_title(fw_name, fontsize=self._typography.title_size, fontweight="bold")
            apply_color_style(ax, "dim_reduction")

        fig.suptitle(
            "UMAP Framework Comparison", fontsize=self._typography.title_size, fontweight="bold"
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"umap_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_parameter_sensitivity(
        self,
        results: list[UMAPResult],
        param_name: str = "n_neighbors",
        cluster_labels: npt.NDArray[np.int_] | None = None,
        format: str = "png",
    ) -> Path:
        """Generate a grid showing UMAP parameter sensitivity.

        Creates a multi-panel figure showing how UMAP embeddings change with
        different parameter values (n_neighbors or min_dist).

        Parameters
        ----------
        results : list[UMAPResult]
            List of UMAP results with varying parameter values.
        param_name : str, default="n_neighbors"
            Name of the parameter that was varied. Options: "n_neighbors", "min_dist".
        cluster_labels : npt.NDArray[np.int_] | None, default=None
            Cluster labels for coloring points.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If param_name is not "n_neighbors" or "min_dist".

        Notes
        -----
        The figure is saved as 'umap_parameter_sensitivity_{param_name}.{format}' in the
        dimensionality reduction figures subdirectory.

        Parameter effects:
        - n_neighbors: Controls local vs global structure balance
        - min_dist: Controls cluster tightness
        """
        import matplotlib.pyplot as plt

        if param_name not in ("n_neighbors", "min_dist"):
            raise ValueError(f"param_name must be 'n_neighbors' or 'min_dist', got '{param_name}'")

        if param_name == "n_neighbors":
            param_values = [r.n_neighbors for r in results]
        else:
            param_values = [r.min_dist for r in results]

        sorted_indices = np.argsort(param_values)
        sorted_results = [results[i] for i in sorted_indices]
        sorted_values = [param_values[i] for i in sorted_indices]

        n_results = len(results)
        n_cols = min(4, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_results == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.ravel()

        for result, param_val, ax in zip(sorted_results, sorted_values, axes_flat, strict=True):
            if cluster_labels is not None:
                unique_clusters = np.unique(cluster_labels)
                for j, cluster_id in enumerate(unique_clusters):
                    mask = cluster_labels == cluster_id
                    color_idx = j % 7
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
                        result.embedding[mask, 0],
                        result.embedding[mask, 1],
                        c=color,
                        alpha=0.6,
                        s=15,
                        edgecolors="none",
                    )
            else:
                ax.scatter(
                    result.embedding[:, 0],
                    result.embedding[:, 1],
                    c=self._colors.primary,
                    alpha=0.6,
                    s=15,
                    edgecolors="none",
                )

            param_label = f"{param_name}={param_val}"
            ax.set_title(param_label, fontsize=self._typography.label_size, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            apply_color_style(ax, "dim_reduction")

        for i in range(n_results, len(axes_flat)):
            axes_flat[i].axis("off")

        framework_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            results[0].framework, results[0].framework.title()
        )

        fig.suptitle(
            f"UMAP Parameter Sensitivity: {param_name} ({framework_name})",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"umap_parameter_sensitivity_{param_name}.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_runtime_comparison(
        self, scptensor_result: UMAPResult, competitor_result: UMAPResult, format: str = "png"
    ) -> Path:
        """Generate a bar chart comparing UMAP runtime between frameworks.

        Creates a simple bar chart showing the runtime comparison for
        UMAP computation between ScpTensor and the competitor framework.

        Parameters
        ----------
        scptensor_result : UMAPResult
            UMAP result from ScpTensor with runtime data.
        competitor_result : UMAPResult
            UMAP result from competitor with runtime data.
        format : str, default="png"
            Output format for the figure. Options: "png", "pdf", "svg".

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If runtime data is not available in either result.

        Notes
        -----
        The figure is saved as 'umap_runtime_comparison.{format}' in the
        dimensionality reduction figures subdirectory.
        """
        import matplotlib.pyplot as plt

        if scptensor_result.runtime_seconds is None or competitor_result.runtime_seconds is None:
            raise ValueError("Runtime data is not available in one or both results.")

        framework_names = [
            self.FRAMEWORK_DISPLAY_NAMES.get(
                scptensor_result.framework, scptensor_result.framework.title()
            ),
            self.FRAMEWORK_DISPLAY_NAMES.get(
                competitor_result.framework, competitor_result.framework.title()
            ),
        ]
        runtimes = [scptensor_result.runtime_seconds, competitor_result.runtime_seconds]

        fig, ax = plt.subplots(figsize=(6, 5))

        x = np.arange(2)
        bars = ax.bar(
            x,
            runtimes,
            color=[self._colors.primary, self._colors.secondary],
            edgecolor=self._colors.neutral,
            alpha=0.8,
        )

        for bar, runtime in zip(bars, runtimes, strict=True):
            ax.annotate(
                f"{runtime:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self._typography.annotation_size,
                fontweight="bold",
            )

        if runtimes[1] > 0:
            speedup = runtimes[1] / runtimes[0]
            ax.annotate(
                f"{speedup:.2f}x speedup",
                xy=(0.5, 0.95),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=self._typography.annotation_size,
                bbox={"boxstyle": "round", "facecolor": self._colors.success, "alpha": 0.3},
            )

        ax.set_ylabel("Runtime (seconds)", fontsize=self._typography.label_size)
        ax.set_title(
            "UMAP Runtime Comparison", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(framework_names)

        apply_color_style(ax, "dim_reduction")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"umap_runtime_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path
