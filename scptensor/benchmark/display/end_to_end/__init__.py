"""End-to-end pipeline display module for ScpTensor benchmark.

Provides visualization capabilities for comparing complete preprocessing pipelines
between ScpTensor and competing frameworks (e.g., Scanpy). This module generates
publication-quality figures showing the full analysis workflow from raw data to
clustering results, including intermediate comparisons and quality metrics.

Classes
-------
EndToEndDisplay
    Visualizes end-to-end pipeline comparison results between ScpTensor and Scanpy.
PipelineResult
    Dataclass for storing complete pipeline execution results.
PipelineStep
    Dataclass for storing individual pipeline step information.
ClusteringMetrics
    Dataclass for storing clustering quality metrics.
IntermediateResults
    Dataclass for storing intermediate pipeline results for each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    "EndToEndDisplay",
    "PipelineResult",
    "PipelineStep",
    "ClusteringMetrics",
    "IntermediateResults",
    "compute_jaccard_index",
]


def compute_jaccard_index(
    labels_a: npt.NDArray[np.int_],
    labels_b: npt.NDArray[np.int_],
) -> float:
    """Compute Jaccard index between two cluster labelings.

    The Jaccard index measures the similarity between two sets of clusters.
    For each pair of samples, check if they are in the same cluster in both
    labelings. The Jaccard index is the ratio of agreements to total pairs.

    Parameters
    ----------
    labels_a : npt.NDArray[np.int_]
        First clustering labels.
    labels_b : npt.NDArray[np.int_]
        Second clustering labels.

    Returns
    -------
    float
        Jaccard index between 0 and 1, where 1 indicates perfect agreement.

    Notes
    -----
    The Jaccard index is defined as:
        J = |A intersect B| / |A union B|
    where A is the set of sample pairs in the same cluster in labels_a,
    and B is the set of sample pairs in the same cluster in labels_b.

    Examples
    --------
    >>> import numpy as np
    >>> labels_a = np.array([0, 0, 1, 1])
    >>> labels_b = np.array([0, 0, 1, 1])
    >>> compute_jaccard_index(labels_a, labels_b)
    1.0
    >>> labels_c = np.array([1, 1, 0, 0])
    >>> compute_jaccard_index(labels_a, labels_c)
    1.0
    """
    if len(labels_a) != len(labels_b):
        msg = f"Label arrays must have same length: {len(labels_a)} vs {len(labels_b)}"
        raise ValueError(msg)

    n = len(labels_a)
    if n < 2:
        return 1.0

    same_a = np.zeros((n, n), dtype=bool)
    same_b = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            same_a[i, j] = labels_a[i] == labels_a[j]
            same_b[i, j] = labels_b[i] == labels_b[j]

    intersection = np.sum(same_a & same_b)
    union = np.sum(same_a | same_b)

    if union == 0:
        return 1.0

    return float(intersection / union)


@dataclass(frozen=True)
class PipelineStep:
    """Information about a single step in the preprocessing pipeline.

    Attributes
    ----------
    name : str
        Name of the preprocessing step (e.g., "qc", "normalization", "imputation").
    display_name : str
        Human-readable display name for the step.
    method_name : str | None
        Name of the specific method used for this step.
    parameters : dict[str, object] | None
        Parameters used for this step.
    runtime_seconds : float | None
        Runtime of this step in seconds.
    memory_mb : float | None
        Memory usage of this step in megabytes.
    """

    name: str
    display_name: str
    method_name: str | None = None
    parameters: dict[str, object] | None = None
    runtime_seconds: float | None = None
    memory_mb: float | None = None


@dataclass(frozen=True)
class ClusteringMetrics:
    """Quality metrics for clustering results.

    Attributes
    ----------
    silhouette_score : float | None
        Silhouette score measuring cluster separation and cohesion.
        Higher values indicate better-defined clusters. Range: [-1, 1].
    davies_bouldin_score : float | None
        Davies-Bouldin index measuring cluster separation.
        Lower values indicate better clustering. Range: [0, inf).
    calinski_harabasz_score : float | None
        Calinski-Harabasz index (variance ratio criterion).
        Higher values indicate better clustering.
    ari_score : float | None
        Adjusted Rand Index comparing clustering to ground truth (if available).
        Range: [-1, 1], where 1 indicates perfect agreement.
    nmi_score : float | None
        Normalized Mutual Information comparing clustering to ground truth.
        Range: [0, 1], where 1 indicates perfect agreement.
    n_clusters : int
        Number of clusters identified.
    """

    silhouette_score: float | None = None
    davies_bouldin_score: float | None = None
    calinski_harabasz_score: float | None = None
    ari_score: float | None = None
    nmi_score: float | None = None
    n_clusters: int = 0


@dataclass(frozen=True)
class IntermediateResults:
    """Intermediate results after each preprocessing step.

    Attributes
    ----------
    step_name : str
        Name of the preprocessing step.
    n_cells : int
        Number of cells after this step.
    n_features : int
        Number of features after this step.
    sparsity : float
        Proportion of missing values in the data.
    total_runtime : float
        Cumulative runtime up to this step in seconds.
    """

    step_name: str
    n_cells: int
    n_features: int
    sparsity: float
    total_runtime: float


@dataclass(frozen=True)
class PipelineResult:
    """Result of a complete end-to-end preprocessing pipeline execution.

    Attributes
    ----------
    framework : str
        Name of the framework used (e.g., "scptensor", "scanpy").
    pipeline_steps : list[PipelineStep]
        Ordered list of preprocessing steps in the pipeline.
    umap_embedding : npt.NDArray[np.float64] | None
        UMAP embedding coordinates of shape (n_samples, 2).
    cluster_labels : npt.NDArray[np.int_] | None
        Cluster labels for each sample.
    clustering_metrics : ClusteringMetrics | None
        Quality metrics for the clustering results.
    intermediate_results : list[IntermediateResults]
        Intermediate results after each preprocessing step.
    total_runtime : float
        Total runtime of the complete pipeline in seconds.
    total_memory_mb : float
        Total memory usage of the complete pipeline in megabytes.
    dataset_name : str
        Name of the dataset used for the pipeline.
    """

    framework: str
    pipeline_steps: list[PipelineStep]
    umap_embedding: npt.NDArray[np.float64] | None = None
    cluster_labels: npt.NDArray[np.int_] | None = None
    clustering_metrics: ClusteringMetrics | None = None
    intermediate_results: list[IntermediateResults] = field(default_factory=list)
    total_runtime: float = 0.0
    total_memory_mb: float = 0.0
    dataset_name: str = "unknown"


class EndToEndDisplay(DisplayBase):
    """Display class for end-to-end pipeline comparison visualizations.

    Generates publication-quality figures comparing complete preprocessing
    pipelines between ScpTensor and competing frameworks, including pipeline
    flowcharts, intermediate result comparisons, clustering visualizations,
    and quality metrics.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated figures.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.display.end_to_end import (
    ...     EndToEndDisplay,
    ...     PipelineResult,
    ...     PipelineStep,
    ...     ClusteringMetrics,
    ...     IntermediateResults,
    ... )
    >>> display = EndToEndDisplay()
    >>> steps = [
    ...     PipelineStep(name="qc", display_name="Quality Control"),
    ...     PipelineStep(name="normalization", display_name="Normalization"),
    ...     PipelineStep(name="imputation", display_name="Imputation"),
    ... ]
    >>> result = PipelineResult(
    ...     framework="scptensor",
    ...     pipeline_steps=steps,
    ...     umap_embedding=np.random.randn(100, 2),
    ...     cluster_labels=np.random.randint(0, 3, 100),
    ...     clustering_metrics=ClusteringMetrics(
    ...         silhouette_score=0.5,
    ...         davies_bouldin_score=0.8,
    ...         ari_score=0.7,
    ...         n_clusters=3
    ...     ),
    ...     total_runtime=10.5
    ... )
    >>> fig_path = display.render_clustering_comparison(result, result)
    """

    FRAMEWORK_DISPLAY_NAMES: dict[str, str] = {
        "scptensor": "ScpTensor",
        "scanpy": "Scanpy",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        super().__init__(output_dir=output_dir)
        self._figures_dir = self.output_dir / "figures" / "end_to_end"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._colors = get_module_colors("end_to_end")
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
        be used directly. Use the specific render methods instead.
        """
        import matplotlib.pyplot as plt

        output_path = self._figures_dir / "end_to_end_summary.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Use render_pipeline_comparison, render_intermediate_comparison,\n"
            "render_clustering_comparison, render_cluster_overlap_analysis,\n"
            "or render_quality_metrics_comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self._typography.annotation_size,
        )
        apply_color_style(ax, "end_to_end")
        apply_typography_theme(fig, self._typography)
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        return output_path

    def render_pipeline_comparison(
        self,
        scptensor_result: PipelineResult,
        competitor_result: PipelineResult,
        format: Literal["png", "pdf", "svg"] = "png",
    ) -> Path:
        """Generate a flowchart showing both pipelines side by side.

        Creates a visual flowchart comparing the preprocessing pipelines
        of ScpTensor and the competitor framework, showing all steps with
        their methods and execution times.

        Parameters
        ----------
        scptensor_result : PipelineResult
            Pipeline result from ScpTensor.
        competitor_result : PipelineResult
            Pipeline result from the competitor framework.
        format : {"png", "pdf", "svg"}, default="png"
            Output format for the figure.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'pipeline_comparison.{format}' in the
        end-to-end figures subdirectory.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        scptensor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            scptensor_result.framework, scptensor_result.framework.title()
        )
        competitor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            competitor_result.framework, competitor_result.framework.title()
        )

        n_steps = max(
            len(scptensor_result.pipeline_steps),
            len(competitor_result.pipeline_steps),
        )
        max_steps = max(n_steps, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max_steps * 1.2 + 2))

        frameworks = [
            (scptensor_result, scptensor_name, ax1),
            (competitor_result, competitor_name, ax2),
        ]

        for result, fw_name, ax in frameworks:
            ax.set_xlim(0, 3)
            ax.set_ylim(max_steps + 1, -1)
            ax.axis("off")
            ax.set_title(fw_name, fontsize=self._typography.title_size, fontweight="bold", pad=10)

            fw_color = (
                self._colors.primary if result.framework == "scptensor" else self._colors.secondary
            )
            ax.text(
                1.5,
                max_steps + 0.5,
                f"Total: {result.total_runtime:.2f}s",
                ha="center",
                va="center",
                fontsize=self._typography.label_size,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": fw_color, "alpha": 0.3},
            )

            for i, step in enumerate(result.pipeline_steps):
                y_pos = max_steps - i - 0.5

                rect = mpatches.FancyBboxPatch(
                    (0.2, y_pos - 0.35),
                    2.6,
                    0.7,
                    boxstyle="round,pad=0.1",
                    edgecolor=fw_color,
                    facecolor=fw_color,
                    alpha=0.2,
                )
                ax.add_patch(rect)

                ax.text(
                    0.5,
                    y_pos,
                    step.display_name,
                    ha="left",
                    va="center",
                    fontsize=self._typography.label_size,
                    fontweight="bold",
                )

                info_text = ""
                if step.method_name:
                    info_text = step.method_name
                if step.runtime_seconds is not None:
                    if info_text:
                        info_text += f" ({step.runtime_seconds:.2f}s)"
                    else:
                        info_text = f"{step.runtime_seconds:.2f}s"

                if info_text:
                    ax.text(
                        0.5,
                        y_pos - 0.15,
                        info_text,
                        ha="left",
                        va="center",
                        fontsize=self._typography.annotation_size,
                        style="italic",
                        color="#555555",
                    )

                if i < len(result.pipeline_steps) - 1:
                    ax.annotate(
                        "",
                        xy=(1.5, y_pos - 0.45),
                        xytext=(1.5, y_pos - 0.75),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": fw_color,
                            "lw": 2,
                        },
                    )

        fig.suptitle(
            "End-to-End Pipeline Comparison",
            fontsize=self._typography.title_size,
            fontweight="bold",
            y=0.98,
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"pipeline_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_intermediate_comparison(
        self,
        scptensor_result: PipelineResult,
        competitor_result: PipelineResult,
        format: Literal["png", "pdf", "svg"] = "png",
    ) -> Path:
        """Generate a table showing results after each step.

        Creates a visualization comparing intermediate results after each
        preprocessing step, including cell counts, feature counts, sparsity,
        and cumulative runtime.

        Parameters
        ----------
        scptensor_result : PipelineResult
            Pipeline result from ScpTensor.
        competitor_result : PipelineResult
            Pipeline result from the competitor framework.
        format : {"png", "pdf", "svg"}, default="png"
            Output format for the figure.

        Returns
        -------
        Path
            Path to the saved figure file.

        Notes
        -----
        The figure is saved as 'intermediate_comparison.{format}' in the
        end-to-end figures subdirectory.
        """
        import matplotlib.pyplot as plt

        scptensor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            scptensor_result.framework, scptensor_result.framework.title()
        )
        competitor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            competitor_result.framework, competitor_result.framework.title()
        )

        scptensor_int = scptensor_result.intermediate_results
        competitor_int = competitor_result.intermediate_results

        if not scptensor_int and not competitor_int:
            output_path = self._figures_dir / f"intermediate_comparison.{format}"
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5,
                0.5,
                "No intermediate results available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self._typography.label_size,
            )
            apply_color_style(ax, "end_to_end")
            apply_typography_theme(fig, self._typography)
            plt.savefig(
                output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
            )
            plt.close(fig)
            return output_path

        step_names = []
        scptensor_cells = []
        scptensor_features = []
        scptensor_sparsity = []
        scptensor_time = []
        competitor_cells = []
        competitor_features = []
        competitor_sparsity = []
        competitor_time = []

        all_steps = set()
        for res in scptensor_int:
            all_steps.add(res.step_name)
        for res in competitor_int:
            all_steps.add(res.step_name)
        step_names = sorted(all_steps)

        for step_name in step_names:
            scptensor_data = next((r for r in scptensor_int if r.step_name == step_name), None)
            competitor_data = next((r for r in competitor_int if r.step_name == step_name), None)

            if scptensor_data:
                scptensor_cells.append(scptensor_data.n_cells)
                scptensor_features.append(scptensor_data.n_features)
                scptensor_sparsity.append(scptensor_data.sparsity * 100)
                scptensor_time.append(scptensor_data.total_runtime)
            else:
                scptensor_cells.append(0)
                scptensor_features.append(0)
                scptensor_sparsity.append(0)
                scptensor_time.append(0)

            if competitor_data:
                competitor_cells.append(competitor_data.n_cells)
                competitor_features.append(competitor_data.n_features)
                competitor_sparsity.append(competitor_data.sparsity * 100)
                competitor_time.append(competitor_data.total_runtime)
            else:
                competitor_cells.append(0)
                competitor_features.append(0)
                competitor_sparsity.append(0)
                competitor_time.append(0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        x = np.arange(len(step_names))
        width = 0.35

        ax = axes[0, 0]
        ax.bar(
            x - width / 2,
            scptensor_cells,
            width,
            label=scptensor_name,
            color=self._colors.primary,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            competitor_cells,
            width,
            label=competitor_name,
            color=self._colors.secondary,
            alpha=0.8,
        )
        ax.set_ylabel("Number of Cells", fontsize=self._typography.label_size)
        ax.set_title(
            "Cell Count After Each Step", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in step_names], rotation=45, ha="right"
        )
        ax.legend(fontsize=self._typography.legend_size)
        apply_color_style(ax, "end_to_end")

        ax = axes[0, 1]
        ax.bar(
            x - width / 2,
            scptensor_features,
            width,
            label=scptensor_name,
            color=self._colors.primary,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            competitor_features,
            width,
            label=competitor_name,
            color=self._colors.secondary,
            alpha=0.8,
        )
        ax.set_ylabel("Number of Features", fontsize=self._typography.label_size)
        ax.set_title(
            "Feature Count After Each Step", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in step_names], rotation=45, ha="right"
        )
        ax.legend(fontsize=self._typography.legend_size)
        apply_color_style(ax, "end_to_end")

        ax = axes[1, 0]
        ax.bar(
            x - width / 2,
            scptensor_sparsity,
            width,
            label=scptensor_name,
            color=self._colors.primary,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            competitor_sparsity,
            width,
            label=competitor_name,
            color=self._colors.secondary,
            alpha=0.8,
        )
        ax.set_ylabel("Sparsity (%)", fontsize=self._typography.label_size)
        ax.set_title(
            "Missing Data Proportion", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in step_names], rotation=45, ha="right"
        )
        ax.legend(fontsize=self._typography.legend_size)
        apply_color_style(ax, "end_to_end")

        ax = axes[1, 1]
        ax.bar(
            x - width / 2,
            scptensor_time,
            width,
            label=scptensor_name,
            color=self._colors.primary,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            competitor_time,
            width,
            label=competitor_name,
            color=self._colors.secondary,
            alpha=0.8,
        )
        ax.set_ylabel("Cumulative Runtime (s)", fontsize=self._typography.label_size)
        ax.set_title(
            "Runtime Accumulation", fontsize=self._typography.title_size, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in step_names], rotation=45, ha="right"
        )
        ax.legend(fontsize=self._typography.legend_size)
        apply_color_style(ax, "end_to_end")

        fig.suptitle(
            "Intermediate Results Comparison",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"intermediate_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_clustering_comparison(
        self,
        scptensor_result: PipelineResult,
        competitor_result: PipelineResult,
        format: Literal["png", "pdf", "svg"] = "png",
    ) -> Path:
        """Generate UMAP plots colored by cluster for both frameworks.

        Creates side-by-side UMAP plots showing clustering results from
        both frameworks for visual comparison.

        Parameters
        ----------
        scptensor_result : PipelineResult
            Pipeline result from ScpTensor with UMAP embedding and cluster labels.
        competitor_result : PipelineResult
            Pipeline result from competitor with UMAP embedding and cluster labels.
        format : {"png", "pdf", "svg"}, default="png"
            Output format for the figure.

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If UMAP embeddings or cluster labels are not available in either result.

        Notes
        -----
        The figure is saved as 'clustering_comparison.{format}' in the
        end-to-end figures subdirectory.
        """
        import matplotlib.pyplot as plt

        if scptensor_result.umap_embedding is None or competitor_result.umap_embedding is None:
            msg = "UMAP embeddings are required for clustering comparison"
            raise ValueError(msg)

        scptensor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            scptensor_result.framework, scptensor_result.framework.title()
        )
        competitor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            competitor_result.framework, competitor_result.framework.title()
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        frameworks = [
            (scptensor_result, scptensor_name, axes[0]),
            (competitor_result, competitor_name, axes[1]),
        ]

        for result, fw_name, ax in frameworks:
            embedding = result.umap_embedding
            labels = result.cluster_labels

            if embedding is None:
                msg = f"UMAP embedding is not available for {fw_name}"
                raise ValueError(msg)

            if embedding.shape[1] != 2:
                msg = f"UMAP embedding must be 2D, got shape {embedding.shape}"
                raise ValueError(msg)

            if labels is not None:
                unique_clusters = np.unique(labels)
                for i, cluster_id in enumerate(unique_clusters):
                    mask = labels == cluster_id
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
                        embedding[mask, 0],
                        embedding[mask, 1],
                        c=color,
                        label=f"C{cluster_id}",
                        alpha=0.6,
                        s=20,
                        edgecolors="none",
                    )
                ax.legend(fontsize=self._typography.legend_size, loc="best", ncol=2)

                n_clusters = len(unique_clusters)
                ax.text(
                    0.02,
                    0.98,
                    f"Clusters: {n_clusters}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    fontsize=self._typography.annotation_size,
                )
            else:
                ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=self._colors.primary,
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

            ax.set_xlabel("UMAP 1", fontsize=self._typography.label_size)
            ax.set_ylabel("UMAP 2", fontsize=self._typography.label_size)
            ax.set_title(fw_name, fontsize=self._typography.title_size, fontweight="bold")
            apply_color_style(ax, "end_to_end")

        fig.suptitle(
            "Clustering Results Comparison",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"clustering_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_cluster_overlap_analysis(
        self,
        scptensor_result: PipelineResult,
        competitor_result: PipelineResult,
        format: Literal["png", "pdf", "svg"] = "png",
    ) -> Path:
        """Generate visualization of cluster overlap using Jaccard index.

        Creates a heatmap showing the pairwise Jaccard index between clusters
        from both frameworks, indicating how similar the cluster assignments are.

        Parameters
        ----------
        scptensor_result : PipelineResult
            Pipeline result from ScpTensor with cluster labels.
        competitor_result : PipelineResult
            Pipeline result from competitor with cluster labels.
        format : {"png", "pdf", "svg"}, default="png"
            Output format for the figure.

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If cluster labels are not available in either result.

        Notes
        -----
        The figure is saved as 'cluster_overlap_analysis.{format}' in the
        end-to-end figures subdirectory.

        The Jaccard index measures the similarity between two sets:
        J = |A intersect B| / |A union B|
        Values range from 0 (no overlap) to 1 (identical clusters).
        """
        import matplotlib.pyplot as plt

        if scptensor_result.cluster_labels is None or competitor_result.cluster_labels is None:
            msg = "Cluster labels are required for overlap analysis"
            raise ValueError(msg)

        scptensor_labels = scptensor_result.cluster_labels
        competitor_labels = competitor_result.cluster_labels

        scptensor_clusters = np.unique(scptensor_labels)
        competitor_clusters = np.unique(competitor_labels)

        jaccard_matrix = np.zeros((len(scptensor_clusters), len(competitor_clusters)))

        for i, sc_cluster in enumerate(scptensor_clusters):
            for j, comp_cluster in enumerate(competitor_clusters):
                sc_binary = (scptensor_labels == sc_cluster).astype(int)
                comp_binary = (competitor_labels == comp_cluster).astype(int)

                intersection = np.sum(sc_binary & comp_binary)
                union = np.sum(sc_binary | comp_binary)

                if union > 0:
                    jaccard_matrix[i, j] = intersection / union
                else:
                    jaccard_matrix[i, j] = 0.0

        scptensor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            scptensor_result.framework, scptensor_result.framework.title()
        )
        competitor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            competitor_result.framework, competitor_result.framework.title()
        )

        fig, ax = plt.subplots(
            figsize=(max(6, len(competitor_clusters) * 0.8), max(5, len(scptensor_clusters) * 0.5))
        )

        im = ax.imshow(
            jaccard_matrix,
            cmap="YlOrRd",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(np.arange(len(competitor_clusters)))
        ax.set_yticks(np.arange(len(scptensor_clusters)))
        ax.set_xticklabels([f"{competitor_name} C{c}" for c in competitor_clusters])
        ax.set_yticklabels([f"{scptensor_name} C{c}" for c in scptensor_clusters])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(scptensor_clusters)):
            for j in range(len(competitor_clusters)):
                value = jaccard_matrix[i, j]
                text_color = "white" if value > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=self._typography.annotation_size,
                )

        ax.set_xlabel(f"{competitor_name} Clusters", fontsize=self._typography.label_size)
        ax.set_ylabel(f"{scptensor_name} Clusters", fontsize=self._typography.label_size)
        ax.set_title(
            "Cluster Overlap Analysis (Jaccard Index)",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Jaccard Index", fontsize=self._typography.label_size)

        overall_jaccard = compute_jaccard_index(scptensor_labels, competitor_labels)
        ax.text(
            0.5,
            -0.15,
            f"Overall Cluster Similarity (Jaccard): {overall_jaccard:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=self._typography.label_size,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": self._colors.success, "alpha": 0.3},
        )

        apply_color_style(ax, "end_to_end")
        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"cluster_overlap_analysis.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path

    def render_quality_metrics_comparison(
        self,
        scptensor_result: PipelineResult,
        competitor_result: PipelineResult,
        format: Literal["png", "pdf", "svg"] = "png",
    ) -> Path:
        """Generate bar chart comparing clustering quality metrics.

        Creates a grouped bar chart comparing Silhouette score, Davies-Bouldin
        index, and Adjusted Rand Index between ScpTensor and the competitor.

        Parameters
        ----------
        scptensor_result : PipelineResult
            Pipeline result from ScpTensor with clustering metrics.
        competitor_result : PipelineResult
            Pipeline result from competitor with clustering metrics.
        format : {"png", "pdf", "svg"}, default="png"
            Output format for the figure.

        Returns
        -------
        Path
            Path to the saved figure file.

        Raises
        ------
        ValueError
            If clustering metrics are not available in either result.

        Notes
        -----
        The figure is saved as 'quality_metrics_comparison.{format}' in the
        end-to-end figures subdirectory.

        Metrics explained:
        - Silhouette Score: Higher is better (range: -1 to 1)
        - Davies-Bouldin Index: Lower is better (range: 0 to inf)
        - Calinski-Harabasz Index: Higher is better (range: 0 to inf)
        - ARI (Adjusted Rand Index): Higher is better (range: -1 to 1)
        """
        import matplotlib.pyplot as plt

        if (
            scptensor_result.clustering_metrics is None
            or competitor_result.clustering_metrics is None
        ):
            msg = "Clustering metrics are required for quality comparison"
            raise ValueError(msg)

        sc_metrics = scptensor_result.clustering_metrics
        comp_metrics = competitor_result.clustering_metrics

        metrics_data: list[tuple[str, float, float, str, bool]] = []

        if sc_metrics.silhouette_score is not None and comp_metrics.silhouette_score is not None:
            metrics_data.append(
                (
                    "Silhouette",
                    sc_metrics.silhouette_score,
                    comp_metrics.silhouette_score,
                    "Higher is better",
                    True,
                )
            )

        if (
            sc_metrics.davies_bouldin_score is not None
            and comp_metrics.davies_bouldin_score is not None
        ):
            metrics_data.append(
                (
                    "Davies-Bouldin",
                    sc_metrics.davies_bouldin_score,
                    comp_metrics.davies_bouldin_score,
                    "Lower is better",
                    False,
                )
            )

        if (
            sc_metrics.calinski_harabasz_score is not None
            and comp_metrics.calinski_harabasz_score is not None
        ):
            metrics_data.append(
                (
                    "Calinski-Harabasz",
                    sc_metrics.calinski_harabasz_score,
                    comp_metrics.calinski_harabasz_score,
                    "Higher is better",
                    True,
                )
            )

        if sc_metrics.ari_score is not None and comp_metrics.ari_score is not None:
            metrics_data.append(
                (
                    "ARI",
                    sc_metrics.ari_score,
                    comp_metrics.ari_score,
                    "Higher is better",
                    True,
                )
            )

        if sc_metrics.nmi_score is not None and comp_metrics.nmi_score is not None:
            metrics_data.append(
                (
                    "NMI",
                    sc_metrics.nmi_score,
                    comp_metrics.nmi_score,
                    "Higher is better",
                    True,
                )
            )

        if not metrics_data:
            msg = "No overlapping metrics available for comparison"
            raise ValueError(msg)

        scptensor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            scptensor_result.framework, scptensor_result.framework.title()
        )
        competitor_name = self.FRAMEWORK_DISPLAY_NAMES.get(
            competitor_result.framework, competitor_result.framework.title()
        )

        n_metrics = len(metrics_data)
        fig, ax = plt.subplots(figsize=(10, max(5, n_metrics * 1.2)))

        x = np.arange(n_metrics)
        width = 0.35

        sc_values = [m[1] for m in metrics_data]
        comp_values = [m[2] for m in metrics_data]
        metric_names = [m[0] for m in metrics_data]

        max_values = [max(sv, cv) for sv, cv in zip(sc_values, comp_values, strict=True)]
        normalized_sc = [
            sv / mv if mv > 0 else 0 for sv, mv in zip(sc_values, max_values, strict=True)
        ]
        normalized_comp = [
            cv / mv if mv > 0 else 0 for cv, mv in zip(comp_values, max_values, strict=True)
        ]

        for i, (_, _, _, _, higher_better) in enumerate(metrics_data):
            if not higher_better:
                normalized_sc[i] = 1 - normalized_sc[i] + (1.0 / (1 + sc_values[i])) * 0.1
                normalized_comp[i] = 1 - normalized_comp[i] + (1.0 / (1 + comp_values[i])) * 0.1

        bars1 = ax.barh(
            x - width / 2,
            sc_values,
            width,
            label=scptensor_name,
            color=self._colors.primary,
            alpha=0.8,
        )
        bars2 = ax.barh(
            x + width / 2,
            comp_values,
            width,
            label=competitor_name,
            color=self._colors.secondary,
            alpha=0.8,
        )

        for bars, values in zip([bars1, bars2], [sc_values, comp_values], strict=True):
            for bar, val in zip(bars, values, strict=True):
                ax.text(
                    bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    ha="left",
                    va="center",
                    fontsize=self._typography.annotation_size,
                )

        ax.set_xlabel("Score Value", fontsize=self._typography.label_size)
        ax.set_yticks(x)
        ax.set_yticklabels(metric_names)
        ax.set_title(
            "Clustering Quality Metrics Comparison",
            fontsize=self._typography.title_size,
            fontweight="bold",
        )
        ax.legend(fontsize=self._typography.legend_size, loc="lower right")

        apply_color_style(ax, "end_to_end")

        best_count_sc = sum(
            1
            for i, (_, sv, cv, _, higher_better) in enumerate(metrics_data)
            if (higher_better and sv > cv) or (not higher_better and sv < cv)
        )
        best_count_comp = n_metrics - best_count_sc

        ax.text(
            0.98,
            0.02,
            f"{scptensor_name}: {best_count_sc}/{n_metrics}\n"
            f"{competitor_name}: {best_count_comp}/{n_metrics}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=self._typography.annotation_size,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )

        apply_typography_theme(fig, self._typography)
        apply_layout_config(fig)

        output_path = self._figures_dir / f"quality_metrics_comparison.{format}"
        plt.savefig(
            output_path, dpi=300.0, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)

        return output_path
