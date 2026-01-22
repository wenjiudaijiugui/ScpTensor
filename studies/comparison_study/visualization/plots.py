"""
Comprehensive visualization module for pipeline comparison.

This module provides functions to generate all figures needed for the
technical comparison report, including batch effects, performance,
distribution, and structure preservation visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from math import pi
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Import scienceplots to register styles
try:
    import scienceplots  # noqa: F401

    # Apply science plots style
    plt.style.use(["science", "no-latex"])
except ImportError:
    # Fallback to default style if scienceplots not available
    plt.style.use("default")


# Pipeline ID mapping
# Maps generic pipeline identifiers to actual pipeline names used in result keys
# These match the names in pipeline_configs.yaml after .replace(" ", "_").lower()
PIPELINE_IDS: dict[str, str] = {
    "pipeline_a": "classic_pipeline",
    "pipeline_b": "batch_correction_pipeline",
    "pipeline_c": "advanced_pipeline",
    "pipeline_d": "performance-optimized_pipeline",  # Note: hyphen from "Performance-Optimized"
    "pipeline_e": "conservative_pipeline",
}


def _extract_metric_value(metric_data: Any) -> float:
    """
    Extract metric value, handling both aggregated and non-aggregated formats.

    Aggregated format (from repeat runs):
        {"mean": 0.46, "std": 0.0, "min": 0.46, "max": 0.46}
    Non-aggregated format (single run):
        0.46

    Parameters
    ----------
    metric_data : Any
        Either a float (non-aggregated) or dict with 'mean' key (aggregated)

    Returns
    -------
    float
        The metric value (mean if aggregated, direct value otherwise)

    Examples
    --------
    >>> _extract_metric_value(0.46)
    0.46
    >>> _extract_metric_value({"mean": 0.46, "std": 0.0})
    0.46
    >>> _extract_metric_value(None)
    0.0
    """
    if metric_data is None:
        return 0.0
    if isinstance(metric_data, dict):
        return float(metric_data.get("mean", 0))
    return float(metric_data)


def _make_result_key(
    dataset: str, pipeline_id: str, repeat: int = 0, *, use_aggregated: bool = False
) -> str:
    """
    Generate the correct result key format for accessing evaluation results.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'small', 'medium', 'large')
    pipeline_id : str
        Pipeline identifier (e.g., 'classic_pipeline', 'batch_correction_pipeline')
    repeat : int, default 0
        Repeat number
    use_aggregated : bool, default False
        If True, generate key for aggregated results (no repeat suffix)

    Returns
    -------
    str
        Result key (e.g., 'small_classic_pipeline_r0' or 'small_classic_pipeline')

    Examples
    --------
    >>> _make_result_key('small', 'classic_pipeline', 0)
    'small_classic_pipeline_r0'
    >>> _make_result_key('small', 'classic_pipeline', use_aggregated=True)
    'small_classic_pipeline'
    """
    if use_aggregated:
        return f"{dataset}_{pipeline_id}"
    return f"{dataset}_{pipeline_id}_r{repeat}"


class ComparisonPlotter:
    """
    Main plotting class for pipeline comparison visualizations.

    Parameters
    ----------
    config : Mapping[str, Any]
        Visualization configuration from evaluation_config.yaml
    output_dir : str, default "outputs/figures"
        Directory to save generated figures
    dpi : int, default 300
        Figure DPI (300 for publication quality)

    Examples
    --------
    >>> config = {"visualization": {"figure": {"colors": {...}}, "font": {...}}}
    >>> plotter = ComparisonPlotter(config)
    >>> path = plotter.plot_batch_effects_comparison(results)
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: str = "outputs/figures",
        dpi: int = 300,
    ) -> None:
        """Initialize the plotter with configuration."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Get color scheme
        self.colors = self._get_colors()

        # Get font settings
        self.font_settings = self._get_font_settings()

    def _get_colors(self) -> dict[str, str]:
        """
        Get pipeline color scheme from configuration.

        Returns
        -------
        Dict[str, str]
            Color mapping for each pipeline
        """
        return (
            self.config.get("visualization", {})
            .get("figure", {})
            .get(
                "colors",
                {
                    "pipeline_a": "#1f77b4",
                    "pipeline_b": "#ff7f0e",
                    "pipeline_c": "#2ca02c",
                    "pipeline_d": "#d62728",
                    "pipeline_e": "#9467bd",
                },
            )
        )

    def _get_font_settings(self) -> dict[str, Any]:
        """
        Get font settings from configuration.

        Returns
        -------
        Dict[str, Any]
            Font configuration with family, title_size, label_size, legend_size
        """
        font_config = self.config.get("visualization", {}).get("font", {})
        return {
            "family": font_config.get("family", "Arial"),
            "title_size": font_config.get("title_size", 16),
            "label_size": font_config.get("label_size", 12),
            "legend_size": font_config.get("legend_size", 10),
        }

    def plot_batch_effects_comparison(
        self,
        results: Mapping[str, Any],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create comprehensive batch effects comparison figure.

        Creates a 2x2 subplot figure showing:
        - kBET scores
        - LISI scores
        - Mixing entropy
        - Variance ratio

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Extract data
        datasets = ["small", "medium", "large"]
        pipelines = [
            "pipeline_a",
            "pipeline_b",
            "pipeline_c",
            "pipeline_d",
            "pipeline_e",
        ]

        metrics = ["kbet", "lisi", "mixing_entropy", "variance_ratio"]
        metric_labels = ["kBET Score", "LISI Score", "Mixing Entropy", "Variance Ratio"]

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
            ax = axes[idx // 2, idx % 2]

            # Prepare data for plotting
            data: list[dict[str, Any]] = []
            for pipeline in pipelines:
                pipeline_id = PIPELINE_IDS[pipeline]
                for dataset in datasets:
                    key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                    if key in results:
                        batch_effects = results[key].get("batch_effects", {})
                        value = batch_effects.get(metric, 0)
                        data.append(
                            {
                                "Pipeline": pipeline.replace("pipeline_", "Pipeline ").upper(),
                                "Dataset": dataset.capitalize(),
                                "Value": value,
                            }
                        )

            # Create grouped bar plot
            if data:
                import pandas as pd

                df = pd.DataFrame(data)

                # Pivot for grouped bar plot
                pivot_df = df.pivot(index="Pipeline", columns="Dataset", values="Value")

                # Plot
                x = np.arange(len(pivot_df.index))
                width = 0.25

                for i, dataset in enumerate(datasets):
                    ax.bar(
                        x + i * width,
                        pivot_df[dataset.capitalize()],
                        width,
                        label=dataset.capitalize(),
                        color=self._get_dataset_color(i),
                        alpha=0.8,
                    )

                ax.set_xlabel("Pipeline", fontsize=self.font_settings["label_size"])
                ax.set_ylabel(label, fontsize=self.font_settings["label_size"])
                ax.set_title(
                    label,
                    fontsize=self.font_settings["title_size"],
                    fontweight="bold",
                )
                ax.set_xticks(x + width)
                ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
                ax.legend(fontsize=self.font_settings["legend_size"])
                ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "batch_effects_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_performance_comparison(
        self,
        results: Mapping[str, Any],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create computational performance comparison figure.

        Shows runtime and memory usage for all pipelines across datasets.

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract performance data
        datasets = ["small", "medium", "large"]
        pipelines = [
            "pipeline_a",
            "pipeline_b",
            "pipeline_c",
            "pipeline_d",
            "pipeline_e",
        ]

        # Runtime plot
        runtime_data: dict[str, list[float]] = {}
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            runtime_data[pipeline] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    perf = results[key].get("performance", {})
                    runtime_value = perf.get("runtime_seconds", 0)
                    runtime = _extract_metric_value(runtime_value)
                    runtime_data[pipeline].append(runtime)
                else:
                    # Fill with 0 if data missing
                    runtime_data[pipeline].append(0.0)

        # Memory plot
        memory_data: dict[str, list[float]] = {}
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            memory_data[pipeline] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    perf = results[key].get("performance", {})
                    memory_value = perf.get("memory_gb", 0)
                    memory = _extract_metric_value(memory_value)
                    memory_data[pipeline].append(memory)
                else:
                    # Fill with 0 if data missing
                    memory_data[pipeline].append(0.0)

        # Plot runtime
        x = np.arange(len(datasets))
        width = 0.15

        for i, pipeline in enumerate(pipelines):
            pipeline_name = pipeline.replace("pipeline_", "Pipeline ").upper()
            color = self.colors.get(pipeline, "#333333")
            ax1.bar(
                x + i * width,
                runtime_data[pipeline],
                width,
                label=pipeline_name,
                color=color,
                alpha=0.8,
            )

        ax1.set_xlabel("Dataset Size", fontsize=self.font_settings["label_size"])
        ax1.set_ylabel("Runtime (seconds)", fontsize=self.font_settings["label_size"])
        ax1.set_title(
            "Pipeline Runtime Comparison",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels([d.capitalize() for d in datasets])
        ax1.legend(fontsize=self.font_settings["legend_size"], ncol=2)
        ax1.grid(axis="y", alpha=0.3)

        # Plot memory
        for i, pipeline in enumerate(pipelines):
            pipeline_name = pipeline.replace("pipeline_", "Pipeline ").upper()
            color = self.colors.get(pipeline, "#333333")
            ax2.bar(
                x + i * width,
                memory_data[pipeline],
                width,
                label=pipeline_name,
                color=color,
                alpha=0.8,
            )

        ax2.set_xlabel("Dataset Size", fontsize=self.font_settings["label_size"])
        ax2.set_ylabel("Peak Memory (GB)", fontsize=self.font_settings["label_size"])
        ax2.set_title(
            "Pipeline Memory Usage Comparison",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels([d.capitalize() for d in datasets])
        ax2.legend(fontsize=self.font_settings["legend_size"], ncol=2)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_distribution_comparison(
        self,
        results: Mapping[str, Any],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create data distribution comparison figure.

        Shows sparsity changes and statistical properties across pipelines.

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Sparsity change
        ax = axes[0, 0]
        datasets = ["small", "medium", "large"]
        pipelines = [
            "pipeline_a",
            "pipeline_b",
            "pipeline_c",
            "pipeline_d",
            "pipeline_e",
        ]

        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            sparsity_changes: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    dist = results[key].get("distribution", {})
                    change_value = dist.get("sparsity_change", 0)
                    change = _extract_metric_value(change_value)
                    sparsity_changes.append(change)
                else:
                    sparsity_changes.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                sparsity_changes,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Sparsity Change (Δ)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Sparsity Change Across Datasets",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        # Mean change
        ax = axes[0, 1]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            mean_changes: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    dist = results[key].get("distribution", {})
                    change_value = dist.get("mean_change", 0)
                    change = _extract_metric_value(change_value)
                    mean_changes.append(change)
                else:
                    mean_changes.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                mean_changes,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Mean Change (Δ)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Mean Value Change Across Datasets",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        # Std change
        ax = axes[1, 0]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            std_changes: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    dist = results[key].get("distribution", {})
                    change_value = dist.get("std_change", 0)
                    change = _extract_metric_value(change_value)
                    std_changes.append(change)
                else:
                    std_changes.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                std_changes,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Std Change (Δ)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Standard Deviation Change Across Datasets",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        # CV change
        ax = axes[1, 1]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            cv_changes: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    dist = results[key].get("distribution", {})
                    change_value = dist.get("cv_change", 0)
                    change = _extract_metric_value(change_value)
                    cv_changes.append(change)
                else:
                    cv_changes.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                cv_changes,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("CV Change (Δ)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Coefficient of Variation Change",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "distribution_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_structure_preservation(
        self,
        results: Mapping[str, Any],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create data structure preservation comparison figure.

        Shows PCA variance, NN consistency, and distance preservation.

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # PCA variance cumulative
        ax = axes[0, 0]
        datasets = ["small", "medium", "large"]
        pipelines = [
            "pipeline_a",
            "pipeline_b",
            "pipeline_c",
            "pipeline_d",
            "pipeline_e",
        ]

        x = np.arange(len(datasets))
        width = 0.15

        for i, pipeline in enumerate(pipelines):
            pipeline_id = PIPELINE_IDS[pipeline]
            pca_variances: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    structure = results[key].get("structure", {})
                    variance_value = structure.get("pca_variance_cumulative", 0)
                    variance = _extract_metric_value(variance_value)
                    pca_variances.append(variance)
                else:
                    pca_variances.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.bar(
                x + i * width,
                pca_variances,
                width,
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                alpha=0.8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Cumulative Variance (PC1-10)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "PCA Variance Explained",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(axis="y", alpha=0.3)

        # NN consistency
        ax = axes[0, 1]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            nn_consistencies: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    structure = results[key].get("structure", {})
                    consistency_value = structure.get("nn_consistency", 0)
                    consistency = _extract_metric_value(consistency_value)
                    nn_consistencies.append(consistency)
                else:
                    nn_consistencies.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                nn_consistencies,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("NN Consistency (Jaccard)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Nearest Neighbor Consistency",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        # Distance preservation
        ax = axes[1, 0]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            distances: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    structure = results[key].get("structure", {})
                    dist_value = structure.get("distance_correlation", 0)
                    dist = _extract_metric_value(dist_value)
                    distances.append(dist)
                else:
                    distances.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                distances,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Distance Correlation", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Distance Preservation",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        # Global structure
        ax = axes[1, 1]
        for pipeline in pipelines:
            pipeline_id = PIPELINE_IDS[pipeline]
            centroid_distances: list[float] = []
            for dataset in datasets:
                key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
                if key in results:
                    structure = results[key].get("structure", {})
                    global_struct = structure.get("global_structure", {})
                    centroid_dist_value = global_struct.get("centroid_distance", 0)
                    centroid_dist = _extract_metric_value(centroid_dist_value)
                    centroid_distances.append(centroid_dist)
                else:
                    centroid_distances.append(0.0)

            color = self.colors.get(pipeline, "#333333")
            ax.plot(
                datasets,
                centroid_distances,
                "o-",
                label=pipeline.replace("pipeline_", "Pipeline ").upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Dataset", fontsize=self.font_settings["label_size"])
        ax.set_ylabel("Centroid Distance", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Global Structure Change",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.legend(fontsize=self.font_settings["legend_size"])
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "structure_preservation.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_comprehensive_radar(
        self,
        results: Mapping[str, Any],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create comprehensive radar chart comparing all pipelines.

        Shows 4 dimensions (batch effects, performance, distribution,
        structure) for all 5 pipelines.

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        # Calculate average scores for each dimension
        pipelines = [
            "pipeline_a",
            "pipeline_b",
            "pipeline_c",
            "pipeline_d",
            "pipeline_e",
        ]
        dimensions = ["Batch Effects", "Performance", "Distribution", "Structure"]

        # Normalize scores to 0-100 scale
        scores: dict[str, dict[str, float]] = {}
        for pipeline in pipelines:
            scores[pipeline] = self._calculate_dimension_scores(results, pipeline)

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

        # Number of variables
        n_dimensions = len(dimensions)

        # Angle for each axis
        angles = [n / float(n_dimensions) * 2 * pi for n in range(n_dimensions)]
        angles += angles[:1]

        # Plot each pipeline
        for pipeline in pipelines:
            values = [scores[pipeline][dim] for dim in dimensions]
            values += values[:1]

            color = self.colors.get(pipeline, "#333333")
            label = pipeline.replace("pipeline_", "Pipeline ").upper()

            ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Add axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)
        ax.grid(True)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        # Add title
        plt.title(
            "Comprehensive Pipeline Comparison",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "comprehensive_radar.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_ranking_barplot(
        self,
        scores: Mapping[str, float],
        save_path: str | Path | None = None,
    ) -> str:
        """
        Create horizontal bar plot showing overall pipeline ranking.

        Parameters
        ----------
        scores : Mapping[str, float]
            Dictionary of pipeline names to overall scores
        save_path : str | None, default None
            Custom save path

        Returns
        -------
        str
            Path to saved figure
        """
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        pipelines = [item[0].replace("pipeline_", "Pipeline ").upper() for item in sorted_scores]
        values = [item[1] for item in sorted_scores]

        # Create color coding based on grade
        colors: list[str] = []
        for value in values:
            if value >= 80:
                colors.append("#2ca02c")  # Green - A grade
            elif value >= 60:
                colors.append("#ff7f0e")  # Orange - B grade
            else:
                colors.append("#d62728")  # Red - C grade

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(pipelines))
        ax.barh(y_pos, values, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pipelines)
        ax.set_xlabel("Overall Score (0-100)", fontsize=self.font_settings["label_size"])
        ax.set_title(
            "Pipeline Overall Ranking",
            fontsize=self.font_settings["title_size"],
            fontweight="bold",
        )
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.1f}", va="center", fontsize=10)

        # Add legend for grades
        legend_elements = [
            mpatches.Patch(color="#2ca02c", label="A Grade (≥80)"),
            mpatches.Patch(color="#ff7f0e", label="B Grade (60-79)"),
            mpatches.Patch(color="#d62728", label="C Grade (<60)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=self.font_settings["legend_size"],
        )

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "ranking_barplot.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def _calculate_dimension_scores(
        self,
        results: Mapping[str, Any],
        pipeline: str,
    ) -> dict[str, float]:
        """
        Calculate normalized scores (0-100) for each dimension.

        Parameters
        ----------
        results : Mapping[str, Any]
            Evaluation results dictionary
        pipeline : str
            Pipeline name (e.g., "pipeline_a")

        Returns
        -------
        Dict[str, float]
            Dictionary with scores for each dimension
        """
        scores: dict[str, float] = {}

        # Average across all datasets
        datasets = ["small", "medium", "large"]

        # Batch effects (higher is better)
        kbet_scores: list[float] = []
        lisi_scores: list[float] = []
        pipeline_id = PIPELINE_IDS[pipeline]
        for dataset in datasets:
            key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
            if key in results:
                batch = results[key].get("batch_effects", {})
                kbet_scores.append(batch.get("kbet", 0))
                lisi_scores.append(batch.get("lisi", 0))

        if kbet_scores and lisi_scores:
            avg_kbet = float(np.mean(kbet_scores))
            avg_lisi = float(np.mean(lisi_scores))
            # Normalize: kBET and LISI both 0-1, target is 1
            scores["Batch Effects"] = ((avg_kbet + avg_lisi) / 2) * 100
        else:
            scores["Batch Effects"] = 0.0

        # Performance (lower is better, invert)
        runtimes: list[float] = []
        memories: list[float] = []
        for dataset in datasets:
            key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
            if key in results:
                perf = results[key].get("performance", {})
                runtime_value = perf.get("runtime_seconds", 0)
                memory_value = perf.get("memory_gb", 0)
                runtimes.append(_extract_metric_value(runtime_value))
                memories.append(_extract_metric_value(memory_value))

        if runtimes and memories:
            # Normalize: use inverse relative to best pipeline
            # This is simplified - proper normalization would compare across pipelines
            avg_runtime = float(np.mean(runtimes))
            avg_memory = float(np.mean(memories))
            # Simple scoring: lower is better, cap at 100
            runtime_score = max(0.0, 100 - avg_runtime / 10)
            memory_score = max(0.0, 100 - avg_memory * 10)
            scores["Performance"] = (runtime_score + memory_score) / 2
        else:
            scores["Performance"] = 0.0

        # Distribution (smaller changes are better)
        sparsity_changes: list[float] = []
        for dataset in datasets:
            key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
            if key in results:
                dist = results[key].get("distribution", {})
                change_value = dist.get("sparsity_change", 0)
                sparsity_changes.append(abs(_extract_metric_value(change_value)))

        if sparsity_changes:
            avg_change = float(np.mean(sparsity_changes))
            # Normalize: smaller change is better
            scores["Distribution"] = max(0.0, 100 - avg_change * 100)
        else:
            scores["Distribution"] = 0.0

        # Structure (higher is better)
        nn_consistencies: list[float] = []
        distance_correlations: list[float] = []
        for dataset in datasets:
            key = _make_result_key(dataset, pipeline_id, use_aggregated=True)
            if key in results:
                struct = results[key].get("structure", {})
                nn_value = struct.get("nn_consistency", 0)
                dist_value = struct.get("distance_correlation", 0)
                nn_consistencies.append(_extract_metric_value(nn_value))
                distance_correlations.append(_extract_metric_value(dist_value))

        if nn_consistencies and distance_correlations:
            avg_nn = float(np.mean(nn_consistencies))
            avg_dist = float(np.mean(distance_correlations))
            # Both are correlations 0-1
            scores["Structure"] = ((avg_nn + avg_dist) / 2) * 100
        else:
            scores["Structure"] = 0.0

        return scores

    def _get_dataset_color(self, index: int) -> str:
        """
        Get color for dataset.

        Parameters
        ----------
        index : int
            Dataset index

        Returns
        -------
        str
            Color hex code
        """
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        return colors[index % len(colors)]


def generate_all_figures(
    results: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: str = "outputs/figures",
) -> list[str]:
    """
    Generate all figures for the comparison report.

    Parameters
    ----------
    results : Mapping[str, Any]
        Evaluation results dictionary
    config : Mapping[str, Any]
        Visualization configuration
    output_dir : str, default "outputs/figures"
        Output directory for figures

    Returns
    -------
    List[str]
        List of paths to generated figures
    """
    plotter = ComparisonPlotter(config, output_dir)

    generated: list[str] = []

    # Generate all main figures
    figures = [
        ("batch_effects", plotter.plot_batch_effects_comparison),
        ("performance", plotter.plot_performance_comparison),
        ("distribution", plotter.plot_distribution_comparison),
        ("structure", plotter.plot_structure_preservation),
        ("radar", plotter.plot_comprehensive_radar),
    ]

    for name, plot_func in figures:
        try:
            path = plot_func(results)
            generated.append(path)
            print(f"✓ Generated {name} figure: {path}")
        except Exception as e:
            print(f"✗ Failed to generate {name} figure: {e}")

    # Generate ranking (requires score calculation)
    try:
        # Import here to avoid circular dependency
        from .report_generator import calculate_overall_scores

        scores = calculate_overall_scores(results, config)
        path = plotter.plot_ranking_barplot(scores)
        generated.append(path)
        print(f"✓ Generated ranking figure: {path}")
    except Exception as e:
        print(f"✗ Failed to generate ranking figure: {e}")

    return generated
