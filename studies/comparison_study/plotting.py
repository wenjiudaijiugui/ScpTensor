"""Simplified visualization module for comparison study.

Provides minimal plotting functions for batch effect and performance comparisons.
Uses pure functions with direct matplotlib calls (no class abstractions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# =============================================================================
# Style Configuration
# =============================================================================

# Simple color scheme (no complex theme management)
COLORS = {
    "scptensor": "#1f77b4",
    "scanpy": "#ff7f0e",
    "scran": "#2ca02c",
    "seurat": "#d62728",
    "batch_before": "#d62728",
    "batch_after": "#2ca02c",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "error": "#d62728",
}

METRIC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]


def _setup_style(dpi: int = 300) -> None:
    """Configure matplotlib with SciencePlots style."""
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
        }
    )


# =============================================================================
# Batch Effect Visualization
# =============================================================================


def plot_batch_effects(
    results_dict: dict[str, dict[str, Any]],
    metrics: list[str] | None = None,
    output_path: str | Path = "batch_effects.png",
) -> Path:
    """Plot batch effect metrics comparison."""
    _setup_style()
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["kbet_score", "ilisi_score", "clisi_score", "asw_score"]

    methods = list(results_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = []
        frameworks = []
        colors = []

        for method in methods:
            if metric in results_dict[method]:
                val = results_dict[method][metric]
                if val is not None:
                    values.append(val)
                    frameworks.append(method)
                    fw = results_dict[method].get("framework", "scptensor")
                    colors.append(COLORS.get(fw, "#1f77b4"))

        if values:
            bars = ax.bar(frameworks, values, color=colors, alpha=0.7)
            ax.set_ylabel(_format_metric_name(metric))
            ax.set_title(_format_metric_name(metric))
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_umap_comparison(
    umap_before: npt.NDArray[np.float64],
    umap_after: npt.NDArray[np.float64],
    batch_labels: npt.NDArray[np.int_],
    method_name: str = "Batch Correction",
    output_path: str | Path = "umap_comparison.png",
) -> Path:
    """Plot UMAP before/after batch correction."""
    _setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before correction
    ax1 = axes[0]
    unique_batches = np.unique(batch_labels)
    for batch_id in unique_batches:
        mask = batch_labels == batch_id
        ax1.scatter(
            umap_before[mask, 0],
            umap_before[mask, 1],
            label=f"Batch {batch_id}",
            alpha=0.6,
            s=20,
        )

    ax1.set_title("Before Batch Correction")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend()

    # After correction
    ax2 = axes[1]
    for batch_id in unique_batches:
        mask = batch_labels == batch_id
        ax2.scatter(
            umap_after[mask, 0],
            umap_after[mask, 1],
            label=f"Batch {batch_id}",
            alpha=0.6,
            s=20,
        )

    ax2.set_title(f"After {method_name}")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Performance Comparison
# =============================================================================


def plot_performance_comparison(
    results_dict: dict[str, dict[str, Any]],
    output_path: str | Path = "performance_comparison.png",
) -> Path:
    """Plot performance metrics (time, memory) comparison."""
    _setup_style()
    import matplotlib.pyplot as plt

    methods = list(results_dict.keys())

    # Extract execution times
    times = []
    time_colors = []
    for method in methods:
        time_val = results_dict[method].get("execution_time")
        if time_val is not None:
            times.append(time_val)
            fw = results_dict[method].get("framework", "scptensor")
            time_colors.append(COLORS.get(fw, "#1f77b4"))
        else:
            times.append(0)
            time_colors.append("#cccccc")

    # Extract memory usage
    memories = []
    mem_colors = []
    for method in methods:
        mem_val = results_dict[method].get("memory_usage")
        if mem_val is not None:
            memories.append(mem_val)
            fw = results_dict[method].get("framework", "scptensor")
            mem_colors.append(COLORS.get(fw, "#1f77b4"))
        else:
            memories.append(0)
            mem_colors.append("#cccccc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Execution time
    ax1 = axes[0]
    bars1 = ax1.bar(methods, times, color=time_colors, alpha=0.7)
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Execution Time")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, times, strict=False):
        if val > 0:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Memory usage
    ax2 = axes[1]
    bars2 = ax2.bar(methods, memories, color=mem_colors, alpha=0.7)
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Memory Usage")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, memories, strict=False):
        if val > 0:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f}MB",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Radar Chart (Comprehensive Assessment)
# =============================================================================


def plot_radar_chart(
    metrics_dict: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    output_path: str | Path = "radar_chart.png",
) -> Path:
    """Plot radar chart for comprehensive metrics comparison."""
    _setup_style()
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["kbet_score", "ilisi_score", "clisi_score", "biological_preservation"]

    methods = list(metrics_dict.keys())

    # Normalize values to 0-1 scale
    normalized_data = {}
    for method in methods:
        vals = []
        for metric in metrics:
            val = metrics_dict[method].get(metric, 0)
            if val is None:
                val = 0
            # Assume most metrics are 0-1, normalize if needed
            vals.append(min(val, 1.0))
        normalized_data[method] = vals

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for idx, method in enumerate(methods):
        values = normalized_data[method]
        values += values[:1]  # Complete the circle

        color = METRIC_COLORS[idx % len(METRIC_COLORS)]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_format_metric_name(m) for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Distribution Comparison
# =============================================================================


def plot_distribution_comparison(
    data_dict: dict[str, npt.NDArray[np.float64]],
    output_path: str | Path = "distribution_comparison.png",
) -> Path:
    """Plot distribution comparison across methods."""
    _setup_style()
    import matplotlib.pyplot as plt

    methods = list(data_dict.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax1 = axes[0]
    for idx, method in enumerate(methods):
        data = data_dict[method].flatten()
        color = METRIC_COLORS[idx % len(METRIC_COLORS)]
        ax1.hist(data, bins=50, alpha=0.5, label=method, color=color, density=True)

    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title("Value Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Box plot
    ax2 = axes[1]
    box_data = [data_dict[method].flatten() for method in methods]
    bp = ax2.boxplot(box_data, labels=methods, patch_artist=True)

    for patch, color in zip(bp["boxes"], METRIC_COLORS[: len(methods)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("Value")
    ax2.set_title("Value Distribution (Box Plot)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Clustering Results
# =============================================================================


def plot_clustering_results(
    x: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    method: str = "PCA",
    output_path: str | Path = "clustering_results.png",
) -> Path:
    """Plot clustering results using dimensionality reduction."""
    _setup_style()
    import matplotlib.pyplot as plt

    # Reduce to 2D if needed
    if x.shape[1] > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        x_2d = pca.fit_transform(x)
        title_suffix = f"(PCA, {pca.explained_variance_ratio_.sum():.1%} variance)"
    else:
        x_2d = x
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        color = METRIC_COLORS[label % len(METRIC_COLORS)]
        ax.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            label=f"Cluster {label}",
            alpha=0.6,
            s=30,
            color=color,
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"Clustering Results {title_suffix}")
    ax.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Heatmap Visualization
# =============================================================================


def plot_metrics_heatmap(
    results_dict: dict[str, dict[str, float]],
    output_path: str | Path = "metrics_heatmap.png",
) -> Path:
    """Plot heatmap of metrics across methods."""
    _setup_style()
    import matplotlib.pyplot as plt

    methods = list(results_dict.keys())
    all_metrics = set()
    for method_data in results_dict.values():
        all_metrics.update(method_data.keys())

    # Filter out non-numeric metrics
    metrics = sorted([m for m in all_metrics if m not in ["framework", "method_name"]])

    # Build matrix
    data_matrix = np.zeros((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            val = results_dict[method].get(metric)
            if val is not None:
                data_matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([_format_metric_name(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = data_matrix[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Normalized Score")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Utility Functions
# =============================================================================


def _format_metric_name(metric_name: str) -> str:
    """Format metric name for display."""
    return metric_name.replace("_", " ").title()


__all__ = [
    "plot_batch_effects",
    "plot_umap_comparison",
    "plot_performance_comparison",
    "plot_radar_chart",
    "plot_distribution_comparison",
    "plot_clustering_results",
    "plot_metrics_heatmap",
]
