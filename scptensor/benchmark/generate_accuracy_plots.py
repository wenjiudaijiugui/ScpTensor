"""Generate accuracy evaluation plots for ScpTensor vs Scanpy comparison.

This module creates comprehensive accuracy assessment visualizations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics."""

    mse: float = 0.0
    mae: float = 0.0
    correlation: float = 0.0
    relative_error: float = 0.0

    def __post_init__(self) -> None:
        # Handle NaN values
        if np.isnan(self.mse):
            self.mse = 0.0
        if np.isnan(self.mae):
            self.mae = 0.0
        if np.isnan(self.correlation):
            self.correlation = 0.0
        if np.isnan(self.relative_error):
            self.relative_error = 0.0


@dataclass
class MethodAccuracyData:
    """Accuracy data for a single method."""

    name: str
    display_name: str
    metrics: AccuracyMetrics
    category: str = "General"


# ==============================================================================
# Data Loading
# ==============================================================================


def load_comparison_results(results_path: Path) -> dict:
    """Load comparison results from JSON.

    Parameters
    ----------
    results_path : Path
        Path to comparison_results.json

    Returns
    -------
    dict
        Parsed results data
    """
    with open(results_path) as f:
        return json.load(f)


def extract_accuracy_metrics(results: dict) -> list[MethodAccuracyData]:
    """Extract accuracy metrics from comparison results.

    Parameters
    ----------
    results : dict
        Raw comparison results

    Returns
    -------
    list[MethodAccuracyData]
        List of accuracy data per method
    """
    accuracy_data = []

    # Method display names and categories
    method_info = {
        "log_normalize": ("Log Normalize", "Normalization"),
        "z_score_normalize": ("Z-Score Normalize", "Normalization"),
        "knn_impute": ("KNN Impute", "Imputation"),
        "pca": ("PCA", "Dimensionality Reduction"),
        "kmeans": ("K-Means", "Clustering"),
        "hvg": ("HVG", "Feature Selection"),
    }

    results_dict = results.get("results", {})

    for method_name, result_list in results_dict.items():
        if not result_list:
            continue

        result = result_list[0]
        comp_metrics = result.get("comparison_metrics", {})

        display_name, category = method_info.get(
            method_name,
            (method_name.replace("_", " ").title(), "Other"),
        )

        metrics = AccuracyMetrics(
            mse=comp_metrics.get("mse", 0.0),
            mae=comp_metrics.get("mae", 0.0),
            correlation=comp_metrics.get("correlation", 0.0),
            relative_error=comp_metrics.get("relative_error", 0.0),
        )

        accuracy_data.append(
            MethodAccuracyData(
                name=method_name,
                display_name=display_name,
                metrics=metrics,
                category=category,
            )
        )

    return accuracy_data


# ==============================================================================
# Plotting Functions
# ==============================================================================


def configure_plots() -> None:
    """Configure matplotlib for publication-quality plots."""
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def plot_mse_comparison(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot MSE comparison across methods.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter methods with valid MSE
    valid_data = [d for d in accuracy_data if d.metrics.mse > 0]

    if not valid_data:
        print("  ⚠ No valid MSE data for plotting")
        return

    methods = [d.display_name for d in valid_data]
    mses = [d.metrics.mse for d in valid_data]

    # Color by category
    colors = []
    for d in valid_data:
        if d.category == "Normalization":
            colors.append("#2E86AB")
        elif d.category == "Imputation":
            colors.append("#A23B72")
        elif d.category == "Dimensionality Reduction":
            colors.append("#6B8E23")
        elif d.category == "Clustering":
            colors.append("#E76F51")
        else:
            colors.append("#888888")

    bars = ax.bar(range(len(methods)), mses, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, mse) in enumerate(zip(bars, mses)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mse:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Output Accuracy: Mean Squared Error Comparison")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="Normalization"),
        Patch(facecolor="#A23B72", label="Imputation"),
        Patch(facecolor="#6B8E23", label="Dim. Reduction"),
        Patch(facecolor="#E76F51", label="Clustering"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "mse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ MSE comparison saved to {output_dir / 'mse_comparison.png'}")


def plot_mae_comparison(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot MAE comparison across methods.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter methods with valid MAE
    valid_data = [d for d in accuracy_data if d.metrics.mae > 0]

    if not valid_data:
        print("  ⚠ No valid MAE data for plotting")
        return

    methods = [d.display_name for d in valid_data]
    maes = [d.metrics.mae for d in valid_data]

    # Color by category
    colors = []
    for d in valid_data:
        if d.category == "Normalization":
            colors.append("#2E86AB")
        elif d.category == "Imputation":
            colors.append("#A23B72")
        elif d.category == "Dimensionality Reduction":
            colors.append("#6B8E23")
        elif d.category == "Clustering":
            colors.append("#E76F51")
        else:
            colors.append("#888888")

    bars = ax.bar(range(len(methods)), maes, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mae:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Output Accuracy: Mean Absolute Error Comparison")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "mae_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ MAE comparison saved to {output_dir / 'mae_comparison.png'}")


def plot_correlation_comparison(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot correlation comparison across methods.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter methods with valid correlation
    valid_data = [d for d in accuracy_data if abs(d.metrics.correlation) > 0]

    if not valid_data:
        print("  ⚠ No valid correlation data for plotting")
        return

    methods = [d.display_name for d in valid_data]
    correlations = [d.metrics.correlation for d in valid_data]

    # Color by correlation quality
    colors = []
    for corr in correlations:
        if abs(corr) >= 0.95:
            colors.append("#6B8E23")  # Green - excellent
        elif abs(corr) >= 0.8:
            colors.append("#2E86AB")  # Blue - good
        elif abs(corr) >= 0.5:
            colors.append("#F4A261")  # Orange - fair
        else:
            colors.append("#E76F51")  # Red - poor

    bars = ax.bar(range(len(methods)), correlations, color=colors, edgecolor="black", linewidth=0.5)

    # Add reference line at 0.95
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="Excellent (0.95)")
    ax.axhline(y=0.80, color="blue", linestyle="--", alpha=0.5, label="Good (0.80)")
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.3)

    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.05 if height >= 0 else -0.05),
            f"{corr:.4f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Pearson Correlation Coefficient")
    ax.set_title("Output Accuracy: Correlation Coefficient Comparison")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "correlation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Correlation comparison saved to {output_dir / 'correlation_comparison.png'}")


def plot_accuracy_radar(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot radar chart for multi-dimensional accuracy comparison.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    # Filter methods with all metrics
    valid_data = [
        d for d in accuracy_data
        if d.metrics.mse > 0 and d.metrics.mae > 0 and abs(d.metrics.correlation) > 0
    ][:5]  # Limit to 5 methods for readability

    if len(valid_data) < 3:
        print("  ⚠ Not enough valid data for radar chart")
        return

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})

    # Define metrics (normalized)
    categories = ["MSE\n(inverted)", "MAE\n(inverted)", "Correlation", "Relative\nAccuracy"]
    n_cats = len(categories)

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Define colors
    method_colors = ["#2E86AB", "#A23B72", "#6B8E23", "#E76F51", "#F4A261"]

    # Plot each method
    for i, data in enumerate(valid_data):
        # Normalize metrics: higher is better
        # For MSE and MAE, use 1/(1+x) to invert
        mse_inv = 1.0 / (1.0 + data.metrics.mse)
        mae_inv = 1.0 / (1.0 + data.metrics.mae)
        corr = abs(data.metrics.correlation)
        rel_acc = 1.0 - min(data.metrics.relative_error, 1.0)

        values = [mse_inv, mae_inv, corr, rel_acc]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=data.display_name,
            color=method_colors[i % len(method_colors)],
        )
        ax.fill(angles, values, alpha=0.15, color=method_colors[i % len(method_colors)])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.title("Multi-Dimensional Accuracy Comparison", pad=20, fontsize=12)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "accuracy_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Accuracy radar saved to {output_dir / 'accuracy_radar.png'}")


def plot_accuracy_heatmap(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot heatmap of accuracy metrics across methods.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data matrix
    valid_data = [d for d in accuracy_data if d.metrics.mse > 0 or d.metrics.mae > 0]

    if len(valid_data) < 2:
        print("  ⚠ Not enough data for accuracy heatmap")
        return

    methods = [d.display_name for d in valid_data]

    # Normalize metrics for comparison (0-1 scale, higher is better)
    data_matrix = []
    for d in valid_data:
        # MSE: invert and normalize
        mse_norm = min(1.0 / (1.0 + d.metrics.mse), 1.0)

        # MAE: invert and normalize
        mae_norm = min(1.0 / (1.0 + d.metrics.mae), 1.0)

        # Correlation: absolute value
        corr_norm = abs(d.metrics.correlation)

        # Relative accuracy: 1 - error
        rel_norm = max(0.0, 1.0 - d.metrics.relative_error)

        data_matrix.append([mse_norm, mae_norm, corr_norm, rel_norm])

    data_matrix = np.array(data_matrix)

    # Create heatmap
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(["MSE\n(Inv)", "MAE\n(Inv)", "Correlation", "Relative\nAccuracy"])
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(4):
            text = ax.text(
                j,
                i,
                f"{data_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if data_matrix[i, j] < 0.5 else "white",
                fontsize=9,
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Score (Higher = Better)")

    ax.set_title("Accuracy Metrics Heatmap")
    plt.tight_layout()

    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Accuracy heatmap saved to {output_dir / 'accuracy_heatmap.png'}")


def plot_accuracy_summary(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Plot comprehensive accuracy summary with multiple panels.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for plots
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    valid_data = [d for d in accuracy_data if d.metrics.mse > 0]

    if not valid_data:
        print("  ⚠ No valid data for accuracy summary")
        return

    methods = [d.display_name for d in valid_data]
    x_pos = np.arange(len(methods))

    # Panel 1: MSE
    ax1 = fig.add_subplot(gs[0, 0])
    mses = [d.metrics.mse for d in valid_data]
    colors1 = ["#2E86AB" if d.category == "Normalization" else
               "#A23B72" if d.category == "Imputation" else
               "#6B8E23" for d in valid_data]
    ax1.bar(x_pos, mses, color=colors1, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("MSE")
    ax1.set_title("Mean Squared Error")
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: MAE
    ax2 = fig.add_subplot(gs[0, 1])
    maes = [d.metrics.mae for d in valid_data]
    ax2.bar(x_pos, maes, color=colors1, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error")
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Correlation
    ax3 = fig.add_subplot(gs[0, 2])
    corr_data = [d for d in valid_data if abs(d.metrics.correlation) > 0]
    if corr_data:
        corr_methods = [d.display_name for d in corr_data]
        corr_x = np.arange(len(corr_methods))
        corrs = [abs(d.metrics.correlation) for d in corr_data]
        colors3 = ["#6B8E23" if c >= 0.95 else "#2E86AB" if c >= 0.8 else "#F4A261" for c in corrs]
        ax3.bar(corr_x, corrs, color=colors3, edgecolor="black", linewidth=0.5)
        ax3.axhline(y=0.95, color="green", linestyle="--", alpha=0.5)
        ax3.set_xticks(corr_x)
        ax3.set_xticklabels(corr_methods, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Correlation")
        ax3.set_title("Pearson Correlation")
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis="y", alpha=0.3)

    # Panel 4: Accuracy score summary
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")

    # Calculate summary statistics
    avg_mse = np.mean(mses)
    avg_mae = np.mean(maes)
    avg_corr = np.mean([abs(d.metrics.correlation) for d in valid_data if abs(d.metrics.correlation) > 0])

    # Count high correlation methods
    high_corr_count = sum(1 for d in valid_data if abs(d.metrics.correlation) >= 0.95)
    good_corr_count = sum(1 for d in valid_data if 0.8 <= abs(d.metrics.correlation) < 0.95)

    summary_text = f"""Accuracy Assessment Summary

Methods Compared: {len(valid_data)}

Error Metrics:
• Mean MSE: {avg_mse:.6f}
• Mean MAE: {avg_mae:.6f}

Correlation Quality:
• Mean Correlation: {avg_corr:.4f}
• High Correlation (≥0.95): {high_corr_count}/{len(valid_data)}
• Good Correlation (0.8-0.95): {good_corr_count}/{len(valid_data)}

Interpretation:
• MSE/MAE: Lower values indicate better accuracy
• Correlation: Higher values indicate better agreement
• Correlation ≥0.95 indicates excellent agreement
"""

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.suptitle("Comprehensive Accuracy Assessment", fontsize=14, y=0.98)

    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "accuracy_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Accuracy summary saved to {output_dir / 'accuracy_summary.png'}")


def generate_accuracy_report(
    accuracy_data: list[MethodAccuracyData],
    output_dir: Path,
) -> None:
    """Generate a text-based accuracy assessment report.

    Parameters
    ----------
    accuracy_data : list[MethodAccuracyData]
        Accuracy metrics for each method
    output_dir : Path
        Output directory for the report
    """
    report_lines = [
        "# Accuracy Assessment Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Method-by-Method Accuracy Analysis",
        "",
    ]

    # Sort by correlation
    sorted_data = sorted(accuracy_data, key=lambda x: abs(x.metrics.correlation), reverse=True)

    for d in sorted_data:
        if d.metrics.mse == 0 and d.metrics.mae == 0:
            continue

        report_lines.extend([
            f"### {d.display_name}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| MSE | {d.metrics.mse:.6f} |",
            f"| MAE | {d.metrics.mae:.6f} |",
            f"| Correlation | {d.metrics.correlation:.4f} |",
            f"| Relative Error | {d.metrics.relative_error:.4f} |",
            "",
        ])

        # Assessment
        corr = abs(d.metrics.correlation)
        if corr >= 0.95:
            assessment = "**Excellent** - High agreement between ScpTensor and Scanpy"
        elif corr >= 0.8:
            assessment = "**Good** - Reasonable agreement between implementations"
        elif corr >= 0.5:
            assessment = "**Fair** - Moderate agreement, some differences expected"
        else:
            assessment = "**Poor** - Low correlation, implementations differ significantly"

        report_lines.extend([
            f"**Assessment**: {assessment}",
            "",
        ])

    # Overall summary
    report_lines.extend([
        "---",
        "",
        "## Overall Summary",
        "",
    ])

    valid_corr = [abs(d.metrics.correlation) for d in accuracy_data if abs(d.metrics.correlation) > 0]
    if valid_corr:
        avg_corr = np.mean(valid_corr)
        high_corr = sum(1 for c in valid_corr if c >= 0.95)
        total = len(valid_corr)

        report_lines.extend([
            f"- **Average Correlation**: {avg_corr:.4f}",
            f"- **High Agreement Methods**: {high_corr}/{total} (correlation ≥ 0.95)",
            "",
        ])

        if avg_corr >= 0.9:
            conclusion = "ScpTensor and Scanpy show **excellent overall agreement** in output accuracy."
        elif avg_corr >= 0.7:
            conclusion = "ScpTensor and Scanpy show **good overall agreement** with minor implementation differences."
        else:
            conclusion = "ScpTensor and Scanpy show **moderate agreement**; differences may be due to algorithmic variations."

        report_lines.extend([
            f"**Conclusion**: {conclusion}",
            "",
        ])

    # Write report
    report_path = output_dir / "accuracy_report.md"
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  ✓ Accuracy report saved to {report_path}")


# ==============================================================================
# Main Function
# ==============================================================================


def main() -> int:
    """Main entry point for accuracy plot generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate accuracy assessment plots for ScpTensor vs Scanpy comparison"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="benchmark_results/comparison_results.json",
        help="Path to comparison results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/figures/accuracy",
        help="Output directory for accuracy plots",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate text-based accuracy report",
    )

    args = parser.parse_args()

    # Configure plots
    configure_plots()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1

    print("=== Loading Comparison Results ===")
    results = load_comparison_results(results_path)

    print("=== Extracting Accuracy Metrics ===")
    accuracy_data = extract_accuracy_metrics(results)
    print(f"  Found {len(accuracy_data)} methods with accuracy data")

    output_dir = Path(args.output)

    # Generate plots
    print("\n=== Generating Accuracy Plots ===")

    plot_mse_comparison(accuracy_data, output_dir)
    plot_mae_comparison(accuracy_data, output_dir)
    plot_correlation_comparison(accuracy_data, output_dir)
    plot_accuracy_radar(accuracy_data, output_dir)
    plot_accuracy_heatmap(accuracy_data, output_dir)
    plot_accuracy_summary(accuracy_data, output_dir)

    # Generate text report if requested
    if args.report:
        print("\n=== Generating Accuracy Report ===")
        report_dir = Path("benchmark_results")
        generate_accuracy_report(accuracy_data, report_dir)

    print("\n=== Accuracy Assessment Complete ===")
    print(f"\nGenerated files in {output_dir}:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
