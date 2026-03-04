#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive test script for normalization autoselect module.

This script tests the automatic method selection for normalization methods
(norm_mean, norm_median, norm_quantile) across various data characteristics.

Test Dimensions:
- Scale: small (50x100), medium (200x500)
- Missing Rate: low (5-10%), medium (20-30%), high (40-50%)
- Missing Pattern: mcar, mar, mnar
- Distribution: normal, log_normal, multimodal
- Batch Effect: with_batch, without_batch

Outputs:
- JSON results file: normalization_test_results.json
- Visualization plots:
  - normalization_heatmap.png (method scores by data characteristics)
  - normalization_selection_count.png (method selection frequency)
  - normalization_missing_rate_scatter.png (performance vs missing rate)
  - normalization_batch_comparison.png (batch effect comparison)
  - normalization_distribution_comparison.png (distribution comparison)
  - normalization_radar.png (radar chart showing 4 evaluation metrics)

Evaluation Metrics (4 metrics):
- intragroup_variation: Reduction of variation within biological groups
- intergroup_preservation: Preservation of biological differences between groups
- technical_variance: Reduction of technical/batch-related variance
- clustering_quality: Quality of clustering in normalized data
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark_data_generator import (
    BenchmarkDataGenerator,
    get_actual_missing_rate,
)
from scptensor.autoselect import StageReport, auto_normalize
from scptensor.transformation import log_transform

# Apply scienceplots style
plt.style.use(["science", "no-latex"])

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent

# Test dimensions
SCALES = ["small", "medium"]  # small: 50x100, medium: 200x500
MISSING_RATES = ["low", "medium", "high"]  # 5-10%, 20-30%, 40-50%
MISSING_PATTERNS = ["mcar", "mar", "mnar"]
DISTRIBUTIONS = ["normal", "log_normal", "multimodal"]
BATCH_VARIANTS = [False, True]  # without_batch, with_batch

# Random seed for reproducibility
SEED = 42

# New 4 evaluation metrics
METRICS = [
    "intragroup_variation",
    "intergroup_preservation",
    "technical_variance",
    "clustering_quality",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TestScenario:
    """Test scenario configuration."""

    name: str
    scale: str
    missing_rate: str
    missing_pattern: str
    distribution: str
    with_batch: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MethodResult:
    """Result for a single normalization method."""

    method_name: str
    overall_score: float
    intragroup_variation: float
    intergroup_preservation: float
    technical_variance: float
    clustering_quality: float
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TestResult:
    """Complete test result for a scenario."""

    scenario: TestScenario
    selected_method: str
    methods: list[MethodResult]
    actual_missing_rate: float
    n_samples: int
    n_features: int
    total_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario.to_dict(),
            "selected_method": self.selected_method,
            "methods": [m.to_dict() for m in self.methods],
            "actual_missing_rate": self.actual_missing_rate,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "total_time": self.total_time,
        }


# =============================================================================
# Scenario Generation
# =============================================================================


def generate_scenarios() -> list[TestScenario]:
    """Generate all test scenarios.

    Returns
    -------
    list[TestScenario]
        List of test scenarios covering all dimension combinations.
    """
    scenarios: list[TestScenario] = []
    scenario_id = 0

    for scale in SCALES:
        for missing_rate in MISSING_RATES:
            for missing_pattern in MISSING_PATTERNS:
                for distribution in DISTRIBUTIONS:
                    for with_batch in BATCH_VARIANTS:
                        # Create scenario name
                        batch_str = "with_batch" if with_batch else "no_batch"
                        name = (
                            f"scenario_{scenario_id:03d}_"
                            f"{scale}_{missing_rate}_{missing_pattern}_"
                            f"{distribution}_{batch_str}"
                        )

                        scenario = TestScenario(
                            name=name,
                            scale=scale,
                            missing_rate=missing_rate,
                            missing_pattern=missing_pattern,
                            distribution=distribution,
                            with_batch=with_batch,
                        )
                        scenarios.append(scenario)
                        scenario_id += 1

    return scenarios


# =============================================================================
# Test Execution
# =============================================================================


def run_single_scenario(
    generator: BenchmarkDataGenerator,
    scenario: TestScenario,
) -> TestResult:
    """Run a single test scenario.

    Parameters
    ----------
    generator : BenchmarkDataGenerator
        Data generator instance.
    scenario : TestScenario
        Test scenario configuration.

    Returns
    -------
    TestResult
        Test result with method scores and selection.
    """
    print(f"\n  Running: {scenario.name}")

    # Generate data
    config = {
        "scale": scenario.scale,
        "missing_rate": scenario.missing_rate,
        "missing_pattern": scenario.missing_pattern,
        "distribution": scenario.distribution,
        "with_batch_effect": scenario.with_batch,
        "n_batches": 3 if scenario.with_batch else 1,
    }
    container = generator.generate_from_config(config)

    # Get data statistics
    actual_missing_rate = get_actual_missing_rate(container)
    n_samples = container.n_samples
    n_features = container.assays["proteins"].n_features

    print(f"    Data: {n_samples} samples x {n_features} features")
    print(f"    Missing rate: {actual_missing_rate:.1%}")

    # Apply log_transform first (preprocessing step)
    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
        base=2.0,
        offset=1.0,
    )

    # Run autoselect for normalization
    start_time = time.perf_counter()
    container, report = auto_normalize(
        container,
        assay_name="proteins",
        source_layer="log",
        keep_all=True,  # Keep all method results for comparison
    )
    total_time = time.perf_counter() - start_time

    # Extract results
    selected_method = report.best_method
    method_results: list[MethodResult] = []

    for result in report.results:
        method_result = MethodResult(
            method_name=result.method_name,
            overall_score=result.overall_score,
            intragroup_variation=result.scores.get("intragroup_variation", 0.0),
            intergroup_preservation=result.scores.get("intergroup_preservation", 0.0),
            technical_variance=result.scores.get("technical_variance", 0.0),
            clustering_quality=result.scores.get("clustering_quality", 0.0),
            execution_time=result.execution_time,
        )
        method_results.append(method_result)

    print(f"    Selected: {selected_method} (score: {report.best_result.overall_score:.4f})")
    print(f"    Time: {total_time:.2f}s")

    return TestResult(
        scenario=scenario,
        selected_method=selected_method,
        methods=method_results,
        actual_missing_rate=actual_missing_rate,
        n_samples=n_samples,
        n_features=n_features,
        total_time=total_time,
    )


def run_all_scenarios(scenarios: list[TestScenario]) -> list[TestResult]:
    """Run all test scenarios.

    Parameters
    ----------
    scenarios : list[TestScenario]
        List of test scenarios.

    Returns
    -------
    list[TestResult]
        List of test results.
    """
    results: list[TestResult] = []
    generator = BenchmarkDataGenerator(seed=SEED)

    total = len(scenarios)
    print(f"\n{'=' * 70}")
    print(f"Running {total} test scenarios")
    print(f"{'=' * 70}")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{total}] {scenario.name}")
        try:
            result = run_single_scenario(generator, scenario)
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            # Add failed result
            results.append(
                TestResult(
                    scenario=scenario,
                    selected_method="ERROR",
                    methods=[],
                    actual_missing_rate=0.0,
                    n_samples=0,
                    n_features=0,
                    total_time=0.0,
                )
            )

    return results


# =============================================================================
# Visualization
# =============================================================================


def create_heatmap(results: list[TestResult], output_path: Path) -> None:
    """Create heatmap showing method scores across scenarios.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    # Aggregate scores by scenario characteristics
    method_names = ["norm_mean", "norm_median", "norm_quantile"]

    # Group by missing rate and missing pattern
    missing_rates = ["low", "medium", "high"]
    missing_patterns = ["mcar", "mar", "mnar"]

    # Create figure with subplots for each method
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, method_name in enumerate(method_names):
        # Create matrix for this method
        score_matrix = np.zeros((len(missing_patterns), len(missing_rates)))
        count_matrix = np.zeros((len(missing_patterns), len(missing_rates)))

        for result in results:
            if result.selected_method == "ERROR":
                continue

            # Find this method's score
            method_score = None
            for m in result.methods:
                if m.method_name == method_name:
                    method_score = m.overall_score
                    break

            if method_score is None:
                continue

            # Get indices
            mr_idx = missing_rates.index(result.scenario.missing_rate)
            mp_idx = missing_patterns.index(result.scenario.missing_pattern)

            score_matrix[mp_idx, mr_idx] += method_score
            count_matrix[mp_idx, mr_idx] += 1

        # Average scores
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_matrix = np.where(count_matrix > 0, score_matrix / count_matrix, np.nan)

        # Plot heatmap
        im = axes[ax_idx].imshow(avg_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Labels
        axes[ax_idx].set_xticks(range(len(missing_rates)))
        axes[ax_idx].set_xticklabels(missing_rates)
        axes[ax_idx].set_yticks(range(len(missing_patterns)))
        axes[ax_idx].set_yticklabels(missing_patterns)
        axes[ax_idx].set_xlabel("Missing Rate")
        axes[ax_idx].set_ylabel("Missing Pattern")
        axes[ax_idx].set_title(f"{method_name}")

        # Add score annotations
        for i in range(len(missing_patterns)):
            for j in range(len(missing_rates)):
                if not np.isnan(avg_matrix[i, j]):
                    text = axes[ax_idx].text(
                        j,
                        i,
                        f"{avg_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                    )

        # Colorbar
        plt.colorbar(im, ax=axes[ax_idx], label="Score")

    plt.suptitle("Normalization Method Scores by Data Characteristics", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {output_path}")


def create_selection_bar_chart(results: list[TestResult], output_path: Path) -> None:
    """Create bar chart showing method selection frequency.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    # Count selections
    selection_counts: dict[str, int] = {}
    for result in results:
        if result.selected_method != "ERROR":
            selection_counts[result.selected_method] = (
                selection_counts.get(result.selected_method, 0) + 1
            )

    # Sort by count
    methods = sorted(selection_counts.keys(), key=lambda x: selection_counts[x], reverse=True)
    counts = [selection_counts[m] for m in methods]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(methods, counts, color=["#2ecc71", "#3498db", "#e74c3c"][: len(methods)])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Normalization Method", fontsize=12)
    ax.set_ylabel("Selection Count", fontsize=12)
    ax.set_title("Method Selection Frequency Across All Scenarios", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar chart: {output_path}")


def create_missing_rate_scatter(results: list[TestResult], output_path: Path) -> None:
    """Create scatter plot showing method performance vs missing rate.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    # Collect data points
    data_points: dict[str, list[tuple[float, float]]] = {
        "norm_mean": [],
        "norm_median": [],
        "norm_quantile": [],
    }

    for result in results:
        if result.selected_method == "ERROR":
            continue

        missing_rate = result.actual_missing_rate

        for method in result.methods:
            if method.method_name in data_points:
                data_points[method.method_name].append((missing_rate, method.overall_score))

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"norm_mean": "#2ecc71", "norm_median": "#3498db", "norm_quantile": "#e74c3c"}
    markers = {"norm_mean": "o", "norm_median": "s", "norm_quantile": "^"}

    for method_name, points in data_points.items():
        if points:
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            ax.scatter(
                x_vals,
                y_vals,
                c=colors[method_name],
                marker=markers[method_name],
                label=method_name,
                alpha=0.6,
                s=50,
            )

            # Add trend line
            if len(points) > 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(
                    x_line,
                    p(x_line),
                    c=colors[method_name],
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                )

    ax.set_xlabel("Actual Missing Rate", fontsize=12)
    ax.set_ylabel("Overall Score", fontsize=12)
    ax.set_title("Method Performance vs Missing Rate", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {output_path}")


def create_batch_effect_comparison(results: list[TestResult], output_path: Path) -> None:
    """Create grouped bar chart comparing methods with/without batch effects.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    # Aggregate scores by batch effect presence
    method_names = ["norm_mean", "norm_median", "norm_quantile"]

    with_batch_scores: dict[str, list[float]] = {m: [] for m in method_names}
    without_batch_scores: dict[str, list[float]] = {m: [] for m in method_names}

    for result in results:
        if result.selected_method == "ERROR":
            continue

        for method in result.methods:
            if method.method_name in method_names:
                if result.scenario.with_batch:
                    with_batch_scores[method.method_name].append(method.overall_score)
                else:
                    without_batch_scores[method.method_name].append(method.overall_score)

    # Calculate averages
    with_batch_avg = [
        np.mean(with_batch_scores[m]) if with_batch_scores[m] else 0 for m in method_names
    ]
    without_batch_avg = [
        np.mean(without_batch_scores[m]) if without_batch_scores[m] else 0 for m in method_names
    ]

    # Create grouped bar chart
    x = np.arange(len(method_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2, without_batch_avg, width, label="No Batch Effect", color="#3498db"
    )
    bars2 = ax.bar(x + width / 2, with_batch_avg, width, label="With Batch Effect", color="#e74c3c")

    # Add value labels
    def add_labels(bars: list) -> None:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel("Normalization Method", fontsize=12)
    ax.set_ylabel("Average Overall Score", fontsize=12)
    ax.set_title("Method Performance: Batch Effect Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved batch effect comparison: {output_path}")


def create_distribution_comparison(results: list[TestResult], output_path: Path) -> None:
    """Create grouped bar chart comparing methods across distributions.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    method_names = ["norm_mean", "norm_median", "norm_quantile"]
    distributions = ["normal", "log_normal", "multimodal"]

    # Aggregate scores
    dist_scores: dict[str, dict[str, list[float]]] = {
        d: {m: [] for m in method_names} for d in distributions
    }

    for result in results:
        if result.selected_method == "ERROR":
            continue

        dist = result.scenario.distribution
        for method in result.methods:
            if method.method_name in method_names:
                dist_scores[dist][method.method_name].append(method.overall_score)

    # Calculate averages
    avg_scores: dict[str, dict[str, float]] = {}
    for dist in distributions:
        avg_scores[dist] = {}
        for method in method_names:
            scores = dist_scores[dist][method]
            avg_scores[dist][method] = np.mean(scores) if scores else 0.0

    # Create grouped bar chart
    x = np.arange(len(distributions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"norm_mean": "#2ecc71", "norm_median": "#3498db", "norm_quantile": "#e74c3c"}

    for i, method in enumerate(method_names):
        scores = [avg_scores[d][method] for d in distributions]
        bars = ax.bar(x + i * width, scores, width, label=method, color=colors[method])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Data Distribution", fontsize=12)
    ax.set_ylabel("Average Overall Score", fontsize=12)
    ax.set_title("Method Performance Across Data Distributions", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(distributions)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved distribution comparison: {output_path}")


def create_radar_chart(results: list[TestResult], output_path: Path) -> None:
    """Create radar chart showing method performance across all 4 metrics.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    method_names = ["norm_mean", "norm_median", "norm_quantile"]

    # Aggregate scores for each method and metric
    metric_scores: dict[str, dict[str, list[float]]] = {
        m: {metric: [] for metric in METRICS} for m in method_names
    }

    for result in results:
        if result.selected_method == "ERROR":
            continue

        for method in result.methods:
            if method.method_name in method_names:
                for metric in METRICS:
                    score = getattr(method, metric, 0.0)
                    metric_scores[method.method_name][metric].append(score)

    # Calculate averages
    avg_scores: dict[str, dict[str, float]] = {}
    for method in method_names:
        avg_scores[method] = {}
        for metric in METRICS:
            scores = metric_scores[method][metric]
            avg_scores[method][metric] = np.mean(scores) if scores else 0.0

    # Number of metrics
    num_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    colors = {"norm_mean": "#2ecc71", "norm_median": "#3498db", "norm_quantile": "#e74c3c"}

    for method in method_names:
        values = [avg_scores[method][metric] for metric in METRICS]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.25, color=colors[method])

    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [metric.replace("_", " ").title() for metric in METRICS], fontsize=10, fontweight="bold"
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.title(
        "Normalization Methods: Performance Radar Chart (4 Metrics)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved radar chart: {output_path}")


def create_metrics_heatmap(results: list[TestResult], output_path: Path) -> None:
    """Create heatmap showing all 4 metrics for each method.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    output_path : Path
        Output file path.
    """
    method_names = ["norm_mean", "norm_median", "norm_quantile"]

    # Aggregate scores for each method and metric
    metric_scores: dict[str, dict[str, list[float]]] = {
        m: {metric: [] for metric in METRICS} for m in method_names
    }

    for result in results:
        if result.selected_method == "ERROR":
            continue

        for method in result.methods:
            if method.method_name in method_names:
                for metric in METRICS:
                    score = getattr(method, metric, 0.0)
                    metric_scores[method.method_name][metric].append(score)

    # Calculate averages
    avg_matrix = np.zeros((len(method_names), len(METRICS)))
    for i, method in enumerate(method_names):
        for j, metric in enumerate(METRICS):
            scores = metric_scores[method][metric]
            avg_matrix[i, j] = np.mean(scores) if scores else 0.0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(avg_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in METRICS], fontsize=10)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)

    # Add score annotations
    for i in range(len(method_names)):
        for j in range(len(METRICS)):
            text = ax.text(
                j,
                i,
                f"{avg_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Score")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Normalization Methods: Metric Scores Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved metrics heatmap: {output_path}")


def create_visualizations(results: list[TestResult]) -> None:
    """Create all visualization plots.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    """
    print("\nCreating visualizations...")

    # Heatmap
    create_heatmap(results, OUTPUT_DIR / "normalization_heatmap.png")

    # Selection bar chart
    create_selection_bar_chart(results, OUTPUT_DIR / "normalization_selection_count.png")

    # Missing rate scatter
    create_missing_rate_scatter(results, OUTPUT_DIR / "normalization_missing_rate_scatter.png")

    # Batch effect comparison
    create_batch_effect_comparison(results, OUTPUT_DIR / "normalization_batch_comparison.png")

    # Distribution comparison
    create_distribution_comparison(
        results, OUTPUT_DIR / "normalization_distribution_comparison.png"
    )

    # Radar chart (new 4 metrics)
    create_radar_chart(results, OUTPUT_DIR / "normalization_radar.png")

    # Metrics heatmap (new)
    create_metrics_heatmap(results, OUTPUT_DIR / "normalization_metrics_heatmap.png")


# =============================================================================
# Summary Statistics
# =============================================================================


def compute_summary_statistics(results: list[TestResult]) -> dict[str, Any]:
    """Compute summary statistics from results.

    Parameters
    ----------
    results : list[TestResult]
        Test results.

    Returns
    -------
    dict[str, Any]
        Summary statistics.
    """
    successful_results = [r for r in results if r.selected_method != "ERROR"]

    # Method selection counts
    selection_counts: dict[str, int] = {}
    for r in successful_results:
        selection_counts[r.selected_method] = selection_counts.get(r.selected_method, 0) + 1

    # Average scores by method
    method_scores: dict[str, list[float]] = {}
    for r in successful_results:
        for m in r.methods:
            if m.method_name not in method_scores:
                method_scores[m.method_name] = []
            method_scores[m.method_name].append(m.overall_score)

    avg_scores = {m: np.mean(scores) for m, scores in method_scores.items()}

    # Average metric scores by method
    method_metric_scores: dict[str, dict[str, list[float]]] = {}
    for r in successful_results:
        for m in r.methods:
            if m.method_name not in method_metric_scores:
                method_metric_scores[m.method_name] = {metric: [] for metric in METRICS}
            for metric in METRICS:
                method_metric_scores[m.method_name][metric].append(getattr(m, metric, 0.0))

    avg_metric_scores: dict[str, dict[str, float]] = {}
    for method, metrics_dict in method_metric_scores.items():
        avg_metric_scores[method] = {}
        for metric, scores in metrics_dict.items():
            avg_metric_scores[method][metric] = np.mean(scores) if scores else 0.0

    # Scores by missing pattern
    pattern_scores: dict[str, dict[str, list[float]]] = {}
    for r in successful_results:
        pattern = r.scenario.missing_pattern
        if pattern not in pattern_scores:
            pattern_scores[pattern] = {}
        for m in r.methods:
            if m.method_name not in pattern_scores[pattern]:
                pattern_scores[pattern][m.method_name] = []
            pattern_scores[pattern][m.method_name].append(m.overall_score)

    # Scores by distribution
    dist_scores: dict[str, dict[str, list[float]]] = {}
    for r in successful_results:
        dist = r.scenario.distribution
        if dist not in dist_scores:
            dist_scores[dist] = {}
        for m in r.methods:
            if m.method_name not in dist_scores[dist]:
                dist_scores[dist][m.method_name] = []
            dist_scores[dist][m.method_name].append(m.overall_score)

    return {
        "total_scenarios": len(results),
        "successful_scenarios": len(successful_results),
        "failed_scenarios": len(results) - len(successful_results),
        "selection_counts": selection_counts,
        "average_scores": avg_scores,
        "average_metric_scores": avg_metric_scores,
        "total_time_seconds": sum(r.total_time for r in successful_results),
        "average_time_seconds": (
            np.mean([r.total_time for r in successful_results]) if successful_results else 0
        ),
    }


def print_detailed_summary(results: list[TestResult], summary: dict[str, Any]) -> None:
    """Print detailed summary of test results.

    Parameters
    ----------
    results : list[TestResult]
        Test results.
    summary : dict[str, Any]
        Summary statistics.
    """
    print("\n" + "=" * 70)
    print("DETAILED SUMMARY")
    print("=" * 70)

    print(f"\nTotal scenarios: {summary['total_scenarios']}")
    print(f"Successful: {summary['successful_scenarios']}")
    print(f"Failed: {summary['failed_scenarios']}")

    # Method selection counts
    print(f"\n{'─' * 50}")
    print("Method Selection Counts:")
    print(f"{'─' * 50}")
    for method, count in sorted(
        summary["selection_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        pct = (
            count / summary["successful_scenarios"] * 100
            if summary["successful_scenarios"] > 0
            else 0
        )
        print(f"  {method:20s}: {count:4d} ({pct:5.1f}%)")

    # Average overall scores
    print(f"\n{'─' * 50}")
    print("Average Overall Scores by Method:")
    print(f"{'─' * 50}")
    for method, score in sorted(
        summary["average_scores"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {method:20s}: {score:.4f}")

    # Average metric scores
    print(f"\n{'─' * 50}")
    print("Average Metric Scores by Method:")
    print(f"{'─' * 50}")

    # Header
    header = f"{'Method':20s}"
    for metric in METRICS:
        header += f" | {metric[:12]:12s}"
    print(f"  {header}")
    print(f"  {'-' * 20}+{'-' * 14 * len(METRICS)}")

    # Data rows
    for method in ["norm_mean", "norm_median", "norm_quantile"]:
        if method in summary["average_metric_scores"]:
            row = f"  {method:20s}"
            for metric in METRICS:
                score = summary["average_metric_scores"][method].get(metric, 0.0)
                row += f" | {score:12.4f}"
            print(row)

    # Performance by missing rate
    print(f"\n{'─' * 50}")
    print("Average Score by Missing Rate:")
    print(f"{'─' * 50}")

    missing_rate_data: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    for r in results:
        if r.selected_method != "ERROR":
            mr = r.scenario.missing_rate
            for m in r.methods:
                if m.method_name == r.selected_method:
                    missing_rate_data[mr].append(m.overall_score)
                    break

    for mr in ["low", "medium", "high"]:
        scores = missing_rate_data[mr]
        if scores:
            print(f"  {mr:10s}: {np.mean(scores):.4f} (n={len(scores)})")

    # Performance by distribution
    print(f"\n{'─' * 50}")
    print("Average Score by Distribution:")
    print(f"{'─' * 50}")

    dist_data: dict[str, list[float]] = {"normal": [], "log_normal": [], "multimodal": []}
    for r in results:
        if r.selected_method != "ERROR":
            dist = r.scenario.distribution
            for m in r.methods:
                if m.method_name == r.selected_method:
                    dist_data[dist].append(m.overall_score)
                    break

    for dist in ["normal", "log_normal", "multimodal"]:
        scores = dist_data[dist]
        if scores:
            print(f"  {dist:15s}: {np.mean(scores):.4f} (n={len(scores)})")

    # Timing
    print(f"\n{'─' * 50}")
    print("Timing:")
    print(f"{'─' * 50}")
    print(f"  Total time: {summary['total_time_seconds']:.2f}s")
    print(f"  Average time per scenario: {summary['average_time_seconds']:.2f}s")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the comprehensive normalization autoselect test."""
    print("=" * 70)
    print("Normalization Autoselect Comprehensive Test")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")

    # Generate scenarios
    scenarios = generate_scenarios()
    print(f"\nGenerated {len(scenarios)} test scenarios")
    print(f"  Scales: {SCALES}")
    print(f"  Missing rates: {MISSING_RATES}")
    print(f"  Missing patterns: {MISSING_PATTERNS}")
    print(f"  Distributions: {DISTRIBUTIONS}")
    print(f"  Batch variants: {BATCH_VARIANTS}")
    print(f"\n  Evaluation metrics (4): {METRICS}")

    # Run all scenarios
    start_time = time.perf_counter()
    results = run_all_scenarios(scenarios)
    total_time = time.perf_counter() - start_time

    print(f"\n{'=' * 70}")
    print(f"All scenarios completed in {total_time:.2f} seconds")
    print(f"{'=' * 70}")

    # Save results to JSON
    summary = compute_summary_statistics(results)
    results_dict = {
        "metadata": {
            "seed": SEED,
            "total_scenarios": len(scenarios),
            "total_time_seconds": total_time,
            "scales": SCALES,
            "missing_rates": MISSING_RATES,
            "missing_patterns": MISSING_PATTERNS,
            "distributions": DISTRIBUTIONS,
            "batch_variants": BATCH_VARIANTS,
            "metrics": METRICS,
        },
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    json_path = OUTPUT_DIR / "normalization_test_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Create visualizations
    create_visualizations(results)

    # Print detailed summary
    print_detailed_summary(results, summary)

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
