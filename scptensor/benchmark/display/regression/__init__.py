"""Performance regression detection for CI/CD benchmark integration.

This module provides tools for detecting performance regressions in benchmark
results and generating trend charts for continuous integration monitoring.

Main exports:
- RegressionThreshold: Dataclass defining regression detection thresholds
- RegressionChecker: Class for comparing current results against baseline
- TrendChartGenerator: Class for generating performance trend visualizations
- load_baseline: Load baseline results from file
- save_baseline: Save baseline results to file

Examples
--------
Load baseline and check for regressions:

>>> from scptensor.benchmark.display.regression import (
...     RegressionChecker,
...     RegressionThreshold,
...     load_baseline,
... )
>>> from scptensor.benchmark.core.result import BenchmarkResults
>>>
>>> # Define thresholds
>>> thresholds = RegressionThreshold(
...     runtime_increase_pct=10.0,  # 10% slower triggers regression
...     memory_increase_pct=15.0,   # 15% more memory triggers regression
...     accuracy_decrease_pct=5.0,  # 5% accuracy drop triggers regression
... )
>>>
>>> # Load baseline and current results
>>> baseline = load_baseline("baseline.json")
>>> current = BenchmarkResults.load_json("current_results.json")
>>>
>>> # Check for regressions
>>> checker = RegressionChecker(thresholds=thresholds)
>>> report = checker.check_regression(current, baseline)
>>>
>>> if report.has_regression:
...     print("Regression detected!")
...     for detail in report.regression_details:
...         print(f"  {detail.method}: {detail.metric} exceeded threshold")

Generate trend charts:

>>> from scptensor.benchmark.display.regression import TrendChartGenerator
>>> import json
>>>
>>> # Load historical data
>>> with open("history.json") as f:
...     history = json.load(f)
>>>
>>> # Generate trend visualization
>>> generator = TrendChartGenerator(output_dir="trends")
>>> generator.generate_runtime_trend(history, "log_normalize")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scptensor.benchmark.core.result import BenchmarkResults, MethodCategory

__all__ = [
    "RegressionThreshold",
    "RegressionDetail",
    "RegressionReport",
    "RegressionChecker",
    "TrendChartGenerator",
    "TrendDataPoint",
    "load_baseline",
    "save_baseline",
    "format_regression_message",
]

# =============================================================================
# Thresholds and Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class RegressionThreshold:
    """Thresholds for detecting performance regressions.

    Defines the percentage change thresholds that trigger regression
    warnings for different performance metrics.

    Parameters
    ----------
    runtime_increase_pct : float, default=10.0
        Maximum acceptable runtime increase as a percentage.
        Values above this trigger a regression warning.
    memory_increase_pct : float, default=15.0
        Maximum acceptable memory increase as a percentage.
        Values above this trigger a regression warning.
    accuracy_decrease_pct : float, default=5.0
        Maximum acceptable accuracy decrease as a percentage.
        Values below this trigger a regression warning.
    correlation_decrease_pct : float, default=3.0
        Maximum acceptable correlation decrease as a percentage.
        Values below this trigger a regression warning.
    min_absolute_runtime_sec : float, default=0.1
        Minimum absolute runtime difference in seconds to consider
        as regression (prevents false positives for very fast methods).
    min_absolute_memory_mb : float, default=10.0
        Minimum absolute memory difference in MB to consider
        as regression (prevents false positives for tiny changes).
    """

    runtime_increase_pct: float = 10.0
    memory_increase_pct: float = 15.0
    accuracy_decrease_pct: float = 5.0
    correlation_decrease_pct: float = 3.0
    min_absolute_runtime_sec: float = 0.1
    min_absolute_memory_mb: float = 10.0

    def __post_init__(self) -> None:
        """Validate threshold values.

        Raises
        ------
        ValueError
            If any threshold is negative.
        """
        for name, value in [
            ("runtime_increase_pct", self.runtime_increase_pct),
            ("memory_increase_pct", self.memory_increase_pct),
            ("accuracy_decrease_pct", self.accuracy_decrease_pct),
            ("correlation_decrease_pct", self.correlation_decrease_pct),
            ("min_absolute_runtime_sec", self.min_absolute_runtime_sec),
            ("min_absolute_memory_mb", self.min_absolute_memory_mb),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(slots=True)
class RegressionDetail:
    """Detail about a single detected regression.

    Attributes
    ----------
    category : MethodCategory
        The method category where regression was detected.
    method : str
        Name of the method with regression.
    metric : str
        Name of the metric that regressed.
    baseline_value : float
        Baseline value for the metric.
    current_value : float
        Current value for the metric.
    change_pct : float
        Percentage change from baseline (positive = worsened).
    severity : str
        Severity level: "minor", "moderate", or "severe".
    message : str
        Human-readable description of the regression.
    """

    category: MethodCategory
    method: str
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: str
    message: str

    def __str__(self) -> str:
        """Return string representation of the regression detail.

        Returns
        -------
        str
            Human-readable regression description.
        """
        return (
            f"[{self.severity.upper()}] {self.category.value}/{self.method}: "
            f"{self.metric} changed by {self.change_pct:+.1f}% "
            f"({self.baseline_value:.4f} -> {self.current_value:.4f})"
        )


@dataclass(slots=True)
class RegressionReport:
    """Report from regression checking.

    Attributes
    ----------
    has_regression : bool
        True if any regression was detected.
    passed : bool
        True if no regressions exceeded thresholds.
    regression_details : list[RegressionDetail]
        List of all detected regressions with details.
    summary : dict[str, Any]
        Summary statistics including counts by severity.
    timestamp : str
        ISO timestamp when the report was generated.
    baseline_version : str | None
        Version or identifier of the baseline used.
    """

    has_regression: bool
    passed: bool
    regression_details: list[RegressionDetail] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    baseline_version: str | None = None

    def generate_regression_report(self) -> str:
        """Generate a human-readable regression report.

        Returns
        -------
        str
            Formatted report with header, summary, and details.
        """
        lines = [
            "=" * 70,
            "Performance Regression Report",
            "=" * 70,
            f"Generated: {self.timestamp}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Baseline: {self.baseline_version or 'Unknown'}",
            "",
        ]

        # Summary section
        lines.extend(
            [
                "Summary",
                "-" * 70,
                f"Total regressions detected: {len(self.regression_details)}",
            ]
        )

        if self.summary:
            for severity, count in self.summary.get("by_severity", {}).items():
                lines.append(f"  {severity}: {count}")

        lines.append("")

        # Details section
        if self.regression_details:
            lines.extend(
                [
                    "Regression Details",
                    "-" * 70,
                ]
            )

            # Group by severity
            by_severity: dict[str, list[RegressionDetail]] = {}
            for detail in self.regression_details:
                by_severity.setdefault(detail.severity, []).append(detail)

            for severity in ["severe", "moderate", "minor"]:
                if severity in by_severity:
                    lines.append(f"\n{severity.upper()} Regressions:")
                    for detail in by_severity[severity]:
                        lines.append(f"  {detail}")
        else:
            lines.extend(
                [
                    "Regression Details",
                    "-" * 70,
                    "No regressions detected. All metrics within thresholds.",
                ]
            )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the report.
        """
        return {
            "has_regression": self.has_regression,
            "passed": self.passed,
            "regression_details": [
                {
                    "category": detail.category.value,
                    "method": detail.method,
                    "metric": detail.metric,
                    "baseline_value": detail.baseline_value,
                    "current_value": detail.current_value,
                    "change_pct": detail.change_pct,
                    "severity": detail.severity,
                    "message": detail.message,
                }
                for detail in self.regression_details
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "baseline_version": self.baseline_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionReport:
        """Create RegressionReport from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing regression report data.

        Returns
        -------
        RegressionReport
            Reconstructed regression report.
        """
        from scptensor.benchmark.core.result import MethodCategory

        details = [
            RegressionDetail(
                category=MethodCategory(d["category"]),
                method=d["method"],
                metric=d["metric"],
                baseline_value=d["baseline_value"],
                current_value=d["current_value"],
                change_pct=d["change_pct"],
                severity=d["severity"],
                message=d["message"],
            )
            for d in data.get("regression_details", [])
        ]

        return cls(
            has_regression=data["has_regression"],
            passed=data["passed"],
            regression_details=details,
            summary=data.get("summary", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            baseline_version=data.get("baseline_version"),
        )


@dataclass(slots=True)
class TrendDataPoint:
    """Single data point for trend charts.

    Attributes
    ----------
    timestamp : str
        ISO timestamp of when the data point was recorded.
    commit_hash : str | None
        Git commit hash for the data point.
    version : str | None
        Version string for the data point.
    runtime : float | None
        Runtime value in seconds.
    memory_mb : float | None
        Memory usage in megabytes.
    mse : float | None
        Mean squared error value.
    correlation : float | None
        Correlation coefficient value.
    metadata : dict[str, Any]
        Additional metadata for the data point.
    """

    timestamp: str
    commit_hash: str | None = None
    version: str | None = None
    runtime: float | None = None
    memory_mb: float | None = None
    mse: float | None = None
    correlation: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the data point.
        """
        return {
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
            "version": self.version,
            "runtime": self.runtime,
            "memory_mb": self.memory_mb,
            "mse": self.mse,
            "correlation": self.correlation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendDataPoint:
        """Create TrendDataPoint from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing data point values.

        Returns
        -------
        TrendDataPoint
            Reconstructed trend data point.
        """
        return cls(
            timestamp=data["timestamp"],
            commit_hash=data.get("commit_hash"),
            version=data.get("version"),
            runtime=data.get("runtime"),
            memory_mb=data.get("memory_mb"),
            mse=data.get("mse"),
            correlation=data.get("correlation"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Regression Checker
# =============================================================================


class RegressionChecker:
    """Checker for performance regressions in benchmark results.

    Compares current benchmark results against a baseline to detect
    performance regressions exceeding defined thresholds.

    Parameters
    ----------
    thresholds : RegressionThreshold, optional
        Threshold configuration for regression detection.
        If None, uses default thresholds.
    baseline_version : str | None, default=None
        Identifier for the baseline version being compared.

    Examples
    --------
    Check for regressions:

    >>> from scptensor.benchmark.display.regression import RegressionChecker
    >>> from scptensor.benchmark.core.result import BenchmarkResults
    >>>
    >>> baseline = BenchmarkResults.load_json("baseline.json")
    >>> current = BenchmarkResults.load_json("current.json")
    >>>
    >>> checker = RegressionChecker()
    >>> report = checker.check_regression(current, baseline)
    >>>
    >>> if not report.passed:
    ...     print(report.generate_regression_report())
    """

    # Metric definitions for comparison
    _METRIC_CONFIGS: dict[str, dict[str, Any]] = {
        "runtime": {
            "attr": "runtime_seconds",
            "threshold_attr": "runtime_increase_pct",
            "direction": "increase",
            "min_absolute_attr": "min_absolute_runtime_sec",
        },
        "memory": {
            "attr": "memory_mb",
            "threshold_attr": "memory_increase_pct",
            "direction": "increase",
            "min_absolute_attr": "min_absolute_memory_mb",
        },
        "mse": {
            "attr": "mse",
            "threshold_attr": "accuracy_decrease_pct",
            "direction": "increase",
            "min_absolute_attr": None,
        },
        "mae": {
            "attr": "mae",
            "threshold_attr": "accuracy_decrease_pct",
            "direction": "increase",
            "min_absolute_attr": None,
        },
        "correlation": {
            "attr": "correlation",
            "threshold_attr": "correlation_decrease_pct",
            "direction": "decrease",
            "min_absolute_attr": None,
        },
        "spearman_correlation": {
            "attr": "spearman_correlation",
            "threshold_attr": "correlation_decrease_pct",
            "direction": "decrease",
            "min_absolute_attr": None,
        },
        "cosine_similarity": {
            "attr": "cosine_similarity",
            "threshold_attr": "correlation_decrease_pct",
            "direction": "decrease",
            "min_absolute_attr": None,
        },
    }

    def __init__(
        self,
        thresholds: RegressionThreshold | None = None,
        baseline_version: str | None = None,
    ) -> None:
        """Initialize the regression checker.

        Parameters
        ----------
        thresholds : RegressionThreshold, optional
            Threshold configuration for regression detection.
            If None, uses default RegressionThreshold.
        baseline_version : str | None, default=None
            Identifier for the baseline version being compared.
        """
        self.thresholds = thresholds or RegressionThreshold()
        self.baseline_version = baseline_version

    def check_regression(
        self,
        current_results: BenchmarkResults,
        baseline: BenchmarkResults,
    ) -> RegressionReport:
        """Check current results against baseline for regressions.

        Compares all matching methods between current and baseline results,
        checking each configured metric against its threshold.

        Parameters
        ----------
        current_results : BenchmarkResults
            Current benchmark results to check.
        baseline : BenchmarkResults
            Baseline results to compare against.

        Returns
        -------
        RegressionReport
            Report containing all detected regressions and summary.
        """
        from scptensor.benchmark.core.result import MethodCategory

        regressions: list[RegressionDetail] = []

        # Check each category
        for category in MethodCategory:
            category_regressions = self._check_category(
                current_results,
                baseline,
                category,
            )
            regressions.extend(category_regressions)

        # Calculate severity and summary
        for r in regressions:
            r.severity = self._calculate_severity(r.change_pct, r.metric)

        summary = self._generate_summary(regressions)

        return RegressionReport(
            has_regression=len(regressions) > 0,
            passed=len(regressions) == 0,
            regression_details=regressions,
            summary=summary,
            baseline_version=self.baseline_version,
        )

    def _check_category(
        self,
        current_results: BenchmarkResults,
        baseline: BenchmarkResults,
        category: MethodCategory,
    ) -> list[RegressionDetail]:
        """Check a single category for regressions.

        Parameters
        ----------
        current_results : BenchmarkResults
            Current benchmark results.
        baseline : BenchmarkResults
            Baseline results.
        category : MethodCategory
            Category to check.

        Returns
        -------
        list[RegressionDetail]
            List of detected regressions in this category.
        """
        regressions: list[RegressionDetail] = []

        current_dict = current_results.get_results_by_category(category)
        baseline_dict = baseline.get_results_by_category(category)

        # Check each method that exists in both
        for method_name in set(current_dict.keys()) & set(baseline_dict.keys()):
            current_result = current_dict[method_name]
            baseline_result = baseline_dict[method_name]

            if not (current_result.is_success and baseline_result.is_success):
                continue

            # Check performance metrics
            metric_regressions = self._check_performance_metrics(
                method_name,
                category,
                current_result.performance,
                baseline_result.performance,
            )
            regressions.extend(metric_regressions)

            # Check accuracy metrics if available
            if current_result.accuracy and baseline_result.accuracy:
                accuracy_regressions = self._check_accuracy_metrics(
                    method_name,
                    category,
                    current_result.accuracy,
                    baseline_result.accuracy,
                )
                regressions.extend(accuracy_regressions)

        return regressions

    def _check_performance_metrics(
        self,
        method_name: str,
        category: MethodCategory,
        current_performance: Any,
        baseline_performance: Any,
    ) -> list[RegressionDetail]:
        """Check performance metrics for regressions.

        Parameters
        ----------
        method_name : str
            Name of the method.
        category : MethodCategory
            Method category.
        current_performance : PerformanceMetrics
            Current performance metrics.
        baseline_performance : PerformanceMetrics
            Baseline performance metrics.

        Returns
        -------
        list[RegressionDetail]
            List of detected performance regressions.
        """
        regressions: list[RegressionDetail] = []

        for metric_name, config in self._METRIC_CONFIGS.items():
            if metric_name not in ["runtime", "memory"]:
                continue

            baseline_value = getattr(baseline_performance, config["attr"])
            current_value = getattr(current_performance, config["attr"])

            change_pct = self._calculate_change_pct(
                baseline_value,
                current_value,
                config["direction"],
            )

            threshold = getattr(self.thresholds, config["threshold_attr"])
            min_absolute = getattr(self.thresholds, config["min_absolute_attr"])
            absolute_diff = abs(current_value - baseline_value)

            # Check if threshold exceeded with minimum absolute difference
            if change_pct > threshold and absolute_diff >= min_absolute:
                regressions.append(
                    RegressionDetail(
                        category=category,
                        method=method_name,
                        metric=metric_name,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change_pct=change_pct,
                        severity="",  # Will be calculated later
                        message=self._generate_regression_message(
                            metric_name, baseline_value, current_value, change_pct, threshold
                        ),
                    )
                )

        return regressions

    def _check_accuracy_metrics(
        self,
        method_name: str,
        category: MethodCategory,
        current_accuracy: Any,
        baseline_accuracy: Any,
    ) -> list[RegressionDetail]:
        """Check accuracy metrics for regressions.

        Parameters
        ----------
        method_name : str
            Name of the method.
        category : MethodCategory
            Method category.
        current_accuracy : AccuracyMetrics
            Current accuracy metrics.
        baseline_accuracy : AccuracyMetrics
            Baseline accuracy metrics.

        Returns
        -------
        list[RegressionDetail]
            List of detected accuracy regressions.
        """
        regressions: list[RegressionDetail] = []

        # For MSE/MAE, increase is bad
        for metric_name in ["mse", "mae"]:
            baseline_value = getattr(baseline_accuracy, metric_name, None)
            current_value = getattr(current_accuracy, metric_name, None)

            if baseline_value is not None and current_value is not None:
                change_pct = self._calculate_change_pct(baseline_value, current_value, "increase")
                threshold = self.thresholds.accuracy_decrease_pct

                if change_pct > threshold:
                    regressions.append(
                        RegressionDetail(
                            category=category,
                            method=method_name,
                            metric=metric_name,
                            baseline_value=baseline_value,
                            current_value=current_value,
                            change_pct=change_pct,
                            severity="",
                            message=self._generate_regression_message(
                                metric_name, baseline_value, current_value, change_pct, threshold
                            ),
                        )
                    )

        # For correlation metrics, decrease is bad
        for metric_name in ["correlation", "spearman_correlation", "cosine_similarity"]:
            baseline_value = getattr(baseline_accuracy, metric_name, None)
            current_value = getattr(current_accuracy, metric_name, None)

            if baseline_value is not None and current_value is not None:
                change_pct = self._calculate_change_pct(baseline_value, current_value, "decrease")
                threshold = self.thresholds.correlation_decrease_pct

                if change_pct > threshold:
                    regressions.append(
                        RegressionDetail(
                            category=category,
                            method=method_name,
                            metric=metric_name,
                            baseline_value=baseline_value,
                            current_value=current_value,
                            change_pct=change_pct,
                            severity="",
                            message=self._generate_regression_message(
                                metric_name, baseline_value, current_value, change_pct, threshold
                            ),
                        )
                    )

        return regressions

    def _calculate_change_pct(
        self,
        baseline: float,
        current: float,
        direction: str,
    ) -> float:
        """Calculate percentage change from baseline.

        Parameters
        ----------
        baseline : float
            Baseline value.
        current : float
            Current value.
        direction : {"increase", "decrease"}
            Direction that indicates a regression.

        Returns
        -------
        float
            Percentage change (always positive for regression).
        """
        if baseline == 0:
            return 0.0 if current == 0 else 100.0

        raw_change = ((current - baseline) / abs(baseline)) * 100

        if direction == "increase":
            return raw_change if raw_change > 0 else 0.0
        else:  # decrease
            return -raw_change if raw_change < 0 else 0.0

    def _calculate_severity(self, change_pct: float, metric: str) -> str:
        """Calculate severity level for a regression.

        Parameters
        ----------
        change_pct : float
            Percentage change from baseline.
        metric : str
            Name of the metric.

        Returns
        -------
        str
            Severity level: "minor", "moderate", or "severe".
        """
        if change_pct >= 50:
            return "severe"
        elif change_pct >= 20:
            return "moderate"
        else:
            return "minor"

    def _generate_regression_message(
        self,
        metric: str,
        baseline: float,
        current: float,
        change_pct: float,
        threshold: float,
    ) -> str:
        """Generate human-readable regression message.

        Parameters
        ----------
        metric : str
            Metric name.
        baseline : float
            Baseline value.
        current : float
            Current value.
        change_pct : float
            Percentage change.
        threshold : float
            Threshold that was exceeded.

        Returns
        -------
        str
            Human-readable message.
        """
        return (
            f"{metric} regressed by {change_pct:.1f}% (threshold: {threshold:.1f}%), "
            f"from {baseline:.4f} to {current:.4f}"
        )

    def _generate_summary(self, regressions: list[RegressionDetail]) -> dict[str, Any]:
        """Generate summary statistics from regression details.

        Parameters
        ----------
        regressions : list[RegressionDetail]
            List of detected regressions.

        Returns
        -------
        dict[str, Any]
            Summary with counts by severity and category.
        """
        from collections import Counter

        by_severity: Counter[str] = Counter()
        by_category: Counter[str] = Counter()
        by_metric: Counter[str] = Counter()

        for r in regressions:
            by_severity[r.severity] += 1
            by_category[r.category.value] += 1
            by_metric[r.metric] += 1

        return {
            "total": len(regressions),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "by_metric": dict(by_metric),
        }


# =============================================================================
# Trend Chart Generator
# =============================================================================


class TrendChartGenerator:
    """Generator for performance trend visualization charts.

    Creates line plots showing performance metrics over time,
    useful for CI/CD monitoring and historical analysis.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated trend charts.
    plot_style : Literal["science", "science_grid", "ieee", "nature"], default="science"
        Matplotlib style to use for plots.
    plot_dpi : int, default=300
        Resolution in dots per inch for saved figures.
    figure_format : Literal["png", "pdf", "svg"], default="png"
        Format for saved figures.

    Examples
    --------
    Generate runtime trend chart:

    >>> from scptensor.benchmark.display.regression import TrendChartGenerator
    >>>
    >>> # Prepare historical data
    >>> history = [
    ...     {"timestamp": "2024-01-01T00:00:00", "runtime": 1.2, "commit": "abc123"},
    ...     {"timestamp": "2024-01-02T00:00:00", "runtime": 1.15, "commit": "def456"},
    ... ]
    >>>
    >>> generator = TrendChartGenerator(output_dir="trends")
    >>> path = generator.generate_runtime_trend(history, "log_normalize")
    >>> print(f"Trend chart saved to: {path}")

    Generate all trend types for multiple methods:

    >>> method_history = {
    ...     "log_normalize": history_log,
    ...     "z_score": history_zscore,
    ... }
    >>>
    >>> for method, data in method_history.items():
    ...     generator.generate_runtime_trend(data, method)
    ...     generator.generate_memory_trend(data, method)
    """

    _VALID_STYLES: frozenset[str] = frozenset({"science", "science_grid", "ieee", "nature"})
    _VALID_FORMATS: frozenset[str] = frozenset({"png", "pdf", "svg"})

    def __init__(
        self,
        output_dir: str | Path = "benchmark_results",
        plot_style: Literal["science", "science_grid", "ieee", "nature"] = "science",
        plot_dpi: int = 300,
        figure_format: Literal["png", "pdf", "svg"] = "png",
    ) -> None:
        """Initialize the trend chart generator.

        Parameters
        ----------
        output_dir : str | Path, default="benchmark_results"
            Directory path for saving generated trend charts.
        plot_style : Literal["science", "science_grid", "ieee", "nature"], default="science"
            Matplotlib style to use for plots.
        plot_dpi : int, default=300
            Resolution in dots per inch for saved figures.
        figure_format : Literal["png", "pdf", "svg"], default="png"
            Format for saved figures.

        Raises
        ------
        ValueError
            If plot_style or figure_format is invalid.
        """
        # Validate inputs
        if plot_style not in self._VALID_STYLES:
            raise ValueError(
                f"Invalid plot_style: {plot_style}. Choose from {sorted(self._VALID_STYLES)}"
            )
        if figure_format not in self._VALID_FORMATS:
            raise ValueError(
                f"Invalid figure_format: {figure_format}. Choose from {sorted(self._VALID_FORMATS)}"
            )

        self.output_dir: Path = Path(output_dir)
        self.plot_style: Literal["science", "science_grid", "ieee", "nature"] = plot_style
        self.plot_dpi = plot_dpi
        self.figure_format: Literal["png", "pdf", "svg"] = figure_format

        # Create output directory
        self._trends_dir = self.output_dir / "trends"
        self._trends_dir.mkdir(parents=True, exist_ok=True)

    def generate_runtime_trend(
        self,
        history_data: list[dict[str, Any]],
        method_name: str,
        output_filename: str | None = None,
    ) -> Path:
        """Generate a line plot of runtime over time.

        Creates a trend chart showing runtime progression across
        historical benchmark runs, useful for identifying performance
        degradation or improvement.

        Parameters
        ----------
        history_data : list[dict[str, Any]]
            List of historical data points. Each dict should contain:
            - "timestamp": ISO timestamp string
            - "runtime": Runtime value in seconds
            - Optional "commit_hash": Git commit hash
            - Optional "version": Version string
        method_name : str
            Name of the method for the chart title and filename.
        output_filename : str | None, default=None
            Custom output filename (without extension).
            If None, generates as "{method_name}_runtime_trend.{format}".

        Returns
        -------
        Path
            Path to the saved trend chart file.

        Raises
        ------
        ValueError
            If history_data is empty or missing required fields.
        """
        if not history_data:
            raise ValueError("history_data cannot be empty")

        # Extract data points
        data_points = [self._parse_data_point(d) for d in history_data]
        data_points = [dp for dp in data_points if dp.runtime is not None]

        if not data_points:
            raise ValueError("No valid runtime data found in history_data")

        # Create plot
        fig, ax = self._create_trend_plot(
            data_points=data_points,
            metric_name="Runtime",
            metric_key="runtime",
            ylabel="Runtime (seconds)",
            color="#2E86AB",
        )

        ax.set_title(f"{method_name}: Runtime Trend", fontsize=12, fontweight="bold")

        # Save figure
        if output_filename is None:
            output_filename = f"{method_name}_runtime_trend"
        output_path = self._trends_dir / f"{output_filename}.{self.figure_format}"

        self._save_figure(fig, output_path)
        return output_path

    def generate_memory_trend(
        self,
        history_data: list[dict[str, Any]],
        method_name: str,
        output_filename: str | None = None,
    ) -> Path:
        """Generate a line plot of memory usage over time.

        Creates a trend chart showing memory usage progression across
        historical benchmark runs, useful for identifying memory leaks
        or optimization opportunities.

        Parameters
        ----------
        history_data : list[dict[str, Any]]
            List of historical data points. Each dict should contain:
            - "timestamp": ISO timestamp string
            - "memory_mb": Memory value in megabytes
            - Optional "commit_hash": Git commit hash
            - Optional "version": Version string
        method_name : str
            Name of the method for the chart title and filename.
        output_filename : str | None, default=None
            Custom output filename (without extension).
            If None, generates as "{method_name}_memory_trend.{format}".

        Returns
        -------
        Path
            Path to the saved trend chart file.

        Raises
        ------
        ValueError
            If history_data is empty or missing required fields.
        """
        if not history_data:
            raise ValueError("history_data cannot be empty")

        # Extract data points
        data_points = [self._parse_data_point(d) for d in history_data]
        data_points = [dp for dp in data_points if dp.memory_mb is not None]

        if not data_points:
            raise ValueError("No valid memory data found in history_data")

        # Create plot
        fig, ax = self._create_trend_plot(
            data_points=data_points,
            metric_name="Memory",
            metric_key="memory_mb",
            ylabel="Memory Usage (MB)",
            color="#A23B72",
        )

        ax.set_title(f"{method_name}: Memory Usage Trend", fontsize=12, fontweight="bold")

        # Save figure
        if output_filename is None:
            output_filename = f"{method_name}_memory_trend"
        output_path = self._trends_dir / f"{output_filename}.{self.figure_format}"

        self._save_figure(fig, output_path)
        return output_path

    def generate_accuracy_trend(
        self,
        history_data: list[dict[str, Any]],
        method_name: str,
        metric: str = "mse",
        output_filename: str | None = None,
    ) -> Path:
        """Generate a line plot of accuracy metrics over time.

        Creates a trend chart showing accuracy metric progression across
        historical benchmark runs. Supports MSE, MAE, and correlation metrics.

        Parameters
        ----------
        history_data : list[dict[str, Any]]
            List of historical data points. Each dict should contain:
            - "timestamp": ISO timestamp string
            - One of: "mse", "mae", "correlation", "spearman_correlation"
            - Optional "commit_hash": Git commit hash
            - Optional "version": Version string
        method_name : str
            Name of the method for the chart title and filename.
        metric : str, default="mse"
            Accuracy metric to plot. Options: "mse", "mae",
            "correlation", "spearman_correlation".
        output_filename : str | None, default=None
            Custom output filename (without extension).
            If None, generates as "{method_name}_{metric}_trend.{format}".

        Returns
        -------
        Path
            Path to the saved trend chart file.

        Raises
        ------
        ValueError
            If history_data is empty or metric is invalid.
        """
        _valid_metrics = {"mse", "mae", "correlation", "spearman_correlation"}
        if metric not in _valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Choose from {_valid_metrics}")

        if not history_data:
            raise ValueError("history_data cannot be empty")

        # Extract data points
        data_points = [self._parse_data_point(d) for d in history_data]

        # Filter by metric availability
        metric_value_key: dict[str, str] = {
            "mse": "mse",
            "mae": "mae",
            "correlation": "correlation",
            "spearman_correlation": "correlation",
        }
        value_key = metric_value_key[metric]
        data_points = [dp for dp in data_points if getattr(dp, value_key, None) is not None]

        if not data_points:
            raise ValueError(f"No valid {metric} data found in history_data")

        # Configure metric-specific settings
        metric_settings: dict[str, dict[str, Any]] = {
            "mse": {
                "ylabel": "Mean Squared Error",
                "color": "#F18F01",
                "lower_is_better": True,
            },
            "mae": {
                "ylabel": "Mean Absolute Error",
                "color": "#C73E1D",
                "lower_is_better": True,
            },
            "correlation": {
                "ylabel": "Pearson Correlation",
                "color": "#6A994E",
                "lower_is_better": False,
            },
            "spearman_correlation": {
                "ylabel": "Spearman Correlation",
                "color": "#BC4B51",
                "lower_is_better": False,
            },
        }

        settings = metric_settings[metric]

        # Create plot
        fig, ax = self._create_trend_plot(
            data_points=data_points,
            metric_name=settings["ylabel"],
            metric_key=value_key,
            ylabel=settings["ylabel"],
            color=settings["color"],
        )

        display_name = metric.replace("_", " ").title()
        ax.set_title(f"{method_name}: {display_name} Trend", fontsize=12, fontweight="bold")

        # Add ideal value reference for correlation metrics
        if not settings["lower_is_better"]:
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
            ax.legend()

        # Save figure
        if output_filename is None:
            output_filename = f"{method_name}_{metric}_trend"
        output_path = self._trends_dir / f"{output_filename}.{self.figure_format}"

        self._save_figure(fig, output_path)
        return output_path

    def _create_trend_plot(
        self,
        data_points: list[TrendDataPoint],
        metric_name: str,
        metric_key: str,
        ylabel: str,
        color: str,
    ) -> tuple[Any, Any]:
        """Create a trend line plot.

        Parameters
        ----------
        data_points : list[TrendDataPoint]
            List of data points to plot.
        metric_name : str
            Name of the metric being plotted.
        metric_key : str
            Key to access metric value in data points.
        ylabel : str
            Y-axis label.
        color : str
            Color for the line plot.

        Returns
        -------
        tuple
            (figure, axes) from matplotlib.
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import date2num

        from scptensor.viz.base.style import PlotStyle

        # Apply style
        PlotStyle.apply_style(theme=self.plot_style, dpi=self.plot_dpi)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Parse timestamps and extract values
        timestamps: list[datetime] = []
        values: list[float] = []
        labels: list[str] = []

        for i, dp in enumerate(data_points):
            try:
                ts = datetime.fromisoformat(dp.timestamp.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.fromisoformat(dp.timestamp)

            timestamps.append(ts)
            value = getattr(dp, metric_key, None)
            if value is not None:
                values.append(float(value))

            # Create label from commit or version
            if dp.commit_hash:
                labels.append(dp.commit_hash[:7])
            elif dp.version:
                labels.append(dp.version)
            else:
                labels.append(str(i))

        # Plot the trend line
        ax.plot(
            timestamps,
            values,
            marker="o",
            linestyle="-",
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Add reference line to first point (baseline)
        if len(values) > 1:
            ax.axhline(
                y=values[0],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Baseline",
            )
            ax.legend()

        # Formatting
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for readability
        fig.autofmt_xdate()

        # Add data point annotations
        for i, (ts, val, label) in enumerate(zip(timestamps, values, labels, strict=True)):
            ax.annotate(
                label,
                (date2num(ts), val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

        return fig, ax

    def _parse_data_point(self, data: dict[str, Any]) -> TrendDataPoint:
        """Parse a data point from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing data point values.

        Returns
        -------
        TrendDataPoint
            Parsed trend data point.
        """
        if isinstance(data, TrendDataPoint):
            return data

        return TrendDataPoint.from_dict(data)

    def _save_figure(self, fig: Any, path: Path) -> None:
        """Save a matplotlib figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        path : Path
            Output file path.
        """
        fig.savefig(
            path,
            dpi=self.plot_dpi,
            format=self.figure_format,
            bbox_inches="tight",
        )
        from matplotlib.pyplot import close

        close(fig)


# =============================================================================
# Baseline File I/O
# =============================================================================


def load_baseline(path: str | Path) -> Any:
    """Load baseline benchmark results from file.

    Loads previously saved baseline results for regression comparison.
    Supports both the legacy BenchmarkResults class and the new one.

    Parameters
    ----------
    path : str | Path
        Path to the baseline file. Can be JSON (.json) or any format
        supported by BenchmarkResults.load_json().

    Returns
    -------
    BenchmarkResults
        Loaded baseline benchmark results.

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    ValueError
        If the file format is not supported or data is invalid.

    Examples
    --------
    Load baseline from JSON file:

    >>> from scptensor.benchmark.display.regression import load_baseline
    >>> baseline = load_baseline("baseline.json")
    >>> print(f"Loaded baseline with {len(baseline.runs)} runs")
    """

    from scptensor.benchmark.core import BenchmarkResults as LegacyBenchmarkResults
    from scptensor.benchmark.core_legacy import MethodRunResult

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    with open(path_obj) as f:
        data = json.load(f)

    # Check if this is legacy format (has 'runs' key)
    if "runs" in data:
        # Load as legacy BenchmarkResults
        results = LegacyBenchmarkResults()
        results.metadata = data.get("metadata", {})

        # Recreate MethodRunResult objects (simplified version)
        # Note: This is a simplified reconstruction - full reconstruction would
        # require recreating ScpContainer objects which may not be available
        for run_data in data.get("runs", []):
            # Create a simplified result dict for compatibility
            results.runs.append(run_data)  # Store raw data for now

        return results
    else:
        # Try new format
        from scptensor.benchmark.core.result import BenchmarkResults as NewBenchmarkResults

        return NewBenchmarkResults.from_dict(data)


def save_baseline(
    path: str | Path,
    results: Any,
) -> None:
    """Save benchmark results as baseline for future regression checks.

    Saves current benchmark results to a file that can be loaded later
    as a baseline for regression detection.

    Parameters
    ----------
    path : str | Path
        Output file path. Should have .json extension for proper format.
    results : BenchmarkResults
        Benchmark results to save as baseline.

    Raises
    ------
    ValueError
        If path does not have .json extension.

    Examples
    --------
    Save current results as baseline:

    >>> from scptensor.benchmark.display.regression import save_baseline
    >>> from scptensor.benchmark.core import BenchmarkResults
    >>>
    >>> current_results = BenchmarkResults()
    >>> # ... populate results ...
    >>> save_baseline("baseline.json", current_results)
    """

    path_obj = Path(path)

    if path_obj.suffix != ".json":
        raise ValueError(f"Baseline file must have .json extension, got: {path_obj.suffix}")

    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Try using save_json method if available
    if hasattr(results, "save_json"):
        results.save_json(str(path_obj))
    else:
        # Fallback: manually serialize
        data = {
            "runs": [r.to_dict() for r in results.runs],
            "datasets": {},
            "metadata": results.metadata,
        }
        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================


def format_regression_message(
    metric: str,
    baseline: float,
    current: float,
    threshold: float,
) -> str:
    """Format a regression message for CI/CD output.

    Creates a concise, human-readable message suitable for
    CI/CD log output or PR comments.

    Parameters
    ----------
    metric : str
        Name of the metric that regressed.
    baseline : float
        Baseline value.
    current : float
        Current value.
    threshold : float
        Threshold that was exceeded.

    Returns
    -------
    str
        Formatted regression message.

    Examples
    --------
    >>> from scptensor.benchmark.display.regression import format_regression_message
    >>> msg = format_regression_message("runtime", 1.0, 1.15, 10.0)
    >>> print(msg)
    runtime: 1.000 -> 1.150 (+15.0%, threshold: 10.0%)
    """
    if baseline == 0:
        change_pct = 100.0 if current != 0 else 0.0
    else:
        change_pct = ((current - baseline) / abs(baseline)) * 100

    direction = "+" if change_pct > 0 else ""
    return (
        f"{metric}: {baseline:.4f} -> {current:.4f} "
        f"({direction}{change_pct:.1f}%, threshold: {threshold:.1f}%)"
    )
