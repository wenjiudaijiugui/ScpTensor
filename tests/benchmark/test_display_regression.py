"""Tests for the regression detection module."""

import pytest

# Import legacy BenchmarkResults for baseline I/O tests
from scptensor.benchmark.core import BenchmarkResults as LegacyBenchmarkResults

# Import new result classes directly
from scptensor.benchmark.core.result import (
    AccuracyMetrics,
    BenchmarkResult,
    BenchmarkResults,
    ComparisonLayer,
    MethodCategory,
    MethodSpec,
    PerformanceMetrics,
)
from scptensor.benchmark.display.regression import (
    RegressionChecker,
    RegressionDetail,
    RegressionReport,
    RegressionThreshold,
    TrendChartGenerator,
    TrendDataPoint,
    format_regression_message,
    load_baseline,
    save_baseline,
)


class TestRegressionThreshold:
    """Tests for RegressionThreshold dataclass."""

    def test_default_thresholds(self):
        """Test that default thresholds are set correctly."""
        threshold = RegressionThreshold()
        assert threshold.runtime_increase_pct == 10.0
        assert threshold.memory_increase_pct == 15.0
        assert threshold.accuracy_decrease_pct == 5.0
        assert threshold.correlation_decrease_pct == 3.0
        assert threshold.min_absolute_runtime_sec == 0.1
        assert threshold.min_absolute_memory_mb == 10.0

    def test_custom_thresholds(self):
        """Test setting custom threshold values."""
        threshold = RegressionThreshold(
            runtime_increase_pct=20.0,
            memory_increase_pct=25.0,
            accuracy_decrease_pct=10.0,
            correlation_decrease_pct=5.0,
        )
        assert threshold.runtime_increase_pct == 20.0
        assert threshold.memory_increase_pct == 25.0
        assert threshold.accuracy_decrease_pct == 10.0
        assert threshold.correlation_decrease_pct == 5.0

    def test_negative_threshold_raises_error(self):
        """Test that negative thresholds raise ValueError."""
        with pytest.raises(ValueError, match="runtime_increase_pct must be non-negative"):
            RegressionThreshold(runtime_increase_pct=-1.0)

        with pytest.raises(ValueError, match="memory_increase_pct must be non-negative"):
            RegressionThreshold(memory_increase_pct=-1.0)

        with pytest.raises(ValueError, match="accuracy_decrease_pct must be non-negative"):
            RegressionThreshold(accuracy_decrease_pct=-1.0)

        with pytest.raises(ValueError, match="correlation_decrease_pct must be non-negative"):
            RegressionThreshold(correlation_decrease_pct=-1.0)

    def test_frozen_immutable(self):
        """Test that RegressionThreshold is frozen (immutable)."""
        threshold = RegressionThreshold()
        with pytest.raises(Exception):  # FrozenInstanceError
            threshold.runtime_increase_pct = 20.0


class TestRegressionDetail:
    """Tests for RegressionDetail dataclass."""

    def test_regression_detail_creation(self):
        """Test creating a regression detail."""
        from scptensor.benchmark.core.result import MethodCategory

        detail = RegressionDetail(
            category=MethodCategory.NORMALIZATION,
            method="test_method",
            metric="runtime",
            baseline_value=1.0,
            current_value=1.2,
            change_pct=20.0,
            severity="moderate",
            message="Runtime increased by 20%",
        )
        assert detail.category == MethodCategory.NORMALIZATION
        assert detail.method == "test_method"
        assert detail.metric == "runtime"
        assert detail.baseline_value == 1.0
        assert detail.current_value == 1.2
        assert detail.change_pct == 20.0
        assert detail.severity == "moderate"

    def test_regression_detail_string_representation(self):
        """Test string representation of regression detail."""
        detail = RegressionDetail(
            category=MethodCategory.NORMALIZATION,
            method="test_method",
            metric="runtime",
            baseline_value=1.0,
            current_value=1.2,
            change_pct=20.0,
            severity="moderate",
            message="Runtime increased by 20%",
        )
        result = str(detail)
        assert "[MODERATE]" in result
        assert "normalization" in result
        assert "test_method" in result
        assert "runtime" in result
        assert "+20.0%" in result


class TestRegressionReport:
    """Tests for RegressionReport dataclass."""

    def test_passed_report(self):
        """Test creating a passed regression report."""
        report = RegressionReport(
            has_regression=False,
            passed=True,
            regression_details=[],
        )
        assert report.has_regression is False
        assert report.passed is True
        assert len(report.regression_details) == 0

    def test_failed_report(self):
        """Test creating a failed regression report with details."""
        from scptensor.benchmark.core.result import MethodCategory

        details = [
            RegressionDetail(
                category=MethodCategory.NORMALIZATION,
                method="test_method",
                metric="runtime",
                baseline_value=1.0,
                current_value=1.2,
                change_pct=20.0,
                severity="moderate",
                message="Runtime increased by 20%",
            )
        ]
        report = RegressionReport(
            has_regression=True,
            passed=False,
            regression_details=details,
        )
        assert report.has_regression is True
        assert report.passed is False
        assert len(report.regression_details) == 1

    def test_generate_regression_report_passed(self):
        """Test generating a human-readable report for passed status."""
        report = RegressionReport(
            has_regression=False,
            passed=True,
            regression_details=[],
            baseline_version="v1.0.0",
        )
        text = report.generate_regression_report()
        assert "PASSED" in text
        assert "v1.0.0" in text
        assert "No regressions detected" in text

    def test_generate_regression_report_failed(self):
        """Test generating a human-readable report for failed status."""
        details = [
            RegressionDetail(
                category=MethodCategory.NORMALIZATION,
                method="test_method",
                metric="runtime",
                baseline_value=1.0,
                current_value=1.2,
                change_pct=20.0,
                severity="moderate",
                message="Runtime increased by 20%",
            )
        ]
        report = RegressionReport(
            has_regression=True,
            passed=False,
            regression_details=details,
        )
        text = report.generate_regression_report()
        assert "FAILED" in text
        assert "Total regressions detected: 1" in text
        assert "MODERATE" in text

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        details = [
            RegressionDetail(
                category=MethodCategory.NORMALIZATION,
                method="test_method",
                metric="runtime",
                baseline_value=1.0,
                current_value=1.2,
                change_pct=20.0,
                severity="moderate",
                message="Runtime increased by 20%",
            )
        ]
        report = RegressionReport(
            has_regression=True,
            passed=False,
            regression_details=details,
            baseline_version="v1.0.0",
        )
        data = report.to_dict()
        assert data["has_regression"] is True
        assert data["passed"] is False
        assert data["baseline_version"] == "v1.0.0"
        assert len(data["regression_details"]) == 1

    def test_report_from_dict(self):
        """Test creating report from dictionary."""
        data = {
            "has_regression": True,
            "passed": False,
            "regression_details": [
                {
                    "category": "normalization",
                    "method": "test_method",
                    "metric": "runtime",
                    "baseline_value": 1.0,
                    "current_value": 1.2,
                    "change_pct": 20.0,
                    "severity": "moderate",
                    "message": "Runtime increased by 20%",
                }
            ],
            "summary": {},
            "timestamp": "2024-01-01T00:00:00",
            "baseline_version": "v1.0.0",
        }
        report = RegressionReport.from_dict(data)
        assert report.has_regression is True
        assert report.passed is False
        assert len(report.regression_details) == 1
        assert report.baseline_version == "v1.0.0"


class TestRegressionChecker:
    """Tests for RegressionChecker class."""

    @pytest.fixture
    def method_spec(self):
        """Create a test method specification."""
        return MethodSpec(
            name="test_method",
            display_name="Test Method",
            category=MethodCategory.NORMALIZATION,
            layer=ComparisonLayer.SHARED,
            framework="scptensor",
            description="Test method",
        )

    @pytest.fixture
    def baseline_result(self, method_spec):
        """Create a baseline benchmark result."""
        return BenchmarkResult(
            method_spec=method_spec,
            performance=PerformanceMetrics(
                runtime_seconds=1.0,
                memory_mb=100.0,
                throughput=1000.0,
            ),
            accuracy=AccuracyMetrics(
                mse=0.01,
                mae=0.05,
                correlation=0.95,
                spearman_correlation=0.94,
                cosine_similarity=0.96,
            ),
        )

    @pytest.fixture
    def current_result_no_regression(self, method_spec):
        """Create current result with no regression."""
        return BenchmarkResult(
            method_spec=method_spec,
            performance=PerformanceMetrics(
                runtime_seconds=0.95,  # 5% faster
                memory_mb=95.0,  # 5% less memory
                throughput=1052.63,
            ),
            accuracy=AccuracyMetrics(
                mse=0.009,  # Better
                mae=0.047,
                correlation=0.96,  # Better
                spearman_correlation=0.95,
                cosine_similarity=0.97,
            ),
        )

    @pytest.fixture
    def current_result_with_regression(self, method_spec):
        """Create current result with regressions."""
        return BenchmarkResult(
            method_spec=method_spec,
            performance=PerformanceMetrics(
                runtime_seconds=1.2,  # 20% slower - regression
                memory_mb=120.0,  # 20% more memory - regression
                throughput=833.33,
            ),
            accuracy=AccuracyMetrics(
                mse=0.012,  # 20% worse - regression
                mae=0.06,
                correlation=0.90,  # 5.26% worse - regression
                spearman_correlation=0.89,
                cosine_similarity=0.91,
            ),
        )

    def test_checker_initialization(self):
        """Test RegressionChecker initialization."""
        checker = RegressionChecker()
        assert checker.thresholds.runtime_increase_pct == 10.0
        assert checker.baseline_version is None

    def test_checker_with_custom_thresholds(self):
        """Test RegressionChecker with custom thresholds."""
        thresholds = RegressionThreshold(
            runtime_increase_pct=20.0,
            memory_increase_pct=25.0,
        )
        checker = RegressionChecker(thresholds=thresholds)
        assert checker.thresholds.runtime_increase_pct == 20.0
        assert checker.thresholds.memory_increase_pct == 25.0

    def test_check_regression_no_regression(self, baseline_result, current_result_no_regression):
        """Test regression check with no regressions."""
        baseline = BenchmarkResults()
        current = BenchmarkResults()

        baseline.add_result(baseline_result)
        current.add_result(current_result_no_regression)

        checker = RegressionChecker()
        report = checker.check_regression(current, baseline)

        assert report.passed is True
        assert report.has_regression is False
        assert len(report.regression_details) == 0

    def test_check_regression_with_regression(
        self, baseline_result, current_result_with_regression
    ):
        """Test regression check with regressions detected."""
        baseline = BenchmarkResults()
        current = BenchmarkResults()

        baseline.add_result(baseline_result)
        current.add_result(current_result_with_regression)

        thresholds = RegressionThreshold(
            runtime_increase_pct=10.0,
            memory_increase_pct=15.0,
            accuracy_decrease_pct=5.0,
            correlation_decrease_pct=3.0,
        )
        checker = RegressionChecker(thresholds=thresholds)
        report = checker.check_regression(current, baseline)

        assert report.passed is False
        assert report.has_regression is True
        assert len(report.regression_details) > 0

    def test_check_regression_below_minimum_absolute(self, baseline_result, method_spec):
        """Test that small absolute differences don't trigger regression."""
        baseline = BenchmarkResults()
        current = BenchmarkResults()

        baseline.add_result(baseline_result)

        # Create result with very small absolute difference
        current_result = BenchmarkResult(
            method_spec=method_spec,
            performance=PerformanceMetrics(
                runtime_seconds=1.05,  # 5% increase but only 0.05 sec
                memory_mb=100.5,  # 0.5% increase but only 0.5 MB
                throughput=952.38,
            ),
        )
        current.add_result(current_result)

        thresholds = RegressionThreshold(
            runtime_increase_pct=10.0,
            min_absolute_runtime_sec=0.1,  # Require 0.1 sec difference
            min_absolute_memory_mb=10.0,  # Require 10 MB difference
        )
        checker = RegressionChecker(thresholds=thresholds)
        report = checker.check_regression(current, baseline)

        # Should not trigger regression due to minimum absolute difference
        assert report.passed is True

    def test_check_regression_multiple_categories(self):
        """Test regression checking across multiple categories."""
        baseline = BenchmarkResults()
        current = BenchmarkResults()

        for category in [MethodCategory.NORMALIZATION, MethodCategory.IMPUTATION]:
            spec = MethodSpec(
                name=f"test_{category.value}",
                display_name=f"Test {category.value}",
                category=category,
                layer=ComparisonLayer.SHARED,
                framework="scptensor",
                description="Test method",
            )

            baseline_result = BenchmarkResult(
                method_spec=spec,
                performance=PerformanceMetrics(
                    runtime_seconds=1.0,
                    memory_mb=100.0,
                    throughput=1000.0,
                ),
            )

            current_result = BenchmarkResult(
                method_spec=spec,
                performance=PerformanceMetrics(
                    runtime_seconds=1.2,  # Regression
                    memory_mb=120.0,  # Regression
                    throughput=833.33,
                ),
            )

            baseline.add_result(baseline_result)
            current.add_result(current_result)

        checker = RegressionChecker()
        report = checker.check_regression(current, baseline)

        # Should detect regressions in both categories
        assert report.has_regression is True
        assert len(report.regression_details) >= 2  # At least runtime and memory per category


class TestTrendDataPoint:
    """Tests for TrendDataPoint dataclass."""

    def test_data_point_creation(self):
        """Test creating a trend data point."""
        dp = TrendDataPoint(
            timestamp="2024-01-01T00:00:00",
            commit_hash="abc123",
            version="v1.0.0",
            runtime=1.2,
            memory_mb=100.0,
            mse=0.01,
            correlation=0.95,
        )
        assert dp.timestamp == "2024-01-01T00:00:00"
        assert dp.commit_hash == "abc123"
        assert dp.version == "v1.0.0"
        assert dp.runtime == 1.2
        assert dp.memory_mb == 100.0
        assert dp.mse == 0.01
        assert dp.correlation == 0.95

    def test_data_point_to_dict(self):
        """Test converting data point to dictionary."""
        dp = TrendDataPoint(
            timestamp="2024-01-01T00:00:00",
            runtime=1.2,
        )
        data = dp.to_dict()
        assert data["timestamp"] == "2024-01-01T00:00:00"
        assert data["runtime"] == 1.2

    def test_data_point_from_dict(self):
        """Test creating data point from dictionary."""
        data = {
            "timestamp": "2024-01-01T00:00:00",
            "commit_hash": "abc123",
            "runtime": 1.2,
            "memory_mb": 100.0,
        }
        dp = TrendDataPoint.from_dict(data)
        assert dp.timestamp == "2024-01-01T00:00:00"
        assert dp.commit_hash == "abc123"
        assert dp.runtime == 1.2
        assert dp.memory_mb == 100.0


class TestTrendChartGenerator:
    """Tests for TrendChartGenerator class."""

    def test_generator_initialization(self, tmp_path):
        """Test TrendChartGenerator initialization."""
        generator = TrendChartGenerator(output_dir=tmp_path)
        assert generator.output_dir == tmp_path
        assert generator.plot_style == "science"
        assert generator.plot_dpi == 300
        assert generator.figure_format == "png"
        assert generator._trends_dir.exists()

    def test_generator_with_custom_settings(self, tmp_path):
        """Test generator with custom settings."""
        generator = TrendChartGenerator(
            output_dir=tmp_path,
            plot_style="ieee",
            plot_dpi=600,
            figure_format="pdf",
        )
        assert generator.plot_style == "ieee"
        assert generator.plot_dpi == 600
        assert generator.figure_format == "pdf"

    def test_invalid_plot_style_raises_error(self, tmp_path):
        """Test that invalid plot style raises ValueError."""
        with pytest.raises(ValueError, match="Invalid plot_style"):
            TrendChartGenerator(output_dir=tmp_path, plot_style="invalid")

    def test_invalid_figure_format_raises_error(self, tmp_path):
        """Test that invalid figure format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid figure_format"):
            TrendChartGenerator(output_dir=tmp_path, figure_format="invalid")

    def test_generate_runtime_trend(self, tmp_path):
        """Test generating runtime trend chart."""
        history = [
            {"timestamp": "2024-01-01T00:00:00", "runtime": 1.2, "commit_hash": "abc123"},
            {"timestamp": "2024-01-02T00:00:00", "runtime": 1.15, "commit_hash": "def456"},
            {"timestamp": "2024-01-03T00:00:00", "runtime": 1.1, "commit_hash": "ghi789"},
        ]

        generator = TrendChartGenerator(output_dir=tmp_path)
        output_path = generator.generate_runtime_trend(history, "test_method")

        assert output_path.exists()
        assert output_path.name == "test_method_runtime_trend.png"

    def test_generate_memory_trend(self, tmp_path):
        """Test generating memory trend chart."""
        history = [
            {"timestamp": "2024-01-01T00:00:00", "memory_mb": 100.0},
            {"timestamp": "2024-01-02T00:00:00", "memory_mb": 95.0},
            {"timestamp": "2024-01-03T00:00:00", "memory_mb": 90.0},
        ]

        generator = TrendChartGenerator(output_dir=tmp_path)
        output_path = generator.generate_memory_trend(history, "test_method")

        assert output_path.exists()
        assert output_path.name == "test_method_memory_trend.png"

    def test_generate_accuracy_trend_mse(self, tmp_path):
        """Test generating MSE accuracy trend chart."""
        history = [
            {"timestamp": "2024-01-01T00:00:00", "mse": 0.01},
            {"timestamp": "2024-01-02T00:00:00", "mse": 0.009},
            {"timestamp": "2024-01-03T00:00:00", "mse": 0.008},
        ]

        generator = TrendChartGenerator(output_dir=tmp_path)
        output_path = generator.generate_accuracy_trend(history, "test_method", metric="mse")

        assert output_path.exists()
        assert output_path.name == "test_method_mse_trend.png"

    def test_generate_accuracy_trend_correlation(self, tmp_path):
        """Test generating correlation accuracy trend chart."""
        history = [
            {"timestamp": "2024-01-01T00:00:00", "correlation": 0.90},
            {"timestamp": "2024-01-02T00:00:00", "correlation": 0.92},
            {"timestamp": "2024-01-03T00:00:00", "correlation": 0.94},
        ]

        generator = TrendChartGenerator(output_dir=tmp_path)
        output_path = generator.generate_accuracy_trend(
            history, "test_method", metric="correlation"
        )

        assert output_path.exists()
        assert output_path.name == "test_method_correlation_trend.png"

    def test_invalid_metric_raises_error(self, tmp_path):
        """Test that invalid metric raises ValueError."""
        history = [
            {"timestamp": "2024-01-01T00:00:00", "invalid": 0.01},
        ]

        generator = TrendChartGenerator(output_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid metric"):
            generator.generate_accuracy_trend(history, "test_method", metric="invalid")

    def test_empty_history_raises_error(self, tmp_path):
        """Test that empty history raises ValueError."""
        generator = TrendChartGenerator(output_dir=tmp_path)

        with pytest.raises(ValueError, match="history_data cannot be empty"):
            generator.generate_runtime_trend([], "test_method")


class TestBaselineFileIO:
    """Tests for baseline file I/O functions."""

    def test_save_and_load_baseline(self, tmp_path):
        """Test saving and loading baseline results."""
        # Create baseline results using legacy BenchmarkResults
        baseline = LegacyBenchmarkResults()
        baseline.metadata = {"version": "1.0.0"}

        # Save baseline
        baseline_path = tmp_path / "baseline.json"
        save_baseline(baseline_path, baseline)

        assert baseline_path.exists()

        # Load baseline
        loaded_baseline = load_baseline(baseline_path)
        assert loaded_baseline is not None
        assert loaded_baseline.metadata == {"version": "1.0.0"}

    def test_load_nonexistent_baseline_raises_error(self, tmp_path):
        """Test loading nonexistent baseline raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")

    def test_save_baseline_with_invalid_extension_raises_error(self, tmp_path):
        """Test saving baseline with invalid extension raises ValueError."""
        baseline = LegacyBenchmarkResults()
        with pytest.raises(ValueError, match="must have .json extension"):
            save_baseline(tmp_path / "baseline.txt", baseline)


class TestFormatRegressionMessage:
    """Tests for format_regression_message utility function."""

    def test_format_increase_regression(self):
        """Test formatting a regression with increase."""
        msg = format_regression_message("runtime", 1.0, 1.2, 10.0)
        assert "runtime" in msg
        assert "1.000" in msg
        assert "1.200" in msg
        assert "+20.0%" in msg
        assert "10.0%" in msg

    def test_format_decrease_regression(self):
        """Test formatting a regression with decrease."""
        msg = format_regression_message("correlation", 0.95, 0.90, 3.0)
        assert "correlation" in msg
        assert "0.9500" in msg
        assert "0.9000" in msg
        assert "-5.3%" in msg

    def test_format_zero_baseline(self):
        """Test formatting with zero baseline value."""
        msg = format_regression_message("metric", 0.0, 1.0, 10.0)
        assert "metric" in msg
        assert "0.0000" in msg
        assert "1.0000" in msg
