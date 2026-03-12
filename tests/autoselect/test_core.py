"""
Tests for autoselect core data classes.

This module contains tests for EvaluationResult, StageReport, and AutoSelectReport.
"""

import pytest


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_create_evaluation_result_success(self):
        """Test creating successful EvaluationResult."""
        from scptensor.autoselect import EvaluationResult

        result = EvaluationResult(
            method_name="log_normalize",
            scores={"metric1": 0.9, "metric2": 0.85},
            overall_score=0.875,
            execution_time=1.23,
            layer_name="log",
        )

        assert result.method_name == "log_normalize"
        assert result.scores == {"metric1": 0.9, "metric2": 0.85}
        assert result.overall_score == 0.875
        assert result.execution_time == 1.23
        assert result.layer_name == "log"
        assert result.error is None

    def test_create_evaluation_result_with_error(self):
        """Test creating EvaluationResult with error."""
        from scptensor.autoselect import EvaluationResult

        result = EvaluationResult(
            method_name="failed_method",
            scores={},
            overall_score=0.0,
            execution_time=0.1,
            layer_name="failed",
            error="Division by zero",
        )

        assert result.method_name == "failed_method"
        assert result.scores == {}
        assert result.overall_score == 0.0
        assert result.error == "Division by zero"

    def test_evaluation_result_to_dict(self):
        """Test converting EvaluationResult to dictionary."""
        from scptensor.autoselect import EvaluationResult

        result = EvaluationResult(
            method_name="knn_impute",
            scores={"rmse": 0.15, "correlation": 0.92},
            overall_score=0.90,
            execution_time=2.5,
            layer_name="imputed",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["method_name"] == "knn_impute"
        assert result_dict["scores"] == {"rmse": 0.15, "correlation": 0.92}
        assert result_dict["overall_score"] == 0.90
        assert result_dict["execution_time"] == 2.5
        assert result_dict["layer_name"] == "imputed"
        assert result_dict["error"] is None

    def test_evaluation_result_to_dict_with_error(self):
        """Test converting EvaluationResult with error to dictionary."""
        from scptensor.autoselect import EvaluationResult

        result = EvaluationResult(
            method_name="bad_method",
            scores={},
            overall_score=0.0,
            execution_time=0.0,
            layer_name="",
            error="Out of memory",
        )

        result_dict = result.to_dict()

        assert result_dict["error"] == "Out of memory"
        assert result_dict["method_name"] == "bad_method"


class TestStageReport:
    """Test StageReport dataclass."""

    def test_create_empty_stage_report(self):
        """Test creating empty StageReport."""
        from scptensor.autoselect import StageReport

        report = StageReport(stage_name="normalization")

        assert report.stage_name == "normalization"
        assert report.results == []
        assert report.best_method == ""
        assert report.best_result is None
        assert report.recommendation_reason == ""
        assert report.method_contracts == {}

    def test_stage_report_success_rate_empty(self):
        """Test success_rate property with empty results."""
        from scptensor.autoselect import StageReport

        report = StageReport(stage_name="qc")
        assert report.success_rate == 0.0

    def test_stage_report_success_rate_all_success(self):
        """Test success_rate property with all successful results."""
        from scptensor.autoselect import EvaluationResult, StageReport

        report = StageReport(
            stage_name="imputation",
            results=[
                EvaluationResult("knn", {"score": 0.9}, 0.9, 1.0, "knn"),
                EvaluationResult("svd", {"score": 0.85}, 0.85, 1.5, "svd"),
                EvaluationResult("mean", {"score": 0.8}, 0.8, 0.5, "mean"),
            ],
        )

        assert report.success_rate == 1.0

    def test_stage_report_success_rate_partial_success(self):
        """Test success_rate property with partial failures."""
        from scptensor.autoselect import EvaluationResult, StageReport

        report = StageReport(
            stage_name="batch_correction",
            results=[
                EvaluationResult("combat", {"score": 0.9}, 0.9, 2.0, "combat"),
                EvaluationResult("harmony", {}, 0.0, 0.1, "harmony", error="Failed"),
                EvaluationResult("mnn", {"score": 0.88}, 0.88, 3.0, "mnn"),
            ],
        )

        # 2 success out of 3
        assert report.success_rate == pytest.approx(2 / 3, rel=1e-3)

    def test_stage_report_with_best_result(self):
        """Test StageReport with best result identified."""
        from scptensor.autoselect import EvaluationResult, StageReport

        best = EvaluationResult("combat", {"score": 0.95}, 0.95, 2.0, "combat")
        report = StageReport(
            stage_name="batch_correction",
            results=[EvaluationResult("harmony", {"score": 0.88}, 0.88, 1.5, "harmony"), best],
            best_method="combat",
            best_result=best,
            recommendation_reason="Highest overall score",
        )

        assert report.best_method == "combat"
        assert report.best_result == best
        assert report.recommendation_reason == "Highest overall score"


class TestAutoSelectReport:
    """Test AutoSelectReport dataclass."""

    def test_create_empty_autoselect_report(self):
        """Test creating empty AutoSelectReport."""
        from scptensor.autoselect import AutoSelectReport

        report = AutoSelectReport()

        assert report.stages == {}
        assert report.total_time == 0.0
        assert report.warnings == []

    def test_autoselect_report_summary_empty(self):
        """Test summary method with empty report."""
        from scptensor.autoselect import AutoSelectReport

        report = AutoSelectReport()
        summary = report.summary()

        assert isinstance(summary, str)
        assert "No stages" in summary or "empty" in summary.lower()

    def test_autoselect_report_summary_with_stages(self):
        """Test summary method with multiple stages."""
        from scptensor.autoselect import (
            AutoSelectReport,
            EvaluationResult,
            StageReport,
        )

        norm_best = EvaluationResult("log", {"score": 0.9}, 0.9, 0.5, "log")
        norm_report = StageReport(
            stage_name="normalization",
            results=[norm_best],
            best_method="log",
            best_result=norm_best,
            recommendation_reason="Best performance",
        )

        imp_best = EvaluationResult("knn", {"score": 0.85}, 0.85, 1.2, "knn")
        imp_report = StageReport(
            stage_name="imputation",
            results=[imp_best],
            best_method="knn",
            best_result=imp_best,
            recommendation_reason="Lowest RMSE",
        )

        report = AutoSelectReport(
            stages={"normalization": norm_report, "imputation": imp_report},
            total_time=5.7,
            warnings=["High missing rate detected"],
        )

        summary = report.summary()

        assert isinstance(summary, str)
        # Stage names are displayed in uppercase in summary
        assert "NORMALIZATION" in summary
        assert "IMPUTATION" in summary
        assert "log" in summary
        assert "knn" in summary
        assert "5.70" in summary  # Formatted to 2 decimal places

    def test_autoselect_report_summary_with_warnings(self):
        """Test summary includes warnings when there are stages."""
        from scptensor.autoselect import (
            AutoSelectReport,
            EvaluationResult,
            StageReport,
        )

        # Warnings are only shown when there are stages
        stage = StageReport(
            stage_name="test", results=[EvaluationResult("method", {}, 0.9, 0.1, "layer")]
        )
        report = AutoSelectReport(stages={"test": stage}, warnings=["Warning 1", "Warning 2"])

        summary = report.summary()

        assert "Warning 1" in summary
        assert "Warning 2" in summary

    def test_autoselect_report_add_stage(self):
        """Test adding stages to report."""
        from scptensor.autoselect import (
            AutoSelectReport,
            EvaluationResult,
            StageReport,
        )

        report = AutoSelectReport()
        stage = StageReport(
            stage_name="qc", results=[EvaluationResult("basic", {}, 0.9, 0.1, "qc")]
        )

        report.stages["qc"] = stage

        assert "qc" in report.stages
        assert report.stages["qc"].stage_name == "qc"
