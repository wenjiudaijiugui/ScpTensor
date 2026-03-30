"""Tests for report generation functions.

This module contains tests for the report export functionality including
Markdown, JSON, and CSV formats.
"""

import json
import os
import subprocess
import sys
import tempfile
from csv import DictReader
from pathlib import Path

import pytest

from scptensor.autoselect import AutoSelectReport, EvaluationResult, StageReport


@pytest.fixture
def sample_report() -> AutoSelectReport:
    """Create sample AutoSelectReport for testing."""
    # Create normalization stage results
    norm_result1 = EvaluationResult(
        method_name="log_normalize",
        scores={"variance": 0.9, "batch_effect": 0.85, "completeness": 0.95},
        report_metrics={"loading_bias_reduction": 0.93, "batch_asw": 0.81},
        overall_score=0.90,
        execution_time=1.2,
        layer_name="log",
        selection_score=0.89,
        n_repeats=3,
        overall_score_std=0.02,
        overall_score_ci_lower=0.86,
        overall_score_ci_upper=0.92,
        repeat_overall_scores=[0.88, 0.90, 0.92],
    )
    norm_result2 = EvaluationResult(
        method_name="median_normalize",
        scores={"variance": 0.85, "batch_effect": 0.9, "completeness": 0.9},
        report_metrics={"loading_bias_reduction": 0.88, "batch_asw": 0.79},
        overall_score=0.88,
        execution_time=0.8,
        layer_name="median",
        selection_score=0.90,
        n_repeats=3,
        overall_score_std=0.01,
        overall_score_ci_lower=0.87,
        overall_score_ci_upper=0.89,
        repeat_overall_scores=[0.87, 0.88, 0.89],
    )

    norm_stage = StageReport(
        stage_name="normalization",
        stage_key="normalize",
        results=[norm_result1, norm_result2],
        best_method="log_normalize",
        best_result=norm_result1,
        recommendation_reason="Highest overall score (0.9000) among 2 successful methods",
        metric_weights={"variance": 0.4, "batch_effect": 0.3, "completeness": 0.3},
        input_assay="proteins",
        input_layer="raw",
        output_assay="proteins",
        output_layer="log",
        selection_strategy="balanced",
        n_repeats=3,
        confidence_level=0.95,
    )

    # Create imputation stage results
    imp_result1 = EvaluationResult(
        method_name="knn_impute",
        scores={"rmse": 0.15, "correlation": 0.92},
        overall_score=0.85,
        execution_time=2.5,
        layer_name="imputed_knn",
        selection_score=0.84,
    )
    imp_result2 = EvaluationResult(
        method_name="svd_impute",
        scores={"rmse": 0.18, "correlation": 0.88},
        overall_score=0.80,
        execution_time=1.8,
        layer_name="imputed_svd",
    )
    imp_result3 = EvaluationResult(
        method_name="failed_impute",
        scores={},
        overall_score=0.0,
        execution_time=0.1,
        layer_name="failed",
        error="Out of memory",
    )

    imp_stage = StageReport(
        stage_name="imputation",
        stage_key="impute",
        results=[imp_result1, imp_result2, imp_result3],
        best_method="knn_impute",
        best_result=imp_result1,
        recommendation_reason="Highest overall score (0.8500) among 2 successful methods",
        metric_weights={"rmse": 0.5, "correlation": 0.5},
        selection_strategy="quality",
    )

    # Create complete report
    report = AutoSelectReport(
        stages={"normalization": norm_stage, "imputation": imp_stage},
        total_time=15.7,
        warnings=["High missing rate detected in batch B2"],
    )

    return report


class TestSaveMarkdown:
    """Test save_markdown function."""

    def test_save_markdown_creates_file(self, sample_report):
        """Test that save_markdown creates a file."""
        from scptensor.autoselect.report import save_markdown

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            save_markdown(sample_report, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_markdown_content(self, sample_report):
        """Test that save_markdown writes correct content."""
        from scptensor.autoselect.report import save_markdown

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            save_markdown(sample_report, filepath)

            content = filepath.read_text(encoding="utf-8")

            # Check for expected sections
            assert "AutoSelect Report" in content
            assert "normalization" in content.lower()
            assert "imputation" in content.lower()
            assert "log_normalize" in content
            assert "knn_impute" in content
            assert "15.70" in content or "15.7" in content  # Total time
            assert "Warning" in content  # Warnings section
            assert "Selection Score" in content
            assert "Strategy Weights" in content
            assert "Metric Details" in content
            assert "Report Metrics" in content
            assert "Stage Valid" in content
            assert "`loading_bias_reduction`=0.9300" in content
            assert "[0.8600, 0.9200]" in content

    def test_save_markdown_empty_report(self):
        """Test saving empty report."""
        from scptensor.autoselect.report import save_markdown

        report = AutoSelectReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_report.md"
            save_markdown(report, filepath)

            content = filepath.read_text(encoding="utf-8")
            assert "AutoSelect Report" in content
            # Empty report should have indication of no stages
            assert "no stages" in content.lower() or len(content) > 0

    def test_save_markdown_handles_non_utf8_default_encoding(self, tmp_path):
        """Test markdown export succeeds even when the default locale is not UTF-8."""
        output_path = tmp_path / "report.md"
        script = f"""
from pathlib import Path
from scptensor.autoselect import AutoSelectReport, EvaluationResult, StageReport
from scptensor.autoselect.report import save_markdown

success = EvaluationResult(
    method_name="ok",
    scores={{}},
    overall_score=1.0,
    execution_time=0.1,
    layer_name="ok_layer",
)
failed = EvaluationResult(
    method_name="bad",
    scores={{}},
    overall_score=0.0,
    execution_time=0.1,
    layer_name="bad_layer",
    error="boom",
)
report = AutoSelectReport(
    stages={{
        "imputation": StageReport(
            stage_name="imputation",
            results=[success, failed],
            best_method="ok",
            best_result=success,
            selection_strategy="balanced",
        )
    }}
)
save_markdown(report, Path({str(output_path)!r}))
"""
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        env["PYTHONUTF8"] = "0"

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        content = output_path.read_text(encoding="utf-8")
        assert "✓ Success" in content
        assert "✗ Failed: boom" in content


class TestSaveJson:
    """Test save_json function."""

    def test_save_json_creates_file(self, sample_report):
        """Test that save_json creates a file."""
        from scptensor.autoselect.report import save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            save_json(sample_report, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_json_is_valid_json(self, sample_report):
        """Test that save_json writes valid JSON."""
        from scptensor.autoselect.report import save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            save_json(sample_report, filepath)

            # Should not raise JSONDecodeError
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_save_json_structure(self, sample_report):
        """Test that save_json has correct structure."""
        from scptensor.autoselect.report import save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            save_json(sample_report, filepath)

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Check for expected keys
            assert "stages" in data
            assert "total_time" in data
            assert "warnings" in data
            assert "timestamp" in data

            # Check stages
            assert "normalization" in data["stages"]
            assert "imputation" in data["stages"]

            # Check stage structure
            norm_stage = data["stages"]["normalization"]
            assert "stage_name" in norm_stage
            assert "results" in norm_stage
            assert "best_method" in norm_stage
            assert "stage_valid" in norm_stage
            assert "invalid_reason" in norm_stage
            assert norm_stage["best_method"] == "log_normalize"
            assert norm_stage["results"][0]["report_metrics"][
                "loading_bias_reduction"
            ] == pytest.approx(0.93)

    def test_save_json_empty_report(self):
        """Test saving empty report as JSON."""
        from scptensor.autoselect.report import save_json

        report = AutoSelectReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_report.json"
            save_json(report, filepath)

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            assert data["stages"] == {}
            assert data["total_time"] == 0.0
            assert data["warnings"] == []


class TestSaveCsv:
    """Test save_csv function."""

    def test_save_csv_creates_file(self, sample_report):
        """Test that save_csv creates a file."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_csv_has_correct_columns(self, sample_report):
        """Test that save_csv has expected columns."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            content = filepath.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            # Check header
            header = lines[0]
            assert "stage_name" in header
            assert "method_name" in header
            assert "overall_score" in header
            assert "execution_time" in header
            assert "error" in header
            assert "is_best" in header
            assert "selection_strategy" in header
            assert "selection_score" in header
            assert "overall_score_ci_lower" in header
            assert "stage_valid" in header
            assert "invalid_reason" in header
            assert "scores" in header
            assert "report_metrics" in header

    def test_save_csv_row_count(self, sample_report):
        """Test that save_csv has correct number of rows."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            content = filepath.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            # 1 header + 2 normalization methods + 3 imputation methods = 6 lines
            assert len(lines) == 6

    def test_save_csv_contains_extended_metadata(self, sample_report):
        """Test CSV row includes repeat/strategy metadata."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            with open(filepath, encoding="utf-8", newline="") as handle:
                rows = list(DictReader(handle))

            assert len(rows) == 5
            strategies = {row["selection_strategy"] for row in rows}
            assert strategies == {"balanced", "quality"}

            first_row = rows[0]
            repeat_scores = json.loads(first_row["repeat_overall_scores"])
            assert repeat_scores == [0.88, 0.9, 0.92]
            report_metrics = json.loads(first_row["report_metrics"])
            assert report_metrics["loading_bias_reduction"] == pytest.approx(0.93)
            metric_weights = json.loads(first_row["metric_weights"])
            assert metric_weights["variance"] == pytest.approx(0.4)

    def test_save_csv_marks_best_methods(self, sample_report):
        """Test that save_csv correctly marks best methods."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            content = filepath.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            # Find best methods (should have is_best=True or similar)
            best_count = sum(1 for line in lines if "True" in line and "is_best" not in line)
            assert best_count >= 2  # At least 2 best methods (one per stage)


class TestAutoSelectReportSave:
    """Test AutoSelectReport.save() method."""

    def test_save_markdown_format(self, sample_report):
        """Test save with markdown format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            sample_report.save(filepath, format="markdown")

            assert filepath.exists()
            content = filepath.read_text(encoding="utf-8")
            assert "AutoSelect Report" in content

    def test_save_json_format(self, sample_report):
        """Test save with json format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            sample_report.save(filepath, format="json")

            assert filepath.exists()
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_save_csv_format(self, sample_report):
        """Test save with csv format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            sample_report.save(filepath, format="csv")

            assert filepath.exists()
            content = filepath.read_text(encoding="utf-8")
            assert "stage_name" in content

    def test_save_default_markdown(self, sample_report):
        """Test save with default format (markdown)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            sample_report.save(filepath)  # No format specified

            assert filepath.exists()
            content = filepath.read_text(encoding="utf-8")
            assert "AutoSelect Report" in content

    def test_save_invalid_format_raises(self, sample_report):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.txt"

            with pytest.raises(ValueError, match="Unsupported format"):
                sample_report.save(filepath, format="invalid")

    def test_save_pathlib_path(self, sample_report):
        """Test save with pathlib.Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            sample_report.save(filepath, format="markdown")

            assert filepath.exists()

    def test_save_string_path(self, sample_report):
        """Test save with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "report.md")
            sample_report.save(filepath, format="markdown")

            assert Path(filepath).exists()
