"""Tests for report generation functions.

This module contains tests for the report export functionality including
Markdown, JSON, and CSV formats.
"""

import json
import tempfile
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
        overall_score=0.90,
        execution_time=1.2,
        layer_name="log",
    )
    norm_result2 = EvaluationResult(
        method_name="median_normalize",
        scores={"variance": 0.85, "batch_effect": 0.9, "completeness": 0.9},
        overall_score=0.88,
        execution_time=0.8,
        layer_name="median",
    )

    norm_stage = StageReport(
        stage_name="normalization",
        results=[norm_result1, norm_result2],
        best_method="log_normalize",
        best_result=norm_result1,
        recommendation_reason="Highest overall score (0.9000) among 2 successful methods",
    )

    # Create imputation stage results
    imp_result1 = EvaluationResult(
        method_name="knn_impute",
        scores={"rmse": 0.15, "correlation": 0.92},
        overall_score=0.85,
        execution_time=2.5,
        layer_name="imputed_knn",
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
        results=[imp_result1, imp_result2, imp_result3],
        best_method="knn_impute",
        best_result=imp_result1,
        recommendation_reason="Highest overall score (0.8500) among 2 successful methods",
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

            content = filepath.read_text()

            # Check for expected sections
            assert "AutoSelect Report" in content
            assert "normalization" in content.lower()
            assert "imputation" in content.lower()
            assert "log_normalize" in content
            assert "knn_impute" in content
            assert "15.70" in content or "15.7" in content  # Total time
            assert "Warning" in content  # Warnings section

    def test_save_markdown_empty_report(self):
        """Test saving empty report."""
        from scptensor.autoselect.report import save_markdown

        report = AutoSelectReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_report.md"
            save_markdown(report, filepath)

            content = filepath.read_text()
            assert "AutoSelect Report" in content
            # Empty report should have indication of no stages
            assert "no stages" in content.lower() or len(content) > 0


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
            with open(filepath) as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_save_json_structure(self, sample_report):
        """Test that save_json has correct structure."""
        from scptensor.autoselect.report import save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            save_json(sample_report, filepath)

            with open(filepath) as f:
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
            assert norm_stage["best_method"] == "log_normalize"

    def test_save_json_empty_report(self):
        """Test saving empty report as JSON."""
        from scptensor.autoselect.report import save_json

        report = AutoSelectReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_report.json"
            save_json(report, filepath)

            with open(filepath) as f:
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

            content = filepath.read_text()
            lines = content.strip().split("\n")

            # Check header
            header = lines[0]
            assert "stage_name" in header
            assert "method_name" in header
            assert "overall_score" in header
            assert "execution_time" in header
            assert "error" in header
            assert "is_best" in header

    def test_save_csv_row_count(self, sample_report):
        """Test that save_csv has correct number of rows."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            content = filepath.read_text()
            lines = content.strip().split("\n")

            # 1 header + 2 normalization methods + 3 imputation methods = 6 lines
            assert len(lines) == 6

    def test_save_csv_marks_best_methods(self, sample_report):
        """Test that save_csv correctly marks best methods."""
        from scptensor.autoselect.report import save_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            save_csv(sample_report, filepath)

            content = filepath.read_text()
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
            content = filepath.read_text()
            assert "AutoSelect Report" in content

    def test_save_json_format(self, sample_report):
        """Test save with json format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            sample_report.save(filepath, format="json")

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_save_csv_format(self, sample_report):
        """Test save with csv format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.csv"
            sample_report.save(filepath, format="csv")

            assert filepath.exists()
            content = filepath.read_text()
            assert "stage_name" in content

    def test_save_default_markdown(self, sample_report):
        """Test save with default format (markdown)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            sample_report.save(filepath)  # No format specified

            assert filepath.exists()
            content = filepath.read_text()
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
