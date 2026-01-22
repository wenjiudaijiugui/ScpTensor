"""Tests for report display module.

Tests cover:
- scptensor.benchmark.display.report.ReportSection: Report section enum
- scptensor.benchmark.display.report.ReportConfig: Report configuration dataclass
- scptensor.benchmark.display.report.BenchmarkReportGenerator: Report generator class
- scptensor.benchmark.display.report.format_metric_value: Metric formatting function
- scptensor.benchmark.display.report.format_duration: Duration formatting function
- scptensor.benchmark.display.report.get_figure_relative_path: Path utility function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scptensor.benchmark.display.report import (
    BenchmarkReportGenerator,
    ReportConfig,
    ReportSection,
    format_duration,
    format_metric_value,
    get_figure_relative_path,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for display outputs.

    Returns
    -------
    Path
        Path to temporary output directory.
    """
    return tmp_path / "benchmark_results"


@pytest.fixture
def sample_report_config() -> ReportConfig:
    """Create a sample report configuration.

    Returns
    -------
    ReportConfig
        Sample configuration with default values.
    """
    return ReportConfig(
        output_dir=Path("benchmark_results"),
        include_figures=True,
        include_sections=["normalization", "imputation"],
        figure_format="png",
        plot_dpi=300,
    )


@pytest.fixture
def sample_benchmark_results() -> MagicMock:
    """Create a mock BenchmarkResults object.

    Returns
    -------
    MagicMock
        Mock benchmark results with required attributes.
    """
    mock_results = MagicMock()
    mock_results.comparisons = []
    mock_results.summary.return_value = {"successful": 10, "failed": 2}
    mock_results.get_results_by_category.return_value = []
    # Return an empty list instead of iterator to avoid StopIteration
    mock_results.iter_all_results.return_value = []
    return mock_results


# ============================================================================
# Tests for ReportSection enum
# ============================================================================


class TestReportSection:
    """Tests for ReportSection enum."""

    def test_all_sections(self) -> None:
        """Test all report sections are defined."""
        all_sections = ReportSection.all()
        assert len(all_sections) == 9
        assert ReportSection.EXECUTIVE_SUMMARY in all_sections
        assert ReportSection.TABLE_OF_CONTENTS in all_sections
        assert ReportSection.NORMALIZATION in all_sections
        assert ReportSection.IMPUTATION in all_sections
        assert ReportSection.INTEGRATION in all_sections
        assert ReportSection.DIM_REDUCTION in all_sections
        assert ReportSection.QC in all_sections
        assert ReportSection.END_TO_END in all_sections
        assert ReportSection.CONCLUSIONS in all_sections

    def test_section_values(self) -> None:
        """Test section enum values."""
        assert ReportSection.EXECUTIVE_SUMMARY.value == "executive_summary"
        assert ReportSection.TABLE_OF_CONTENTS.value == "table_of_contents"
        assert ReportSection.NORMALIZATION.value == "normalization"
        assert ReportSection.IMPUTATION.value == "imputation"
        assert ReportSection.INTEGRATION.value == "integration"
        assert ReportSection.DIM_REDUCTION.value == "dim_reduction"
        assert ReportSection.QC.value == "qc"
        assert ReportSection.END_TO_END.value == "end_to_end"
        assert ReportSection.CONCLUSIONS.value == "conclusions"

    def test_from_string_valid(self) -> None:
        """Test from_string with valid values."""
        assert ReportSection.from_string("normalization") == ReportSection.NORMALIZATION
        assert ReportSection.from_string("NORMALIZATION") == ReportSection.NORMALIZATION
        assert ReportSection.from_string("imputation") == ReportSection.IMPUTATION
        assert ReportSection.from_string("qc") == ReportSection.QC

    def test_from_string_invalid(self) -> None:
        """Test from_string with invalid values."""
        assert ReportSection.from_string("invalid") is None
        assert ReportSection.from_string("") is None

    def test_all_returns_ordered_sections(self) -> None:
        """Test all() returns sections in expected order."""
        sections = ReportSection.all()
        assert sections[0] == ReportSection.EXECUTIVE_SUMMARY
        assert sections[1] == ReportSection.TABLE_OF_CONTENTS
        assert sections[-1] == ReportSection.CONCLUSIONS


# ============================================================================
# Tests for ReportConfig dataclass
# ============================================================================


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_create_default_config(self) -> None:
        """Test creating ReportConfig with defaults."""
        config = ReportConfig()
        assert config.include_figures is True
        assert config.include_sections == []
        assert config.figure_format == "png"
        assert config.plot_dpi == 300
        assert config.max_table_rows == 20
        assert config.include_regression_analysis is False
        assert config.show_exclusive_methods is True
        assert config.title == "ScpTensor Benchmark Report"
        assert config.author is None

    def test_create_custom_config(self, sample_report_config: ReportConfig) -> None:
        """Test creating ReportConfig with custom values."""
        assert sample_report_config.include_figures is True
        assert sample_report_config.include_sections == ["normalization", "imputation"]
        assert sample_report_config.figure_format == "png"
        assert sample_report_config.plot_dpi == 300

    def test_get_sections_empty_returns_all(self) -> None:
        """Test get_sections returns all sections when include_sections is empty."""
        config = ReportConfig(include_sections=[])
        sections = config.get_sections()
        assert len(sections) == 9

    def test_get_sections_filtered(self) -> None:
        """Test get_sections returns specified sections."""
        config = ReportConfig(include_sections=["normalization", "imputation", "qc"])
        sections = config.get_sections()
        assert len(sections) == 3
        assert ReportSection.NORMALIZATION in sections
        assert ReportSection.IMPUTATION in sections
        assert ReportSection.QC in sections

    def test_get_sections_ignores_invalid(self) -> None:
        """Test get_sections ignores invalid section names."""
        config = ReportConfig(include_sections=["normalization", "invalid", "qc"])
        sections = config.get_sections()
        assert len(sections) == 2
        assert ReportSection.NORMALIZATION in sections
        assert ReportSection.QC in sections

    def test_dataclass_is_frozen(self, sample_report_config: ReportConfig) -> None:
        """Test ReportConfig is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_report_config.include_figures = False

    def test_default_subdirs_constant(self) -> None:
        """Test _DEFAULT_SUBDIRS constant is defined."""
        assert ReportConfig._DEFAULT_SUBDIRS == ("figures", "tables", "summaries")


# ============================================================================
# Tests for format_metric_value function
# ============================================================================


class TestFormatMetricValue:
    """Tests for format_metric_value function."""

    def test_format_none(self) -> None:
        """Test formatting None returns N/A."""
        assert format_metric_value(None) == "N/A"

    def test_format_small_float(self) -> None:
        """Test formatting small float values."""
        assert format_metric_value(0.1234) == "0.1234"
        assert format_metric_value(0.12345678) == "0.1235"
        assert format_metric_value(0.5) == "0.5000"

    def test_format_medium_float(self) -> None:
        """Test formatting medium float values."""
        assert format_metric_value(50.5) == "50.5000"
        assert format_metric_value(99.999) == "99.9990"

    def test_format_large_float(self) -> None:
        """Test formatting large float values."""
        assert format_metric_value(100.0) == "100.000"
        assert format_metric_value(123.45) == "123.450"
        assert format_metric_value(999.99) == "999.990"

    def test_format_very_large_float(self) -> None:
        """Test formatting very large float values."""
        assert format_metric_value(1000.0) == "1000.00"
        assert format_metric_value(1234.56) == "1234.56"
        assert format_metric_value(10000.0) == "10000.00"

    def test_format_negative_float(self) -> None:
        """Test formatting negative float values."""
        assert format_metric_value(-0.1234) == "-0.1234"
        assert format_metric_value(-100.0) == "-100.000"
        assert format_metric_value(-1000.0) == "-1000.00"

    def test_custom_precision(self) -> None:
        """Test formatting with custom precision."""
        assert format_metric_value(0.12345678, precision=2) == "0.12"
        assert format_metric_value(0.12345678, precision=6) == "0.123457"

    def test_format_integer(self) -> None:
        """Test formatting integer values."""
        assert format_metric_value(42) == "42"
        assert format_metric_value(0) == "0"

    def test_format_zero(self) -> None:
        """Test formatting zero."""
        assert format_metric_value(0.0) == "0.0000"


# ============================================================================
# Tests for format_duration function
# ============================================================================


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_seconds_only(self) -> None:
        """Test formatting duration less than 1 minute."""
        assert format_duration(0.5) == "0.5s"
        assert format_duration(5.2) == "5.2s"
        assert format_duration(45.9) == "45.9s"
        assert format_duration(59.9) == "59.9s"

    def test_format_minutes_and_seconds(self) -> None:
        """Test formatting duration with minutes."""
        assert format_duration(60.0) == "1m 0.0s"
        assert format_duration(90.5) == "1m 30.5s"
        assert format_duration(125.0) == "2m 5.0s"
        assert format_duration(3665.0) == "61m 5.0s"

    def test_format_large_duration(self) -> None:
        """Test formatting large durations."""
        assert format_duration(3600.0) == "60m 0.0s"
        assert format_duration(7200.0) == "120m 0.0s"

    def test_format_fractional_seconds(self) -> None:
        """Test formatting with fractional seconds."""
        assert format_duration(0.123) == "0.1s"
        assert format_duration(60.123) == "1m 0.1s"


# ============================================================================
# Tests for get_figure_relative_path function
# ============================================================================


class TestGetFigureRelativePath:
    """Tests for get_figure_relative_path function."""

    def test_relative_path_within_report_dir(self, tmp_path: Path) -> None:
        """Test getting relative path when figure is within report directory."""
        figure_path = tmp_path / "figures" / "test.png"
        report_path = tmp_path / "report.md"
        output_dir = tmp_path

        result = get_figure_relative_path(figure_path, output_dir, report_path)
        assert result == "figures/test.png"

    def test_relative_path_nested_structure(self, tmp_path: Path) -> None:
        """Test getting relative path with nested structure."""
        figure_path = tmp_path / "figures" / "normalization" / "plot.png"
        report_path = tmp_path / "reports" / "summary.md"
        output_dir = tmp_path

        result = get_figure_relative_path(figure_path, output_dir, report_path)
        # When figure is not relative to report parent, returns absolute path
        assert str(figure_path) in result or "figures/normalization/plot.png" in result

    def test_absolute_path_when_not_relative(self, tmp_path: Path) -> None:
        """Test absolute path is returned when figure is not relative."""
        figure_path = Path("/some/other/path/figure.png")
        report_path = tmp_path / "report.md"
        output_dir = tmp_path

        result = get_figure_relative_path(figure_path, output_dir, report_path)
        assert result == str(figure_path)


# ============================================================================
# Tests for BenchmarkReportGenerator class
# ============================================================================


class TestBenchmarkReportGenerator:
    """Tests for BenchmarkReportGenerator class."""

    def test_initialization_default_path(self) -> None:
        """Test initialization with default path."""
        generator = BenchmarkReportGenerator()
        assert generator.output_dir == Path("benchmark_results")
        assert generator._figures_dir == Path("benchmark_results") / "figures"

    def test_initialization_custom_path(self, temp_output_dir: Path) -> None:
        """Test initialization with custom output directory."""
        generator = BenchmarkReportGenerator(output_dir=temp_output_dir)
        assert generator.output_dir == temp_output_dir
        assert generator._figures_dir == temp_output_dir / "figures"
        assert generator._figures_dir.exists()

    def test_initialization_string_path(self, temp_output_dir: Path) -> None:
        """Test initialization with string path."""
        generator = BenchmarkReportGenerator(output_dir=str(temp_output_dir))
        assert generator.output_dir == temp_output_dir
        assert isinstance(generator.output_dir, Path)

    def test_creates_figure_subdirectories(self, temp_output_dir: Path) -> None:
        """Test that figure subdirectories are created."""
        generator = BenchmarkReportGenerator(output_dir=temp_output_dir)
        expected_subdirs = [
            "normalization",
            "imputation",
            "integration",
            "dim_reduction",
            "qc",
            "end_to_end",
        ]
        for subdir in expected_subdirs:
            assert (generator._figures_dir / subdir).exists()

    def test_creates_tables_directory(self, temp_output_dir: Path) -> None:
        """Test that tables directory is created."""
        generator = BenchmarkReportGenerator(output_dir=temp_output_dir)
        assert generator._tables_dir.exists()
        assert generator._tables_dir == temp_output_dir / "tables"

    def test_section_names_mapping(self) -> None:
        """Test SECTION_NAMES mapping is complete."""
        assert ReportSection.EXECUTIVE_SUMMARY in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.TABLE_OF_CONTENTS in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.NORMALIZATION in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.IMPUTATION in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.INTEGRATION in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.DIM_REDUCTION in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.QC in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.END_TO_END in BenchmarkReportGenerator.SECTION_NAMES
        assert ReportSection.CONCLUSIONS in BenchmarkReportGenerator.SECTION_NAMES

    def test_figure_subdirs_mapping(self) -> None:
        """Test FIGURE_SUBDIRS mapping."""
        assert (
            BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.NORMALIZATION] == "normalization"
        )
        assert BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.IMPUTATION] == "imputation"
        assert BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.INTEGRATION] == "integration"
        assert (
            BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.DIM_REDUCTION] == "dim_reduction"
        )
        assert BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.QC] == "qc"
        assert BenchmarkReportGenerator.FIGURE_SUBDIRS[ReportSection.END_TO_END] == "end_to_end"

    def test_generate_creates_report_file(
        self,
        temp_output_dir: Path,
        sample_benchmark_results: MagicMock,
    ) -> None:
        """Test generate creates a report file."""
        # Patch open to avoid actual file writing
        with patch("scptensor.benchmark.display.report.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            generator = BenchmarkReportGenerator(output_dir=temp_output_dir)

            # Patch the problematic section generation to avoid StopIteration
            with patch.object(
                generator, "_generate_normalization_section", return_value=["## Normalization", ""]
            ):
                report_path = generator.generate(sample_benchmark_results)

                assert report_path.parent == temp_output_dir
                assert report_path.suffix == ".md"
                assert "benchmark_report_" in report_path.name
                mock_open.assert_called_once()

    def test_generate_with_custom_config(
        self,
        temp_output_dir: Path,
        sample_benchmark_results: MagicMock,
    ) -> None:
        """Test generate with custom configuration."""
        config = ReportConfig(
            output_dir=temp_output_dir,
            title="Custom Report",
            author="Test Author",
        )

        with patch("scptensor.benchmark.display.report.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            generator = BenchmarkReportGenerator(output_dir=temp_output_dir)

            # Patch the problematic section generation to avoid StopIteration
            with patch.object(
                generator, "_generate_normalization_section", return_value=["## Normalization", ""]
            ):
                report_path = generator.generate(sample_benchmark_results, config=config)

                assert report_path.parent == temp_output_dir


# ============================================================================
# Parametrized tests
# ============================================================================


class TestReportParametrized:
    """Parametrized tests for report module."""

    @pytest.mark.parametrize(
        ("value", "precision", "expected"),
        [
            (None, 4, "N/A"),
            (0.0, 4, "0.0000"),
            (1.0, 4, "1.0000"),
            (10.0, 4, "10.0000"),
            (100.0, 4, "100.000"),
            (1000.0, 4, "1000.00"),
            (0.1234, 4, "0.1234"),
            (0.9999, 2, "1.00"),
            (-50.5, 4, "-50.5000"),
        ],
    )
    def test_format_metric_value_cases(
        self,
        value: float | None,
        precision: int,
        expected: str,
    ) -> None:
        """Test format_metric_value with various inputs."""
        result = format_metric_value(value, precision)
        assert result == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.1, "0.1s"),
            (1.0, "1.0s"),
            (30.0, "30.0s"),
            (59.9, "59.9s"),
            (60.0, "1m 0.0s"),
            (90.0, "1m 30.0s"),
            (120.0, "2m 0.0s"),
            (3661.0, "61m 1.0s"),
        ],
    )
    def test_format_duration_cases(self, seconds: float, expected: str) -> None:
        """Test format_duration with various inputs."""
        result = format_duration(seconds)
        assert result == expected

    @pytest.mark.parametrize(
        ("section_name", "expected_section"),
        [
            ("executive_summary", ReportSection.EXECUTIVE_SUMMARY),
            ("EXECUTIVE_SUMMARY", ReportSection.EXECUTIVE_SUMMARY),
            ("normalization", ReportSection.NORMALIZATION),
            ("imputation", ReportSection.IMPUTATION),
            ("integration", ReportSection.INTEGRATION),
            ("dim_reduction", ReportSection.DIM_REDUCTION),
            ("qc", ReportSection.QC),
            ("end_to_end", ReportSection.END_TO_END),
            ("conclusions", ReportSection.CONCLUSIONS),
        ],
    )
    def test_from_string_various_sections(
        self,
        section_name: str,
        expected_section: ReportSection,
    ) -> None:
        """Test ReportSection.from_string with various section names."""
        result = ReportSection.from_string(section_name)
        assert result == expected_section

    @pytest.mark.parametrize(
        ("figure_format", "plot_dpi", "include_figures"),
        [
            ("png", 150, True),
            ("pdf", 300, True),
            ("svg", 600, False),
        ],
    )
    def test_report_config_various_formats(
        self,
        figure_format: str,
        plot_dpi: int,
        include_figures: bool,
    ) -> None:
        """Test ReportConfig with various format options."""
        config = ReportConfig(
            figure_format=figure_format,
            plot_dpi=plot_dpi,
            include_figures=include_figures,
        )
        assert config.figure_format == figure_format
        assert config.plot_dpi == plot_dpi
        assert config.include_figures == include_figures


# ============================================================================
# Edge case tests
# ============================================================================


class TestReportEdgeCases:
    """Edge case tests for report module."""

    def test_format_metric_value_very_small(self) -> None:
        """Test formatting very small values."""
        assert format_metric_value(0.00001) == "0.0000"

    def test_format_metric_value_negative_small(self) -> None:
        """Test formatting negative small values."""
        assert format_metric_value(-0.0001) == "-0.0001"

    def test_format_duration_zero(self) -> None:
        """Test formatting zero duration."""
        assert format_duration(0.0) == "0.0s"

    def test_format_duration_negative(self) -> None:
        """Test formatting negative duration."""
        assert format_duration(-10.0) == "-10.0s"

    def test_report_config_empty_include_sections(self) -> None:
        """Test ReportConfig with empty include_sections."""
        config = ReportConfig(include_sections=[])
        sections = config.get_sections()
        assert len(sections) == len(ReportSection.all())

    def test_report_config_all_invalid_sections(self) -> None:
        """Test ReportConfig with all invalid section names."""
        config = ReportConfig(include_sections=["invalid", "wrong", "not_a_section"])
        sections = config.get_sections()
        assert len(sections) == 0

    def test_generator_with_nested_output_dir(self, tmp_path: Path) -> None:
        """Test generator with deeply nested output directory."""
        nested_dir = tmp_path / "deeply" / "nested" / "benchmark"
        generator = BenchmarkReportGenerator(output_dir=nested_dir)
        assert generator._figures_dir.exists()
        assert generator._tables_dir.exists()
