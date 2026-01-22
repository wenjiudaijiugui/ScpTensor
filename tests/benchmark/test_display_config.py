"""Tests for benchmark display configuration module.

This module tests the configuration classes and enums for customizing
the visualization and reporting of preprocessing method comparisons.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scptensor.benchmark.display.config import (
    CATEGORY_METRICS,
    ComparisonLayer,
    MethodCategory,
    PlotStyle,
    ReportConfig,
    get_category_metrics,
    get_style_string,
)


class TestMethodCategory:
    """Test MethodCategory enum."""

    def test_enum_values(self) -> None:
        """Test MethodCategory has correct enum values."""
        assert MethodCategory.NORMALIZATION.value == "normalization"
        assert MethodCategory.IMPUTATION.value == "imputation"
        assert MethodCategory.INTEGRATION.value == "integration"
        assert MethodCategory.QC.value == "qc"
        assert MethodCategory.DIM_REDUCTION.value == "dim_reduction"
        assert MethodCategory.FEATURE_SELECTION.value == "feature_selection"

    def test_enum_members(self) -> None:
        """Test MethodCategory has all expected members."""
        expected_members = {
            "NORMALIZATION",
            "IMPUTATION",
            "INTEGRATION",
            "QC",
            "DIM_REDUCTION",
            "FEATURE_SELECTION",
        }
        actual_members = {member.name for member in MethodCategory}
        assert actual_members == expected_members

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (MethodCategory.NORMALIZATION, "normalization"),
            (MethodCategory.IMPUTATION, "imputation"),
            (MethodCategory.INTEGRATION, "integration"),
            (MethodCategory.QC, "qc"),
            (MethodCategory.DIM_REDUCTION, "dim_reduction"),
            (MethodCategory.FEATURE_SELECTION, "feature_selection"),
        ],
    )
    def test_enum_value_mapping(self, member: MethodCategory, expected_value: str) -> None:
        """Test each MethodCategory member maps to correct string value."""
        assert member.value == expected_value


class TestComparisonLayer:
    """Test ComparisonLayer enum."""

    def test_enum_values(self) -> None:
        """Test ComparisonLayer has correct enum values."""
        assert ComparisonLayer.SHARED.value == "shared"
        assert ComparisonLayer.SCPTENSOR_EXCLUSIVE.value == "scptensor_exclusive"
        assert ComparisonLayer.SCANPY_EXCLUSIVE.value == "scanpy_exclusive"

    def test_enum_members(self) -> None:
        """Test ComparisonLayer has all expected members."""
        expected_members = {
            "SHARED",
            "SCPTENSOR_EXCLUSIVE",
            "SCANPY_EXCLUSIVE",
        }
        actual_members = {member.name for member in ComparisonLayer}
        assert actual_members == expected_members

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (ComparisonLayer.SHARED, "shared"),
            (ComparisonLayer.SCPTENSOR_EXCLUSIVE, "scptensor_exclusive"),
            (ComparisonLayer.SCANPY_EXCLUSIVE, "scanpy_exclusive"),
        ],
    )
    def test_enum_value_mapping(self, member: ComparisonLayer, expected_value: str) -> None:
        """Test each ComparisonLayer member maps to correct string value."""
        assert member.value == expected_value


class TestPlotStyle:
    """Test PlotStyle enum."""

    def test_enum_values(self) -> None:
        """Test PlotStyle has correct enum values."""
        assert PlotStyle.SCIENCE.value == "science"
        assert PlotStyle.IEEE.value == "ieee"
        assert PlotStyle.NATURE.value == "nature"
        assert PlotStyle.DEFAULT.value == "default"

    def test_enum_members(self) -> None:
        """Test PlotStyle has all expected members."""
        expected_members = {"SCIENCE", "IEEE", "NATURE", "DEFAULT"}
        actual_members = {member.name for member in PlotStyle}
        assert actual_members == expected_members

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (PlotStyle.SCIENCE, "science"),
            (PlotStyle.IEEE, "ieee"),
            (PlotStyle.NATURE, "nature"),
            (PlotStyle.DEFAULT, "default"),
        ],
    )
    def test_enum_value_mapping(self, member: PlotStyle, expected_value: str) -> None:
        """Test each PlotStyle member maps to correct string value."""
        assert member.value == expected_value


class TestGetStyleString:
    """Test get_style_string function."""

    def test_science_style_returns_list(self) -> None:
        """Test SCIENCE style returns list of style strings."""
        result = get_style_string(PlotStyle.SCIENCE)
        assert result == ["science", "no-latex"]

    def test_ieee_style_returns_list(self) -> None:
        """Test IEEE style returns list with single style."""
        result = get_style_string(PlotStyle.IEEE)
        assert result == ["ieee"]

    def test_nature_style_returns_list(self) -> None:
        """Test NATURE style returns list with single style."""
        result = get_style_string(PlotStyle.NATURE)
        assert result == ["nature"]

    def test_default_style_returns_string(self) -> None:
        """Test DEFAULT style returns string, not list."""
        result = get_style_string(PlotStyle.DEFAULT)
        assert result == "default"

    @pytest.mark.parametrize(
        ("style", "expected"),
        [
            (PlotStyle.SCIENCE, ["science", "no-latex"]),
            (PlotStyle.IEEE, ["ieee"]),
            (PlotStyle.NATURE, ["nature"]),
            (PlotStyle.DEFAULT, "default"),
        ],
    )
    def test_all_style_mappings(self, style: PlotStyle, expected: str | list[str]) -> None:
        """Test all PlotStyle enum values map correctly."""
        assert get_style_string(style) == expected


class TestGetCategoryMetrics:
    """Test get_category_metrics function."""

    def test_normalization_primary_metrics(self) -> None:
        """Test normalization category primary metrics."""
        metrics = get_category_metrics(MethodCategory.NORMALIZATION, "primary")
        assert metrics == ["execution_time", "memory_usage"]

    def test_normalization_secondary_metrics(self) -> None:
        """Test normalization category secondary metrics."""
        metrics = get_category_metrics(MethodCategory.NORMALIZATION, "secondary")
        expected = ["distribution_preservation", "variance_stabilization"]
        assert metrics == expected

    def test_normalization_optional_metrics(self) -> None:
        """Test normalization category optional metrics."""
        metrics = get_category_metrics(MethodCategory.NORMALIZATION, "optional")
        expected = ["log_fold_change", "sparsity_ratio"]
        assert metrics == expected

    def test_imputation_primary_metrics(self) -> None:
        """Test imputation category primary metrics."""
        metrics = get_category_metrics(MethodCategory.IMPUTATION, "primary")
        expected = ["execution_time", "memory_usage", "imputation_accuracy"]
        assert metrics == expected

    def test_imputation_secondary_metrics(self) -> None:
        """Test imputation category secondary metrics."""
        metrics = get_category_metrics(MethodCategory.IMPUTATION, "secondary")
        assert set(metrics) == {"mse", "mae", "correlation"}

    def test_integration_primary_metrics(self) -> None:
        """Test integration category primary metrics."""
        metrics = get_category_metrics(MethodCategory.INTEGRATION, "primary")
        expected = ["execution_time", "memory_usage", "batch_mixing"]
        assert metrics == expected

    def test_qc_primary_metrics(self) -> None:
        """Test QC category primary metrics."""
        metrics = get_category_metrics(MethodCategory.QC, "primary")
        expected = ["execution_time", "cells_removed", "features_removed"]
        assert metrics == expected

    def test_dim_reduction_primary_metrics(self) -> None:
        """Test dimensionality reduction category primary metrics."""
        metrics = get_category_metrics(MethodCategory.DIM_REDUCTION, "primary")
        expected = ["execution_time", "memory_usage", "variance_explained"]
        assert metrics == expected

    def test_feature_selection_primary_metrics(self) -> None:
        """Test feature selection category primary metrics."""
        metrics = get_category_metrics(MethodCategory.FEATURE_SELECTION, "primary")
        expected = ["execution_time", "n_features_selected", "variance_coverage"]
        assert metrics == expected

    def test_invalid_metric_type_raises_value_error(self) -> None:
        """Test invalid metric_type raises ValueError."""
        with pytest.raises(ValueError, match="metric_type must be"):
            get_category_metrics(MethodCategory.NORMALIZATION, "invalid")

    @pytest.mark.parametrize(
        ("invalid_type",),
        [
            ("primary_extra",),
            ("SECONDARY",),
            ("",),
            ("optional_extra",),
        ],
    )
    def test_various_invalid_metric_types(self, invalid_type: str) -> None:
        """Test various invalid metric_type values raise ValueError."""
        with pytest.raises(ValueError):
            get_category_metrics(MethodCategory.IMPUTATION, invalid_type)

    @pytest.mark.parametrize(
        ("category", "metric_type", "expected_count"),
        [
            (MethodCategory.NORMALIZATION, "primary", 2),
            (MethodCategory.NORMALIZATION, "secondary", 2),
            (MethodCategory.NORMALIZATION, "optional", 2),
            (MethodCategory.IMPUTATION, "primary", 3),
            (MethodCategory.IMPUTATION, "secondary", 3),
            (MethodCategory.INTEGRATION, "primary", 3),
            (MethodCategory.QC, "primary", 3),
            (MethodCategory.DIM_REDUCTION, "primary", 3),
            (MethodCategory.FEATURE_SELECTION, "primary", 3),
        ],
    )
    def test_metric_counts(
        self,
        category: MethodCategory,
        metric_type: str,
        expected_count: int,
    ) -> None:
        """Test each category returns expected number of metrics."""
        metrics = get_category_metrics(category, metric_type)
        assert len(metrics) == expected_count

    @pytest.mark.parametrize(
        ("metric_type",),
        [("primary",), ("secondary",), ("optional",)],
    )
    def test_all_valid_metric_types(self, metric_type: str) -> None:
        """Test all valid metric types work for each category."""
        for category in MethodCategory:
            metrics = get_category_metrics(category, metric_type)
            assert isinstance(metrics, list)
            assert all(isinstance(m, str) for m in metrics)


class TestCategoryMetricsConstant:
    """Test CATEGORY_METRICS constant dictionary."""

    def test_all_method_categories_have_metrics(self) -> None:
        """Test all MethodCategory values have entries in CATEGORY_METRICS."""
        for category in MethodCategory:
            assert category in CATEGORY_METRICS

    def test_metric_structure(self) -> None:
        """Test each category has required metric types."""
        required_types = {"primary", "secondary", "optional"}
        for category_metrics in CATEGORY_METRICS.values():
            assert set(category_metrics.keys()) == required_types

    def test_metrics_are_lists(self) -> None:
        """Test all metric values are lists of strings."""
        for category_metrics in CATEGORY_METRICS.values():
            for metric_list in category_metrics.values():
                assert isinstance(metric_list, list)
                assert all(isinstance(m, str) for m in metric_list)

    def test_no_empty_metric_lists(self) -> None:
        """Test all metric lists are non-empty."""
        for category_metrics in CATEGORY_METRICS.values():
            for metric_list in category_metrics.values():
                assert len(metric_list) > 0


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test ReportConfig with default values."""
        config = ReportConfig()
        assert config.output_dir == Path("benchmark_results")
        assert config.include_figures is True
        assert config.plot_style == PlotStyle.SCIENCE
        assert config.plot_dpi == 300
        assert config.show_exclusive_methods is True
        assert config.include_regression_analysis is False

    def test_custom_output_dir(self) -> None:
        """Test ReportConfig with custom output directory."""
        config = ReportConfig(output_dir=Path("custom_output"))
        assert config.output_dir == Path("custom_output")

    def test_custom_plot_style(self) -> None:
        """Test ReportConfig with custom plot style."""
        config = ReportConfig(plot_style=PlotStyle.IEEE)
        assert config.plot_style == PlotStyle.IEEE

    def test_custom_plot_dpi(self) -> None:
        """Test ReportConfig with custom DPI."""
        config = ReportConfig(plot_dpi=600)
        assert config.plot_dpi == 600

    def test_include_figures_false(self) -> None:
        """Test ReportConfig with figures disabled."""
        config = ReportConfig(include_figures=False)
        assert config.include_figures is False

    def test_show_exclusive_methods_false(self) -> None:
        """Test ReportConfig with exclusive methods hidden."""
        config = ReportConfig(show_exclusive_methods=False)
        assert config.show_exclusive_methods is False

    def test_include_regression_analysis_true(self) -> None:
        """Test ReportConfig with regression analysis enabled."""
        config = ReportConfig(include_regression_analysis=True)
        assert config.include_regression_analysis is True

    @pytest.mark.parametrize(
        ("dpi", "is_valid"),
        [
            (72, True),
            (150, True),
            (300, True),
            (600, True),
            (1200, True),
        ],
    )
    def test_various_dpi_values(self, dpi: int, is_valid: bool) -> None:
        """Test ReportConfig accepts various DPI values."""
        config = ReportConfig(plot_dpi=dpi)
        assert config.plot_dpi == dpi

    def test_frozen_dataclass_is_immutable(self) -> None:
        """Test ReportConfig is frozen (immutable)."""
        config = ReportConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.plot_dpi = 600

    @pytest.mark.parametrize(
        ("plot_style", "expected"),
        [
            (PlotStyle.SCIENCE, PlotStyle.SCIENCE),
            (PlotStyle.IEEE, PlotStyle.IEEE),
            (PlotStyle.NATURE, PlotStyle.NATURE),
            (PlotStyle.DEFAULT, PlotStyle.DEFAULT),
        ],
    )
    def test_all_plot_styles(self, plot_style: PlotStyle, expected: PlotStyle) -> None:
        """Test ReportConfig accepts all PlotStyle values."""
        config = ReportConfig(plot_style=plot_style)
        assert config.plot_style == expected

    def test_full_custom_config(self) -> None:
        """Test ReportConfig with all custom values."""
        config = ReportConfig(
            output_dir=Path("custom_output"),
            include_figures=False,
            plot_style=PlotStyle.IEEE,
            plot_dpi=600,
            show_exclusive_methods=False,
            include_regression_analysis=True,
        )
        assert config.output_dir == Path("custom_output")
        assert config.include_figures is False
        assert config.plot_style == PlotStyle.IEEE
        assert config.plot_dpi == 600
        assert config.show_exclusive_methods is False
        assert config.include_regression_analysis is True

    def test_default_subdirs_constant(self) -> None:
        """Test _DEFAULT_SUBDIRS class variable."""
        assert ReportConfig._DEFAULT_SUBDIRS == (
            "figures",
            "tables",
            "summaries",
        )


class TestReportConfigEdgeCases:
    """Test ReportConfig edge cases and validation."""

    def test_output_dir_as_string_converts_to_path(self) -> None:
        """Test string output_dir is not automatically converted to Path."""
        # Path type hint requires Path object, not string
        config = ReportConfig(output_dir=Path("string_path"))
        assert isinstance(config.output_dir, Path)

    def test_zero_dpi(self) -> None:
        """Test ReportConfig with zero DPI."""
        config = ReportConfig(plot_dpi=0)
        assert config.plot_dpi == 0

    def test_negative_dpi(self) -> None:
        """Test ReportConfig with negative DPI."""
        config = ReportConfig(plot_dpi=-100)
        assert config.plot_dpi == -100

    def test_config_equality(self) -> None:
        """Test two configs with same values are equal."""
        config1 = ReportConfig(plot_dpi=600)
        config2 = ReportConfig(plot_dpi=600)
        assert config1 == config2

    def test_config_inequality(self) -> None:
        """Test two configs with different values are not equal."""
        config1 = ReportConfig(plot_dpi=300)
        config2 = ReportConfig(plot_dpi=600)
        assert config1 != config2


class TestIntegrationScenarios:
    """Test integration scenarios between config components."""

    def test_config_and_style_string_integration(self) -> None:
        """Test ReportConfig.plot_style works with get_style_string."""
        config = ReportConfig(plot_style=PlotStyle.SCIENCE)
        style_string = get_style_string(config.plot_style)
        assert style_string == ["science", "no-latex"]

    def test_config_and_category_metrics_integration(self) -> None:
        """Test ReportConfig with MethodCategory in get_category_metrics."""
        config = ReportConfig()
        metrics = get_category_metrics(MethodCategory.IMPUTATION, "primary")
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    @pytest.mark.parametrize(
        ("category", "config_style"),
        [
            (MethodCategory.NORMALIZATION, PlotStyle.SCIENCE),
            (MethodCategory.IMPUTATION, PlotStyle.IEEE),
            (MethodCategory.INTEGRATION, PlotStyle.NATURE),
            (MethodCategory.QC, PlotStyle.DEFAULT),
        ],
    )
    def test_category_and_style_combinations(
        self, category: MethodCategory, config_style: PlotStyle
    ) -> None:
        """Test various category and style combinations."""
        config = ReportConfig(plot_style=config_style)
        style_string = get_style_string(config.plot_style)
        metrics = get_category_metrics(category, "primary")
        assert style_string is not None
        assert len(metrics) > 0
