"""Tests for display module public API exports.

This module verifies that all expected exports are available from
scptensor.benchmark.display and that backward compatibility type
aliases work correctly.
"""

from __future__ import annotations


class TestDisplayImports:
    """Test that all expected exports are available from display module."""

    def test_visualization_exports(self) -> None:
        """Test visualization-related exports are available."""
        from scptensor.benchmark.display import (
            ComparisonVisualizer,
            PlotStyle,
            configure_plots,
            get_visualizer,
        )

        # Verify ComparisonVisualizer is a class
        assert hasattr(ComparisonVisualizer, "__init__")

        # Verify PlotStyle is an Enum subclass
        assert hasattr(PlotStyle, "__members__")
        assert hasattr(PlotStyle, "SCIENCE")
        assert hasattr(PlotStyle, "IEEE")
        assert hasattr(PlotStyle, "NATURE")
        assert hasattr(PlotStyle, "DEFAULT")

        # Verify configure_plots is callable
        assert callable(configure_plots)

        # Verify get_visualizer is callable
        assert callable(get_visualizer)

    def test_configuration_exports(self) -> None:
        """Test configuration enum exports are available."""
        from scptensor.benchmark.display import ComparisonLayer, MethodCategory

        # Verify MethodCategory is an Enum subclass
        assert hasattr(MethodCategory, "__members__")
        assert hasattr(MethodCategory, "NORMALIZATION")
        assert hasattr(MethodCategory, "IMPUTATION")
        assert hasattr(MethodCategory, "DIM_REDUCTION")
        assert hasattr(MethodCategory, "INTEGRATION")
        assert hasattr(MethodCategory, "QC")
        assert hasattr(MethodCategory, "FEATURE_SELECTION")

        # Verify ComparisonLayer is an Enum subclass
        assert hasattr(ComparisonLayer, "__members__")
        assert hasattr(ComparisonLayer, "SHARED")
        assert hasattr(ComparisonLayer, "SCPTENSOR_EXCLUSIVE")
        assert hasattr(ComparisonLayer, "SCANPY_EXCLUSIVE")

    def test_result_class_exports(self) -> None:
        """Test result dataclass exports are available."""
        from scptensor.benchmark.display import (
            BenchmarkResult,
            BenchmarkResults,
            ComparisonResult,
            MethodRunResult,
            MethodSpec,
        )

        # Verify MethodSpec (alias for MethodResult) is available
        assert hasattr(MethodSpec, "__dataclass_fields__")

        # Verify BenchmarkResult (alias for MethodRunResult) is available
        assert hasattr(BenchmarkResult, "__dataclass_fields__")

        # Verify MethodRunResult is available
        assert hasattr(MethodRunResult, "__dataclass_fields__")

        # Verify ComparisonResult is available
        assert hasattr(ComparisonResult, "__dataclass_fields__")

        # Verify BenchmarkResults is available (legacy class, not dataclass)
        assert hasattr(BenchmarkResults, "__init__")

    def test_all_export_list(self) -> None:
        """Test that __all__ contains expected exports."""
        from scptensor.benchmark import display

        expected_exports = {
            # Visualization
            "ComparisonVisualizer",
            "get_visualizer",
            "configure_plots",
            # Configuration
            "MethodCategory",
            "ComparisonLayer",
            "PlotStyle",
            # Result classes
            "MethodSpec",
            "BenchmarkResult",
            "ComparisonResult",
            "BenchmarkResults",
            "MethodRunResult",
        }

        actual_exports = set(display.__all__)
        # Check that expected exports are a subset of actual exports
        assert expected_exports.issubset(actual_exports)


class TestBackwardCompatibilityAliases:
    """Test backward compatibility type aliases."""

    def test_method_spec_is_method_result(self) -> None:
        """Test MethodSpec is an alias for MethodResult."""
        from scptensor.benchmark.comparison_engine import MethodResult
        from scptensor.benchmark.display import MethodSpec

        # MethodSpec should be the same as MethodResult
        assert MethodSpec is MethodResult

    def test_benchmark_result_is_method_run_result(self) -> None:
        """Test BenchmarkResult is an alias for MethodRunResult."""
        from scptensor.benchmark.core import MethodRunResult
        from scptensor.benchmark.display import BenchmarkResult

        # BenchmarkResult should be the same as MethodRunResult
        assert BenchmarkResult is MethodRunResult

    def test_comparison_result_from_engine(self) -> None:
        """Test ComparisonResult is imported from comparison_engine."""
        from scptensor.benchmark.comparison_engine import ComparisonResult as EngineComparisonResult
        from scptensor.benchmark.display import ComparisonResult

        # ComparisonResult should be the same as engine's ComparisonResult
        assert ComparisonResult is EngineComparisonResult

    def test_benchmark_results_from_core(self) -> None:
        """Test BenchmarkResults is imported from core."""
        from scptensor.benchmark.core import BenchmarkResults as CoreBenchmarkResults
        from scptensor.benchmark.display import BenchmarkResults

        # BenchmarkResults should be the same as core's BenchmarkResults
        assert BenchmarkResults is CoreBenchmarkResults


class TestPlotStyleEnum:
    """Test PlotStyle enum values."""

    def test_plot_style_values(self) -> None:
        """Test PlotStyle enum has correct string values."""
        from scptensor.benchmark.display import PlotStyle

        assert PlotStyle.SCIENCE.value == "science"
        assert PlotStyle.IEEE.value == "ieee"
        assert PlotStyle.NATURE.value == "nature"
        assert PlotStyle.DEFAULT.value == "default"


class TestMethodCategoryEnum:
    """Test MethodCategory enum values."""

    def test_method_category_values(self) -> None:
        """Test MethodCategory enum has correct string values."""
        from scptensor.benchmark.display import MethodCategory

        assert MethodCategory.NORMALIZATION.value == "normalization"
        assert MethodCategory.IMPUTATION.value == "imputation"
        assert MethodCategory.DIM_REDUCTION.value == "dim_reduction"
        assert MethodCategory.INTEGRATION.value == "integration"
        assert MethodCategory.QC.value == "qc"
        assert MethodCategory.FEATURE_SELECTION.value == "feature_selection"


class TestComparisonLayerEnum:
    """Test ComparisonLayer enum values."""

    def test_comparison_layer_values(self) -> None:
        """Test ComparisonLayer enum has correct string values."""
        from scptensor.benchmark.display import ComparisonLayer

        assert ComparisonLayer.SHARED.value == "shared"
        assert ComparisonLayer.SCPTENSOR_EXCLUSIVE.value == "scptensor_exclusive"
        assert ComparisonLayer.SCANPY_EXCLUSIVE.value == "scanpy_exclusive"


class TestVisualizerFactory:
    """Test get_visualizer factory function."""

    def test_get_visualizer_returns_instance(self) -> None:
        """Test get_visualizer returns ComparisonVisualizer instance."""
        from scptensor.benchmark.display import ComparisonVisualizer, get_visualizer

        visualizer = get_visualizer()
        assert isinstance(visualizer, ComparisonVisualizer)

    def test_get_visualizer_with_custom_output_dir(self, tmp_path) -> None:
        """Test get_visualizer with custom output directory."""
        from scptensor.benchmark.display import ComparisonVisualizer, get_visualizer

        output_dir = tmp_path / "custom_figures"
        visualizer = get_visualizer(output_dir=str(output_dir))
        assert isinstance(visualizer, ComparisonVisualizer)

    def test_get_visualizer_with_custom_style(self) -> None:
        """Test get_visualizer with custom plot style."""
        from scptensor.benchmark.display import ComparisonVisualizer, PlotStyle, get_visualizer

        visualizer = get_visualizer(style=PlotStyle.IEEE)
        assert isinstance(visualizer, ComparisonVisualizer)
