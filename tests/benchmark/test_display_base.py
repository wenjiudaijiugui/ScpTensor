"""Tests for benchmark display base classes.

Tests cover:
- scptensor.benchmark.display.base.DisplayBase: Abstract base class
- scptensor.benchmark.display.base.ComparisonDisplay: Abstract comparison display
- Instantiation prevention for abstract classes
- Output directory creation and management
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from scptensor.benchmark.display.base import ComparisonDisplay, DisplayBase

# ============================================================================
# Concrete implementations for testing abstract classes
# ============================================================================


class ConcreteDisplay(DisplayBase):
    """Concrete implementation of DisplayBase for testing."""

    def render(self) -> Path:
        """Render a test output file.

        Returns
        -------
        Path
            Path to the rendered output file.
        """
        output_path = self.output_dir / "test_output.txt"
        output_path.write_text("test content")
        return output_path


class ConcreteComparisonDisplay(ComparisonDisplay):
    """Concrete implementation of ComparisonDisplay for testing."""

    def render(self) -> Path:
        """Render a test output file.

        Returns
        -------
        Path
            Path to the rendered output file.
        """
        output_path = self.output_dir / "comparison_summary.txt"
        output_path.write_text("comparison content")
        return output_path

    def render_comparison(self, result: dict[str, Any]) -> Path:
        """Render a single comparison result.

        Parameters
        ----------
        result : dict[str, Any]
            Dictionary containing comparison result data.

        Returns
        -------
        Path
            Path to the rendered comparison output file.
        """
        method_name = result.get("method", "unknown")
        output_path = self.output_dir / f"comparison_{method_name}.txt"
        output_path.write_text(f"comparison result for {method_name}")
        return output_path

    def render_summary(self, results: list[dict[str, Any]]) -> Path:
        """Render a summary of multiple comparison results.

        Parameters
        ----------
        results : list[dict[str, Any]]
            List of comparison result dictionaries.

        Returns
        -------
        Path
            Path to the rendered summary output file.
        """
        output_path = self.output_dir / "summary.txt"
        content = f"summary of {len(results)} results"
        output_path.write_text(content)
        return output_path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary directory for display outputs.

    Returns
    -------
    Path
            Path to temporary output directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample comparison result dictionary.

    Returns
    -------
    dict[str, Any]
        Sample result with method name and metrics.
    """
    return {
        "method": "log_normalize",
        "dataset": "test_dataset",
        "runtime_seconds": 1.5,
        "memory_mb": 100.0,
        "accuracy": 0.95,
    }


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Create a list of sample comparison result dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        List of sample results.
    """
    return [
        {
            "method": "log_normalize",
            "dataset": "test_dataset",
            "runtime_seconds": 1.5,
            "memory_mb": 100.0,
            "accuracy": 0.95,
        },
        {
            "method": "z_score_normalize",
            "dataset": "test_dataset",
            "runtime_seconds": 1.2,
            "memory_mb": 90.0,
            "accuracy": 0.92,
        },
    ]


# ============================================================================
# Tests for DisplayBase abstract class
# ============================================================================


class TestDisplayBaseAbstract:
    """Tests for DisplayBase abstract class behavior."""

    def test_display_base_cannot_instantiate(self):
        """DisplayBase should not be directly instantiable."""
        with pytest.raises(TypeError):
            DisplayBase()  # type: ignore[abstract]

    def test_display_base_is_abstract(self):
        """DisplayBase should be recognized as an abstract class."""
        # The abstractmethod decorator should make the class abstract
        assert hasattr(DisplayBase, "__abstractmethods__")
        assert "render" in DisplayBase.__abstractmethods__


class TestDisplayBaseInitialization:
    """Tests for DisplayBase initialization."""

    def test_display_base_with_default_output_dir(self, temp_output_dir: Path):
        """Test DisplayBase with default output directory."""
        display = ConcreteDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display.output_dir.exists()

    def test_display_base_with_string_output_dir(self, temp_output_dir: Path):
        """Test DisplayBase with string output directory."""
        output_dir_str = str(temp_output_dir)
        display = ConcreteDisplay(output_dir=output_dir_str)

        assert display.output_dir == temp_output_dir
        assert isinstance(display.output_dir, Path)

    def test_display_base_creates_directory_if_not_exists(self, temp_output_dir: Path):
        """Test DisplayBase creates output directory if it does not exist."""
        new_dir = temp_output_dir / "new_subdir" / "nested"
        assert not new_dir.exists()

        display = ConcreteDisplay(output_dir=new_dir)

        assert display.output_dir.exists()
        assert display.output_dir.is_dir()

    def test_display_base_with_existing_directory(self, temp_output_dir: Path):
        """Test DisplayBase works with existing directory."""
        display = ConcreteDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display.output_dir.exists()

    @pytest.mark.parametrize(
        ("output_dir_path", "expected_exists"),
        [
            (Path("outputs"), True),
            (Path("benchmark_results"), True),
            (Path("custom/path/to/output"), True),
        ],
    )
    def test_display_base_creates_various_paths(
        self, output_dir_path: Path, expected_exists: bool, tmp_path: Path
    ):
        """Test DisplayBase creates various path types."""
        full_path = tmp_path / output_dir_path
        display = ConcreteDisplay(output_dir=full_path)

        assert display.output_dir.exists() == expected_exists
        assert display.output_dir.is_dir()


# ============================================================================
# Tests for DisplayBase.render
# ============================================================================


class TestDisplayBaseRender:
    """Tests for DisplayBase.render method."""

    def test_render_creates_output_file(self, temp_output_dir: Path):
        """Test render creates an output file."""
        display = ConcreteDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        assert output_path.exists()
        assert output_path.is_file()

    def test_render_returns_path(self, temp_output_dir: Path):
        """Test render returns a Path object."""
        display = ConcreteDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        assert isinstance(output_path, Path)

    def test_render_output_in_output_dir(self, temp_output_dir: Path):
        """Test render creates output in the configured directory."""
        display = ConcreteDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        assert output_path.parent == display.output_dir

    def test_render_content(self, temp_output_dir: Path):
        """Test render writes expected content."""
        display = ConcreteDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        content = output_path.read_text()
        assert content == "test content"


# ============================================================================
# Tests for ComparisonDisplay abstract class
# ============================================================================


class TestComparisonDisplayAbstract:
    """Tests for ComparisonDisplay abstract class behavior."""

    def test_comparison_display_cannot_instantiate(self):
        """ComparisonDisplay should not be directly instantiable."""
        with pytest.raises(TypeError):
            ComparisonDisplay()  # type: ignore[abstract]

    def test_comparison_display_is_abstract(self):
        """ComparisonDisplay should be recognized as an abstract class."""
        assert hasattr(ComparisonDisplay, "__abstractmethods__")
        assert "render_comparison" in ComparisonDisplay.__abstractmethods__
        assert "render_summary" in ComparisonDisplay.__abstractmethods__

    def test_comparison_display_inherits_display_base(self):
        """ComparisonDisplay should inherit from DisplayBase."""
        assert issubclass(ComparisonDisplay, DisplayBase)


class TestComparisonDisplayInitialization:
    """Tests for ComparisonDisplay initialization."""

    def test_comparison_display_init(self, temp_output_dir: Path):
        """Test ComparisonDisplay initialization."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display.output_dir.exists()

    def test_comparison_display_init_with_string(self, temp_output_dir: Path):
        """Test ComparisonDisplay with string output directory."""
        output_dir_str = str(temp_output_dir)
        display = ConcreteComparisonDisplay(output_dir=output_dir_str)

        assert isinstance(display.output_dir, Path)
        assert display.output_dir == temp_output_dir


# ============================================================================
# Tests for ComparisonDisplay.render_comparison
# ============================================================================


class TestComparisonDisplayRenderComparison:
    """Tests for ComparisonDisplay.render_comparison method."""

    def test_render_comparison_creates_file(
        self, temp_output_dir: Path, sample_result: dict[str, Any]
    ):
        """Test render_comparison creates an output file."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_comparison(sample_result)

        assert output_path.exists()
        assert output_path.is_file()

    def test_render_comparison_returns_path(
        self, temp_output_dir: Path, sample_result: dict[str, Any]
    ):
        """Test render_comparison returns a Path object."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_comparison(sample_result)

        assert isinstance(output_path, Path)

    def test_render_comparison_uses_method_name(
        self, temp_output_dir: Path, sample_result: dict[str, Any]
    ):
        """Test render_comparison uses method name in output filename."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_comparison(sample_result)

        assert "log_normalize" in output_path.name

    def test_render_comparison_with_missing_method(self, temp_output_dir: Path):
        """Test render_comparison handles result without method name."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        result = {"dataset": "test_dataset"}  # No method key
        output_path = display.render_comparison(result)

        assert output_path.exists()
        assert "unknown" in output_path.name

    def test_render_comparison_in_output_dir(
        self, temp_output_dir: Path, sample_result: dict[str, Any]
    ):
        """Test render_comparison creates output in the configured directory."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_comparison(sample_result)

        assert output_path.parent == display.output_dir


# ============================================================================
# Tests for ComparisonDisplay.render_summary
# ============================================================================


class TestComparisonDisplayRenderSummary:
    """Tests for ComparisonDisplay.render_summary method."""

    def test_render_summary_creates_file(
        self, temp_output_dir: Path, sample_results: list[dict[str, Any]]
    ):
        """Test render_summary creates an output file."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary(sample_results)

        assert output_path.exists()
        assert output_path.is_file()

    def test_render_summary_returns_path(
        self, temp_output_dir: Path, sample_results: list[dict[str, Any]]
    ):
        """Test render_summary returns a Path object."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary(sample_results)

        assert isinstance(output_path, Path)

    def test_render_summary_content(
        self, temp_output_dir: Path, sample_results: list[dict[str, Any]]
    ):
        """Test render_summary writes expected content."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary(sample_results)

        content = output_path.read_text()
        assert "2" in content  # Number of results

    def test_render_summary_with_empty_list(self, temp_output_dir: Path):
        """Test render_summary with empty results list."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary([])

        assert output_path.exists()
        content = output_path.read_text()
        assert "0" in content  # Zero results

    def test_render_summary_in_output_dir(
        self, temp_output_dir: Path, sample_results: list[dict[str, Any]]
    ):
        """Test render_summary creates output in the configured directory."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary(sample_results)

        assert output_path.parent == display.output_dir


# ============================================================================
# Tests for ComparisonDisplay.render (inherited)
# ============================================================================


class TestComparisonDisplayRender:
    """Tests for ComparisonDisplay.render inherited method."""

    def test_render_creates_file(self, temp_output_dir: Path):
        """Test render creates an output file."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        assert output_path.exists()
        assert output_path.is_file()

    def test_render_returns_path(self, temp_output_dir: Path):
        """Test render returns a Path object."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render()

        assert isinstance(output_path, Path)


# ============================================================================
# Parametrized tests
# ============================================================================


class TestDisplayBaseParametrized:
    """Parametrized tests for DisplayBase."""

    @pytest.mark.parametrize(
        "dir_name",
        [
            "simple",
            "with_underscore",
            "with-dash",
            "with.dot",
            "CamelCase",
            "123numbers",
        ],
    )
    def test_various_directory_names(self, dir_name: str, tmp_path: Path):
        """Test DisplayBase with various directory name formats."""
        output_dir = tmp_path / dir_name
        display = ConcreteDisplay(output_dir=output_dir)

        assert display.output_dir.exists()
        assert display.output_dir.is_dir()

    @pytest.mark.parametrize(
        ("nested_path", "expected_parts"),
        [
            ("a/b", ["a", "a/b"]),
            ("a/b/c", ["a", "a/b", "a/b/c"]),
            (
                "deep/nested/path/structure",
                ["deep", "deep/nested", "deep/nested/path", "deep/nested/path/structure"],
            ),
        ],
    )
    def test_nested_directory_creation(
        self, nested_path: str, expected_parts: list[str], tmp_path: Path
    ):
        """Test DisplayBase creates nested directories."""
        output_dir = tmp_path / nested_path
        display = ConcreteDisplay(output_dir=output_dir)

        for part in expected_parts:
            assert (tmp_path / part).exists()


class TestComparisonDisplayParametrized:
    """Parametrized tests for ComparisonDisplay."""

    @pytest.mark.parametrize(
        "result",
        [
            {"method": "method1", "value": 1},
            {"method": "method2", "value": 2},
            {"method": "method3", "value": 3, "extra": "data"},
        ],
    )
    def test_render_comparison_various_results(self, result: dict[str, Any], temp_output_dir: Path):
        """Test render_comparison with various result formats."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_comparison(result)

        assert output_path.exists()
        assert result["method"] in output_path.name

    @pytest.mark.parametrize(
        "results",
        [
            [],
            [{"method": "m1"}],
            [{"method": "m1"}, {"method": "m2"}],
            [{"method": "m1"}, {"method": "m2"}, {"method": "m3"}],
        ],
    )
    def test_render_summary_various_result_counts(
        self, results: list[dict[str, Any]], temp_output_dir: Path
    ):
        """Test render_summary with various result counts."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        output_path = display.render_summary(results)

        assert output_path.exists()
        content = output_path.read_text()
        assert str(len(results)) in content


# ============================================================================
# Edge case tests
# ============================================================================


class TestDisplayBaseEdgeCases:
    """Edge case tests for DisplayBase."""

    def test_render_multiple_times(self, temp_output_dir: Path):
        """Test calling render multiple times."""
        display = ConcreteDisplay(output_dir=temp_output_dir)

        output1 = display.render()
        output2 = display.render()

        assert output1 == output2
        assert output1.exists()

    def test_same_output_directory_multiple_instances(self, temp_output_dir: Path):
        """Test multiple instances with same output directory."""
        display1 = ConcreteDisplay(output_dir=temp_output_dir)
        display2 = ConcreteDisplay(output_dir=temp_output_dir)

        assert display1.output_dir == display2.output_dir
        assert display1.output_dir.exists()


class TestComparisonDisplayEdgeCases:
    """Edge case tests for ComparisonDisplay."""

    def test_render_comparison_with_complex_result(self, temp_output_dir: Path):
        """Test render_comparison with complex nested result."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        complex_result = {
            "method": "complex",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42.5,
        }
        output_path = display.render_comparison(complex_result)

        assert output_path.exists()

    def test_render_summary_with_single_result(self, temp_output_dir: Path):
        """Test render_summary with a single result."""
        display = ConcreteComparisonDisplay(output_dir=temp_output_dir)
        single_result = [{"method": "single"}]
        output_path = display.render_summary(single_result)

        assert output_path.exists()
        content = output_path.read_text()
        assert "1" in content
