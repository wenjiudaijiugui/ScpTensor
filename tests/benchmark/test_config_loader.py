"""Tests for benchmark configuration loader."""

import pytest
import tempfile
from pathlib import Path

from scptensor.benchmark.config import (
    BenchmarkConfig,
    ChartConfig,
    ConfigurationError,
    ModuleConfigEntry,
    OutputConfig,
    get_default_config,
    load_charts_config,
    load_config,
    save_config,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write("""
datasets:
  - test_dataset_1
  - test_dataset_2

output:
  directory: test_output
  generate_plots: true
  generate_report: true

cache_enabled: true
cache_dir: .cache/test
max_workers: 2
random_seed: 42
verbose: false

methods:
  normalization:
    enabled: true
    methods:
      - log_normalize
      - z_score_normalize
    datasets:
      - test_dataset_1
    params:
      log_normalize:
        base: 2.0
  imputation:
    methods:
      - knn_impute
""")
        temp_path = f.name
    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_charts_file():
    """Create a temporary charts config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write("""
performance_bar:
  chart_type: bar
  title: Performance
  x_label: Method
  y_label: Time
  figsize: [10, 6]
  dpi: 300

accuracy_line:
  chart_type: line
  title: Accuracy
  grid_enabled: false
""")
        temp_path = f.name
    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


def test_load_config(temp_config_file):
    """Test loading configuration from file."""
    config = load_config(temp_config_file)

    assert isinstance(config, BenchmarkConfig)
    assert len(config.datasets) == 2
    assert "test_dataset_1" in config.datasets
    assert config.cache_enabled is True
    assert config.cache_dir == ".cache/test"
    assert config.max_workers == 2
    assert config.random_seed == 42
    assert config.verbose is False


def test_load_config_output_section(temp_config_file):
    """Test loading output configuration."""
    config = load_config(temp_config_file)

    assert isinstance(config.output, OutputConfig)
    assert config.output.directory == "test_output"
    assert config.output.generate_plots is True
    assert config.output.generate_report is True


def test_load_config_modules(temp_config_file):
    """Test loading module configurations."""
    config = load_config(temp_config_file)

    assert len(config.modules) >= 2

    # Find normalization module
    norm_mod = next((m for m in config.modules if m.name == "normalization"), None)
    assert norm_mod is not None
    assert norm_mod.enabled is True
    assert "log_normalize" in norm_mod.methods
    assert "z_score_normalize" in norm_mod.methods
    assert norm_mod.datasets == ["test_dataset_1"]
    assert norm_mod.params["log_normalize"]["base"] == 2.0

    # Find imputation module
    imp_mod = next((m for m in config.modules if m.name == "imputation"), None)
    assert imp_mod is not None
    assert "knn_impute" in imp_mod.methods


def test_load_config_creates_default():
    """Test that loading non-existent file creates default config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "non_existent.yaml"
        config = load_config(non_existent)

        assert isinstance(config, BenchmarkConfig)
        assert len(config.datasets) >= 1
        # File should be created
        assert non_existent.exists()


def test_save_config(temp_config_file):
    """Test saving configuration to file."""
    config = load_config(temp_config_file)
    config.max_workers = 8
    config.cache_dir = ".cache/custom"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "saved_config.yaml"
        save_config(config, save_path)

        # Load and verify
        loaded = load_config(save_path)
        assert loaded.max_workers == 8
        assert loaded.cache_dir == ".cache/custom"


def test_get_default_config():
    """Test getting default configuration."""
    config = get_default_config()

    assert isinstance(config, BenchmarkConfig)
    assert len(config.datasets) >= 3
    assert "synthetic_small" in config.datasets
    assert config.cache_enabled is True
    assert config.max_workers == 4
    assert len(config.modules) >= 4


def test_output_config_dataclass():
    """Test OutputConfig dataclass."""
    output = OutputConfig(
        directory="custom_output",
        generate_plots=False,
        formats=["svg"],
    )

    assert output.directory == "custom_output"
    assert output.generate_plots is False
    assert output.formats == ["svg"]
    assert output.generate_report is True  # default


def test_module_config_entry_dataclass():
    """Test ModuleConfigEntry dataclass."""
    entry = ModuleConfigEntry(
        name="test_module",
        enabled=False,
        methods=["method1", "method2"],
        datasets=["ds1", "ds2"],
        params={"method1": {"param1": 10}},
        priority=5,
    )

    assert entry.name == "test_module"
    assert entry.enabled is False
    assert len(entry.methods) == 2
    assert len(entry.datasets) == 2
    assert entry.params["method1"]["param1"] == 10
    assert entry.priority == 5


def test_chart_config_dataclass():
    """Test ChartConfig dataclass."""
    chart = ChartConfig(
        chart_type="bar",
        title="Test Chart",
        x_label="X",
        y_label="Y",
        figsize=(12.0, 8.0),
        dpi=600,
    )

    assert chart.chart_type == "bar"
    assert chart.title == "Test Chart"
    assert chart.x_label == "X"
    assert chart.y_label == "Y"
    assert chart.figsize == (12.0, 8.0)
    assert chart.dpi == 600
    assert chart.legend_enabled is True  # default
    assert chart.grid_enabled is True  # default


def test_load_charts_config(temp_charts_file):
    """Test loading charts configuration."""
    charts = load_charts_config(temp_charts_file)

    assert "performance_bar" in charts
    assert "accuracy_line" in charts

    perf_bar = charts["performance_bar"]
    assert perf_bar.chart_type == "bar"
    assert perf_bar.title == "Performance"
    assert perf_bar.x_label == "Method"
    assert perf_bar.y_label == "Time"
    assert perf_bar.figsize == (10.0, 6.0)
    assert perf_bar.dpi == 300
    assert perf_bar.grid_enabled is True  # default

    acc_line = charts["accuracy_line"]
    assert acc_line.chart_type == "line"
    assert acc_line.grid_enabled is False


def test_load_charts_config_nonexistent():
    """Test loading charts from non-existent file returns defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "non_existent_charts.yaml"
        charts = load_charts_config(non_existent)

        # Should return default charts
        assert len(charts) >= 5
        assert "performance_bar" in charts
        assert "accuracy_line" in charts
        assert "memory_box" in charts


def test_configuration_error():
    """Test ConfigurationError exception."""
    error = ConfigurationError("Test error message")

    assert str(error) == "Test error message"
    assert error.config_path is None

    error_with_path = ConfigurationError("Test error", Path("/test/path.yaml"))
    assert error_with_path.config_path == Path("/test/path.yaml")


def test_benchmark_config_dataclass():
    """Test BenchmarkConfig dataclass defaults."""
    config = BenchmarkConfig()

    assert config.datasets == []
    assert config.cache_enabled is True
    assert config.cache_dir == ".cache/benchmark"
    assert config.max_workers == 4
    assert config.random_seed == 42
    assert config.verbose is False
    assert len(config.modules) == 0
