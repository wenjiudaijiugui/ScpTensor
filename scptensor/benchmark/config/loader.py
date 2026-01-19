"""Configuration file loader for benchmark module.

Provides YAML-based configuration loading with default value support
and type-safe configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors.

    Attributes
    ----------
    message : str
        Error message describing the configuration issue.
    config_path : Path | None
        Path to the configuration file that caused the error.
    """

    def __init__(self, message: str, config_path: Path | None = None) -> None:
        super().__init__(message)
        self.config_path = config_path


# Default paths
DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "benchmark_config.yaml"
DEFAULT_CHARTS_PATH = DEFAULT_CONFIG_DIR / "charts.yaml"


@dataclass(slots=True)
class ChartConfig:
    """Configuration for a single chart type.

    Attributes
    ----------
    chart_type : str
        Type of chart: bar, line, heatmap, box, scatter, etc.
    title : str
        Chart title.
    x_label : str | None
        Label for x-axis. None if not applicable.
    y_label : str | None
        Label for y-axis. None if not applicable.
    figsize : tuple[float, float]
        Figure size as (width, height) in inches.
    style : str
        Matplotlib style to use (e.g., 'science', 'no-latex').
    dpi : int
        Dots per inch for output images.
    color_palette : str | None
        Color palette name. None uses default.
    legend_enabled : bool
        Whether to show legend.
    grid_enabled : bool
        Whether to show grid lines.
    """

    chart_type: str
    title: str
    x_label: str | None = None
    y_label: str | None = None
    figsize: tuple[float, float] = (8.0, 6.0)
    style: str = "science"
    dpi: int = 300
    color_palette: str | None = None
    legend_enabled: bool = True
    grid_enabled: bool = True


@dataclass(slots=True)
class ModuleConfigEntry:
    """Configuration entry for a single benchmark module.

    Attributes
    ----------
    name : str
        Module name (e.g., 'normalization', 'imputation').
    enabled : bool
        Whether this module is enabled for benchmarking.
    datasets : list[str]
        List of dataset names to use for this module.
    methods : list[str]
        List of method names to benchmark within this module.
    params : dict[str, dict[str, object]]
        Method-specific parameters as {method_name: {param: value}}.
    priority : int
        Execution priority (lower = earlier execution).
    """

    name: str
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    params: dict[str, dict[str, object]] = field(default_factory=dict)
    priority: int = 0


@dataclass(slots=True)
class OutputConfig:
    """Configuration for benchmark output.

    Attributes
    ----------
    directory : str
        Output directory path for results.
    generate_plots : bool
        Whether to generate visualization plots.
    generate_report : bool
        Whether to generate HTML/Markdown reports.
    save_intermediate : bool
        Whether to save intermediate results.
    formats : list[str]
        Output formats for plots (e.g., ['png', 'pdf', 'svg']).
    """

    directory: str = "benchmark_results"
    generate_plots: bool = True
    generate_report: bool = True
    save_intermediate: bool = False
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass(slots=True)
class BenchmarkConfig:
    """Main configuration for benchmark suite.

    Attributes
    ----------
    datasets : list[str]
        List of dataset names to use in benchmarks.
    output : OutputConfig
        Output configuration settings.
    cache_enabled : bool
        Whether to enable result caching.
    cache_dir : str
        Directory for cached results.
    max_workers : int
        Maximum number of parallel workers.
    modules : list[ModuleConfigEntry]
        Per-module benchmark configurations.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to enable verbose logging.
    """

    datasets: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache_enabled: bool = True
    cache_dir: str = ".cache/benchmark"
    max_workers: int = 4
    modules: list[ModuleConfigEntry] = field(default_factory=list)
    random_seed: int = 42
    verbose: bool = False


def _parse_output_config(data: dict | None) -> OutputConfig:
    """Parse output configuration from YAML data.

    Parameters
    ----------
    data : dict | None
        Raw YAML output section.

    Returns
    -------
    OutputConfig
        Parsed output configuration.
    """
    if data is None:
        return OutputConfig()

    return OutputConfig(
        directory=data.get("directory", "benchmark_results"),
        generate_plots=data.get("generate_plots", True),
        generate_report=data.get("generate_report", True),
        save_intermediate=data.get("save_intermediate", False),
        formats=data.get("formats", ["png", "pdf"]),
    )


def _parse_module_configs(data: dict | None) -> list[ModuleConfigEntry]:
    """Parse module configurations from YAML data.

    Parameters
    ----------
    data : dict | None
        Raw YAML methods/sections data.

    Returns
    -------
    list[ModuleConfigEntry]
        Parsed module configurations.
    """
    if data is None or not data:
        return []

    modules = []
    for name, methods in data.items():
        if isinstance(methods, list):
            modules.append(
                ModuleConfigEntry(
                    name=name,
                    methods=methods,
                )
            )
        elif isinstance(methods, dict):
            modules.append(
                ModuleConfigEntry(
                    name=name,
                    enabled=methods.get("enabled", True),
                    methods=methods.get("methods", []),
                    datasets=methods.get("datasets", []),
                    params=methods.get("params", {}),
                    priority=methods.get("priority", 0),
                )
            )

    return modules


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file.

    If the file does not exist, creates and returns a default configuration.
    The default configuration is saved to the specified path.

    Parameters
    ----------
    config_path : str | Path
        Path to the configuration YAML file.

    Returns
    -------
    BenchmarkConfig
        Loaded benchmark configuration.

    Raises
    ------
    ConfigurationError
        If YAML parsing fails or file is unreadable.
    """
    path = Path(config_path)

    if not path.exists():
        default_config = get_default_config()
        save_config(default_config, path)
        return default_config

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML config: {e}",
            config_path=path,
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"Failed to read config file: {e}",
            config_path=path,
        ) from e

    if data is None:
        data = {}

    return BenchmarkConfig(
        datasets=data.get("datasets", []),
        output=_parse_output_config(data.get("output")),
        cache_enabled=data.get("cache_enabled", True),
        cache_dir=data.get("cache_dir", ".cache/benchmark"),
        max_workers=data.get("max_workers", 4),
        modules=_parse_module_configs(data.get("methods")),
        random_seed=data.get("random_seed", 42),
        verbose=data.get("verbose", False),
    )


def load_charts_config(config_path: str | Path = DEFAULT_CHARTS_PATH) -> dict[str, ChartConfig]:
    """Load chart configuration from YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the charts YAML file.

    Returns
    -------
    dict[str, ChartConfig]
        Dictionary mapping chart names to their configurations.

    Raises
    ------
    ConfigurationError
        If YAML parsing fails or file is unreadable.
    """
    path = Path(config_path)

    if not path.exists():
        return _get_default_charts_config()

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse charts YAML: {e}",
            config_path=path,
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"Failed to read charts file: {e}",
            config_path=path,
        ) from e

    if data is None:
        return _get_default_charts_config()

    charts: dict[str, ChartConfig] = {}
    for name, chart_data in data.items():
        if isinstance(chart_data, dict):
            charts[name] = ChartConfig(
                chart_type=chart_data.get("chart_type", "bar"),
                title=chart_data.get("title", name),
                x_label=chart_data.get("x_label"),
                y_label=chart_data.get("y_label"),
                figsize=tuple(chart_data.get("figsize", (8.0, 6.0))),
                style=chart_data.get("style", "science"),
                dpi=chart_data.get("dpi", 300),
                color_palette=chart_data.get("color_palette"),
                legend_enabled=chart_data.get("legend_enabled", True),
                grid_enabled=chart_data.get("grid_enabled", True),
            )

    return charts


def save_config(config: BenchmarkConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration to save.
    config_path : str | Path
        Path where configuration should be saved.

    Raises
    ------
    ConfigurationError
        If file cannot be written.
    """
    path = Path(config_path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for YAML serialization
    data = {
        "datasets": config.datasets,
        "output": {
            "directory": config.output.directory,
            "generate_plots": config.output.generate_plots,
            "generate_report": config.output.generate_report,
            "save_intermediate": config.output.save_intermediate,
            "formats": config.output.formats,
        },
        "cache_enabled": config.cache_enabled,
        "cache_dir": config.cache_dir,
        "max_workers": config.max_workers,
        "methods": {
            module.name: {
                "enabled": module.enabled,
                "methods": module.methods,
                "datasets": module.datasets,
                "params": module.params,
                "priority": module.priority,
            }
            for module in config.modules
        },
        "random_seed": config.random_seed,
        "verbose": config.verbose,
    }

    try:
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigurationError(
            f"Failed to save config file: {e}",
            config_path=path,
        ) from e


def get_default_config() -> BenchmarkConfig:
    """Get default benchmark configuration.

    Returns
    -------
    BenchmarkConfig
        Default configuration with sensible values.
    """
    return BenchmarkConfig(
        datasets=[
            "synthetic_small",
            "synthetic_medium",
            "synthetic_large",
        ],
        output=OutputConfig(
            directory="benchmark_results",
            generate_plots=True,
            generate_report=True,
            save_intermediate=False,
            formats=["png", "pdf"],
        ),
        cache_enabled=True,
        cache_dir=".cache/benchmark",
        max_workers=4,
        modules=[
            ModuleConfigEntry(
                name="normalization",
                methods=["log_normalize", "z_score_normalize"],
                priority=1,
            ),
            ModuleConfigEntry(
                name="imputation",
                methods=["knn_impute", "svd_impute"],
                priority=2,
            ),
            ModuleConfigEntry(
                name="dimensionality_reduction",
                methods=["pca", "umap"],
                priority=3,
            ),
            ModuleConfigEntry(
                name="clustering",
                methods=["kmeans"],
                priority=4,
            ),
        ],
        random_seed=42,
        verbose=False,
    )


def _get_default_charts_config() -> dict[str, ChartConfig]:
    """Get default chart configurations.

    Returns
    -------
    dict[str, ChartConfig]
        Default chart configurations.
    """
    return {
        "performance_bar": ChartConfig(
            chart_type="bar",
            title="Execution Time Comparison",
            x_label="Method",
            y_label="Time (seconds)",
            figsize=(10.0, 6.0),
            style="science",
            dpi=300,
            color_palette="Set2",
            legend_enabled=False,
            grid_enabled=True,
        ),
        "accuracy_line": ChartConfig(
            chart_type="line",
            title="Accuracy Trend",
            x_label="Parameter Value",
            y_label="Accuracy Score",
            figsize=(8.0, 6.0),
            style="science",
            dpi=300,
            legend_enabled=True,
            grid_enabled=True,
        ),
        "memory_box": ChartConfig(
            chart_type="box",
            title="Memory Usage Distribution",
            x_label="Method",
            y_label="Memory (MB)",
            figsize=(10.0, 6.0),
            style="science",
            dpi=300,
            legend_enabled=False,
            grid_enabled=True,
        ),
        "correlation_heatmap": ChartConfig(
            chart_type="heatmap",
            title="Method Correlation Matrix",
            x_label=None,
            y_label=None,
            figsize=(8.0, 8.0),
            style="science",
            dpi=300,
            legend_enabled=True,
            grid_enabled=False,
        ),
        "scatter_comparison": ChartConfig(
            chart_type="scatter",
            title="Method Comparison",
            x_label="X",
            y_label="Y",
            figsize=(8.0, 8.0),
            style="science",
            dpi=300,
            legend_enabled=True,
            grid_enabled=True,
        ),
    }


__all__ = [
    "BenchmarkConfig",
    "ChartConfig",
    "ModuleConfigEntry",
    "OutputConfig",
    "load_config",
    "load_charts_config",
    "save_config",
    "get_default_config",
    "ConfigurationError",
]
