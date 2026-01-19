"""Configuration loader for benchmark module."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "benchmark_config.yaml"
DEFAULT_CHARTS_PATH = DEFAULT_CONFIG_DIR / "charts.yaml"


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_path: Path | None = None) -> None:
        self.config_path = config_path
        super().__init__(message)


@dataclass(slots=True)
class ChartConfig:
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
    name: str
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    params: dict[str, dict[str, object]] = field(default_factory=dict)
    priority: int = 0


@dataclass(slots=True)
class OutputConfig:
    directory: str = "benchmark_results"
    generate_plots: bool = True
    generate_report: bool = True
    save_intermediate: bool = False
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass(slots=True)
class BenchmarkConfig:
    datasets: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache_enabled: bool = True
    cache_dir: str = ".cache/benchmark"
    max_workers: int = 4
    modules: list[ModuleConfigEntry] = field(default_factory=list)
    random_seed: int = 42
    verbose: bool = False


def _parse_output_config(data: dict) -> OutputConfig:
    return OutputConfig(
        directory=data.get("directory", "benchmark_results"),
        generate_plots=data.get("generate_plots", True),
        generate_report=data.get("generate_report", True),
        save_intermediate=data.get("save_intermediate", False),
        formats=data.get("formats", ["png", "pdf"]),
    )


def _parse_module_configs(data: dict | None) -> list[ModuleConfigEntry]:
    if not data:
        return []

    modules = []
    for name, methods in data.items():
        if isinstance(methods, list):
            modules.append(ModuleConfigEntry(name=name, methods=methods))
        elif isinstance(methods, dict):
            modules.append(ModuleConfigEntry(
                name=name,
                enabled=methods.get("enabled", True),
                methods=methods.get("methods", []),
                datasets=methods.get("datasets", []),
                params=methods.get("params", {}),
                priority=methods.get("priority", 0),
            ))
    return modules


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfig:
    path = Path(config_path)

    if not path.exists():
        default_config = get_default_config()
        save_config(default_config, path)
        return default_config

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return BenchmarkConfig(
        datasets=data.get("datasets", []),
        output=_parse_output_config(data.get("output", {})),
        cache_enabled=data.get("cache_enabled", True),
        cache_dir=data.get("cache_dir", ".cache/benchmark"),
        max_workers=data.get("max_workers", 4),
        modules=_parse_module_configs(data.get("methods")),
        random_seed=data.get("random_seed", 42),
        verbose=data.get("verbose", False),
    )


def load_charts_config(config_path: str | Path = DEFAULT_CHARTS_PATH) -> dict[str, ChartConfig]:
    path = Path(config_path)

    if not path.exists():
        return _get_default_charts_config()

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return _get_default_charts_config()

    return {
        name: ChartConfig(
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
        for name, chart_data in data.items()
        if isinstance(chart_data, dict)
    }


def save_config(config: BenchmarkConfig, config_path: str | Path) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
            m.name: {
                "enabled": m.enabled,
                "methods": m.methods,
                "datasets": m.datasets,
                "params": m.params,
                "priority": m.priority,
            }
            for m in config.modules
        },
        "random_seed": config.random_seed,
        "verbose": config.verbose,
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=["synthetic_small", "synthetic_medium", "synthetic_large"],
        output=OutputConfig(directory="benchmark_results", formats=["png", "pdf"]),
        modules=[
            ModuleConfigEntry(name="normalization", methods=["log_normalize", "z_score_normalize"], priority=1),
            ModuleConfigEntry(name="imputation", methods=["knn_impute", "svd_impute"], priority=2),
            ModuleConfigEntry(name="dimensionality_reduction", methods=["pca", "umap"], priority=3),
            ModuleConfigEntry(name="clustering", methods=["kmeans"], priority=4),
        ],
    )


def _get_default_charts_config() -> dict[str, ChartConfig]:
    return {
        "performance_bar": ChartConfig(chart_type="bar", title="Execution Time Comparison",
                                       x_label="Method", y_label="Time (seconds)", figsize=(10.0, 6.0),
                                       color_palette="Set2", legend_enabled=False),
        "accuracy_line": ChartConfig(chart_type="line", title="Accuracy Trend",
                                     x_label="Parameter Value", y_label="Accuracy Score"),
        "memory_box": ChartConfig(chart_type="box", title="Memory Usage Distribution",
                                  x_label="Method", y_label="Memory (MB)", figsize=(10.0, 6.0), legend_enabled=False),
        "correlation_heatmap": ChartConfig(chart_type="heatmap", title="Method Correlation Matrix",
                                           figsize=(8.0, 8.0), grid_enabled=False),
        "scatter_comparison": ChartConfig(chart_type="scatter", title="Method Comparison",
                                          x_label="X", y_label="Y", figsize=(8.0, 8.0)),
    }


__all__ = [
    "BenchmarkConfig", "ChartConfig", "ModuleConfigEntry", "OutputConfig",
    "load_config", "load_charts_config", "save_config", "get_default_config",
]
