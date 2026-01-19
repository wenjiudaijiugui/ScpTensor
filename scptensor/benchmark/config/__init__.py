"""Configuration module for benchmark suite.

Provides YAML-based configuration loading with type-safe dataclasses.
"""

from .loader import (
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

__all__ = [
    "BenchmarkConfig",
    "ChartConfig",
    "ConfigurationError",
    "ModuleConfigEntry",
    "OutputConfig",
    "load_config",
    "load_charts_config",
    "save_config",
    "get_default_config",
]
