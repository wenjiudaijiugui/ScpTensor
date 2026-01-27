"""Comparison study module for evaluating analysis methods."""

from . import data  # noqa: F401
from . import evaluation  # noqa: F401
from . import visualization  # noqa: F401
from .comparison_engine import compare_pipelines, generate_comparison_report  # noqa: F401
from .data_generation import (  # noqa: F401
    generate_large_dataset,
    generate_medium_dataset,
    generate_small_dataset,
    generate_synthetic_data,
)
from .metrics import (
    calculate_all_metrics,
    calculate_asw,
    calculate_clisi,
    calculate_correlation,
    calculate_ilisi,
    calculate_integration_metrics,
    calculate_kbet,
    calculate_mae,
    calculate_mse,
    measure_memory_usage,
    measure_runtime,
)
from .plotting import (  # noqa: F401
    plot_batch_effects,
    plot_performance_comparison,
    plot_radar_chart,
)

__all__ = [
    # Data generation
    "generate_small_dataset",
    "generate_medium_dataset",
    "generate_large_dataset",
    "generate_synthetic_data",
    # Metrics
    "calculate_kbet",
    "calculate_ilisi",
    "calculate_clisi",
    "calculate_asw",
    "calculate_mse",
    "calculate_mae",
    "calculate_correlation",
    "measure_runtime",
    "measure_memory_usage",
    "calculate_all_metrics",
    "calculate_integration_metrics",
    # Comparison engine
    "compare_pipelines",
    "generate_comparison_report",
    # Plotting
    "plot_batch_effects",
    "plot_performance_comparison",
    "plot_radar_chart",
    # Submodules
    "data",
    "evaluation",
    "visualization",
]
