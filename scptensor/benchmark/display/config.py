"""Configuration for benchmark display and reporting.

This module provides configuration classes and enums for customizing
the visualization and reporting of preprocessing method comparisons
between ScpTensor and Scanpy.

Components
----------
- MethodCategory: Enum of preprocessing method categories
- ComparisonLayer: Enum for method availability comparison
- PlotStyle: Enum for matplotlib style presets
- ReportConfig: Dataclass for report generation settings
- CATEGORY_METRICS: Metric requirements per category

Examples
--------
Create a custom report configuration:

>>> from scptensor.benchmark.display.config import ReportConfig, PlotStyle
>>> config = ReportConfig(
...     output_dir=Path("custom_output"),
...     plot_style=PlotStyle.IEEE,
...     plot_dpi=600,
...     include_regression_analysis=True
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

__all__ = [
    "MethodCategory",
    "ComparisonLayer",
    "PlotStyle",
    "ReportConfig",
    "CATEGORY_METRICS",
    "get_style_string",
    "get_category_metrics",
]


class MethodCategory(Enum):
    """Categories of preprocessing methods for benchmark comparison.

    Each category represents a distinct preprocessing step in the
    single-cell analysis pipeline, with specific metrics and
    evaluation criteria.
    """

    NORMALIZATION = "normalization"
    IMPUTATION = "imputation"
    INTEGRATION = "integration"
    QC = "qc"
    DIM_REDUCTION = "dim_reduction"
    FEATURE_SELECTION = "feature_selection"


class ComparisonLayer(Enum):
    """Method availability comparison layer.

    Indicates whether a preprocessing method is available in both
    frameworks (shared) or exclusive to one framework.
    """

    SHARED = "shared"
    SCPTENSOR_EXCLUSIVE = "scptensor_exclusive"
    SCANPY_EXCLUSIVE = "scanpy_exclusive"


class PlotStyle(Enum):
    """Matplotlib style presets for publication-quality figures."""

    SCIENCE = "science"
    IEEE = "ieee"
    NATURE = "nature"
    DEFAULT = "default"


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for benchmark report generation.

    Parameters
    ----------
    output_dir : Path
        Directory path for saving generated reports and figures.
        Default is "benchmark_results".
    include_figures : bool
        Whether to include visualizations in the report.
        Default is True.
    plot_style : PlotStyle
        Matplotlib style preset for all generated figures.
        Default is PlotStyle.SCIENCE.
    plot_dpi : int
        Resolution in dots per inch for saved figures.
        Default is 300 for publication quality.
    show_exclusive_methods : bool
        Whether to include framework-exclusive method comparisons.
        Default is True.
    include_regression_analysis : bool
        Whether to include regression analysis for performance trends.
        Default is False.
    """

    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    include_figures: bool = True
    plot_style: PlotStyle = PlotStyle.SCIENCE
    plot_dpi: int = 300
    show_exclusive_methods: bool = True
    include_regression_analysis: bool = False

    _DEFAULT_SUBDIRS: ClassVar[tuple[str, ...]] = (
        "figures",
        "tables",
        "summaries",
    )


CATEGORY_METRICS: dict[MethodCategory, dict[str, list[str]]] = {
    MethodCategory.NORMALIZATION: {
        "primary": ["execution_time", "memory_usage"],
        "secondary": ["distribution_preservation", "variance_stabilization"],
        "optional": ["log_fold_change", "sparsity_ratio"],
    },
    MethodCategory.IMPUTATION: {
        "primary": ["execution_time", "memory_usage", "imputation_accuracy"],
        "secondary": ["mse", "mae", "correlation"],
        "optional": ["downstream_task_impact", "biological_preservation"],
    },
    MethodCategory.INTEGRATION: {
        "primary": ["execution_time", "memory_usage", "batch_mixing"],
        "secondary": ["kbet_score", "graph_connectivity", "asil_score"],
        "optional": ["biological_variance_retention", "conservation_score"],
    },
    MethodCategory.QC: {
        "primary": ["execution_time", "cells_removed", "features_removed"],
        "secondary": ["mitochondrial_content", "gene_detection_rate"],
        "optional": ["doublet_detection", "cell_cycle_score"],
    },
    MethodCategory.DIM_REDUCTION: {
        "primary": ["execution_time", "memory_usage", "variance_explained"],
        "secondary": ["reconstruction_error", "neighborhood_preservation"],
        "optional": ["cluster_separation", "global_structure"],
    },
    MethodCategory.FEATURE_SELECTION: {
        "primary": ["execution_time", "n_features_selected", "variance_coverage"],
        "secondary": ["highly_variable_ranking", "dispersion_trend"],
        "optional": ["biological_relevance", "downstream_clustering_impact"],
    },
}


_STYLE_MAPPING: dict[PlotStyle, str | list[str]] = {
    PlotStyle.SCIENCE: ["science", "no-latex"],
    PlotStyle.IEEE: ["ieee"],
    PlotStyle.NATURE: ["nature"],
    PlotStyle.DEFAULT: "default",
}


def get_style_string(style: PlotStyle) -> str | list[str]:
    """Get matplotlib style string(s) for a given PlotStyle enum.

    Parameters
    ----------
    style : PlotStyle
        The plot style enum value.

    Returns
    -------
    str | list[str]
        Matplotlib style string or list of styles to apply.

    Examples
    --------
    >>> get_style_string(PlotStyle.SCIENCE)
    ['science', 'no-latex']
    >>> get_style_string(PlotStyle.DEFAULT)
    'default'
    """
    return _STYLE_MAPPING[style]


_VALID_METRIC_TYPES = {"primary", "secondary", "optional"}


def get_category_metrics(
    category: MethodCategory,
    metric_type: str = "primary",
) -> list[str]:
    """Get required metrics for a specific method category.

    Parameters
    ----------
    category : MethodCategory
        The preprocessing method category.
    metric_type : str, default "primary"
        Type of metrics to retrieve: "primary", "secondary", or "optional".

    Returns
    -------
    list[str]
        List of metric names for the specified category and type.

    Raises
    ------
    ValueError
        If metric_type is not one of "primary", "secondary", or "optional".

    Examples
    --------
    >>> get_category_metrics(MethodCategory.IMPUTATION, "primary")
    ['execution_time', 'memory_usage', 'imputation_accuracy']
    """
    if metric_type not in _VALID_METRIC_TYPES:
        raise ValueError(
            f"metric_type must be 'primary', 'secondary', or 'optional', got '{metric_type}'"
        )
    return CATEGORY_METRICS[category][metric_type]
