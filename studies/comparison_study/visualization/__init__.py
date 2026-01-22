"""
Visualization and report generation module for pipeline comparison.

This module provides comprehensive visualization tools for generating
high-quality scientific figures and reports for pipeline comparison studies.

Main Components
---------------
- ComparisonPlotter: Generate publication-quality figures (300 DPI)
- ReportGenerator: Create comprehensive comparison reports
- calculate_overall_scores: Compute weighted pipeline rankings

Usage Examples
--------------
>>> from docs.comparison_study.visualization import ComparisonPlotter
>>> plotter = ComparisonPlotter(config, output_dir="outputs/figures")
>>> path = plotter.plot_batch_effects_comparison(results)

>>> from docs.comparison_study.visualization import generate_all_figures
>>> figures = generate_all_figures(results, config)

>>> from docs.comparison_study.visualization import ReportGenerator
>>> generator = ReportGenerator(config)
>>> pdf_path = generator.generate_report(results, figures)
"""

from __future__ import annotations

from .plots import ComparisonPlotter, generate_all_figures
from .report_generator import ReportGenerator, calculate_overall_scores

__all__ = [
    "ComparisonPlotter",
    "generate_all_figures",
    "ReportGenerator",
    "calculate_overall_scores",
]
