"""Base visualization components.

This module exports core visualization utilities for ScpTensor:
- Style management with SciencePlots integration
- Data extraction from ScpContainer
- Multi-panel layout management
- Missing value handling
- Plot functions (scatter, heatmap, violin)
- Validation utilities for input checking
"""

from .data_extractor import DataExtractor
from .heatmap import heatmap
from .missing_value import MissingValueHandler
from .multi_panel import PanelLayout
from .scatter import scatter
from .style import CMAP_PROTEOMICS, THEMES, PlotStyle, setup_style
from .validation import (
    validate_container,
    validate_features,
    validate_groupby,
    validate_layer,
    validate_plot_data,
)
from .violin import violin

__all__ = [
    # Style
    "PlotStyle",
    "THEMES",
    "CMAP_PROTEOMICS",
    "setup_style",
    # Components
    "PanelLayout",
    "DataExtractor",
    "MissingValueHandler",
    # Plot functions
    "scatter",
    "heatmap",
    "violin",
    # Validation
    "validate_container",
    "validate_layer",
    "validate_features",
    "validate_groupby",
    "validate_plot_data",
]
