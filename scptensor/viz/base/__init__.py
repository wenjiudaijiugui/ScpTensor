"""Base visualization components.

This module exports core visualization utilities for ScpTensor:
- Style management with SciencePlots integration
- Data extraction from ScpContainer
- Multi-panel layout management
- Missing value handling
- Plot functions (scatter, heatmap, violin)
"""

from .data_extractor import DataExtractor
from .heatmap import heatmap
from .missing_value import MissingValueHandler
from .multi_panel import PanelLayout
from .scatter import scatter
from .style import CMAP_PROTEOMICS, THEMES, PlotStyle, setup_style
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
]
