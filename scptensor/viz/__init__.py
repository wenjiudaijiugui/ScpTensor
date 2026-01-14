from .base import heatmap, scatter, violin
from .recipes import embedding, qc_completeness, qc_matrix_spy, volcano

__all__ = [
    # Base primitives
    "scatter",
    "heatmap",
    "violin",
    # Recipe plots
    "embedding",
    "qc_completeness",
    "qc_matrix_spy",
    "volcano",
]
