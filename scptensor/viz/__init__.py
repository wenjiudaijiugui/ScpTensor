from .base import scatter, heatmap, violin
from .recipes.qc import qc_completeness, qc_matrix_spy
from .recipes.embedding import embedding
from .recipes.stats import volcano

__all__ = [
    'scatter',
    'heatmap',
    'violin',
    'qc_completeness',
    'qc_matrix_spy',
    'embedding',
    'volcano'
]
