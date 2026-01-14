from .base import heatmap, violin
from .base import scatter as base_scatter
from .recipes import (
    embedding,
    pca,
    qc_completeness,
    qc_matrix_spy,
    scatter,
    tsne,
    umap,
    volcano,
)

__all__ = [
    # Base primitives
    "base_scatter",
    "heatmap",
    "violin",
    # Recipe plots
    "scatter",
    "umap",
    "pca",
    "tsne",
    "embedding",
    "qc_completeness",
    "qc_matrix_spy",
    "volcano",
]
