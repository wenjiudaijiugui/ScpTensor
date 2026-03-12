"""Experimental dimensionality-reduction module for ScpTensor.

This module provides dimensionality reduction methods aligned with scanpy's API:
- reduce_pca: Principal Component Analysis
- reduce_tsne: t-SNE embedding
- reduce_umap: UMAP embedding

Status
------
This module is available for exploratory downstream analysis and is currently
classified as *experimental* in ScpTensor release scope.
"""

from .pca import SolverType, reduce_pca
from .tsne import reduce_tsne
from .umap import reduce_umap

__all__ = [
    "reduce_pca",
    "reduce_tsne",
    "reduce_umap",
    "SolverType",
]
