"""Dimensionality reduction module for ScpTensor.

This module provides dimensionality reduction methods aligned with scanpy's API:
- reduce_pca: Principal Component Analysis
- reduce_umap: UMAP embedding

Main functions follow scanpy naming convention (reduce_*).
"""

from .pca import SolverType, reduce_pca
from .umap import reduce_umap

__all__ = [
    "reduce_pca",
    "reduce_umap",
    "SolverType",
]
