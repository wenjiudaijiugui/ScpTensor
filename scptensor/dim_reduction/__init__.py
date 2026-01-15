"""Dimensionality reduction module for ScpTensor.

This module provides dimensionality reduction methods for single-cell
proteomics data, including PCA and UMAP.
"""

from .pca import get_solver_info, pca, reduce_pca
from .umap import reduce_umap, umap

__all__ = [
    "reduce_pca",
    "reduce_umap",
    "pca",  # Deprecated: use reduce_pca
    "umap",  # Deprecated: use reduce_umap
    "get_solver_info",
]
