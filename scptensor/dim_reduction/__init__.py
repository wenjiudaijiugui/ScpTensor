"""Dimensionality reduction module for ScpTensor.

This module provides dimensionality reduction methods for single-cell
proteomics data, including PCA and UMAP.
"""

from .pca import get_solver_info, pca
from .umap import umap

__all__ = ["pca", "umap", "get_solver_info"]
