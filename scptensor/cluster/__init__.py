"""Clustering module for ScpTensor.

This module provides clustering methods aligned with scanpy's API:
- cluster_kmeans: K-means clustering
- cluster_leiden: Leiden graph clustering

Functions follow scanpy naming convention (cluster_*).
"""

from .graph import cluster_leiden
from .kmeans import cluster_kmeans

__all__ = [
    "cluster_kmeans",
    "cluster_leiden",
]
