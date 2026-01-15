from .basic import cluster_kmeans
from .graph import cluster_leiden
from .kmeans import cluster_kmeans as cluster_kmeans_assay

# Deprecated aliases for backward compatibility
from .basic import kmeans
from .graph import leiden
from .kmeans import run_kmeans

__all__ = [
    # New API
    "cluster_kmeans",
    "cluster_leiden",
    "cluster_kmeans_assay",
    # Deprecated (remove in v0.2.0)
    "kmeans",
    "leiden",
    "run_kmeans",
]
