# New API
from .basic import cluster_kmeans
from .graph import cluster_leiden
from .kmeans import cluster_kmeans as cluster_kmeans_assay

__all__ = [
    "cluster_kmeans",
    "cluster_leiden",
    "cluster_kmeans_assay",
]

# Deprecated aliases are available via submodule imports:
# - from scptensor.cluster.basic import kmeans
# - from scptensor.cluster.graph import leiden
# - from scptensor.cluster.kmeans import run_kmeans
# These will be removed in v0.2.0
