"""Experimental analysis modules for ScpTensor.

This namespace hosts experimental helpers that are intentionally kept outside
the stable preprocessing release contract.

Current experimental APIs:
- downstream analysis helpers:
  ``reduce_pca``, ``reduce_tsne``, ``reduce_umap``,
  ``cluster_kmeans``, ``cluster_leiden``
- experimental pre-aggregation helper:
  ``qc_psm``

``qc_psm`` stays in this namespace for boundary clarity and discoverability,
but it should not be interpreted as a downstream embedding/clustering helper.
"""

from scptensor.cluster import cluster_kmeans, cluster_leiden
from scptensor.dim_reduction import SolverType, reduce_pca, reduce_tsne, reduce_umap
from scptensor.qc import qc_psm

__all__ = [
    "reduce_pca",
    "reduce_tsne",
    "reduce_umap",
    "SolverType",
    "cluster_kmeans",
    "cluster_leiden",
    "qc_psm",
]
