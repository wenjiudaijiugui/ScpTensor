"""Experimental analysis modules for ScpTensor.

This namespace hosts modules that are available for exploratory downstream
analysis, but are not part of the core preprocessing release contract.

Current experimental APIs:
- dimensionality reduction: ``reduce_pca``, ``reduce_tsne``, ``reduce_umap``
- clustering: ``cluster_kmeans``, ``cluster_leiden``
- peptide/PSM QC: ``qc_psm``
"""

import scptensor.qc.qc_psm as qc_psm
from scptensor.cluster import cluster_kmeans, cluster_leiden
from scptensor.dim_reduction import SolverType, reduce_pca, reduce_tsne, reduce_umap

__all__ = [
    "reduce_pca",
    "reduce_tsne",
    "reduce_umap",
    "SolverType",
    "cluster_kmeans",
    "cluster_leiden",
    "qc_psm",
]
