"""Tests for experimental namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.experimental as experimental
import scptensor.qc as stable_qc
import scptensor.qc.qc_psm as qc_psm_core
from scptensor.cluster import cluster_kmeans as cluster_kmeans_core
from scptensor.cluster import cluster_leiden as cluster_leiden_core
from scptensor.dim_reduction import SolverType as SolverType_core
from scptensor.dim_reduction import reduce_pca as reduce_pca_core
from scptensor.dim_reduction import reduce_tsne as reduce_tsne_core
from scptensor.dim_reduction import reduce_umap as reduce_umap_core
from scptensor.experimental import (
    SolverType,
    cluster_kmeans,
    cluster_leiden,
    qc_psm,
    reduce_pca,
    reduce_tsne,
    reduce_umap,
)


def test_reduce_and_cluster_not_exported_from_top_level() -> None:
    for name in (
        "reduce_pca",
        "reduce_tsne",
        "reduce_umap",
        "cluster_kmeans",
        "cluster_leiden",
        "SolverType",
        "qc_psm",
    ):
        assert not hasattr(scp, name)


def test_experimental_namespace_reexports_core_implementations() -> None:
    assert experimental.__all__ == [
        "reduce_pca",
        "reduce_tsne",
        "reduce_umap",
        "SolverType",
        "cluster_kmeans",
        "cluster_leiden",
        "qc_psm",
    ]
    assert reduce_pca is reduce_pca_core
    assert reduce_tsne is reduce_tsne_core
    assert reduce_umap is reduce_umap_core
    assert SolverType is SolverType_core
    assert cluster_kmeans is cluster_kmeans_core
    assert cluster_leiden is cluster_leiden_core
    assert qc_psm is qc_psm_core


def test_qc_psm_not_exported_from_stable_qc_namespace() -> None:
    assert "qc_psm" not in stable_qc.__all__
