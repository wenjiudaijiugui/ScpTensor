"""Tests for experimental namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.qc.qc_psm as qc_psm_core
from scptensor.cluster import cluster_kmeans as cluster_kmeans_core
from scptensor.dim_reduction import reduce_pca as reduce_pca_core
from scptensor.experimental import cluster_kmeans, qc_psm, reduce_pca


def test_reduce_and_cluster_not_exported_from_top_level() -> None:
    assert not hasattr(scp, "reduce_pca")
    assert not hasattr(scp, "cluster_kmeans")
    assert not hasattr(scp, "qc_psm")


def test_experimental_namespace_reexports_core_implementations() -> None:
    assert reduce_pca is reduce_pca_core
    assert cluster_kmeans is cluster_kmeans_core
    assert qc_psm is qc_psm_core
