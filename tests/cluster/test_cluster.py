"""
Tests for clustering modules.

Tests cover:
- cluster_kmeans (simplified API, aligned with scanpy)
- cluster_leiden (graph clustering with optional dependencies)
"""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.cluster import cluster_kmeans, cluster_leiden
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError

LEIDEN_AVAILABLE = (
    importlib.util.find_spec("leidenalg") is not None
    and importlib.util.find_spec("igraph") is not None
)

# =============================================================================
# Fixtures for clustering tests
# =============================================================================


@pytest.fixture
def pca_container(sample_obs):
    """Create a container with PCA data for clustering tests."""
    np.random.seed(42)
    n_samples = len(sample_obs)
    n_pcs = 10

    X_pca = np.random.randn(n_samples, n_pcs)
    for i in range(n_samples):
        cluster_id = i % 3
        X_pca[i, :] += cluster_id * 2.0

    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(n_pcs)]})
    assay_pca = Assay(var=var_pca, layers={"X": ScpMatrix(X=X_pca)})

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca})


@pytest.fixture
def sparse_pca_container(sample_obs):
    """Create a container with sparse PCA data."""
    np.random.seed(42)
    n_samples = len(sample_obs)
    n_pcs = 10

    X_dense = np.random.randn(n_samples, n_pcs)
    X_dense[X_dense < 0.3] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(n_pcs)]})
    assay_pca = Assay(var=var_pca, layers={"X": ScpMatrix(X=X_sparse)})

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca})


@pytest.fixture
def multi_assay_container(sample_obs):
    """Create a container with multiple assays."""
    np.random.seed(42)
    n_samples = len(sample_obs)

    X_pca = np.random.randn(n_samples, 5)
    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(5)]})
    assay_pca = Assay(var=var_pca, layers={"X": ScpMatrix(X=X_pca)})

    X_umap = np.random.randn(n_samples, 2)
    var_umap = pl.DataFrame({"_index": ["UMAP1", "UMAP2"]})
    assay_umap = Assay(var=var_umap, layers={"X": ScpMatrix(X=X_umap)})

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca, "reduce_umap": assay_umap})


@pytest.fixture
def multi_layer_container(sample_obs):
    """Create a container with multiple layers."""
    np.random.seed(42)
    n_samples = len(sample_obs)
    n_pcs = 10

    X_pca = np.random.randn(n_samples, n_pcs)
    X_normed = X_pca / np.linalg.norm(X_pca, axis=1, keepdims=True)

    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(n_pcs)]})

    assay_pca = Assay(
        var=var_pca,
        layers={
            "X": ScpMatrix(X=X_pca),
            "normalized": ScpMatrix(X=X_normed),
        },
    )

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca})


# =============================================================================
# cluster_kmeans tests
# =============================================================================


class TestClusterKmeans:
    """Tests for cluster_kmeans function."""

    def test_kmeans_basic(self, pca_container):
        """Test basic K-means clustering."""
        result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        expected_col = "kmeans_k3"
        assert expected_col in result.obs.columns
        assert result.obs[expected_col].dtype == pl.String

    def test_kmeans_different_n_clusters(self, pca_container):
        """Test K-means with different k values."""
        for k in [2, 4, 5]:
            result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=k)
            expected_col = f"kmeans_k{k}"
            assert expected_col in result.obs.columns

            labels = result.obs[expected_col].cast(pl.Int32).to_numpy()
            assert np.all((labels >= 0) & (labels < k))

    def test_kmeans_custom_key_added(self, pca_container):
        """Test K-means with custom key_added."""
        result = cluster_kmeans(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            key_added="my_clusters",
        )

        assert "my_clusters" in result.obs.columns

    def test_kmeans_custom_assay_and_layer(self, multi_assay_container):
        """Test K-means on custom assay and layer."""
        result = cluster_kmeans(
            multi_assay_container,
            assay_name="reduce_umap",
            base_layer="X",
            n_clusters=2,
        )

        assert "kmeans_k2" in result.obs.columns

    def test_kmeans_random_state_reproducibility(self, pca_container):
        """Test K-means reproducibility with random_state."""
        result1 = cluster_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, random_state=123
        )
        result2 = cluster_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, random_state=123
        )

        labels1 = result1.obs["kmeans_k3"].to_list()
        labels2 = result2.obs["kmeans_k3"].to_list()

        assert labels1 == labels2

    def test_kmeans_sparse_matrix(self, sparse_pca_container):
        """Test K-means with sparse input matrix."""
        result = cluster_kmeans(sparse_pca_container, assay_name="reduce_pca", n_clusters=3)

        assert "kmeans_k3" in result.obs.columns

    def test_kmeans_new_container_returned(self, pca_container):
        """Test that a new container is returned (immutable pattern)."""
        original_obs_columns = set(pca_container.obs.columns)
        result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert set(pca_container.obs.columns) == original_obs_columns
        assert "kmeans_k3" in result.obs.columns

    def test_kmeans_history_logging(self, pca_container):
        """Test that K-means operation is logged in history."""
        initial_history_len = len(pca_container.history)
        result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert len(result.history) == initial_history_len + 1
        log_entry = result.history[-1]
        assert log_entry.action == "cluster_kmeans"
        assert log_entry.params["n_clusters"] == 3

    def test_kmeans_history_is_not_shared_with_input(self, pca_container):
        """Result container history should not mutate the input container."""
        result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert len(pca_container.history) == 0
        assert len(result.history) == 1
        assert pca_container.history is not result.history

    def test_kmeans_invalid_n_clusters_zero(self, pca_container):
        """Test K-means with n_clusters=0 raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=0)

        assert "n_clusters must be positive" in str(exc_info.value)

    def test_kmeans_invalid_n_clusters_negative(self, pca_container):
        """Test K-means with negative n_clusters raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=-5)

        assert "n_clusters must be positive" in str(exc_info.value)

    def test_kmeans_assay_not_found(self, pca_container):
        """Test K-means with non-existent assay raises error."""
        with pytest.raises(AssayNotFoundError):
            cluster_kmeans(pca_container, assay_name="nonexistent")

    def test_kmeans_layer_not_found(self, pca_container):
        """Test K-means with non-existent layer raises error."""
        with pytest.raises(LayerNotFoundError):
            cluster_kmeans(pca_container, assay_name="reduce_pca", base_layer="nonexistent")

    def test_kmeans_backend_sklearn_default(self, pca_container):
        """Test K-means with default sklearn backend."""
        result = cluster_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, backend="sklearn"
        )

        assert "kmeans_k3" in result.obs.columns
        assert result.history[-1].params["backend"] == "sklearn"

    def test_kmeans_resolves_assay_alias_and_uses_unified_history_keys(self, sample_obs):
        """Clustering should resolve aliases and record unified provenance keys."""
        X = np.random.default_rng(0).normal(size=(len(sample_obs), 4))
        var = pl.DataFrame({"_index": [f"F{i}" for i in range(4)]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=sample_obs, assays={"proteins": assay})

        result = cluster_kmeans(container, assay_name="protein", base_layer="X", n_clusters=3)

        assert "kmeans_k3" in result.obs.columns
        assert result.assays["proteins"] is container.assays["proteins"]

        params = result.history[-1].params
        assert params["source_assay"] == "proteins"
        assert params["source_layer"] == "X"
        assert params["output_key"] == "kmeans_k3"
        assert "assay" not in params
        assert "layer" not in params

    def test_kmeans_freezes_copy_contract(self, pca_container):
        """Clustering keeps assays shared but creates fresh obs/history containers."""
        result = cluster_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert result is not pca_container
        assert result.obs is not pca_container.obs
        assert result.history is not pca_container.history
        assert result.assays is pca_container.assays
        assert result.assays["reduce_pca"] is pca_container.assays["reduce_pca"]
        assert "kmeans_k3" not in pca_container.obs.columns
        assert "kmeans_k3" in result.obs.columns


# =============================================================================
# cluster_leiden tests
# =============================================================================


class TestClusterLeiden:
    """Tests for cluster_leiden function."""

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_basic(self, pca_container):
        """Test basic Leiden clustering."""
        result = cluster_leiden(pca_container, resolution=1.0)

        expected_col = "leiden_r1.0"
        assert expected_col in result.obs.columns

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_different_resolution(self, pca_container):
        """Test Leiden with different resolution parameters."""
        for res in [0.5, 1.0, 2.0]:
            result = cluster_leiden(pca_container, resolution=res)
            expected_col = f"leiden_r{res}"
            assert expected_col in result.obs.columns

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_custom_key_added(self, pca_container):
        """Test Leiden with custom key_added."""
        result = cluster_leiden(
            pca_container,
            resolution=1.0,
            key_added="my_leiden",
        )

        assert "my_leiden" in result.obs.columns

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_invalid_n_neighbors_zero(self, pca_container):
        """Test Leiden with n_neighbors=0 raises error."""
        with pytest.raises(ScpValueError):
            cluster_leiden(pca_container, n_neighbors=0)

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_invalid_resolution_zero(self, pca_container):
        """Test Leiden with resolution=0 raises error."""
        with pytest.raises(ScpValueError):
            cluster_leiden(pca_container, resolution=0)

    @pytest.mark.skipif(
        not LEIDEN_AVAILABLE, reason="Requires optional dependencies: leidenalg and python-igraph"
    )
    def test_leiden_uses_unified_history_keys(self, pca_container):
        """Leiden provenance should use the same source/output key schema."""
        result = cluster_leiden(pca_container, resolution=1.0)

        params = result.history[-1].params
        assert params["source_assay"] == "reduce_pca"
        assert params["source_layer"] == "X"
        assert params["output_key"] == "leiden_r1.0"
        assert "assay" not in params
        assert "layer" not in params


# =============================================================================
# Edge case tests
# =============================================================================


class TestClusteringEdgeCases:
    """Edge case tests for clustering functions."""

    def test_kmeans_single_sample(self):
        """Test K-means with single sample."""
        obs = pl.DataFrame({"_index": ["S1"]})
        X = np.array([[1.0, 2.0]])
        var = pl.DataFrame({"_index": ["PC1", "PC2"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        result = cluster_kmeans(container, assay_name="reduce_pca", n_clusters=1)
        assert "kmeans_k1" in result.obs.columns

    def test_kmeans_two_samples(self):
        """Test K-means with two samples."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        var = pl.DataFrame({"_index": ["PC1", "PC2"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        result = cluster_kmeans(container, assay_name="reduce_pca", n_clusters=2)
        assert "kmeans_k2" in result.obs.columns
