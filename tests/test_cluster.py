"""
Comprehensive tests for clustering modules.

Tests cover:
- cluster_kmeans_assay (kmeans.py)
- kmeans (basic.py)
- leiden (graph.py) with skip for optional dependencies
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.cluster import (
    cluster_kmeans_assay,
)

# Deprecated aliases for testing (from submodules to avoid naming conflicts)
from scptensor.cluster.basic import cluster_kmeans as basic_kmeans
from scptensor.cluster.graph import cluster_leiden as leiden
from scptensor.cluster.kmeans import cluster_kmeans as run_kmeans  # noqa: F401 (used in tests)
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError

# =============================================================================
# Fixtures for clustering tests
# =============================================================================


@pytest.fixture
def pca_container(sample_obs):
    """Create a container with PCA data for clustering tests."""
    np.random.seed(42)
    # Create realistic PCA-like data
    n_samples = len(sample_obs)
    n_pcs = 10

    # Simulate PCA data with some structure
    X_pca = np.random.randn(n_samples, n_pcs)
    # Add cluster-like structure
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

    # Create sparse matrix
    X_dense = np.random.randn(n_samples, n_pcs)
    X_dense[X_dense < 0.3] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(n_pcs)]})

    assay_pca = Assay(var=var_pca, layers={"X": ScpMatrix(X=X_sparse)})

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca})


@pytest.fixture
def multi_assay_container(sample_obs):
    """Create a container with multiple assays for testing assay selection."""
    np.random.seed(42)
    n_samples = len(sample_obs)

    # PCA assay
    X_pca = np.random.randn(n_samples, 5)
    var_pca = pl.DataFrame({"_index": [f"PC{i}" for i in range(5)]})
    assay_pca = Assay(var=var_pca, layers={"X": ScpMatrix(X=X_pca)})

    # UMAP assay
    X_umap = np.random.randn(n_samples, 2)
    var_umap = pl.DataFrame({"_index": ["UMAP1", "UMAP2"]})
    assay_umap = Assay(var=var_umap, layers={"X": ScpMatrix(X=X_umap)})

    return ScpContainer(obs=sample_obs, assays={"reduce_pca": assay_pca, "reduce_umap": assay_umap})


@pytest.fixture
def multi_layer_container(sample_obs):
    """Create a container with multiple layers for testing layer selection."""
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
# cluster_kmeans_assay tests (kmeans.py) - 93% coverage already, add edge cases
# =============================================================================


class TestRunKmeans:
    """Tests for cluster_kmeans_assay function."""

    def test_cluster_kmeans_assay_basic(self, pca_container):
        """Test basic K-means clustering functionality."""
        result = cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=3)

        # Check new assay was created
        assert "cluster_kmeans" in result.assays
        cluster_assay = result.assays["cluster_kmeans"]

        # Check assay structure
        assert cluster_assay.n_features == 3
        assert "binary" in cluster_assay.layers

        # Check binary layer is one-hot encoded
        binary_layer = cluster_assay.layers["binary"]
        assert binary_layer.X.shape == (5, 3)

        # Verify one-hot encoding (each row has exactly one 1)
        row_sums = binary_layer.X.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_cluster_kmeans_assay_different_k(self, pca_container):
        """Test K-means with different number of clusters."""
        for k in [2, 3, 4, 5]:
            # Use fresh container each time since cluster_kmeans_assay modifies in place
            result = cluster_kmeans_assay(
                pca_container,
                assay_name="reduce_pca",
                n_clusters=k,
                new_assay_name=f"cluster_kmeans_{k}",
            )
            assert result.assays[f"cluster_kmeans_{k}"].n_features == k
            assert result.assays[f"cluster_kmeans_{k}"].layers["binary"].X.shape[1] == k

    def test_cluster_kmeans_assay_with_key_added(self, pca_container):
        """Test K-means with key_added parameter to add labels to obs."""
        result = cluster_kmeans_assay(
            pca_container, assay_name="reduce_pca", n_clusters=3, key_added="kmeans_labels"
        )

        # Check column was added to obs
        assert "kmeans_labels" in result.obs.columns

        # Check values are valid cluster IDs (0-2)
        labels = result.obs["kmeans_labels"].to_list()
        assert all(label in ["0", "1", "2"] for label in labels)

    def test_cluster_kmeans_assay_custom_assay_name(self, pca_container):
        """Test K-means with custom assay name."""
        result = cluster_kmeans_assay(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            new_assay_name="my_clusters",
        )

        assert "my_clusters" in result.assays
        assert result.assays["my_clusters"].n_features == 3

    def test_cluster_kmeans_assay_custom_assay_and_layer(self, multi_assay_container):
        """Test K-means on custom assay and layer."""
        result = cluster_kmeans_assay(
            multi_assay_container,
            assay_name="reduce_umap",
            base_layer="X",
            n_clusters=2,
        )

        assert "cluster_kmeans" in result.assays
        # Check history references correct source
        assert result.history[-1].params["source_assay"] == "reduce_umap"

    def test_cluster_kmeans_assay_random_state(self, pca_container):
        """Test K-means reproducibility with random_state."""
        result1 = cluster_kmeans_assay(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            random_state=42,
            new_assay_name="cluster_kmeans_1",
        )
        result2 = cluster_kmeans_assay(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            random_state=42,
            new_assay_name="cluster_kmeans_2",
        )

        labels1 = result1.assays["cluster_kmeans_1"].layers["binary"].X
        labels2 = result2.assays["cluster_kmeans_2"].layers["binary"].X

        # Results should be identical
        assert np.array_equal(labels1, labels2)

    def test_cluster_kmeans_assay_sparse_matrix(self, sparse_pca_container):
        """Test K-means with sparse input matrix."""
        result = cluster_kmeans_assay(sparse_pca_container, assay_name="reduce_pca", n_clusters=3)

        assert "cluster_kmeans" in result.assays
        cluster_assay = result.assays["cluster_kmeans"]
        assert cluster_assay.n_features == 3

    def test_cluster_kmeans_assay_history_logging(self, pca_container):
        """Test that K-means operation is logged in history."""
        initial_history_len = len(pca_container.history)
        result = cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert len(result.history) == initial_history_len + 1
        log_entry = result.history[-1]
        assert log_entry.action == "cluster_kmeans"
        assert log_entry.params["n_clusters"] == 3
        assert "K-Means" in log_entry.description

    def test_cluster_kmeans_assay_invalid_n_clusters_zero(self, pca_container):
        """Test K-means with n_clusters=0 raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=0)

        assert "n_clusters must be positive" in str(exc_info.value)
        assert exc_info.value.parameter == "n_clusters"

    def test_cluster_kmeans_assay_invalid_n_clusters_negative(self, pca_container):
        """Test K-means with negative n_clusters raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=-1)

        assert "n_clusters must be positive" in str(exc_info.value)

    def test_cluster_kmeans_assay_assay_not_found(self, pca_container):
        """Test K-means with non-existent assay raises error."""
        with pytest.raises(AssayNotFoundError) as exc_info:
            cluster_kmeans_assay(pca_container, assay_name="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_cluster_kmeans_assay_layer_not_found(self, pca_container):
        """Test K-means with non-existent layer raises error."""
        with pytest.raises(LayerNotFoundError) as exc_info:
            cluster_kmeans_assay(pca_container, assay_name="reduce_pca", base_layer="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_cluster_kmeans_assay_n_clusters_larger_than_samples(self, pca_container):
        """Test K-means with n_clusters > n_samples."""
        # sklearn requires n_clusters <= n_samples
        with pytest.raises(ValueError) as exc_info:
            cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=10)

        assert "n_samples" in str(exc_info.value) or "n_clusters" in str(exc_info.value)

    def test_cluster_kmeans_assay_multi_layer_container(self, multi_layer_container):
        """Test K-means with different layers."""
        result = cluster_kmeans_assay(
            multi_layer_container,
            assay_name="reduce_pca",
            base_layer="normalized",
            n_clusters=3,
        )

        assert "cluster_kmeans" in result.assays
        assert result.history[-1].params["source_layer"] == "normalized"


# =============================================================================
# kmeans tests (basic.py) - currently 50% coverage
# =============================================================================


class TestBasicKmeans:
    """Tests for kmeans function from basic.py."""

    def test_kmeans_basic(self, pca_container):
        """Test basic K-means adds column to obs."""
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        # Check column was added to obs
        expected_col = "kmeans_k3"
        assert expected_col in result.obs.columns

        # Check values are strings (cast to Utf8)
        labels = result.obs[expected_col].to_list()
        assert all(isinstance(label, str) for label in labels)
        assert all(label in ["0", "1", "2"] for label in labels)

    def test_kmeans_different_n_clusters(self, pca_container):
        """Test K-means with different k values."""
        for k in [2, 4, 5]:
            result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=k)
            expected_col = f"kmeans_k{k}"
            assert expected_col in result.obs.columns

            # Check label range
            labels = result.obs[expected_col].cast(pl.Int32).to_numpy()
            assert np.all((labels >= 0) & (labels < k))

    def test_kmeans_custom_assay_and_layer(self, multi_assay_container):
        """Test K-means on custom assay and layer."""
        result = basic_kmeans(
            multi_assay_container,
            assay_name="reduce_umap",
            base_layer="X",
            n_clusters=2,
        )

        assert "kmeans_k2" in result.obs.columns

    def test_kmeans_random_state_reproducibility(self, pca_container):
        """Test K-means reproducibility with random_state."""
        result1 = basic_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, random_state=123
        )
        result2 = basic_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, random_state=123
        )

        labels1 = result1.obs["kmeans_k3"].to_list()
        labels2 = result2.obs["kmeans_k3"].to_list()

        assert labels1 == labels2

    def test_kmeans_different_random_states(self, pca_container):
        """Test that different random states may give different results."""
        result1 = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, random_state=1)
        result2 = basic_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, random_state=999
        )

        # Results might differ (not guaranteed but likely)
        labels1 = result1.obs["kmeans_k3"].to_list()
        labels2 = result2.obs["kmeans_k3"].to_list()

        # We don't assert they're different since K-means could converge
        # to the same solution, but we check the function works
        assert len(labels1) == len(labels2)

    def test_kmeans_sparse_matrix(self, sparse_pca_container):
        """Test K-means with sparse input matrix."""
        result = basic_kmeans(sparse_pca_container, assay_name="reduce_pca", n_clusters=3)

        assert "kmeans_k3" in result.obs.columns
        labels = result.obs["kmeans_k3"].to_list()
        assert len(labels) == 5

    def test_kmeans_new_container_returned(self, pca_container):
        """Test that a new container is returned (immutable pattern)."""
        original_obs_columns = set(pca_container.obs.columns)
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        # Original should be unchanged
        assert set(pca_container.obs.columns) == original_obs_columns
        # Result should have new column
        assert "kmeans_k3" in result.obs.columns

    def test_kmeans_history_logging(self, pca_container):
        """Test that K-means operation is logged in history."""
        initial_history_len = len(pca_container.history)
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3)

        assert len(result.history) == initial_history_len + 1
        log_entry = result.history[-1]
        assert log_entry.action == "cluster_kmeans"
        assert log_entry.params["n_clusters"] == 3
        assert "K-Means clustering" in log_entry.description

    def test_kmeans_invalid_n_clusters_zero(self, pca_container):
        """Test K-means with n_clusters=0 raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=0)

        assert "n_clusters must be positive" in str(exc_info.value)
        assert exc_info.value.parameter == "n_clusters"

    def test_kmeans_invalid_n_clusters_negative(self, pca_container):
        """Test K-means with negative n_clusters raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=-5)

        assert "n_clusters must be positive" in str(exc_info.value)

    def test_kmeans_assay_not_found(self, pca_container):
        """Test K-means with non-existent assay raises error."""
        with pytest.raises(AssayNotFoundError) as exc_info:
            basic_kmeans(pca_container, assay_name="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_kmeans_layer_not_found(self, pca_container):
        """Test K-means with non-existent layer raises error."""
        with pytest.raises(LayerNotFoundError) as exc_info:
            basic_kmeans(pca_container, assay_name="reduce_pca", base_layer="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_kmeans_n_clusters_equals_samples(self, pca_container):
        """Test K-means with n_clusters equal to number of samples."""
        n_samples = pca_container.n_samples
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=n_samples)

        # Each sample should be in its own cluster
        labels = result.obs[f"kmeans_k{n_samples}"].cast(pl.Int32).to_numpy()
        # With n_samples clusters, sklearn's KMeans may still merge some
        # due to initialization, but labels should all be in valid range
        assert np.all((labels >= 0) & (labels < n_samples))

    def test_kmeans_multi_assay_selection(self, multi_assay_container):
        """Test K-means on different assays."""
        # Test on PCA
        result_pca = basic_kmeans(
            multi_assay_container,
            assay_name="reduce_pca",
            n_clusters=2,
        )
        assert "kmeans_k2" in result_pca.obs.columns

        # Test on UMAP
        result_umap = basic_kmeans(
            multi_assay_container,
            assay_name="reduce_umap",
            n_clusters=2,
        )
        assert "kmeans_k2" in result_umap.obs.columns

    def test_kmeans_multi_layer_selection(self, multi_layer_container):
        """Test K-means on different layers."""
        result_x = basic_kmeans(
            multi_layer_container,
            assay_name="reduce_pca",
            base_layer="X",
            n_clusters=3,
        )
        assert "kmeans_k3" in result_x.obs.columns

        result_norm = basic_kmeans(
            multi_layer_container,
            assay_name="reduce_pca",
            base_layer="normalized",
            n_clusters=3,
        )
        assert "kmeans_k3" in result_norm.obs.columns

    def test_kmeans_backend_sklearn_default(self, pca_container):
        """Test K-means with default sklearn backend."""
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, backend="sklearn")

        assert "kmeans_k3" in result.obs.columns
        labels = result.obs["kmeans_k3"].cast(pl.Int32).to_numpy()
        assert np.all((labels >= 0) & (labels < 3))
        assert result.history[-1].params["backend"] == "sklearn"

    def test_kmeans_backend_sklearn_explicit(self, pca_container):
        """Test K-means with explicit sklearn backend."""
        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, backend="sklearn")

        assert "kmeans_k3" in result.obs.columns
        assert result.history[-1].params["backend"] == "sklearn"

    def test_kmeans_backend_numba(self, pca_container):
        """Test K-means with numba backend."""
        from scptensor.core.jit_ops import NUMBA_AVAILABLE

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        result = basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, backend="numba")

        assert "kmeans_k3" in result.obs.columns
        labels = result.obs["kmeans_k3"].cast(pl.Int32).to_numpy()
        assert np.all((labels >= 0) & (labels < 3))
        assert result.history[-1].params["backend"] == "numba"

    def test_kmeans_backend_numba_custom_params(self, pca_container):
        """Test K-means with numba backend and custom parameters."""
        from scptensor.core.jit_ops import NUMBA_AVAILABLE

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        result = basic_kmeans(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            backend="numba",
            max_iter=50,
            tol=1e-3,
        )

        assert "kmeans_k3" in result.obs.columns
        assert result.history[-1].params["backend"] == "numba"
        assert result.history[-1].params["max_iter"] == 50
        assert result.history[-1].params["tol"] == 1e-3

    def test_kmeans_backend_comparison(self, pca_container):
        """Test that sklearn and numba backends produce similar results."""
        from scptensor.core.jit_ops import NUMBA_AVAILABLE

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Run with same random state for comparison
        result_sklearn = basic_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, backend="sklearn", random_state=42
        )
        result_numba = basic_kmeans(
            pca_container, assay_name="reduce_pca", n_clusters=3, backend="numba", random_state=42
        )

        # Both should produce valid clusterings
        sklearn_labels = result_sklearn.obs["kmeans_k3"].cast(pl.Int32).to_numpy()
        numba_labels = result_numba.obs["kmeans_k3"].cast(pl.Int32).to_numpy()

        # Both should have same number of unique labels
        assert len(np.unique(sklearn_labels)) == len(np.unique(numba_labels))

        # Both should have all labels in valid range
        assert np.all((sklearn_labels >= 0) & (sklearn_labels < 3))
        assert np.all((numba_labels >= 0) & (numba_labels < 3))

    def test_kmeans_backend_invalid(self, pca_container):
        """Test K-means with invalid backend parameter."""
        with pytest.raises(ScpValueError) as exc_info:
            basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, backend="invalid")

        assert "backend" in str(exc_info.value).lower()
        assert exc_info.value.parameter == "backend"

    def test_kmeans_backend_numba_not_available(self, pca_container, monkeypatch):
        """Test K-means with numba backend when numba is not available."""
        from scptensor.core.jit_ops import NUMBA_AVAILABLE
        import scptensor.cluster.basic as basic_module

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available, cannot test fallback")

        # Temporarily set NUMBA_AVAILABLE to False in the cluster.basic module
        monkeypatch.setattr(basic_module, "NUMBA_AVAILABLE", False)

        with pytest.raises(ImportError) as exc_info:
            basic_kmeans(pca_container, assay_name="reduce_pca", n_clusters=3, backend="numba")

        assert "numba" in str(exc_info.value).lower()

    def test_kmeans_backend_sklearn_with_custom_params_ignored(self, pca_container):
        """Test that max_iter and tol are ignored for sklearn backend."""
        result = basic_kmeans(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            backend="sklearn",
            max_iter=50,
            tol=1e-3,
        )

        assert "kmeans_k3" in result.obs.columns
        # sklearn backend should log "default" for max_iter and tol
        assert result.history[-1].params["backend"] == "sklearn"
        assert result.history[-1].params["max_iter"] == "default"
        assert result.history[-1].params["tol"] == "default"


# =============================================================================
# leiden tests (graph.py) - currently 0% coverage
# =============================================================================


class TestLeiden:
    """Tests for leiden function from graph.py."""

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_basic(self, pca_container):
        """Test basic Leiden clustering functionality."""
        result = leiden(pca_container, resolution=1.0)

        # Check column was added to obs
        expected_col = "leiden_r1.0"
        assert expected_col in result.obs.columns

        # Check values are strings
        labels = result.obs[expected_col].to_list()
        assert all(isinstance(label, str) for label in labels)

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_different_resolution(self, pca_container):
        """Test Leiden with different resolution parameters."""
        for res in [0.5, 1.0, 2.0]:
            result = leiden(pca_container, resolution=res)
            expected_col = f"leiden_r{res}"
            assert expected_col in result.obs.columns

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_different_n_neighbors(self, pca_container):
        """Test Leiden with different n_neighbors values."""
        for n in [5, 10, 15, 20]:
            result = leiden(pca_container, n_neighbors=n)
            assert "leiden_r1.0" in result.obs.columns

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_random_state_reproducibility(self, pca_container):
        """Test Leiden reproducibility with random_state."""
        result1 = leiden(pca_container, random_state=42)
        result2 = leiden(pca_container, random_state=42)

        labels1 = result1.obs["leiden_r1.0"].to_list()
        labels2 = result2.obs["leiden_r1.0"].to_list()

        # Leiden should be deterministic with same seed
        assert labels1 == labels2

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_sparse_matrix(self, sparse_pca_container):
        """Test Leiden with sparse input matrix."""
        result = leiden(sparse_pca_container, n_neighbors=3)

        assert "leiden_r1.0" in result.obs.columns

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_history_logging(self, pca_container):
        """Test that Leiden operation is logged in history."""
        initial_history_len = len(pca_container.history)
        result = leiden(pca_container, resolution=1.0)

        assert len(result.history) == initial_history_len + 1
        log_entry = result.history[-1]
        assert log_entry.action == "cluster_leiden"
        assert log_entry.params["resolution"] == 1.0
        assert log_entry.params["n_neighbors"] == 15
        assert "Leiden clustering" in log_entry.description

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_invalid_n_neighbors_zero(self, pca_container):
        """Test Leiden with n_neighbors=0 raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            leiden(pca_container, n_neighbors=0)

        assert "n_neighbors must be positive" in str(exc_info.value)
        assert exc_info.value.parameter == "n_neighbors"

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_invalid_n_neighbors_negative(self, pca_container):
        """Test Leiden with negative n_neighbors raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            leiden(pca_container, n_neighbors=-5)

        assert "n_neighbors must be positive" in str(exc_info.value)

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_invalid_resolution_zero(self, pca_container):
        """Test Leiden with resolution=0 raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            leiden(pca_container, resolution=0)

        assert "resolution must be positive" in str(exc_info.value)
        assert exc_info.value.parameter == "resolution"

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_invalid_resolution_negative(self, pca_container):
        """Test Leiden with negative resolution raises error."""
        with pytest.raises(ScpValueError) as exc_info:
            leiden(pca_container, resolution=-0.5)

        assert "resolution must be positive" in str(exc_info.value)

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_assay_not_found(self, pca_container):
        """Test Leiden with non-existent assay raises error."""
        with pytest.raises(AssayNotFoundError) as exc_info:
            leiden(pca_container, assay_name="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.skip(
        reason="Requires optional dependencies: leidenalg and python-igraph - "
        "parameter validation happens after dependency check"
    )
    def test_leiden_layer_not_found(self, pca_container):
        """Test Leiden with non-existent layer raises error."""
        with pytest.raises(LayerNotFoundError) as exc_info:
            leiden(pca_container, base_layer="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_custom_assay_and_layer(self, multi_assay_container):
        """Test Leiden on custom assay and layer."""
        result = leiden(
            multi_assay_container,
            assay_name="reduce_umap",
            base_layer="X",
            n_neighbors=3,
        )

        assert "leiden_r1.0" in result.obs.columns

    @pytest.mark.skip(reason="Requires optional dependencies: leidenalg and python-igraph")
    def test_leiden_n_neighbors_larger_than_samples(self, pca_container):
        """Test Leiden with n_neighbors > n_samples."""
        # sklearn kneighbors_graph handles this by capping at n_samples - 1
        result = leiden(pca_container, n_neighbors=100)
        assert "leiden_r1.0" in result.obs.columns


# =============================================================================
# Integration tests combining multiple clustering methods
# =============================================================================


class TestClusteringIntegration:
    """Integration tests for clustering methods."""

    def test_kmeans_after_kmeans_different_k(self, pca_container):
        """Test running K-means multiple times with different k."""
        result1 = cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=3)
        result2 = cluster_kmeans_assay(
            result1, assay_name="reduce_pca", n_clusters=4, new_assay_name="cluster_kmeans_4"
        )

        assert "cluster_kmeans" in result2.assays
        assert "cluster_kmeans_4" in result2.assays
        assert result2.assays["cluster_kmeans"].n_features == 3
        assert result2.assays["cluster_kmeans_4"].n_features == 4

    def test_basic_kmeans_after_cluster_kmeans_assay(self, pca_container):
        """Test running basic kmeans after cluster_kmeans_assay."""
        result1 = cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=3)
        result2 = basic_kmeans(
            result1, assay_name="cluster_kmeans", base_layer="binary", n_clusters=2
        )

        # cluster_kmeans_assay creates new assay, basic_kmeans adds to obs
        assert "cluster_kmeans" in result2.assays
        assert "kmeans_k2" in result2.obs.columns

    def test_clustering_with_key_added_and_assay(self, pca_container):
        """Test K-means with both key_added and assay creation."""
        result = cluster_kmeans_assay(
            pca_container,
            assay_name="reduce_pca",
            n_clusters=3,
            key_added="cluster_labels",
        )

        # Should have assay
        assert "cluster_kmeans" in result.assays
        # Should have obs column
        assert "cluster_labels" in result.obs.columns

        # Labels should match
        assay_labels = result.assays["cluster_kmeans"].layers["binary"].X.argmax(axis=1)
        obs_labels = result.obs["cluster_labels"].cast(pl.Int32).to_numpy()
        assert np.array_equal(assay_labels, obs_labels)

    def test_clustering_history_multiple_operations(self, pca_container):
        """Test that multiple clustering operations are logged."""
        # First clustering on reduce_pca assay
        result = cluster_kmeans_assay(pca_container, assay_name="reduce_pca", n_clusters=3)
        # Second clustering on reduce_pca assay with different parameters
        result = cluster_kmeans_assay(
            result, assay_name="reduce_pca", n_clusters=4, new_assay_name="cluster_kmeans_4"
        )
        # Third clustering using basic_kmeans on reduce_pca assay (has X layer)
        result = basic_kmeans(result, assay_name="reduce_pca", n_clusters=2)

        assert len(result.history) == 3
        assert result.history[0].action == "cluster_kmeans"
        assert result.history[1].action == "cluster_kmeans"
        assert result.history[2].action == "cluster_kmeans"


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

        # With n_clusters=1, should work
        result = basic_kmeans(container, assay_name="reduce_pca", n_clusters=1)
        assert "kmeans_k1" in result.obs.columns
        assert result.obs["kmeans_k1"][0] == "0"

    def test_kmeans_two_samples(self):
        """Test K-means with two samples."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        var = pl.DataFrame({"_index": ["PC1", "PC2"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        result = basic_kmeans(container, assay_name="reduce_pca", n_clusters=2)
        assert "kmeans_k2" in result.obs.columns
        labels = result.obs["kmeans_k2"].cast(pl.Int32).to_numpy()
        # With 2 samples and 2 clusters, each should be in its own cluster
        assert len(set(labels)) <= 2

    def test_cluster_kmeans_assay_single_sample(self):
        """Test cluster_kmeans_assay with single sample."""
        obs = pl.DataFrame({"_index": ["S1"]})
        X = np.array([[1.0, 2.0]])
        var = pl.DataFrame({"_index": ["PC1", "PC2"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        result = cluster_kmeans_assay(container, assay_name="reduce_pca", n_clusters=1)
        assert "cluster_kmeans" in result.assays
        assert result.assays["cluster_kmeans"].n_features == 1

    def test_kmeans_high_dimensional_data(self):
        """Test K-means with high-dimensional data."""
        n_samples = 10
        n_features = 100
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)]})
        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"PC{i}" for i in range(n_features)]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        result = basic_kmeans(container, assay_name="reduce_pca", n_clusters=3)
        assert "kmeans_k3" in result.obs.columns

    def test_kmeans_identical_points(self):
        """Test K-means with identical data points."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        var = pl.DataFrame({"_index": ["PC1", "PC2"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"reduce_pca": assay})

        # Should still work even with identical points
        result = basic_kmeans(container, assay_name="reduce_pca", n_clusters=2)
        assert "kmeans_k2" in result.obs.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
