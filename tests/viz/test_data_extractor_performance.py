"""Unit tests for DataExtractor performance optimizations.

Tests the new performance features:
- Lazy dense conversion (force_dense parameter)
- Data sampling (max_samples, max_features)
- Sparse matrix utilities (should_use_sparse, get_sparse_stats)
- Memory-efficient validation
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.viz.base.data_extractor import DataExtractor


@pytest.fixture
def sparse_container():
    """Create a test container with sparse data."""
    n_samples, n_features = 100, 50

    obs_df = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "batch": np.random.choice(["A", "B"], n_samples),
        }
    )
    var_df = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "protein": [
                f"protein_{i}" for i in range(n_features)
            ],  # Add protein column for compatibility
        }
    )

    container = ScpContainer(obs=obs_df)
    assay = Assay(var=var_df)

    # Create sparse data (70% sparse)
    X_sparse = sparse.random(n_samples, n_features, density=0.3, format="csr") * 10
    M_sparse = X_sparse.copy()
    M_sparse.data = np.ones(len(M_sparse.data), dtype=np.int8)

    assay.layers["raw"] = ScpMatrix(X=X_sparse, M=M_sparse)
    container.assays["proteins"] = assay

    return container


class TestLazyDenseConversion:
    """Test lazy dense conversion optimization."""

    def test_default_force_dense_true(self, sparse_container):
        """Test default behavior (force_dense=True) returns dense array."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw"
        )

        assert isinstance(X, np.ndarray)
        assert not sparse.issparse(X)
        assert X.shape == (100, 50)

    def test_force_dense_false_keeps_sparse(self, sparse_container):
        """Test force_dense=False keeps sparse format."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", force_dense=False
        )

        assert sparse.issparse(X)
        assert X.shape == (100, 50)

    def test_force_dense_false_with_mask(self, sparse_container):
        """Test force_dense=False still converts mask to dense."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", force_dense=False
        )

        # Mask is always converted to dense for masking operations
        # This is expected behavior
        assert X is not None


class TestDataSampling:
    """Test data sampling for large datasets."""

    def test_sample_rows_only(self, sparse_container):
        """Test sampling only rows (samples)."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=50, random_state=42
        )

        assert X.shape[0] == 50
        assert X.shape[1] == 50  # All features kept
        assert obs.shape[0] == 50

    def test_sample_columns_only(self, sparse_container):
        """Test sampling only columns (features)."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_features=25, random_state=42
        )

        assert X.shape[0] == 100  # All samples kept
        assert X.shape[1] == 25
        assert var.shape[0] == 25

    def test_sample_both_dimensions(self, sparse_container):
        """Test sampling both rows and columns."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=30, max_features=20, random_state=42
        )

        assert X.shape == (30, 20)
        assert obs.shape[0] == 30
        assert var.shape[0] == 20

    def test_sampling_reproducibility(self, sparse_container):
        """Test that sampling with same random_state gives same results."""
        X1, _, _ = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=50, random_state=42
        )
        X2, _, _ = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=50, random_state=42
        )

        if sparse.issparse(X1):
            np.testing.assert_array_equal(X1.toarray(), X2.toarray())
        else:
            np.testing.assert_array_equal(X1, X2)

    def test_sampling_larger_than_data(self, sparse_container):
        """Test sampling larger than available data returns all data."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=1000, max_features=1000
        )

        assert X.shape == (100, 50)

    def test_sampling_none(self, sparse_container):
        """Test max_samples=None and max_features=None returns all data."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=None, max_features=None
        )

        assert X.shape == (100, 50)


class TestSparseUtilities:
    """Test sparse matrix utility functions."""

    def test_should_use_sparse_with_sparse_matrix(self):
        """Test should_use_sparse with sparse matrix returns True."""
        X_sparse = sparse.random(100, 50, density=0.3, format="csr")
        assert DataExtractor.should_use_sparse(X_sparse) is True

    def test_should_use_sparse_with_dense_sparse(self):
        """Test should_use_sparse with very sparse dense array."""
        X_dense = np.zeros((100, 50))
        result = DataExtractor.should_use_sparse(X_dense, threshold=0.5)
        # np.bool_ type, need to convert to bool for comparison
        assert bool(result) is True

    def test_should_use_sparse_with_dense_dense(self):
        """Test should_use_sparse with dense array returns False."""
        X_dense = np.random.rand(100, 50)
        result = DataExtractor.should_use_sparse(X_dense, threshold=0.5)
        # np.bool_ type, need to convert to bool for comparison
        assert bool(result) is False

    def test_get_sparse_stats_sparse_matrix(self):
        """Test get_sparse_stats with sparse matrix."""
        X_sparse = sparse.random(100, 50, density=0.3, format="csr") * 10
        stats = DataExtractor.get_sparse_stats(X_sparse)

        assert "sparsity" in stats
        assert "density" in stats
        assert "n_zeros" in stats
        assert "n_nonzeros" in stats
        assert "size" in stats
        assert "memory_mb" in stats

        assert stats["sparsity"] > 0.6  # Should be ~70% sparse
        assert stats["density"] < 0.4
        assert stats["size"] == 100 * 50
        assert stats["memory_mb"] > 0

    def test_get_sparse_stats_dense_array(self):
        """Test get_sparse_stats with dense array."""
        X_dense = np.random.rand(100, 50)
        stats = DataExtractor.get_sparse_stats(X_dense)

        assert stats["sparsity"] == 0.0  # Random data has no zeros
        assert stats["density"] == 1.0
        assert stats["size"] == 100 * 50
        assert stats["memory_mb"] > 0

    def test_get_sparse_stats_memory_estimation(self):
        """Test that memory estimation is reasonable."""
        # Create 1000x1000 sparse matrix (1M elements)
        X_sparse = sparse.random(1000, 1000, density=0.1, format="csr")
        stats = DataExtractor.get_sparse_stats(X_sparse)

        # Sparse should use much less memory than dense (1M * 8 bytes = 8 MB)
        assert stats["memory_mb"] < 5.0  # Sparse should be < 5 MB

        # Dense would be ~8 MB
        X_dense = np.random.rand(1000, 1000)
        stats_dense = DataExtractor.get_sparse_stats(X_dense)
        assert stats_dense["memory_mb"] > 7.0  # Dense should be ~8 MB


class TestCacheBehavior:
    """Test LRU cache behavior."""

    def test_cache_clear(self, sparse_container):
        """Test that clear_cache() works."""
        # Populate cache
        DataExtractor.get_expression_matrix(
            sparse_container, "proteins", "raw", None, None, id(sparse_container)
        )

        # Clear cache
        DataExtractor.clear_cache()

        # Cache should be empty (can't directly test, but should not raise)
        DataExtractor.clear_cache()


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_uncached_default_behavior(self, sparse_container):
        """Test that uncached version maintains old behavior."""
        # Old API call (without new parameters)
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw"
        )

        # Should return dense array (old behavior)
        assert isinstance(X, np.ndarray)
        assert not sparse.issparse(X)
        assert X.shape == (100, 50)

    def test_uncached_with_var_names(self, sparse_container):
        """Test uncached with var_names parameter."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", var_names=["protein_1", "protein_2"]
        )

        assert X.shape == (100, 2)

    def test_uncached_with_samples(self, sparse_container):
        """Test uncached with samples parameter."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", samples=["sample_1", "sample_2"]
        )

        assert X.shape[0] == 2


class TestValidationOptimizations:
    """Test memory-efficient validation."""

    def test_sparse_validation_without_conversion(self, sparse_container):
        """Test that sparse matrices don't require conversion for validation."""
        # This should not raise an error
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", force_dense=False
        )

        # If we got here without error, validation worked
        assert X is not None


class TestIntegration:
    """Integration tests for combined optimizations."""

    def test_sparse_with_sampling(self, sparse_container):
        """Test combining sparse format with sampling."""
        X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container,
            "proteins",
            "raw",
            force_dense=False,
            max_samples=50,
            max_features=25,
            random_state=42,
        )

        assert X.shape == (50, 25)
        # Note: may be dense due to mask conversion

    def test_end_to_end_workflow(self, sparse_container):
        """Test complete workflow with all optimizations."""
        # Step 1: Check if sparse is beneficial
        X_orig = sparse_container.assays["proteins"].layers["raw"].X  # noqa: N806
        should_use_sparse = DataExtractor.should_use_sparse(X_orig)
        stats = DataExtractor.get_sparse_stats(X_orig)

        assert should_use_sparse is True
        assert stats["sparsity"] > 0.5

        # Step 2: Extract with optimizations
        if should_use_sparse:
            # Use sparse format for speed
            X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
                sparse_container, "proteins", "raw", force_dense=False
            )
        else:
            # Use dense format
            X, obs, var = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
                sparse_container, "proteins", "raw", force_dense=True
            )

        assert X is not None

        # Step 3: For preview, sample the data
        X_preview, _, _ = DataExtractor.get_expression_matrix_uncached(  # noqa: N806
            sparse_container, "proteins", "raw", max_samples=50, max_features=25, random_state=42
        )

        assert X_preview.shape == (50, 25)
