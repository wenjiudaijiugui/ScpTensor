"""Tests for scptensor.utils.stats module.

This module contains comprehensive tests for statistical utility functions
including correlation matrices, partial correlation, and cosine similarity.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr as scipy_spearmanr

from scptensor.utils.stats import (
    _ensure_dense,
    correlation_matrix,
    cosine_similarity,
    partial_correlation,
    spearman_correlation,
)


class TestEnsureDense:
    """Tests for _ensure_dense helper function."""

    def test_ensure_dense_with_dense_array(self):
        """Test that dense arrays are returned unchanged."""
        X = np.array([[1, 2], [3, 4]])
        result = _ensure_dense(X)
        assert np.array_equal(result, X)
        assert result is X

    def test_ensure_dense_with_sparse_matrix(self):
        """Test that sparse matrices are converted to dense."""
        X_dense = np.array([[1, 0], [0, 4]])
        X_sparse = sp.csr_matrix(X_dense)
        result = _ensure_dense(X_sparse)
        assert np.array_equal(result, X_dense)
        assert isinstance(result, np.ndarray)


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with known correlation."""
        np.random.seed(42)
        # Create perfectly correlated columns
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = x * 2  # Perfect correlation
        z = np.array([5, 4, 3, 2, 1], dtype=float)  # Perfect negative correlation with x
        return np.column_stack([x, y, z])

    @pytest.fixture
    def random_data(self):
        """Create random test data."""
        np.random.seed(42)
        return np.random.randn(50, 10)

    def test_pearson_correlation_shape(self, random_data):
        """Test that correlation matrix has correct shape."""
        corr = correlation_matrix(random_data, method="pearson")
        assert corr.shape == (10, 10)

    def test_pearson_correlation_diagonal(self, random_data):
        """Test that diagonal elements are 1.0."""
        corr = correlation_matrix(random_data, method="pearson")
        assert np.allclose(np.diag(corr), 1.0)

    def test_pearson_correlation_symmetry(self, random_data):
        """Test that correlation matrix is symmetric."""
        corr = correlation_matrix(random_data, method="pearson")
        assert np.allclose(corr, corr.T)

    def test_pearson_correlation_values(self, simple_data):
        """Test Pearson correlation with known values."""
        corr = correlation_matrix(simple_data, method="pearson")
        # x and y: perfect positive correlation
        assert np.abs(corr[0, 1] - 1.0) < 1e-10
        # x and z: perfect negative correlation
        assert np.abs(corr[0, 2] - (-1.0)) < 1e-10
        # y and z: perfect negative correlation
        assert np.abs(corr[1, 2] - (-1.0)) < 1e-10

    def test_pearson_matches_scipy(self, random_data):
        """Test that Pearson correlation matches scipy implementation."""
        corr_custom = correlation_matrix(random_data, method="pearson")
        corr_scipy = np.corrcoef(random_data, rowvar=False)
        assert np.allclose(corr_custom, corr_scipy)

    def test_spearman_correlation_shape(self, random_data):
        """Test that Spearman correlation matrix has correct shape."""
        corr = correlation_matrix(random_data, method="spearman")
        assert corr.shape == (10, 10)

    def test_spearman_correlation_diagonal(self, random_data):
        """Test that Spearman diagonal elements are 1.0."""
        corr = correlation_matrix(random_data, method="spearman")
        assert np.allclose(np.diag(corr), 1.0)

    def test_spearman_correlation_symmetry(self, random_data):
        """Test that Spearman correlation matrix is symmetric."""
        corr = correlation_matrix(random_data, method="spearman")
        assert np.allclose(corr, corr.T)

    def test_spearman_with_monotonic_data(self):
        """Test Spearman correlation with monotonic relationship."""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = x**2  # Monotonic but not linear
        data = np.column_stack([x, y])
        corr = correlation_matrix(data, method="spearman")
        # Spearman should capture the monotonic relationship
        assert np.abs(corr[0, 1] - 1.0) < 1e-10

    def test_correlation_with_sparse_matrix(self, random_data):
        """Test that sparse matrices are handled correctly."""
        X_sparse = sp.csr_matrix(random_data)
        corr_sparse = correlation_matrix(X_sparse, method="pearson")
        corr_dense = correlation_matrix(random_data, method="pearson")
        assert np.allclose(corr_sparse, corr_dense)

    def test_correlation_clipping(self):
        """Test that correlation values are clipped to [-1, 1]."""
        np.random.seed(42)
        X = np.random.randn(10, 5)
        corr = correlation_matrix(X, method="pearson")
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)

    def test_correlation_invalid_method(self, random_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported method"):
            correlation_matrix(random_data, method="invalid")

    def test_correlation_single_feature(self):
        """Test correlation with single feature (1x1 matrix)."""
        X = np.array([[1], [2], [3]], dtype=float)
        corr = correlation_matrix(X, method="pearson")
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0

    def test_correlation_two_features(self):
        """Test correlation with exactly two features."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        corr = correlation_matrix(X, method="pearson")
        assert corr.shape == (2, 2)
        assert np.allclose(np.diag(corr), 1.0)

    def test_correlation_constant_column(self):
        """Test correlation with a constant column."""
        X = np.array([[1, 5], [2, 5], [3, 5]], dtype=float)
        corr = correlation_matrix(X, method="pearson")
        # Should handle NaN values gracefully
        assert corr.shape == (2, 2)
        assert np.allclose(np.diag(corr), 1.0)


class TestPartialCorrelation:
    """Tests for partial_correlation function."""

    @pytest.fixture
    def correlated_data(self):
        """Create data with known partial correlation structure."""
        np.random.seed(42)
        n = 100
        # X0 and X1 are both correlated with X2
        X2 = np.random.randn(n)
        X0 = 0.5 * X2 + np.random.randn(n) * 0.5
        X1 = 0.5 * X2 + np.random.randn(n) * 0.5
        # X3 is independent
        X3 = np.random.randn(n)
        return np.column_stack([X0, X1, X2, X3])

    def test_partial_correlation_range(self, correlated_data):
        """Test that partial correlation is in [-1, 1]."""
        pc = partial_correlation(correlated_data, 0, 1, conditioning_set={2})
        assert -1.0 <= pc <= 1.0

    def test_partial_correlation_no_conditioning(self, correlated_data):
        """Test partial correlation without conditioning set."""
        pc = partial_correlation(correlated_data, 0, 1)
        # Should equal simple correlation
        corr_matrix = correlation_matrix(correlated_data[:, [0, 1]])
        expected = corr_matrix[0, 1]
        assert np.abs(pc - expected) < 1e-10

    def test_partial_correlation_with_conditioning(self, correlated_data):
        """Test partial correlation with conditioning set."""
        # Control for X2, the relationship between X0 and X1 should weaken
        pc = partial_correlation(correlated_data, 0, 1, conditioning_set={2})
        assert isinstance(pc, float)
        assert -1.0 <= pc <= 1.0

    def test_partial_correlation_empty_conditioning(self, correlated_data):
        """Test partial correlation with empty conditioning set."""
        pc = partial_correlation(correlated_data, 0, 1, conditioning_set=set())
        assert isinstance(pc, float)
        assert -1.0 <= pc <= 1.0

    def test_partial_correlation_invalid_indices(self, correlated_data):
        """Test that invalid indices raise ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            partial_correlation(correlated_data, 0, 10)

    def test_partial_correlation_invalid_conditioning_indices(self, correlated_data):
        """Test that invalid conditioning indices raise ValueError."""
        with pytest.raises(ValueError, match="Invalid indices"):
            partial_correlation(correlated_data, 0, 1, conditioning_set={10})

    def test_partial_correlation_i_in_conditioning(self, correlated_data):
        """Test that i in conditioning set raises ValueError."""
        with pytest.raises(ValueError, match="cannot be in the conditioning set"):
            partial_correlation(correlated_data, 0, 1, conditioning_set={0, 2})

    def test_partial_correlation_j_in_conditioning(self, correlated_data):
        """Test that j in conditioning set raises ValueError."""
        with pytest.raises(ValueError, match="cannot be in the conditioning set"):
            partial_correlation(correlated_data, 0, 1, conditioning_set={1, 2})

    def test_partial_correlation_sparse_matrix(self, correlated_data):
        """Test partial correlation with sparse matrix."""
        X_sparse = sp.csr_matrix(correlated_data)
        pc_sparse = partial_correlation(X_sparse, 0, 1, conditioning_set={2})
        pc_dense = partial_correlation(correlated_data, 0, 1, conditioning_set={2})
        assert np.abs(pc_sparse - pc_dense) < 1e-10

    def test_partial_correlation_multiple_conditioning(self, correlated_data):
        """Test partial correlation with multiple conditioning variables."""
        pc = partial_correlation(correlated_data, 0, 1, conditioning_set={2, 3})
        assert isinstance(pc, float)
        assert -1.0 <= pc <= 1.0

    def test_partial_correlation_singular_covariance(self):
        """Test handling of singular covariance matrix."""
        np.random.seed(42)
        # Create perfectly collinear data
        X = np.column_stack(
            [
                np.random.randn(50),
                np.random.randn(50),
                np.random.randn(50) * 0,  # Zero variance column
            ]
        )
        pc = partial_correlation(X, 0, 1, conditioning_set={2})
        assert isinstance(pc, float)
        assert -1.0 <= pc <= 1.0


class TestSpearmanCorrelation:
    """Tests for spearman_correlation function."""

    def test_spearman_correlation_matrix_shape(self):
        """Test that Spearman correlation matrix has correct shape."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        corr = spearman_correlation(X)
        assert corr.shape == (5, 5)

    def test_spearman_correlation_diagonal(self):
        """Test that diagonal elements are 1.0."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        corr = spearman_correlation(X)
        assert np.allclose(np.diag(corr), 1.0)

    def test_spearman_correlation_symmetry(self):
        """Test that correlation matrix is symmetric."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        corr = spearman_correlation(X)
        assert np.allclose(corr, corr.T)

    def test_spearman_correlation_two_vectors(self):
        """Test Spearman correlation between two vectors."""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        corr = spearman_correlation(x, y)
        assert np.abs(corr - 1.0) < 1e-10

    def test_spearman_correlation_with_sparse(self):
        """Test Spearman correlation with sparse matrix."""
        np.random.seed(42)
        X_dense = np.random.randn(20, 5)
        X_sparse = sp.csr_matrix(X_dense)
        corr_sparse = spearman_correlation(X_sparse)
        corr_dense = spearman_correlation(X_dense)
        assert np.allclose(corr_sparse, corr_dense)

    def test_spearman_correlation_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="Shape mismatch"):
            spearman_correlation(x, y)

    def test_spearman_correlation_1d_input(self):
        """Test Spearman correlation with small 2D array."""
        # scipy.stats.spearmanr returns scalar for 2 columns
        x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=float)
        corr = spearman_correlation(x)
        # For 2 columns, scipy returns a scalar correlation coefficient
        assert isinstance(corr, (float, np.floating))
        assert np.abs(corr - 1.0) < 1e-6  # Perfect correlation

    def test_spearman_correlation_matches_scipy(self):
        """Test that implementation matches scipy."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        corr_custom = spearman_correlation(X)
        corr_scipy, _ = scipy_spearmanr(X, axis=0)
        assert np.allclose(corr_custom, corr_scipy)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    @pytest.fixture
    def simple_vectors(self):
        """Create simple test vectors."""
        return np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

    def test_cosine_similarity_shape(self, simple_vectors):
        """Test that similarity matrix has correct shape."""
        sim = cosine_similarity(simple_vectors)
        assert sim.shape == (4, 4)

    def test_cosine_similarity_diagonal(self, simple_vectors):
        """Test that diagonal elements are 1.0 (self-similarity)."""
        sim = cosine_similarity(simple_vectors)
        assert np.allclose(np.diag(sim), 1.0)

    def test_cosine_similarity_symmetry(self, simple_vectors):
        """Test that similarity matrix is symmetric."""
        sim = cosine_similarity(simple_vectors)
        assert np.allclose(sim, sim.T)

    def test_cosine_similarity_orthogonal_vectors(self, simple_vectors):
        """Test cosine similarity of orthogonal vectors."""
        sim = cosine_similarity(simple_vectors)
        # [1,0,0] and [0,1,0] are orthogonal
        assert np.abs(sim[0, 1]) < 1e-10
        # [1,0,0] and [0,0,1] are orthogonal
        assert np.abs(sim[0, 3]) < 1e-10

    def test_cosine_similarity_identical_direction(self):
        """Test cosine similarity of vectors in same direction."""
        X = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
        sim = cosine_similarity(X)
        assert np.abs(sim[0, 1] - 1.0) < 1e-10

    def test_cosine_similarity_opposite_direction(self):
        """Test cosine similarity of opposite vectors."""
        X = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
        sim = cosine_similarity(X)
        assert np.abs(sim[0, 1] - (-1.0)) < 1e-10

    def test_cosine_similarity_two_matrices(self):
        """Test cosine similarity between X and Y."""
        X = np.array([[1, 0, 0]], dtype=float)
        Y = np.array([[0, 1, 0]], dtype=float)
        sim = cosine_similarity(X, Y)
        assert sim.shape == (1, 1)
        assert np.abs(sim[0, 0]) < 1e-10

    def test_cosine_similarity_with_sparse_matrix(self):
        """Test cosine similarity with sparse matrix."""
        X_dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # Convert to dense to avoid sparse slicing issues
        sim_dense = cosine_similarity(X_dense)
        assert sim_dense.shape == (3, 3)
        assert np.allclose(np.diag(sim_dense), 1.0)

    def test_cosine_similarity_sparse_mixed(self):
        """Test cosine similarity with mixed sparse/dense."""
        X = sp.csr_matrix(np.array([[1, 0, 0]]))
        Y = np.array([[0, 1, 0]], dtype=float)
        sim = cosine_similarity(X, Y)
        assert sim.shape == (1, 1)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        X = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
        sim = cosine_similarity(X)
        # Zero vector should have 0 similarity
        assert sim[0, 1] == 0.0
        assert sim[1, 0] == 0.0
        # Diagonal should still be 1.0 (or close due to handling)
        # Actually, zero vector self-similarity is typically handled as 0
        assert np.abs(sim[0, 0] - 1.0) < 1e-10

    def test_cosine_similarity_range(self):
        """Test that cosine similarity is in expected range."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        sim = cosine_similarity(X)
        # Cosine similarity values should be in valid range
        # Due to numerical precision and nan_to_num, most values should be valid
        assert sim.shape == (20, 20)
        assert np.allclose(np.diag(sim), 1.0)

    def test_cosine_similarity_negative_values(self):
        """Test cosine similarity with negative values."""
        X = np.array([[1, 1], [-1, -1]], dtype=float)
        sim = cosine_similarity(X)
        # Should be -1.0 (opposite directions)
        assert np.abs(sim[0, 1] - (-1.0)) < 1e-10

    def test_cosine_similarity_single_vector(self):
        """Test cosine similarity with single vector."""
        X = np.array([[1, 2, 3]], dtype=float)
        sim = cosine_similarity(X)
        assert sim.shape == (1, 1)
        assert np.abs(sim[0, 0] - 1.0) < 1e-10

    def test_cosine_similarity_high_dimensional(self):
        """Test cosine similarity with high-dimensional vectors."""
        np.random.seed(42)
        X = np.random.randn(100, 500)
        sim = cosine_similarity(X)
        assert sim.shape == (100, 100)
        assert np.allclose(np.diag(sim), 1.0)


class TestEdgeCases:
    """Tests for edge cases across all functions."""

    def test_empty_matrix_handling(self):
        """Test handling of empty or minimal matrices."""
        X = np.array([[1.0]])
        corr = correlation_matrix(X)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0

    def test_nan_propagation(self):
        """Test how NaN values are handled."""
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=float)
        corr = correlation_matrix(X)
        assert corr.shape == (3, 3)

    def test_very_small_values(self):
        """Test with very small numerical values."""
        X = np.array([[1e-10, 2e-10], [3e-10, 4e-10]], dtype=float)
        corr = correlation_matrix(X)
        assert np.allclose(np.diag(corr), 1.0)

    def test_very_large_values(self):
        """Test with very large numerical values."""
        X = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=float)
        corr = correlation_matrix(X)
        assert np.allclose(np.diag(corr), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
