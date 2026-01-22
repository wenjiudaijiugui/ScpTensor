"""Comprehensive tests for quantile normalization module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.normalization.quantile_normalization import norm_quantile

# =============================================================================
# Helper Functions
# =============================================================================


def create_quantile_test_container(
    n_samples: int = 10,
    n_features: int = 20,
    sparse: bool = False,
    with_mask: bool = True,
    with_nan: bool = False,
    seed: int = 42,
) -> ScpContainer:
    """Create a test container for quantile normalization tests.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        sparse: Whether to use sparse matrices
        with_mask: Whether to include mask matrices
        with_nan: Whether to include NaN values for testing
        seed: Random seed for reproducibility

    Returns:
        A test ScpContainer
    """
    rng = np.random.default_rng(seed)

    # Create sample metadata
    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "batch": rng.choice(["A", "B"], n_samples),
            "condition": rng.choice(["control", "treatment"], n_samples),
        }
    )

    # Create feature metadata
    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "mean_intensity": rng.uniform(10, 30, n_features),
        }
    )

    # Create data matrix with different distributions per sample
    # This helps test normalization effectiveness
    sample_scales = rng.uniform(0.5, 2.0, n_samples)
    X = rng.uniform(10, 30, (n_samples, n_features))
    for i in range(n_samples):
        X[i, :] *= sample_scales[i]

    if with_nan:
        # Add some NaN values
        nan_mask = rng.random((n_samples, n_features)) < 0.05
        X[nan_mask] = np.nan

    if sparse:
        X = sp.csr_matrix(X)

    # Create mask matrix
    if with_mask:
        M = rng.choice([0, 1, 2], size=(n_samples, n_features), p=[0.85, 0.1, 0.05]).astype(np.int8)
    else:
        M = None

    # Create assay
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"protein": assay}, sample_id_col="_index")


# =============================================================================
# Test Suite
# =============================================================================


class TestQuantileNormalization:
    """Tests for norm_quantile function."""

    def test_quantile_basic(self):
        """Test basic quantile normalization functionality."""
        container = create_quantile_test_container(n_samples=5, n_features=10, seed=42)

        result = norm_quantile(container)

        # Check that new layer was created
        assert "quantile_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["quantile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

    def test_quantile_distributions_match(self):
        """Test that quantile normalization makes all distributions identical."""
        # Create simple test case
        obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})

        X = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = norm_quantile(container)
        X_norm = result.assays["protein"].layers["quantile_norm"].X

        # After quantile normalization, sorted columns should be identical
        for j in range(X_norm.shape[1]):
            sorted_col = np.sort(X_norm[:, j])
            # All sorted columns should be identical
            if j > 0:
                assert np.allclose(sorted_col, np.sort(X_norm[:, 0]))

    def test_quantile_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_quantile_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = norm_quantile(container)

        normalized_M = result.assays["protein"].layers["quantile_norm"].M
        assert np.array_equal(original_M, normalized_M)

    def test_quantile_sparse_matrix(self):
        """Test quantile normalization with sparse matrices."""
        container = create_quantile_test_container(sparse=True, seed=42)

        result = norm_quantile(container)

        # Check that layer was created
        assert "quantile_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["quantile_norm"].X
        assert X_norm.shape == (10, 20)

    def test_quantile_custom_layer_names(self):
        """Test quantile normalization with custom layer names."""
        container = create_quantile_test_container(seed=42)

        result = norm_quantile(
            container, assay_name="protein", source_layer="raw", new_layer_name="custom_quantile"
        )

        assert "custom_quantile" in result.assays["protein"].layers
        assert "quantile_norm" not in result.assays["protein"].layers

    def test_quantile_with_nan_values(self):
        """Test quantile normalization with NaN values."""
        container = create_quantile_test_container(with_nan=True, seed=42)

        result = norm_quantile(container)

        X_norm = result.assays["protein"].layers["quantile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # NaN positions should be preserved
        X_original = container.assays["protein"].layers["raw"].X
        if sp.issparse(X_original):
            X_original = X_original.toarray()

        original_nan_mask = np.isnan(X_original)
        normalized_nan_mask = np.isnan(X_norm)

        assert np.array_equal(original_nan_mask, normalized_nan_mask)

    def test_quantile_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_quantile_test_container(seed=42)

        initial_history_len = len(container.history)
        result = norm_quantile(container)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_quantile"
        assert result.history[-1].params["assay"] == "protein"
        assert result.history[-1].params["new_layer_name"] == "quantile_norm"

    def test_quantile_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_quantile_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            norm_quantile(container, assay_name="nonexistent")

    def test_quantile_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_quantile_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            norm_quantile(container, source_layer="nonexistent")

    def test_quantile_immutability(self):
        """Test that original container is not modified."""
        container = create_quantile_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        result = norm_quantile(container)

        # Check original is unchanged
        result_X = result.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_quantile_single_sample(self):
        """Test quantile normalization with a single sample."""
        container = create_quantile_test_container(n_samples=1, n_features=10, seed=42)

        result = norm_quantile(container)

        X_norm = result.assays["protein"].layers["quantile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Single sample should be normalized
        # Values should be sorted
        assert X_norm.shape == (1, 10)

    def test_quantile_all_nan_column(self):
        """Test quantile normalization when a column is all NaN."""
        obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})

        X = np.array(
            [
                [10.0, 20.0, 30.0],
                [np.nan, np.nan, np.nan],
                [15.0, 25.0, 35.0],
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = norm_quantile(container)
        X_norm = result.assays["protein"].layers["quantile_norm"].X

        # All NaN column should remain all NaN
        assert np.all(np.isnan(X_norm[1, :]))

    def test_quantile_tie_handling(self):
        """Test that tied values are handled correctly."""
        obs = pl.DataFrame({"_index": ["s1", "s2"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})

        X = np.array(
            [
                [5.0, 5.0, 3.0, 1.0],  # Tied values: 5.0 appears twice
                [2.0, 4.0, 6.0, 8.0],
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = norm_quantile(container)
        X_norm = result.assays["protein"].layers["quantile_norm"].X

        # Tied values should receive the same normalized value
        # (average of their corresponding quantiles)
        assert np.isclose(X_norm[0, 0], X_norm[0, 1])

    def test_quantile_rank_preservation(self):
        """Test that ranks are preserved within each column."""
        container = create_quantile_test_container(n_samples=5, n_features=20, seed=42)

        result = norm_quantile(container)
        X_norm = result.assays["protein"].layers["quantile_norm"].X

        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        X_original = container.assays["protein"].layers["raw"].X
        if sp.issparse(X_original):
            X_original = X_original.toarray()

        # For each column, ranks should be preserved
        for j in range(X_norm.shape[1]):
            original_ranks = np.argsort(np.argsort(X_original[:, j]))
            normalized_ranks = np.argsort(np.argsort(X_norm[:, j]))

            # Handle NaN values
            valid_mask = ~np.isnan(X_original[:, j]) & ~np.isnan(X_norm[:, j])

            assert np.array_equal(original_ranks[valid_mask], normalized_ranks[valid_mask])

    def test_quantile_identical_distributions(self):
        """Test that all samples have identical distributions after normalization."""
        container = create_quantile_test_container(n_samples=10, n_features=50, seed=42)

        result = norm_quantile(container)
        X_norm = result.assays["protein"].layers["quantile_norm"].X

        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Check that sorted columns are identical
        sorted_first = np.sort(X_norm[:, 0])
        for j in range(1, X_norm.shape[1]):
            sorted_j = np.sort(X_norm[:, j])
            assert np.allclose(sorted_first, sorted_j, rtol=1e-10)


# =============================================================================
# Run All Tests (for direct execution)
# =============================================================================


def run_all_tests() -> None:
    """Run all quantile normalization tests."""
    print("=" * 60)
    print("Testing quantile normalization")
    print("=" * 60)

    test_cls = TestQuantileNormalization()
    test_cls.test_quantile_basic()
    print("✓ test_quantile_basic")

    test_cls.test_quantile_distributions_match()
    print("✓ test_quantile_distributions_match")

    test_cls.test_quantile_with_mask()
    print("✓ test_quantile_with_mask")

    test_cls.test_quantile_sparse_matrix()
    print("✓ test_quantile_sparse_matrix")

    test_cls.test_quantile_custom_layer_names()
    print("✓ test_quantile_custom_layer_names")

    test_cls.test_quantile_with_nan_values()
    print("✓ test_quantile_with_nan_values")

    test_cls.test_quantile_provenance_logging()
    print("✓ test_quantile_provenance_logging")

    test_cls.test_quantile_assay_not_found()
    print("✓ test_quantile_assay_not_found")

    test_cls.test_quantile_layer_not_found()
    print("✓ test_quantile_layer_not_found")

    test_cls.test_quantile_immutability()
    print("✓ test_quantile_immutability")

    test_cls.test_quantile_single_sample()
    print("✓ test_quantile_single_sample")

    test_cls.test_quantile_all_nan_column()
    print("✓ test_quantile_all_nan_column")

    test_cls.test_quantile_tie_handling()
    print("✓ test_quantile_tie_handling")

    test_cls.test_quantile_rank_preservation()
    print("✓ test_quantile_rank_preservation")

    test_cls.test_quantile_identical_distributions()
    print("✓ test_quantile_identical_distributions")

    print()
    print("=" * 60)
    print("All quantile normalization tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
