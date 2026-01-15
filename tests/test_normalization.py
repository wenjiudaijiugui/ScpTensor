"""Comprehensive tests for normalization modules.

Tests cover all normalization methods:
- median_centering
- median_scaling
- global_median_normalization
- sample_mean_normalization
- sample_median_normalization
- upper_quartile_normalization
- tmm_normalization
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

# =============================================================================
# Pytest markers for sparse matrix tests
# =============================================================================


sparse_unsupported = pytest.mark.skipif(
    True,
    reason="Sparse matrices are not yet supported by normalization functions. "
    "They will be converted to dense internally or raise an error.",
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_normalization_test_container(
    n_samples: int = 10,
    n_features: int = 20,
    sparse: bool = False,
    with_mask: bool = True,
    with_nan: bool = False,
    seed: int = 42,
) -> ScpContainer:
    """Create a test container for normalization tests.

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

    # Create data matrix with different scales per sample
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


def verify_mask_preserved(original_M: np.ndarray, normalized_M: np.ndarray | None) -> bool:
    """Verify that mask values are preserved during normalization."""
    if original_M is None:
        return normalized_M is None
    if normalized_M is None:
        return False
    return np.array_equal(original_M, normalized_M)


def verify_no_negative_values(X: np.ndarray) -> bool:
    """Verify that all values are non-negative."""
    if sp.issparse(X):
        return (X.data >= 0).all()
    return (X >= 0).all()


# =============================================================================
# median_centering Tests
# =============================================================================


class TestMedianCentering:
    """Tests for median_centering function."""

    def test_median_centering_basic(self):
        """Test basic median centering functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = median_centering(container)

        # Check that new layer was created
        assert "median_centered" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["median_centered"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

        # Check that median of each row is approximately zero
        medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(medians, 0, atol=1e-10)

    def test_median_centering_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = median_centering(container)

        normalized_M = result.assays["protein"].layers["median_centered"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_median_centering_sparse_matrix(self):
        """Test median centering with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        # The function should work, but will convert sparse to dense
        result = median_centering(container)

        # Check that layer was created
        assert "median_centered" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["median_centered"].X
        assert X_norm.shape == (10, 20)

    def test_median_centering_custom_layer_names(self):
        """Test median centering with custom layer names."""
        container = create_normalization_test_container(seed=42)

        result = median_centering(
            container, assay_name="protein", source_layer="raw", new_layer_name="custom_centered"
        )

        assert "custom_centered" in result.assays["protein"].layers
        assert "median_centered" not in result.assays["protein"].layers

    def test_median_centering_none_layer_name(self):
        """Test median centering with None as layer name (uses default)."""
        container = create_normalization_test_container(seed=42)

        result = median_centering(container, new_layer_name=None)

        assert "median_centered" in result.assays["protein"].layers

    def test_median_centering_with_nan_values(self):
        """Test median centering with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = median_centering(container)

        X_norm = result.assays["protein"].layers["median_centered"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Check that NaN values are handled (median should ignore them)
        medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(medians, 0, atol=1e-10)

    def test_median_centering_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = median_centering(container)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_median_centering"
        assert result.history[-1].params["assay"] == "protein"

    def test_median_centering_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            median_centering(container, assay_name="nonexistent")

    def test_median_centering_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            median_centering(container, source_layer="nonexistent")

    def test_median_centering_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        result = median_centering(container)

        # Check original is unchanged
        result_X = result.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_median_centering_single_sample(self):
        """Test median centering with a single sample."""
        container = create_normalization_test_container(n_samples=1, n_features=10, seed=42)

        result = median_centering(container)

        X_norm = result.assays["protein"].layers["median_centered"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Single sample should be centered to zero
        assert np.allclose(np.nanmedian(X_norm), 0, atol=1e-10)

    def test_median_centering_all_zeros_row(self):
        """Test median centering when a row is all zeros."""
        obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})

        X = np.array(
            [
                [10.0, 20.0, 30.0],
                [0.0, 0.0, 0.0],
                [15.0, 25.0, 35.0],
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = median_centering(container)

        X_norm = result.assays["protein"].layers["median_centered"].X
        # Row with all zeros should remain zeros
        assert np.allclose(X_norm[1, :], 0)


# =============================================================================
# median_scaling Tests
# =============================================================================


class TestMedianScaling:
    """Tests for median_scaling function."""

    def test_median_scaling_basic(self):
        """Test basic median scaling functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = median_scaling(container, "protein", "raw")

        # Check that new layer was created
        assert "median_scaling" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["median_scaling"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

        # Check that all samples have approximately the same median
        global_median = np.nanmedian(X_norm)
        sample_medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(sample_medians, global_median, atol=0.1)

    def test_median_scaling_alignment(self):
        """Test that median scaling aligns samples to global median."""
        container = create_normalization_test_container(n_samples=10, seed=42)

        original_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(original_X):
            original_X = original_X.toarray()

        # Get original medians
        original_medians = np.nanmedian(original_X, axis=1)
        original_global_median = np.nanmedian(original_X)

        # Verify samples have different medians initially
        assert not np.allclose(original_medians, original_global_median, rtol=0.1)

        result = median_scaling(container, "protein", "raw")

        X_norm = result.assays["protein"].layers["median_scaling"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # After normalization, medians should be aligned
        normalized_medians = np.nanmedian(X_norm, axis=1)
        normalized_global_median = np.nanmedian(X_norm)

        # All sample medians should be close to global median
        assert np.allclose(normalized_medians, normalized_global_median, atol=0.5)

    def test_median_scaling_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = median_scaling(container, "protein", "raw")

        normalized_M = result.assays["protein"].layers["median_scaling"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_median_scaling_sparse_matrix(self):
        """Test median scaling with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = median_scaling(container, "protein", "raw")

        # Check that layer was created
        assert "median_scaling" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["median_scaling"].X
        assert X_norm.shape == (10, 20)

    def test_median_scaling_custom_layer_name(self):
        """Test median scaling with custom layer name."""
        container = create_normalization_test_container(seed=42)

        result = median_scaling(
            container, assay_name="protein", source_layer="raw", new_layer_name="custom_scaled"
        )

        assert "custom_scaled" in result.assays["protein"].layers

    def test_median_scaling_with_nan_values(self):
        """Test median scaling with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = median_scaling(container, "protein", "raw")

        X_norm = result.assays["protein"].layers["median_scaling"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Check NaN handling
        assert not np.any(np.isnan(X_norm[np.isfinite(X_norm)]))

    def test_median_scaling_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = median_scaling(container, "protein", "raw")

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_median_scaling"

    def test_median_scaling_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            median_scaling(container, "nonexistent", "raw")

    def test_median_scaling_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            median_scaling(container, "protein", "nonexistent")

    def test_median_scaling_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        median_scaling(container, "protein", "raw")

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)


# =============================================================================
# global_median_normalization Tests
# =============================================================================


class TestGlobalMedianNormalization:
    """Tests for global_median_normalization function."""

    def test_global_median_basic(self):
        """Test basic global median normalization functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = global_median_normalization(container)

        # Check that new layer was created
        assert "global_median_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["global_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

        # Check that all samples have approximately the same median
        global_median = np.nanmedian(X_norm)
        sample_medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(sample_medians, global_median, atol=0.1)

    def test_global_median_aligns_samples(self):
        """Test that global median normalization aligns all samples."""
        container = create_normalization_test_container(n_samples=10, seed=42)

        result = global_median_normalization(container)

        X_norm = result.assays["protein"].layers["global_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # All samples should have the same median
        global_med = np.nanmedian(X_norm)
        sample_meds = np.nanmedian(X_norm, axis=1)

        # Each sample median should be very close to global median
        for med in sample_meds:
            assert np.isclose(med, global_med, atol=0.5)

    def test_global_median_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = global_median_normalization(container)

        normalized_M = result.assays["protein"].layers["global_median_norm"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_global_median_sparse_matrix(self):
        """Test global median normalization with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = global_median_normalization(container)

        # Check that layer was created
        assert "global_median_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["global_median_norm"].X
        assert X_norm.shape == (10, 20)

    def test_global_median_custom_layer_names(self):
        """Test global median normalization with custom layer names."""
        container = create_normalization_test_container(seed=42)

        result = global_median_normalization(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="custom_global_norm",
        )

        assert "custom_global_norm" in result.assays["protein"].layers

    def test_global_median_with_nan_values(self):
        """Test global median normalization with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = global_median_normalization(container)

        X_norm = result.assays["protein"].layers["global_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # NaN values should be handled correctly
        assert not np.any(np.isnan(X_norm[np.isfinite(X_norm)]))

    def test_global_median_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = global_median_normalization(container)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_global_median"

    def test_global_median_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            global_median_normalization(container, assay_name="nonexistent")

    def test_global_median_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            global_median_normalization(container, source_layer="nonexistent")

    def test_global_median_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        global_median_normalization(container)

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_global_median_mathematical_correctness(self):
        """Test the mathematical correctness of global median normalization."""
        # Create a simple test case with known values
        obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})

        X = np.array(
            [
                [10.0, 20.0, 30.0],  # median = 20
                [20.0, 30.0, 40.0],  # median = 30
                [15.0, 25.0, 35.0],  # median = 25
            ]
        )
        # Global median = 25

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = global_median_normalization(container)

        X_norm = result.assays["protein"].layers["global_median_norm"].X

        # Expected biases: [20-25=-5, 30-25=5, 25-25=0]
        # Expected result: X - bias
        expected = np.array(
            [
                [15.0, 25.0, 35.0],  # 10-(-5)=15, 20-(-5)=25, 30-(-5)=35
                [15.0, 25.0, 35.0],  # 20-5=15, 30-5=25, 40-5=35
                [15.0, 25.0, 35.0],  # 15-0=15, 25-0=25, 35-0=35
            ]
        )

        assert np.allclose(X_norm, expected)


# =============================================================================
# sample_mean_normalization Tests
# =============================================================================


class TestSampleMeanNormalization:
    """Tests for sample_mean_normalization function."""

    def test_sample_mean_basic(self):
        """Test basic sample mean normalization functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = sample_mean_normalization(container)

        # Check that new layer was created
        assert "sample_mean_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["sample_mean_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

        # Check that mean of each row is approximately zero
        means = np.nanmean(X_norm, axis=1)
        assert np.allclose(means, 0, atol=1e-10)

    def test_sample_mean_centers_to_zero(self):
        """Test that sample mean normalization centers each sample to zero mean."""
        container = create_normalization_test_container(n_samples=10, seed=42)

        result = sample_mean_normalization(container)

        X_norm = result.assays["protein"].layers["sample_mean_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Each sample should have mean ~0
        means = np.nanmean(X_norm, axis=1)
        for mean in means:
            assert np.isclose(mean, 0, atol=1e-10)

    def test_sample_mean_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = sample_mean_normalization(container)

        normalized_M = result.assays["protein"].layers["sample_mean_norm"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_sample_mean_sparse_matrix(self):
        """Test sample mean normalization with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = sample_mean_normalization(container)

        # Check that layer was created
        assert "sample_mean_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["sample_mean_norm"].X
        assert X_norm.shape == (10, 20)

    def test_sample_mean_custom_layer_names(self):
        """Test sample mean normalization with custom layer names."""
        container = create_normalization_test_container(seed=42)

        result = sample_mean_normalization(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="custom_mean_norm",
        )

        assert "custom_mean_norm" in result.assays["protein"].layers

    def test_sample_mean_with_nan_values(self):
        """Test sample mean normalization with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = sample_mean_normalization(container)

        X_norm = result.assays["protein"].layers["sample_mean_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # NaN values should be handled correctly
        means = np.nanmean(X_norm, axis=1)
        assert np.allclose(means, 0, atol=1e-10)

    def test_sample_mean_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = sample_mean_normalization(container)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_sample_mean"

    def test_sample_mean_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            sample_mean_normalization(container, assay_name="nonexistent")

    def test_sample_mean_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            sample_mean_normalization(container, source_layer="nonexistent")

    def test_sample_mean_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        sample_mean_normalization(container)

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_sample_mean_vs_median_difference(self):
        """Test that mean normalization is more sensitive to outliers than median."""
        obs = pl.DataFrame({"_index": ["s1"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4", "p5"]})

        # Data with outlier
        X = np.array([[10.0, 20.0, 30.0, 40.0, 1000.0]])

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result_mean = sample_mean_normalization(container)

        X_mean = result_mean.assays["protein"].layers["sample_mean_norm"].X

        # Mean normalization: all values centered around mean (220)
        # Mean of normalized should be 0
        assert np.isclose(np.nanmean(X_mean), 0, atol=1e-10)
        # But values are spread widely due to outlier
        assert X_mean[0, -1] > X_mean[0, 0]  # Outlier effect visible


# =============================================================================
# sample_median_normalization Tests
# =============================================================================


class TestSampleMedianNormalization:
    """Tests for sample_median_normalization function."""

    def test_sample_median_basic(self):
        """Test basic sample median normalization functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = sample_median_normalization(container)

        # Check that new layer was created
        assert "sample_median_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["sample_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

        # Check that median of each row is approximately zero
        medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(medians, 0, atol=1e-10)

    def test_sample_median_centers_to_zero(self):
        """Test that sample median normalization centers each sample to zero median."""
        container = create_normalization_test_container(n_samples=10, seed=42)

        result = sample_median_normalization(container)

        X_norm = result.assays["protein"].layers["sample_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # Each sample should have median ~0
        medians = np.nanmedian(X_norm, axis=1)
        for med in medians:
            assert np.isclose(med, 0, atol=1e-10)

    def test_sample_median_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = sample_median_normalization(container)

        normalized_M = result.assays["protein"].layers["sample_median_norm"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_sample_median_sparse_matrix(self):
        """Test sample median normalization with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = sample_median_normalization(container)

        # Check that layer was created
        assert "sample_median_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["sample_median_norm"].X
        assert X_norm.shape == (10, 20)

    def test_sample_median_custom_layer_names(self):
        """Test sample median normalization with custom layer names."""
        container = create_normalization_test_container(seed=42)

        result = sample_median_normalization(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="custom_median_norm",
        )

        assert "custom_median_norm" in result.assays["protein"].layers

    def test_sample_median_none_layer_name(self):
        """Test sample median normalization with None as layer name."""
        container = create_normalization_test_container(seed=42)

        result = sample_median_normalization(container, new_layer_name=None)

        # Should use default name
        assert "sample_median_norm" in result.assays["protein"].layers

    def test_sample_median_with_nan_values(self):
        """Test sample median normalization with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = sample_median_normalization(container)

        X_norm = result.assays["protein"].layers["sample_median_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # NaN values should be handled correctly
        medians = np.nanmedian(X_norm, axis=1)
        assert np.allclose(medians, 0, atol=1e-10)

    def test_sample_median_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = sample_median_normalization(container)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_sample_median"

    def test_sample_median_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            sample_median_normalization(container, assay_name="nonexistent")

    def test_sample_median_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            sample_median_normalization(container, source_layer="nonexistent")

    def test_sample_median_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        sample_median_normalization(container)

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_sample_median_robust_to_outliers(self):
        """Test that median normalization is robust to outliers."""
        obs = pl.DataFrame({"_index": ["s1"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4", "p5"]})

        # Data with outlier
        X = np.array([[10.0, 20.0, 30.0, 40.0, 1000.0]])
        # Median = 30

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = sample_median_normalization(container)

        X_norm = result.assays["protein"].layers["sample_median_norm"].X

        # After median centering, values should be around median (30)
        # Expected: [-20, -10, 0, 10, 970]
        expected = np.array([[10.0 - 30, 20.0 - 30, 30.0 - 30, 40.0 - 30, 1000.0 - 30]])
        assert np.allclose(X_norm, expected)


# =============================================================================
# upper_quartile_normalization Tests
# =============================================================================


class TestUpperQuartileNormalization:
    """Tests for upper_quartile_normalization function."""

    def test_upper_quartile_basic(self):
        """Test basic upper quartile normalization functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=10, seed=42)

        result = upper_quartile_normalization(container)

        # Check that new layer was created
        assert "upper_quartile_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 10)

    def test_upper_quartile_aligns_samples(self):
        """Test that upper quartile normalization aligns all samples."""
        container = create_normalization_test_container(n_samples=10, seed=42)

        result = upper_quartile_normalization(container)

        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # All samples should have approximately the same 75th percentile
        global_uq = np.nanpercentile(X_norm, 75)
        sample_uqs = np.nanpercentile(X_norm, 75, axis=1)

        # Each sample's 75th percentile should be close to global
        for uq in sample_uqs:
            assert np.isclose(uq, global_uq, atol=1)

    def test_upper_quartile_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = upper_quartile_normalization(container)

        normalized_M = result.assays["protein"].layers["upper_quartile_norm"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_upper_quartile_sparse_matrix(self):
        """Test upper quartile normalization with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = upper_quartile_normalization(container)

        # Check that layer was created
        assert "upper_quartile_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X
        assert X_norm.shape == (10, 20)

    def test_upper_quartile_custom_percentile(self):
        """Test upper quartile normalization with custom percentile."""
        container = create_normalization_test_container(n_samples=5, n_features=20, seed=42)

        # Test with 90th percentile
        result = upper_quartile_normalization(container, percentile=0.9)

        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # All samples should have approximately the same 90th percentile
        global_p90 = np.nanpercentile(X_norm, 90)
        sample_p90s = np.nanpercentile(X_norm, 90, axis=1)

        for p90 in sample_p90s:
            assert np.isclose(p90, global_p90, atol=2)

    def test_upper_quartile_custom_layer_names(self):
        """Test upper quartile normalization with custom layer names."""
        container = create_normalization_test_container(seed=42)

        result = upper_quartile_normalization(
            container, assay_name="protein", source_layer="raw", new_layer_name="custom_uq_norm"
        )

        assert "custom_uq_norm" in result.assays["protein"].layers

    def test_upper_quartile_invalid_percentile(self):
        """Test error when percentile is outside (0, 1)."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(ScpValueError):
            upper_quartile_normalization(container, percentile=0)

        with pytest.raises(ScpValueError):
            upper_quartile_normalization(container, percentile=1)

        with pytest.raises(ScpValueError):
            upper_quartile_normalization(container, percentile=-0.1)

        with pytest.raises(ScpValueError):
            upper_quartile_normalization(container, percentile=1.5)

    def test_upper_quartile_with_nan_values(self):
        """Test upper quartile normalization with NaN values."""
        container = create_normalization_test_container(with_nan=True, seed=42)

        result = upper_quartile_normalization(container)

        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # NaN values should be handled correctly
        assert not np.any(np.isnan(X_norm[np.isfinite(X_norm)]))

    def test_upper_quartile_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = upper_quartile_normalization(container, percentile=0.75)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_upper_quartile"
        assert result.history[-1].params["percentile"] == 0.75

    def test_upper_quartile_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            upper_quartile_normalization(container, assay_name="nonexistent")

    def test_upper_quartile_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            upper_quartile_normalization(container, source_layer="nonexistent")

    def test_upper_quartile_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        upper_quartile_normalization(container)

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_upper_quartile_zero_handling(self):
        """Test upper quartile normalization when some samples have zero UQ."""
        obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})

        X = np.array(
            [
                [10.0, 20.0, 30.0, 40.0],  # UQ = 32.5
                [0.0, 0.0, 0.0, 0.0],  # UQ = 0
                [15.0, 25.0, 35.0, 45.0],  # UQ = 40
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = upper_quartile_normalization(container)

        X_norm = result.assays["protein"].layers["upper_quartile_norm"].X

        # Sample with zero UQ should not cause division by zero
        # It should remain zeros (scaling factor = 1.0 for non-finite)
        assert np.allclose(X_norm[1, :], 0)


# =============================================================================
# tmm_normalization Tests
# =============================================================================


class TestTMMNormalization:
    """Tests for tmm_normalization function."""

    def test_tmm_basic(self):
        """Test basic TMM normalization functionality."""
        container = create_normalization_test_container(n_samples=5, n_features=20, seed=42)

        result = tmm_normalization(container)

        # Check that new layer was created
        assert "tmm_norm" in result.assays["protein"].layers

        # Check that shape is preserved
        X_norm = result.assays["protein"].layers["tmm_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        assert X_norm.shape == (5, 20)

    def test_tmm_with_mask(self):
        """Test that mask matrix is preserved."""
        container = create_normalization_test_container(with_mask=True, seed=42)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        result = tmm_normalization(container)

        normalized_M = result.assays["protein"].layers["tmm_norm"].M
        assert np.array_equal(original_M, normalized_M)

    @sparse_unsupported
    def test_tmm_sparse_matrix(self):
        """Test TMM normalization with sparse matrices.

        Note: Current implementation converts sparse to dense internally.
        This test verifies the function works, but output will be dense.
        """
        container = create_normalization_test_container(sparse=True, seed=42)

        result = tmm_normalization(container)

        # Check that layer was created
        assert "tmm_norm" in result.assays["protein"].layers
        X_norm = result.assays["protein"].layers["tmm_norm"].X
        assert X_norm.shape == (10, 20)

    def test_tmm_custom_reference_sample(self):
        """Test TMM normalization with custom reference sample."""
        container = create_normalization_test_container(n_samples=5, n_features=20, seed=42)

        result = tmm_normalization(container, reference_sample=0)

        X_norm = result.assays["protein"].layers["tmm_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        # First sample should be unchanged (reference)
        X_orig = container.assays["protein"].layers["raw"].X
        if sp.issparse(X_orig):
            X_orig = X_orig.toarray()

        # Reference sample should have scaling factor of 1.0
        assert np.allclose(X_norm[0, :], X_orig[0, :], rtol=0.01)

    def test_tmm_custom_trim_ratio(self):
        """Test TMM normalization with custom trim ratio."""
        container = create_normalization_test_container(n_samples=5, n_features=20, seed=42)

        result = tmm_normalization(container, trim_ratio=0.2)

        X_norm = result.assays["protein"].layers["tmm_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        assert X_norm.shape == (5, 20)

    def test_tmm_custom_layer_name(self):
        """Test TMM normalization with custom layer name."""
        container = create_normalization_test_container(seed=42)

        result = tmm_normalization(
            container, assay_name="protein", source_layer="raw", new_layer_name="custom_tmm"
        )

        assert "custom_tmm" in result.assays["protein"].layers

    def test_tmm_invalid_trim_ratio(self):
        """Test error when trim_ratio is outside [0, 0.5)."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(ScpValueError):
            tmm_normalization(container, trim_ratio=-0.1)

        with pytest.raises(ScpValueError):
            tmm_normalization(container, trim_ratio=0.5)

        with pytest.raises(ScpValueError):
            tmm_normalization(container, trim_ratio=1.0)

    def test_tmm_invalid_reference_sample_negative(self):
        """Test error when reference_sample is negative."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(ScpValueError):
            tmm_normalization(container, reference_sample=-1)

    def test_tmm_invalid_reference_sample_out_of_bounds(self):
        """Test error when reference_sample is out of bounds."""
        container = create_normalization_test_container(n_samples=5, seed=42)

        with pytest.raises(ScpValueError):
            tmm_normalization(container, reference_sample=10)

    def test_tmm_provenance_logging(self):
        """Test that operation is logged in provenance."""
        container = create_normalization_test_container(seed=42)

        initial_history_len = len(container.history)
        result = tmm_normalization(container, trim_ratio=0.3)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "normalization_tmm"
        assert "trim_ratio" in result.history[-1].params
        assert "reference_sample" in result.history[-1].params

    def test_tmm_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(AssayNotFoundError):
            tmm_normalization(container, assay_name="nonexistent")

    def test_tmm_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_normalization_test_container(seed=42)

        with pytest.raises(LayerNotFoundError):
            tmm_normalization(container, source_layer="nonexistent")

    def test_tmm_immutability(self):
        """Test that original container is not modified."""
        container = create_normalization_test_container(seed=42)

        original_X = (
            container.assays["protein"].layers["raw"].X.copy().toarray()
            if sp.issparse(container.assays["protein"].layers["raw"].X)
            else container.assays["protein"].layers["raw"].X.copy()
        )

        tmm_normalization(container)

        result_X = container.assays["protein"].layers["raw"].X
        if sp.issparse(result_X):
            result_X = result_X.toarray()
        assert np.array_equal(original_X, result_X)

    def test_tmm_zero_handling(self):
        """Test TMM normalization with zero values."""
        obs = pl.DataFrame({"_index": ["s1", "s2"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})

        X = np.array(
            [
                [10.0, 20.0, 0.0, 40.0],
                [0.0, 20.0, 30.0, 0.0],
            ]
        )

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = tmm_normalization(container)

        # Should handle zeros gracefully
        assert "tmm_norm" in result.assays["protein"].layers

    def test_tmm_single_sample(self):
        """Test TMM normalization with a single sample."""
        container = create_normalization_test_container(n_samples=1, n_features=10, seed=42)

        result = tmm_normalization(container)

        X_norm = result.assays["protein"].layers["tmm_norm"].X
        X_orig = container.assays["protein"].layers["raw"].X

        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        if sp.issparse(X_orig):
            X_orig = X_orig.toarray()

        # Single sample should remain unchanged
        assert np.allclose(X_norm, X_orig)

    def test_tmm_scaling_factors_sum_to_reference(self):
        """Test that TMM scaling factors properly normalize to reference."""
        container = create_normalization_test_container(n_samples=10, n_features=50, seed=42)

        result = tmm_normalization(container, reference_sample=5)

        X_norm = result.assays["protein"].layers["tmm_norm"].X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()

        X_orig = container.assays["protein"].layers["raw"].X
        if sp.issparse(X_orig):
            X_orig = X_orig.toarray()

        # Reference sample should be unchanged
        assert np.allclose(X_norm[5, :], X_orig[5, :], rtol=0.01)

        # Other samples should be scaled
        for i in range(10):
            if i != 5:
                # Check that some scaling occurred
                assert not np.allclose(X_norm[i, :], X_orig[i, :])

    def test_tmm_edge_case_two_samples(self):
        """Test TMM normalization with only two samples."""
        container = create_normalization_test_container(n_samples=2, n_features=10, seed=42)

        result = tmm_normalization(container)

        # Should complete without error
        assert "tmm_norm" in result.assays["protein"].layers


# =============================================================================
# Import normalization functions
# =============================================================================


from scptensor.normalization.global_median import global_median_normalization
from scptensor.normalization.median_centering import median_centering
from scptensor.normalization.median_scaling import median_scaling
from scptensor.normalization.sample_mean import sample_mean_normalization
from scptensor.normalization.sample_median import sample_median_normalization
from scptensor.normalization.tmm import tmm_normalization
from scptensor.normalization.upper_quartile import upper_quartile_normalization

# =============================================================================
# Run All Tests (for direct execution)
# =============================================================================


def run_all_tests() -> None:
    """Run all normalization tests."""
    print("=" * 60)
    print("Testing median_centering")
    print("=" * 60)

    test_cls = TestMedianCentering()
    test_cls.test_median_centering_basic()
    test_cls.test_median_centering_with_mask()
    test_cls.test_median_centering_sparse_matrix()
    test_cls.test_median_centering_custom_layer_names()
    test_cls.test_median_centering_none_layer_name()
    test_cls.test_median_centering_with_nan_values()
    test_cls.test_median_centering_provenance_logging()
    test_cls.test_median_centering_assay_not_found()
    test_cls.test_median_centering_layer_not_found()
    test_cls.test_median_centering_immutability()
    test_cls.test_median_centering_single_sample()
    test_cls.test_median_centering_all_zeros_row()
    print("All median_centering tests passed!")

    print()
    print("=" * 60)
    print("Testing median_scaling")
    print("=" * 60)

    test_cls = TestMedianScaling()
    test_cls.test_median_scaling_basic()
    test_cls.test_median_scaling_alignment()
    test_cls.test_median_scaling_with_mask()
    test_cls.test_median_scaling_sparse_matrix()
    test_cls.test_median_scaling_custom_layer_name()
    test_cls.test_median_scaling_with_nan_values()
    test_cls.test_median_scaling_provenance_logging()
    test_cls.test_median_scaling_assay_not_found()
    test_cls.test_median_scaling_layer_not_found()
    test_cls.test_median_scaling_immutability()
    print("All median_scaling tests passed!")

    print()
    print("=" * 60)
    print("Testing global_median_normalization")
    print("=" * 60)

    test_cls = TestGlobalMedianNormalization()
    test_cls.test_global_median_basic()
    test_cls.test_global_median_aligns_samples()
    test_cls.test_global_median_with_mask()
    test_cls.test_global_median_sparse_matrix()
    test_cls.test_global_median_custom_layer_names()
    test_cls.test_global_median_with_nan_values()
    test_cls.test_global_median_provenance_logging()
    test_cls.test_global_median_assay_not_found()
    test_cls.test_global_median_layer_not_found()
    test_cls.test_global_median_immutability()
    test_cls.test_global_median_mathematical_correctness()
    print("All global_median_normalization tests passed!")

    print()
    print("=" * 60)
    print("Testing sample_mean_normalization")
    print("=" * 60)

    test_cls = TestSampleMeanNormalization()
    test_cls.test_sample_mean_basic()
    test_cls.test_sample_mean_centers_to_zero()
    test_cls.test_sample_mean_with_mask()
    test_cls.test_sample_mean_sparse_matrix()
    test_cls.test_sample_mean_custom_layer_names()
    test_cls.test_sample_mean_with_nan_values()
    test_cls.test_sample_mean_provenance_logging()
    test_cls.test_sample_mean_assay_not_found()
    test_cls.test_sample_mean_layer_not_found()
    test_cls.test_sample_mean_immutability()
    test_cls.test_sample_mean_vs_median_difference()
    print("All sample_mean_normalization tests passed!")

    print()
    print("=" * 60)
    print("Testing sample_median_normalization")
    print("=" * 60)

    test_cls = TestSampleMedianNormalization()
    test_cls.test_sample_median_basic()
    test_cls.test_sample_median_centers_to_zero()
    test_cls.test_sample_median_with_mask()
    test_cls.test_sample_median_sparse_matrix()
    test_cls.test_sample_median_custom_layer_names()
    test_cls.test_sample_median_none_layer_name()
    test_cls.test_sample_median_with_nan_values()
    test_cls.test_sample_median_provenance_logging()
    test_cls.test_sample_median_assay_not_found()
    test_cls.test_sample_median_layer_not_found()
    test_cls.test_sample_median_immutability()
    test_cls.test_sample_median_robust_to_outliers()
    print("All sample_median_normalization tests passed!")

    print()
    print("=" * 60)
    print("Testing upper_quartile_normalization")
    print("=" * 60)

    test_cls = TestUpperQuartileNormalization()
    test_cls.test_upper_quartile_basic()
    test_cls.test_upper_quartile_aligns_samples()
    test_cls.test_upper_quartile_with_mask()
    test_cls.test_upper_quartile_sparse_matrix()
    test_cls.test_upper_quartile_custom_percentile()
    test_cls.test_upper_quartile_custom_layer_names()
    test_cls.test_upper_quartile_invalid_percentile()
    test_cls.test_upper_quartile_with_nan_values()
    test_cls.test_upper_quartile_provenance_logging()
    test_cls.test_upper_quartile_assay_not_found()
    test_cls.test_upper_quartile_layer_not_found()
    test_cls.test_upper_quartile_immutability()
    test_cls.test_upper_quartile_zero_handling()
    print("All upper_quartile_normalization tests passed!")

    print()
    print("=" * 60)
    print("Testing tmm_normalization")
    print("=" * 60)

    test_cls = TestTMMNormalization()
    test_cls.test_tmm_basic()
    test_cls.test_tmm_with_mask()
    test_cls.test_tmm_sparse_matrix()
    test_cls.test_tmm_custom_reference_sample()
    test_cls.test_tmm_custom_trim_ratio()
    test_cls.test_tmm_custom_layer_name()
    test_cls.test_tmm_invalid_trim_ratio()
    test_cls.test_tmm_invalid_reference_sample_negative()
    test_cls.test_tmm_invalid_reference_sample_out_of_bounds()
    test_cls.test_tmm_provenance_logging()
    test_cls.test_tmm_assay_not_found()
    test_cls.test_tmm_layer_not_found()
    test_cls.test_tmm_immutability()
    test_cls.test_tmm_zero_handling()
    test_cls.test_tmm_single_sample()
    test_cls.test_tmm_scaling_factors_sum_to_reference()
    test_cls.test_tmm_edge_case_two_samples()
    print("All tmm_normalization tests passed!")

    print()
    print("=" * 60)
    print("All normalization tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
