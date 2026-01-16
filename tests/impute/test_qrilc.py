"""
Comprehensive tests for QRILC imputation.

Tests cover:
- Basic functionality
- Different missing rates
- Different q (quantile) values
- Parameter validation
- Edge cases (all missing, no missing, tiny data)
- Mask matrix updates
- Reproducibility with random state
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp_sparse
from scipy.stats import norm

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.impute.qrilc import impute_qrilc

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_qrilc_container():
    """Factory function to create containers with MNAR missing data."""

    def _create(
        n_samples=100,
        n_features=50,
        missing_rate=0.2,
        random_state=42,
        with_mask=True,
        use_sparse=False,
        mnar=True,
    ):
        np.random.seed(random_state)

        # Generate log-normal data (typical for proteomics)
        X_true = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)

        if mnar:
            # Add MNAR missingness (lower values more likely to be missing)
            X_missing = X_true.copy()
            missing_prob = 1 - norm.cdf(X_true, loc=np.mean(X_true), scale=np.std(X_true))
            missing_prob = (missing_prob - missing_prob.min()) / (
                missing_prob.max() - missing_prob.min()
            )
            missing_mask = np.random.rand(n_samples, n_features) < missing_prob * missing_rate
        else:
            # Random missing (MCAR/MAR)
            X_missing = X_true.copy()
            missing_mask = np.random.rand(n_samples, n_features) < missing_rate

        X_missing[missing_mask] = np.nan

        if use_sparse:
            X_missing = sp_sparse.csr_matrix(X_missing)

        # Create mask matrix if requested
        if with_mask:
            M = np.zeros(X_true.shape, dtype=np.int8)
            M[missing_mask] = np.where(
                np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
            )
        else:
            M = None

        # Create container
        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_missing, M=M))

        container = ScpContainer(obs=obs, assays={"protein": assay})

        return container, X_true, missing_mask

    return _create


@pytest.fixture
def qrilc_container(create_qrilc_container):
    """Standard test container with MNAR missing data."""
    return create_qrilc_container(n_samples=100, n_features=50, missing_rate=0.2, with_mask=False)


@pytest.fixture
def container_no_missing():
    """Container with no missing values."""

    def _create(use_sparse=False):
        np.random.seed(42)
        n_samples, n_features = 50, 30
        X = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)

        if use_sparse:
            X = sp_sparse.csr_matrix(X)

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


@pytest.fixture
def container_all_missing():
    """Container with all values missing."""

    def _create():
        n_samples, n_features = 30, 20
        X = np.full((n_samples, n_features), np.nan)

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


@pytest.fixture
def tiny_container():
    """Minimal container for edge case testing."""

    def _create(n_samples=5, n_features=5, missing_rate=0.3):
        np.random.seed(42)
        X = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestQRILCBasic:
    """Test basic QRILC imputation functionality."""

    def test_qrilc_basic_imputation(self, qrilc_container):
        """Test basic QRILC imputation."""
        container, X_true, missing_mask = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )

        assert "qrilc" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["qrilc"]
        X_imputed = result_matrix.X
        M_imputed = result_matrix.M

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

        # Check original values preserved
        assert np.allclose(X_imputed[~missing_mask], X_true[~missing_mask])

        # Check mask was created and updated correctly
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_qrilc_imputed_values_non_negative(self, qrilc_container):
        """Test that imputed values are non-negative."""
        container, X_true, missing_mask = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )

        X_imputed = result.assays["protein"].layers["qrilc"].X

        # All imputed values should be non-negative
        assert np.all(X_imputed >= 0)

    def test_qrilc_imputed_values_below_detected(self, qrilc_container):
        """Test that imputed values are typically below detection threshold."""
        container, X_true, missing_mask = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )

        X_imputed = result.assays["protein"].layers["qrilc"].X

        # For most features, imputed values should be below median of detected
        imputed_values = X_imputed[missing_mask]
        detected_values = X_true[~missing_mask]

        # At least 50% of imputed values should be below median detected
        assert np.sum(imputed_values < np.median(detected_values)) / len(imputed_values) >= 0.5

    def test_qrilc_different_missing_rates(self, create_qrilc_container):
        """Test QRILC with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_qrilc_container(
                n_samples=100, n_features=50, missing_rate=rate
            )

            result = impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=0.01,
            )
            X_imputed = result.assays["protein"].layers["qrilc"].X

            assert not np.any(np.isnan(X_imputed))
            # Imputed values should be in reasonable range
            assert np.all(np.isfinite(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_qrilc_sparse_matrix(self, create_qrilc_container):
        """Test QRILC with sparse input matrix."""
        container, X_true, missing_mask = create_qrilc_container(
            n_samples=100, n_features=50, missing_rate=0.2, use_sparse=True
        )

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape
        assert np.all(X_imputed >= 0)

    def test_qrilc_custom_layer_name(self, qrilc_container):
        """Test QRILC with custom output layer name."""
        container, _, _ = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="my_imputed",
        )

        assert "my_imputed" in result.assays["protein"].layers
        assert "qrilc" not in result.assays["protein"].layers

    def test_qrilc_no_missing_values(self, container_no_missing):
        """Test QRILC with no missing values."""
        container = container_no_missing()

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )

        assert "qrilc" in result.assays["protein"].layers

        # Values should be unchanged
        X_original = container.assays["protein"].layers["raw"].X
        X_imputed = result.assays["protein"].layers["qrilc"].X
        if hasattr(X_original, "toarray"):
            X_original = X_original.toarray()
        np.testing.assert_array_equal(X_imputed, X_original)


# =============================================================================
# Parameter Tests
# =============================================================================


class TestQRILCParameters:
    """Test QRILC with different parameter values."""

    def test_qrilc_different_q_values(self, qrilc_container):
        """Test QRILC with different quantile thresholds."""
        container, _, missing_mask = qrilc_container

        for q_val in [0.001, 0.01, 0.05, 0.1]:
            result = impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=q_val,
            )
            X_imputed = result.assays["protein"].layers["qrilc"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

            # Higher q should generally give higher imputed values
            imputed_values = X_imputed[missing_mask]
            assert len(imputed_values) > 0

    def test_qrilc_random_state_reproducibility(self, qrilc_container):
        """Test that random state gives reproducible results."""
        container, _, _ = qrilc_container

        result1 = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
            random_state=42,
        )

        result2 = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["qrilc"].X
        X2 = result2.assays["protein"].layers["qrilc"].X

        # Results should be identical with same random state
        np.testing.assert_array_almost_equal(X1, X2)

    def test_qrilc_different_random_states(self, create_qrilc_container):
        """Test that different random states give different results."""
        # Create two separate containers to avoid in-place modification issues
        container1, _, _ = create_qrilc_container(
            n_samples=100, n_features=50, missing_rate=0.2, with_mask=False, random_state=42
        )
        container2, _, _ = create_qrilc_container(
            n_samples=100, n_features=50, missing_rate=0.2, with_mask=False, random_state=42
        )

        result1 = impute_qrilc(
            container1,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
            random_state=42,
        )

        result2 = impute_qrilc(
            container2,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
            random_state=123,
        )

        X1 = result1.assays["protein"].layers["qrilc"].X
        X2 = result2.assays["protein"].layers["qrilc"].X

        # Results should differ with different random states
        # (but may not be completely different due to deterministic parts)
        assert not np.array_equal(X1, X2)


# =============================================================================
# Mask Update Tests
# =============================================================================


class TestQRILCMaskUpdate:
    """Test mask matrix handling in QRILC."""

    def test_qrilc_mask_created_when_none(self, qrilc_container):
        """Test that QRILC creates mask when original M is None."""
        container, X_true, missing_mask = qrilc_container

        # Ensure no original mask
        assert container.assays["protein"].layers["raw"].M is None

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
        )

        M_result = result.assays["protein"].layers["qrilc"].M

        # Mask should be created
        assert M_result is not None
        assert np.all(M_result[missing_mask] == MaskCode.IMPUTED)
        assert np.all(M_result[~missing_mask] == MaskCode.VALID)

    def test_qrilc_existing_mask_updated(self, create_qrilc_container):
        """Test that QRILC updates existing mask correctly."""
        container, X_true, missing_mask = create_qrilc_container(
            n_samples=100, n_features=50, missing_rate=0.2, with_mask=True
        )

        M_original = container.assays["protein"].layers["raw"].M

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
        )

        M_result = result.assays["protein"].layers["qrilc"].M

        # All originally missing values should now have IMPUTED code
        assert M_result is not None
        assert np.all(M_result[missing_mask] == MaskCode.IMPUTED)
        # Non-missing values should keep their original codes
        assert np.array_equal(M_result[~missing_mask], M_original[~missing_mask])


# =============================================================================
# Validation Tests
# =============================================================================


class TestQRILCValidation:
    """Test parameter validation for QRILC."""

    def test_qrilc_invalid_q_zero(self, qrilc_container):
        """Test QRILC with q=0 raises error."""
        container, _, _ = qrilc_container

        with pytest.raises(ScpValueError, match="Quantile q must be between 0 and 1"):
            impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=0,
            )

    def test_qrilc_invalid_q_one(self, qrilc_container):
        """Test QRILC with q=1 raises error."""
        container, _, _ = qrilc_container

        with pytest.raises(ScpValueError, match="Quantile q must be between 0 and 1"):
            impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=1,
            )

    def test_qrilc_invalid_q_negative(self, qrilc_container):
        """Test QRILC with negative q raises error."""
        container, _, _ = qrilc_container

        with pytest.raises(ScpValueError, match="Quantile q must be between 0 and 1"):
            impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=-0.01,
            )

    def test_qrilc_invalid_q_greater_than_one(self, qrilc_container):
        """Test QRILC with q>1 raises error."""
        container, _, _ = qrilc_container

        with pytest.raises(ScpValueError, match="Quantile q must be between 0 and 1"):
            impute_qrilc(
                container,
                assay_name="protein",
                source_layer="raw",
                q=1.5,
            )

    def test_qrilc_assay_not_found(self, qrilc_container):
        """Test QRILC with non-existent assay."""
        container, _, _ = qrilc_container

        with pytest.raises(AssayNotFoundError):
            impute_qrilc(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_qrilc_layer_not_found(self, qrilc_container):
        """Test QRILC with non-existent layer."""
        container, _, _ = qrilc_container

        with pytest.raises(LayerNotFoundError):
            impute_qrilc(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestQRILCEdgeCases:
    """Test edge cases for QRILC."""

    def test_qrilc_all_missing(self, container_all_missing):
        """Test QRILC with all values missing."""
        container = container_all_missing()

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        # Should handle all missing case
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_qrilc_tiny_dataset(self, tiny_container):
        """Test QRILC with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_qrilc_single_feature(self):
        """Test QRILC with single feature."""
        np.random.seed(42)
        n_samples = 50
        n_features = 1

        X_true = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)
        X_missing = X_true.copy()
        missing_mask = np.random.rand(n_samples, n_features) < 0.2
        X_missing[missing_mask] = np.nan

        var = pl.DataFrame({"_index": ["prot_0"]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == (n_samples, n_features)

    def test_qrilc_few_detected_values(self):
        """Test QRILC with features having very few detected values."""
        np.random.seed(42)
        n_samples = 20
        n_features = 10

        # Create data where most features have very few detected values
        X_true = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)
        X_missing = X_true.copy()

        # Make 80% missing for each feature
        for feat_idx in range(n_features):
            missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=False)
            X_missing[missing_indices, feat_idx] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        # Should still impute all values
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_qrilc_constant_feature(self):
        """Test QRILC with feature having constant detected values."""
        np.random.seed(42)
        n_samples = 50
        n_features = 5

        X = np.random.randn(n_samples, n_features) * 0.5 + 2
        X[:, 0] = 5.0  # Constant feature

        # Add missing values
        missing_mask = np.random.rand(n_samples, n_features) < 0.2
        X[missing_mask] = np.nan

        # Ensure at least some detected values for constant feature
        X[0:10, 0] = 5.0
        missing_mask[0:10, 0] = False

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)


# =============================================================================
# Logging Tests
# =============================================================================


class TestQRILCLogging:
    """Test operation logging for QRILC."""

    def test_qrilc_logs_operation(self, qrilc_container):
        """Test that QRILC logs operation."""
        container, _, _ = qrilc_container
        initial_history_len = len(container.history)

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_qrilc"

    def test_qrilc_log_parameters(self, qrilc_container):
        """Test that QRILC logs parameters correctly."""
        container, _, _ = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.05,
        )

        log_entry = result.history[-1]
        assert log_entry.params["q"] == 0.05
        assert log_entry.params["assay"] == "protein"
        assert log_entry.params["source_layer"] == "raw"


# =============================================================================
# Distribution Preservation Tests
# =============================================================================


class TestQRILCDistributionPreservation:
    """Test that QRILC preserves data distribution characteristics."""

    def test_qrilc_preserves_mean_trend(self, qrilc_container):
        """Test that QRILC preserves mean trend across features."""
        container, X_true, missing_mask = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        # Compare feature means
        true_means = np.mean(X_true, axis=0)
        imputed_means = np.mean(X_imputed, axis=0)

        # Should have reasonable correlation
        correlation = np.corrcoef(true_means, imputed_means)[0, 1]
        assert correlation > 0.7  # Reasonable preservation

    def test_qrilc_no_negative_values(self, qrilc_container):
        """Test that QRILC never produces negative values."""
        container, _, _ = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        assert np.all(X_imputed >= 0)

    def test_qrilc_variance_reasonable(self, qrilc_container):
        """Test that QRILC produces reasonable variance."""
        container, X_true, missing_mask = qrilc_container

        result = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=0.01,
        )
        X_imputed = result.assays["protein"].layers["qrilc"].X

        # Imputed data should have finite variance
        imputed_var = np.var(X_imputed, axis=0)
        assert np.all(np.isfinite(imputed_var))
        assert np.all(imputed_var >= 0)
