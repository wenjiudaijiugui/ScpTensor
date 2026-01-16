"""
Comprehensive tests for MinProb and MinDet imputation methods.

Tests cover:
- MinProb probabilistic imputation
- MinDet deterministic imputation
- Different sigma values
- Parameter validation
- Edge cases (all missing, no missing, single sample)
- Mask matrix updates
- Random state reproducibility
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp_sparse

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.impute.minprob import impute_mindet, impute_minprob

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_minprob_container():
    """Factory function to create containers with MNAR missing data."""

    def _create(
        n_samples=50,
        n_features=20,
        missing_rate=0.2,
        random_state=42,
        with_mask=True,
        use_sparse=False,
        mnar=True,
    ):
        np.random.seed(random_state)

        if mnar:
            # Generate data with MNAR pattern - lower values more likely missing
            X_true = np.random.exponential(scale=10, size=(n_samples, n_features))

            # Introduce MNAR missingness: lower values more likely to be missing
            missing_prob = 1 / (1 + np.exp((X_true - 5) / 2))
            missing_mask = np.random.rand(n_samples, n_features) < missing_prob
        else:
            # Random missingness
            X_true = np.random.randn(n_samples, n_features) * 10 + 10
            missing_mask = np.random.rand(n_samples, n_features) < missing_rate

        X_missing = X_true.copy()
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
def minprob_container(create_minprob_container):
    """Standard test container with MNAR missing data."""
    return create_minprob_container(n_samples=50, n_features=20, missing_rate=0.2)


@pytest.fixture
def container_no_missing():
    """Container with no missing values."""

    def _create(use_sparse=False):
        np.random.seed(42)
        n_samples, n_features = 30, 15
        X = np.random.exponential(scale=10, size=(n_samples, n_features))

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
        n_samples, n_features = 20, 10
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

    def _create(n_samples=3, n_features=3, missing_rate=0.3):
        np.random.seed(42)
        X = np.random.exponential(scale=10, size=(n_samples, n_features))
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


# =============================================================================
# MinProb Imputation Tests
# =============================================================================


class TestMinProbImputation:
    """Test MinProb imputation functionality."""

    def test_minprob_basic_imputation(self, minprob_container):
        """Test basic MinProb imputation."""
        container, X_true, missing_mask = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
        )

        assert "imputed_minprob" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_minprob"]
        X_imputed = result_matrix.X
        M_imputed = result_matrix.M

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

        # Check mask was created and updated correctly
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

        # Check imputed values are positive
        assert np.all(X_imputed >= 0)

    def test_minprob_imputed_values_below_detection(self, minprob_container):
        """Test that MinProb imputes values below detection limit."""
        container, X_true, missing_mask = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )

        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        # For each feature, check that imputed values are below or near min detected
        for j in range(X_true.shape[1]):
            if np.any(missing_mask[:, j]):
                min_detected = np.min(X_true[~missing_mask[:, j], j])
                imputed_vals = X_imputed[missing_mask[:, j], j]
                # Imputed values should be near or below min detected
                assert np.mean(imputed_vals) <= min_detected * 1.2

    def test_minprob_different_missing_rates(self, create_minprob_container):
        """Test MinProb with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_minprob_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = impute_minprob(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=2.0,
            )
            X_imputed = result.assays["protein"].layers["imputed_minprob"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(np.isfinite(X_imputed))

    def test_minprob_sparse_matrix(self, create_minprob_container):
        """Test MinProb with sparse input matrix."""
        container, X_true, missing_mask = create_minprob_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_minprob_different_sigma_values(self, minprob_container):
        """Test MinProb with different sigma values."""
        container, _, _ = minprob_container

        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            result = impute_minprob(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=sigma,
            )
            X_imputed = result.assays["protein"].layers["imputed_minprob"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_minprob_custom_layer_name(self, minprob_container):
        """Test MinProb with custom output layer name."""
        container, _, _ = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="minprob_result",
        )

        assert "minprob_result" in result.assays["protein"].layers
        assert "imputed_minprob" not in result.assays["protein"].layers

    def test_minprob_no_missing_values(self, container_no_missing):
        """Test MinProb with no missing values."""
        container = container_no_missing()

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
        )

        assert "imputed_minprob" in result.assays["protein"].layers

    def test_minprob_existing_mask_update(self, create_minprob_container):
        """Test MinProb with existing mask matrix."""
        container, X_true, missing_mask = create_minprob_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
        )
        result_matrix = result.assays["protein"].layers["imputed_minprob"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_minprob_parameter_validation(self, minprob_container):
        """Test MinProb parameter validation."""
        container, _, _ = minprob_container

        # Invalid sigma (zero)
        with pytest.raises(ScpValueError, match="sigma must be positive"):
            impute_minprob(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=0,
            )

        # Invalid sigma (negative)
        with pytest.raises(ScpValueError, match="sigma must be positive"):
            impute_minprob(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=-1.0,
            )

    def test_minprob_assay_not_found(self, minprob_container):
        """Test MinProb with non-existent assay."""
        container, _, _ = minprob_container

        with pytest.raises(AssayNotFoundError):
            impute_minprob(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_minprob_layer_not_found(self, minprob_container):
        """Test MinProb with non-existent layer."""
        container, _, _ = minprob_container

        with pytest.raises(LayerNotFoundError):
            impute_minprob(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_minprob_tiny_dataset(self, tiny_container):
        """Test MinProb with very small dataset."""
        container = tiny_container(n_samples=3, n_features=3, missing_rate=0.3)

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert not np.any(np.isnan(X_imputed))

    def test_minprob_random_state_reproducibility(self, minprob_container):
        """Test MinProb with random state for reproducibility."""
        container, _, _ = minprob_container

        result1 = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )

        result2 = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["imputed_minprob"].X
        X2 = result2.assays["protein"].layers["imputed_minprob"].X

        # Results should be identical with same random state
        np.testing.assert_array_equal(X1, X2)

    def test_minprob_different_random_states(self, create_minprob_container):
        """Test MinProb with different random states produces different results."""
        # Create two fresh containers for this test
        container1, _, missing_mask = create_minprob_container()
        container2, _, _ = create_minprob_container()

        result1 = impute_minprob(
            container1,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )

        result2 = impute_minprob(
            container2,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=123,
        )

        X1 = result1.assays["protein"].layers["imputed_minprob"].X
        X2 = result2.assays["protein"].layers["imputed_minprob"].X

        # Results should be different with different random states
        # Check that at least some imputed values differ
        imputed_diff = X1[missing_mask] != X2[missing_mask]
        # At least 10% of imputed values should differ (allows for some coincidence)
        assert np.sum(imputed_diff) >= 0.1 * np.sum(missing_mask), (
            f"Different random states should produce different imputed values. "
            f"Only {np.sum(imputed_diff)}/{np.sum(missing_mask)} values differ."
        )

    def test_minprob_no_random_state(self, minprob_container):
        """Test MinProb without random state (should still work)."""
        container, _, _ = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=None,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert not np.any(np.isnan(X_imputed))

    def test_minprob_logging(self, minprob_container):
        """Test that MinProb logs operation."""
        container, _, _ = minprob_container
        initial_history_len = len(container.history)

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_minprob"


# =============================================================================
# MinDet Imputation Tests
# =============================================================================


class TestMinDetImputation:
    """Test MinDet imputation functionality."""

    def test_mindet_basic_imputation(self, minprob_container):
        """Test basic MinDet imputation."""
        container, X_true, missing_mask = minprob_container

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=1.0,
        )

        assert "imputed_mindet" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_mindet"]
        X_imputed = result_matrix.X
        M_imputed = result_matrix.M

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

        # Check mask was created and updated correctly
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

        # Check imputed values are positive
        assert np.all(X_imputed >= 0)

    def test_mindet_deterministic(self, minprob_container):
        """Test that MinDet is deterministic."""
        container, _, _ = minprob_container

        result1 = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=1.0,
        )

        result2 = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=1.0,
        )

        X1 = result1.assays["protein"].layers["imputed_mindet"].X
        X2 = result2.assays["protein"].layers["imputed_mindet"].X

        # Results should be identical (deterministic)
        np.testing.assert_array_equal(X1, X2)

    def test_mindet_same_value_per_feature(self, minprob_container):
        """Test that MinDet uses same value for all missing entries in a feature."""
        container, _, missing_mask = minprob_container

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=1.0,
        )

        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        # For each feature, all imputed values should be the same
        for j in range(X_imputed.shape[1]):
            if np.any(missing_mask[:, j]):
                imputed_vals = X_imputed[missing_mask[:, j], j]
                # All imputed values for this feature should be identical
                assert np.allclose(imputed_vals, imputed_vals[0])

    def test_mindet_different_missing_rates(self, create_minprob_container):
        """Test MinDet with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, _ = create_minprob_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = impute_mindet(
                container,
                assay_name="protein",
                source_layer="raw",
            )
            X_imputed = result.assays["protein"].layers["imputed_mindet"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(np.isfinite(X_imputed))

    def test_mindet_sparse_matrix(self, create_minprob_container):
        """Test MinDet with sparse input matrix."""
        container, X_true, missing_mask = create_minprob_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_mindet_different_sigma_values(self, minprob_container):
        """Test MinDet with different sigma values."""
        container, _, _ = minprob_container

        for sigma in [0.5, 1.0, 2.0, 3.0]:
            result = impute_mindet(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=sigma,
            )
            X_imputed = result.assays["protein"].layers["imputed_mindet"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_mindet_custom_layer_name(self, minprob_container):
        """Test MinDet with custom output layer name."""
        container, _, _ = minprob_container

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="mindet_result",
        )

        assert "mindet_result" in result.assays["protein"].layers
        assert "imputed_mindet" not in result.assays["protein"].layers

    def test_mindet_no_missing_values(self, container_no_missing):
        """Test MinDet with no missing values."""
        container = container_no_missing()

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )

        assert "imputed_mindet" in result.assays["protein"].layers

    def test_mindet_existing_mask_update(self, create_minprob_container):
        """Test MinDet with existing mask matrix."""
        container, X_true, missing_mask = create_minprob_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        result_matrix = result.assays["protein"].layers["imputed_mindet"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_mindet_parameter_validation(self, minprob_container):
        """Test MinDet parameter validation."""
        container, _, _ = minprob_container

        # Invalid sigma (zero)
        with pytest.raises(ScpValueError, match="sigma must be positive"):
            impute_mindet(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=0,
            )

        # Invalid sigma (negative)
        with pytest.raises(ScpValueError, match="sigma must be positive"):
            impute_mindet(
                container,
                assay_name="protein",
                source_layer="raw",
                sigma=-1.0,
            )

    def test_mindet_assay_not_found(self, minprob_container):
        """Test MinDet with non-existent assay."""
        container, _, _ = minprob_container

        with pytest.raises(AssayNotFoundError):
            impute_mindet(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_mindet_layer_not_found(self, minprob_container):
        """Test MinDet with non-existent layer."""
        container, _, _ = minprob_container

        with pytest.raises(LayerNotFoundError):
            impute_mindet(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_mindet_tiny_dataset(self, tiny_container):
        """Test MinDet with very small dataset."""
        container = tiny_container(n_samples=3, n_features=3, missing_rate=0.3)

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        assert not np.any(np.isnan(X_imputed))

    def test_mindet_logging(self, minprob_container):
        """Test that MinDet logs operation."""
        container, _, _ = minprob_container
        initial_history_len = len(container.history)

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_mindet"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestMinProbMinDetEdgeCases:
    """Test edge cases for MinProb and MinDet."""

    def test_all_missing_minprob(self, container_all_missing):
        """Test MinProb with all values missing."""
        container = container_all_missing()

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        # Should fill with small positive values
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_all_missing_mindet(self, container_all_missing):
        """Test MinDet with all values missing."""
        container = container_all_missing()

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        # Should fill with small positive value
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_single_sample_minprob(self):
        """Test MinProb with single sample."""
        np.random.seed(42)
        X = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(5)]})
        obs = pl.DataFrame({"_index": ["cell_1"]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            random_state=42,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert X_imputed.shape == (1, 5)
        assert not np.any(np.isnan(X_imputed))

    def test_single_sample_mindet(self):
        """Test MinDet with single sample."""
        np.random.seed(42)
        X = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(5)]})
        obs = pl.DataFrame({"_index": ["cell_1"]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        assert X_imputed.shape == (1, 5)
        assert not np.any(np.isnan(X_imputed))

    def test_single_feature_missing(self):
        """Test imputation with single feature having all missing values."""
        np.random.seed(42)
        n_samples, n_features = 50, 20
        X = np.random.exponential(scale=10, size=(n_samples, n_features))
        X[:, 0] = np.nan  # First feature all missing

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Test MinProb
        result_minprob = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            random_state=42,
        )
        X_minprob = result_minprob.assays["protein"].layers["imputed_minprob"].X
        assert not np.any(np.isnan(X_minprob[:, 0]))
        assert np.all(X_minprob[:, 0] >= 0)

        # Test MinDet
        result_mindet = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_mindet = result_mindet.assays["protein"].layers["imputed_mindet"].X
        assert not np.any(np.isnan(X_mindet[:, 0]))
        assert np.all(X_mindet[:, 0] >= 0)

    def test_comparison_minprob_vs_mindet(self, minprob_container):
        """Compare MinProb and MinDet results."""
        container, X_true, missing_mask = minprob_container

        result_minprob = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=2.0,
            random_state=42,
        )

        # Create fresh container for MinDet
        container2, _, _ = minprob_container
        result_mindet = impute_mindet(
            container2,
            assay_name="protein",
            source_layer="raw",
            sigma=1.0,
        )

        X_minprob = result_minprob.assays["protein"].layers["imputed_minprob"].X
        X_mindet = result_mindet.assays["protein"].layers["imputed_mindet"].X

        # Both should have no NaNs
        assert not np.any(np.isnan(X_minprob))
        assert not np.any(np.isnan(X_mindet))

        # For each feature, MinDet variance should be 0 (same value for all)
        for j in range(X_true.shape[1]):
            if np.any(missing_mask[:, j]):
                imputed_vals_mindet = X_mindet[missing_mask[:, j], j]
                assert np.var(imputed_vals_mindet) < 1e-10  # Essentially 0

    def test_original_data_preserved(self, create_minprob_container):
        """Test that imputation doesn't modify original data."""
        container, X_true, missing_mask = create_minprob_container()

        # Store original values
        original_X = container.assays["protein"].layers["raw"].X.copy()
        if hasattr(original_X, "toarray"):
            original_X = original_X.toarray()

        # Run imputation
        impute_minprob(container, assay_name="protein", source_layer="raw")

        # Original should be unchanged
        X_after = container.assays["protein"].layers["raw"].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()

        np.testing.assert_array_equal(X_after, original_X)

    def test_minprob_with_large_sigma(self, minprob_container):
        """Test MinProb with very large sigma value."""
        container, _, _ = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=10.0,
            random_state=42,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_mindet_with_large_sigma(self, minprob_container):
        """Test MinDet with very large sigma value."""
        container, _, _ = minprob_container

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=10.0,
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_minprob_with_small_sigma(self, minprob_container):
        """Test MinProb with very small sigma value."""
        container, _, _ = minprob_container

        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=0.1,
            random_state=42,
        )
        X_imputed = result.assays["protein"].layers["imputed_minprob"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_mindet_with_small_sigma(self, minprob_container):
        """Test MinDet with very small sigma value."""
        container, _, _ = minprob_container

        result = impute_mindet(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=0.1,
        )
        X_imputed = result.assays["protein"].layers["imputed_mindet"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_imputed_values_preserve_observed(self, minprob_container):
        """Test that observed values are not modified by imputation."""
        container, X_true, missing_mask = minprob_container

        result_minprob = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
        )
        X_minprob = result_minprob.assays["protein"].layers["imputed_minprob"].X

        # Observed values should be unchanged
        np.testing.assert_array_equal(X_minprob[~missing_mask], X_true[~missing_mask])

        # Same for MinDet
        container2, _, _ = minprob_container
        result_mindet = impute_mindet(
            container2,
            assay_name="protein",
            source_layer="raw",
        )
        X_mindet = result_mindet.assays["protein"].layers["imputed_mindet"].X

        np.testing.assert_array_equal(X_mindet[~missing_mask], X_true[~missing_mask])
