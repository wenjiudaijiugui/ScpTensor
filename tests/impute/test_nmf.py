"""
Comprehensive tests for NMF (Non-negative Matrix Factorization) imputation module.

Tests cover:
- Basic imputation functionality
- Different missing rates
- Sparse/dense matrices
- Parameter validation
- Edge cases (all missing, no missing, tiny data)
- Mask matrix updates
- Convergence behavior
- Random state reproducibility
- Non-negative data constraints
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp_sparse

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.impute import impute_nmf

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_nmf_container():
    """Factory function to create containers with missing data for NMF testing."""

    def _create(
        n_samples=50,
        n_features=20,
        missing_rate=0.2,
        random_state=42,
        with_mask=True,
        use_sparse=False,
    ):
        np.random.seed(random_state)

        # Create correlated non-negative data (low-rank structure)
        U_true = np.abs(np.random.randn(n_samples, 5))
        V_true = np.abs(np.random.randn(n_features, 5))
        X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1
        X_true = np.maximum(X_true, 0)  # Ensure non-negative

        # Add missing values
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
def nmf_container(create_nmf_container):
    """Standard test container with missing data."""
    return create_nmf_container(n_samples=50, n_features=20, missing_rate=0.2)


@pytest.fixture
def container_no_missing():
    """Container with no missing values."""

    def _create(use_sparse=False):
        np.random.seed(42)
        n_samples, n_features = 30, 15
        X = np.abs(np.random.randn(n_samples, n_features))  # Non-negative

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
        X = np.abs(np.random.randn(n_samples, n_features))  # Non-negative
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


# =============================================================================
# NMF Imputation Tests
# =============================================================================


class TestNMFImputation:
    """Test NMF imputation functionality."""

    def test_nmf_basic_imputation(self, nmf_container):
        """Test basic NMF imputation."""
        container, X_true, missing_mask = nmf_container

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )

        assert "imputed_nmf" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_nmf"]
        X_imputed = result_matrix.X
        M_imputed = result_matrix.M

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

        # Check non-negativity of imputed values
        assert np.all(X_imputed >= 0)

        # Check mask was created and updated correctly
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_nmf_different_missing_rates(self, create_nmf_container):
        """Test NMF with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_nmf_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=100,
            )
            X_imputed = result.assays["protein"].layers["imputed_nmf"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)  # Non-negative constraint
            assert np.all(np.isfinite(X_imputed))

    def test_nmf_sparse_matrix(self, create_nmf_container):
        """Test NMF with sparse input matrix."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )
        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape
        assert np.all(X_imputed >= 0)

    def test_nmf_different_n_components(self, nmf_container):
        """Test NMF with different n_components values."""
        container, _, _ = nmf_container

        for n_comp in [2, 5, 10]:
            result = impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=n_comp,
                max_iter=100,
            )
            X_imputed = result.assays["protein"].layers["imputed_nmf"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_nmf_n_components_none(self, nmf_container):
        """Test NMF with n_components=None (auto selection)."""
        container, _, _ = nmf_container

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=None,
            max_iter=100,
        )
        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

        # Check that n_components was logged
        log_entry = result.history[-1]
        assert "n_components" in log_entry.params
        assert log_entry.params["n_components"] > 0

    def test_nmf_custom_max_iter(self, nmf_container):
        """Test NMF with different max_iter values."""
        container, _, _ = nmf_container

        for max_iter in [50, 100, 200]:
            result = impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=max_iter,
            )
            X_imputed = result.assays["protein"].layers["imputed_nmf"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_nmf_custom_tolerance(self, nmf_container):
        """Test NMF with different tolerance values."""
        container, _, _ = nmf_container

        for tol in [1e-5, 1e-4, 1e-3]:
            result = impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=100,
                tol=tol,
            )
            X_imputed = result.assays["protein"].layers["imputed_nmf"].X

            assert not np.any(np.isnan(X_imputed))
            assert np.all(X_imputed >= 0)

    def test_nmf_custom_layer_name(self, nmf_container):
        """Test NMF with custom output layer name."""
        container, _, _ = nmf_container

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="nmf_result",
        )

        assert "nmf_result" in result.assays["protein"].layers
        assert "imputed_nmf" not in result.assays["protein"].layers

    def test_nmf_no_missing_values(self, container_no_missing):
        """Test NMF with no missing values."""
        container = container_no_missing()

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert "imputed_nmf" in result.assays["protein"].layers

    def test_nmf_existing_mask_update(self, create_nmf_container):
        """Test NMF with existing mask matrix."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )
        result_matrix = result.assays["protein"].layers["imputed_nmf"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_nmf_sparse_mask_update(self, create_nmf_container):
        """Test NMF with sparse mask matrix."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=False
        )

        # Create sparse mask
        M_sparse = sp_sparse.csr_matrix(
            container.assays["protein"].layers["raw"].X.shape, dtype=np.int8
        )
        container.assays["protein"].layers["raw"].M = M_sparse

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )
        M_imputed = result.assays["protein"].layers["imputed_nmf"].M

        # Check mask was created
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)

    def test_nmf_parameter_validation(self, nmf_container):
        """Test NMF parameter validation."""
        container, _, _ = nmf_container

        # Invalid n_components
        with pytest.raises(ScpValueError, match="n_components must be positive"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=0,
            )

        with pytest.raises(ScpValueError, match="n_components must be positive"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=-5,
            )

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=0,
            )

        # Invalid tol
        with pytest.raises(ScpValueError, match="tol must be positive"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=0,
            )

        with pytest.raises(ScpValueError, match="tol must be positive"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=-1e-6,
            )

    def test_nmf_dimension_error(self, nmf_container):
        """Test NMF with n_components too large."""
        container, _, _ = nmf_container

        # n_samples=50, n_features=20, so min is 20
        with pytest.raises(DimensionError, match="n_components.*must be less than"):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=25,
            )

    def test_nmf_assay_not_found(self, nmf_container):
        """Test NMF with non-existent assay."""
        container, _, _ = nmf_container

        with pytest.raises(AssayNotFoundError):
            impute_nmf(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_nmf_layer_not_found(self, nmf_container):
        """Test NMF with non-existent layer."""
        container, _, _ = nmf_container

        with pytest.raises(LayerNotFoundError):
            impute_nmf(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_nmf_tiny_dataset(self, tiny_container):
        """Test NMF with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_nmf_random_state(self, nmf_container):
        """Test NMF with random state for reproducibility."""
        container, _, _ = nmf_container

        result1 = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        result2 = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["imputed_nmf"].X
        X2 = result2.assays["protein"].layers["imputed_nmf"].X

        # Results should be identical with same random state
        np.testing.assert_array_almost_equal(X1, X2)

    def test_nmf_different_random_states(self, nmf_container):
        """Test NMF accepts different random states."""
        container, _, _ = nmf_container

        # Just verify it runs with different random states
        result1 = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        result2 = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=123,
        )

        X1 = result1.assays["protein"].layers["imputed_nmf"].X
        X2 = result2.assays["protein"].layers["imputed_nmf"].X

        # Both should produce valid imputations
        assert not np.any(np.isnan(X1))
        assert not np.any(np.isnan(X2))
        assert np.all(X1 >= 0)
        assert np.all(X2 >= 0)

    def test_nmf_logging(self, nmf_container):
        """Test that NMF logs operation."""
        container, _, _ = nmf_container
        initial_history_len = len(container.history)

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_nmf"

    def test_nmf_convergence(self, nmf_container):
        """Test that NMF converges properly."""
        container, _, _ = nmf_container

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=200,
            tol=1e-4,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

        # Check that n_iterations is logged
        log_entry = result.history[-1]
        assert "n_iterations" in log_entry.params

    def test_nmf_imputation_quality(self, create_nmf_container):
        """Test that NMF produces reasonable imputation quality."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=100, n_features=50, missing_rate=0.2
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=200,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        # Check correlation on imputed values
        imputed_values = X_imputed[missing_mask]
        true_values = X_true[missing_mask]
        correlation = np.corrcoef(imputed_values, true_values)[0, 1]

        # NMF should achieve reasonable correlation for low-rank non-negative data
        assert correlation > 0.3

    def test_nmf_preserves_observed_values(self, nmf_container):
        """Test that NMF preserves observed values."""
        container, X_true, missing_mask = nmf_container

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        # Observed values should be preserved
        np.testing.assert_array_almost_equal(
            X_imputed[~missing_mask], X_true[~missing_mask], decimal=10
        )

    def test_nmf_non_negative_constraint(self, create_nmf_container):
        """Test that NMF maintains non-negative constraint."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=100, n_features=50, missing_rate=0.3
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=200,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        # All values should be non-negative
        assert np.all(X_imputed >= 0), "NMF imputation produced negative values"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestNMFEdgeCases:
    """Test edge cases for NMF imputation."""

    def test_all_missing_nmf(self, container_all_missing):
        """Test NMF with all values missing."""
        container = container_all_missing()

        # NMF should handle all missing case (filled with small values)
        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_nmf"].X

        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_imputation_preserves_original_data(self, create_nmf_container):
        """Test that NMF doesn't modify original data."""
        container, X_true, missing_mask = create_nmf_container()

        # Store original values
        original_X = container.assays["protein"].layers["raw"].X.copy()
        if hasattr(original_X, "toarray"):
            original_X = original_X.toarray()

        # Run imputation
        impute_nmf(container, assay_name="protein", source_layer="raw", n_components=5)

        # Original should be unchanged
        X_after = container.assays["protein"].layers["raw"].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()

        np.testing.assert_array_equal(X_after, original_X)

    def test_high_missing_rate(self, create_nmf_container):
        """Test NMF with very high missing rate (80%)."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=100, n_features=50, missing_rate=0.8
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=200,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(np.isfinite(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_single_row_missing_values(self):
        """Test NMF imputation when only one sample has missing values."""
        np.random.seed(42)
        n_samples, n_features = 20, 10

        # Create data where only first row has missing values
        X = np.abs(np.random.randn(n_samples, n_features))  # Non-negative
        X[0, [2, 5, 7]] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_single_column_missing_values(self):
        """Test NMF imputation when only one feature has missing values."""
        np.random.seed(42)
        n_samples, n_features = 20, 10

        # Create data where only one column has missing values
        X = np.abs(np.random.randn(n_samples, n_features))  # Non-negative
        X[[2, 5, 7], 0] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_very_small_missing_rate(self, create_nmf_container):
        """Test NMF with very small missing rate (1%)."""
        container, X_true, missing_mask = create_nmf_container(
            n_samples=50, n_features=20, missing_rate=0.01
        )

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(X_imputed >= 0)

    def test_nmf_with_negative_input_values(self):
        """Test NMF handles input with negative values by shifting."""
        np.random.seed(42)
        n_samples, n_features = 30, 15

        # Create data with negative values
        X = np.random.randn(n_samples, n_features)
        # Add some missing values
        missing_mask = np.random.rand(n_samples, n_features) < 0.2
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
        )

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))

    def test_nmf_auto_n_components_large_matrix(self):
        """Test auto n_components selection for large matrices."""
        np.random.seed(42)
        n_samples, n_features = 100, 80

        X = np.abs(np.random.randn(n_samples, n_features))
        missing_mask = np.random.rand(n_samples, n_features) < 0.2
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=None,  # Auto selection
            max_iter=100,
        )

        # min(100, 80) = 80, so auto should use 80 // 4 = 20
        log_entry = result.history[-1]
        assert log_entry.params["n_components"] == 20

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))

    def test_nmf_auto_n_components_small_matrix(self):
        """Test auto n_components selection for small matrices."""
        np.random.seed(42)
        n_samples, n_features = 20, 15

        X = np.abs(np.random.randn(n_samples, n_features))
        missing_mask = np.random.rand(n_samples, n_features) < 0.2
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=None,  # Auto selection
            max_iter=100,
        )

        # min(20, 15) = 15, so auto should use 15 // 2 = 7 (or max(2, ...) = 7)
        log_entry = result.history[-1]
        assert log_entry.params["n_components"] == 7

        X_imputed = result.assays["protein"].layers["imputed_nmf"].X
        assert not np.any(np.isnan(X_imputed))
