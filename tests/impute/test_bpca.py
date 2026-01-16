"""
Comprehensive tests for BPCA (Bayesian PCA) imputation module.

Tests cover:
- Basic imputation functionality
- Different missing rates
- Sparse/dense matrices
- Parameter validation
- Edge cases (all missing, no missing, tiny data)
- Mask matrix updates
- Convergence behavior
- Random state reproducibility
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
from scptensor.impute import impute_bpca as bpca

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_bpca_container():
    """Factory function to create containers with missing data for BPCA testing."""

    def _create(
        n_samples=50,
        n_features=20,
        missing_rate=0.2,
        random_state=42,
        with_mask=True,
        use_sparse=False,
    ):
        np.random.seed(random_state)

        # Create correlated data (low-rank structure)
        U_true = np.random.randn(n_samples, 5)
        V_true = np.random.randn(n_features, 5)
        X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1

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
def bpca_container(create_bpca_container):
    """Standard test container with missing data."""
    return create_bpca_container(n_samples=50, n_features=20, missing_rate=0.2)


@pytest.fixture
def container_no_missing():
    """Container with no missing values."""

    def _create(use_sparse=False):
        np.random.seed(42)
        n_samples, n_features = 30, 15
        X = np.random.randn(n_samples, n_features)

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
        X = np.random.randn(n_samples, n_features)
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X[missing_mask] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))

        return ScpContainer(obs=obs, assays={"protein": assay})

    return _create


# =============================================================================
# BPCA Imputation Tests
# =============================================================================


class TestBPCAImputation:
    """Test BPCA imputation functionality."""

    def test_bpca_basic_imputation(self, bpca_container):
        """Test basic BPCA imputation."""
        container, X_true, missing_mask = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )

        assert "imputed_bpca" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_bpca"]
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

    def test_bpca_different_missing_rates(self, create_bpca_container):
        """Test BPCA with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_bpca_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_bpca"].X

            assert not np.any(np.isnan(X_imputed))
            # Imputed values should be in reasonable range
            assert np.all(np.isfinite(X_imputed))

    def test_bpca_sparse_matrix(self, create_bpca_container):
        """Test BPCA with sparse input matrix."""
        container, X_true, missing_mask = create_bpca_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_bpca_different_n_components(self, bpca_container):
        """Test BPCA with different n_components values."""
        container, _, _ = bpca_container

        for n_comp in [2, 5, 10]:
            result = bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=n_comp,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_bpca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_bpca_n_components_none(self, bpca_container):
        """Test BPCA with n_components=None (auto selection)."""
        container, _, _ = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=None,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        assert not np.any(np.isnan(X_imputed))

    def test_bpca_custom_max_iter(self, bpca_container):
        """Test BPCA with different max_iter values."""
        container, _, _ = bpca_container

        for max_iter in [10, 50, 100]:
            result = bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=max_iter,
            )
            X_imputed = result.assays["protein"].layers["imputed_bpca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_bpca_custom_tolerance(self, bpca_container):
        """Test BPCA with different tolerance values."""
        container, _, _ = bpca_container

        for tol in [1e-4, 1e-6, 1e-8]:
            result = bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
                tol=tol,
            )
            X_imputed = result.assays["protein"].layers["imputed_bpca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_bpca_custom_layer_name(self, bpca_container):
        """Test BPCA with custom output layer name."""
        container, _, _ = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="bpca_result",
        )

        assert "bpca_result" in result.assays["protein"].layers
        assert "imputed_bpca" not in result.assays["protein"].layers

    def test_bpca_no_missing_values(self, container_no_missing):
        """Test BPCA with no missing values."""
        container = container_no_missing()

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert "imputed_bpca" in result.assays["protein"].layers

    def test_bpca_existing_mask_update(self, create_bpca_container):
        """Test BPCA with existing mask matrix."""
        container, X_true, missing_mask = create_bpca_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        result_matrix = result.assays["protein"].layers["imputed_bpca"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_bpca_sparse_mask_update(self, create_bpca_container):
        """Test BPCA with sparse mask matrix."""
        container, X_true, missing_mask = create_bpca_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=False
        )

        # Create sparse mask
        M_sparse = sp_sparse.csr_matrix(
            container.assays["protein"].layers["raw"].X.shape, dtype=np.int8
        )
        container.assays["protein"].layers["raw"].M = M_sparse

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        M_imputed = result.assays["protein"].layers["imputed_bpca"].M

        # Check mask was created
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)

    def test_bpca_parameter_validation(self, bpca_container):
        """Test BPCA parameter validation."""
        container, _, _ = bpca_container

        # Invalid n_components
        with pytest.raises(ScpValueError, match="n_components must be positive"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=0,
            )

        with pytest.raises(ScpValueError, match="n_components must be positive"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=-5,
            )

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=0,
            )

        # Invalid tol
        with pytest.raises(ScpValueError, match="tol must be positive"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=0,
            )

        with pytest.raises(ScpValueError, match="tol must be positive"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=-1e-6,
            )

    def test_bpca_dimension_error(self, bpca_container):
        """Test BPCA with n_components too large."""
        container, _, _ = bpca_container

        # n_samples=50, n_features=20, so min is 20
        with pytest.raises(DimensionError, match="n_components.*must be less than"):
            bpca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=25,
            )

    def test_bpca_assay_not_found(self, bpca_container):
        """Test BPCA with non-existent assay."""
        container, _, _ = bpca_container

        with pytest.raises(AssayNotFoundError):
            bpca(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_bpca_layer_not_found(self, bpca_container):
        """Test BPCA with non-existent layer."""
        container, _, _ = bpca_container

        with pytest.raises(LayerNotFoundError):
            bpca(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_bpca_tiny_dataset(self, tiny_container):
        """Test BPCA with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=20,
        )
        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        assert not np.any(np.isnan(X_imputed))

    def test_bpca_random_state(self, bpca_container):
        """Test BPCA with random state for reproducibility."""
        container, _, _ = bpca_container

        result1 = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        result2 = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["imputed_bpca"].X
        X2 = result2.assays["protein"].layers["imputed_bpca"].X

        # Results should be identical with same random state
        np.testing.assert_array_almost_equal(X1, X2)

    def test_bpca_different_random_states(self, bpca_container):
        """Test BPCA accepts different random states."""
        container, _, _ = bpca_container

        # Just verify it runs with different random states
        result1 = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        result2 = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=123,
        )

        X1 = result1.assays["protein"].layers["imputed_bpca"].X
        X2 = result2.assays["protein"].layers["imputed_bpca"].X

        # Both should produce valid imputations
        assert not np.any(np.isnan(X1))
        assert not np.any(np.isnan(X2))

    def test_bpca_logging(self, bpca_container):
        """Test that BPCA logs operation."""
        container, _, _ = bpca_container
        initial_history_len = len(container.history)

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_bpca"

    def test_bpca_convergence(self, bpca_container):
        """Test that BPCA converges properly."""
        container, _, _ = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
            tol=1e-6,
        )

        X_imputed = result.assays["protein"].layers["imputed_bpca"].X
        assert not np.any(np.isnan(X_imputed))

    def test_bpca_effective_components_logged(self, bpca_container):
        """Test that effective components are logged."""
        container, _, _ = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=50,
        )

        # Check that effective_components is in the log
        log_entry = result.history[-1]
        assert "effective_components" in log_entry.params
        assert log_entry.params["effective_components"] <= log_entry.params["n_components"]

    def test_bpca_imputation_quality(self, create_bpca_container):
        """Test that BPCA produces reasonable imputation quality."""
        container, X_true, missing_mask = create_bpca_container(
            n_samples=100, n_features=50, missing_rate=0.2
        )

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=100,
        )

        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        # Check correlation on imputed values
        imputed_values = X_imputed[missing_mask]
        true_values = X_true[missing_mask]
        correlation = np.corrcoef(imputed_values, true_values)[0, 1]

        # BPCA should achieve reasonable correlation for low-rank data
        assert correlation > 0.5

    def test_bpca_preserves_observed_values(self, bpca_container):
        """Test that BPCA preserves observed values."""
        container, X_true, missing_mask = bpca_container

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        # Observed values should be preserved
        np.testing.assert_array_almost_equal(
            X_imputed[~missing_mask], X_true[~missing_mask], decimal=10
        )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestBPCAEdgeCases:
    """Test edge cases for BPCA imputation."""

    def test_all_missing_bpca(self, container_all_missing):
        """Test BPCA with all values missing."""
        container = container_all_missing()

        # BPCA should handle all missing case (filled with zeros/means)
        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_bpca"].X

        assert not np.any(np.isnan(X_imputed))

    def test_imputation_preserves_original_data(self, create_bpca_container):
        """Test that BPCA doesn't modify original data."""
        container, X_true, missing_mask = create_bpca_container()

        # Store original values
        original_X = container.assays["protein"].layers["raw"].X.copy()
        if hasattr(original_X, "toarray"):
            original_X = original_X.toarray()

        # Run imputation
        bpca(container, assay_name="protein", source_layer="raw", n_components=5)

        # Original should be unchanged
        X_after = container.assays["protein"].layers["raw"].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()

        np.testing.assert_array_equal(X_after, original_X)

    def test_high_missing_rate(self, create_bpca_container):
        """Test BPCA with very high missing rate (80%)."""
        container, X_true, missing_mask = create_bpca_container(
            n_samples=100, n_features=50, missing_rate=0.8
        )

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=10,
            max_iter=100,
        )

        X_imputed = result.assays["protein"].layers["imputed_bpca"].X
        assert not np.any(np.isnan(X_imputed))
        assert np.all(np.isfinite(X_imputed))

    def test_single_row_missing_values(self):
        """Test BPCA imputation when only one sample has missing values."""
        np.random.seed(42)
        n_samples, n_features = 20, 10

        # Create data where only first row has missing values
        X = np.random.randn(n_samples, n_features)
        X[0, [2, 5, 7]] = np.nan

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = bpca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        X_imputed = result.assays["protein"].layers["imputed_bpca"].X
        assert not np.any(np.isnan(X_imputed))
