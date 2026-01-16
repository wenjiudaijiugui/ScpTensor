"""
Tests for LLS (Local Least Squares) imputation.

Tests cover:
- Basic imputation functionality
- Different missing rates
- Parameter validation (k, max_iter, tol)
- Edge cases (all missing, no missing, tiny data)
- Mask matrix updates
- Convergence behavior
- Reproducibility
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
from scptensor.impute import impute_lls

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_lls_container():
    """Factory function to create containers with missing data."""

    def _create(
        n_samples=50,
        n_features=20,
        missing_rate=0.2,
        random_state=42,
        with_mask=True,
        use_sparse=False,
    ):
        np.random.seed(random_state)

        # Create correlated data using low-rank structure
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
def lls_container(create_lls_container):
    """Standard test container with missing data."""
    return create_lls_container(n_samples=50, n_features=20, missing_rate=0.2)


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
# LLS Imputation Tests
# =============================================================================


class TestLLSImputation:
    """Test LLS imputation functionality."""

    def test_lls_basic_imputation(self, lls_container):
        """Test basic LLS imputation."""
        container, X_true, missing_mask = lls_container

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=10,
        )

        assert "imputed_lls" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_lls"]
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

        # Check original values preserved
        assert np.allclose(X_imputed[~missing_mask], X_true[~missing_mask])

    def test_lls_different_missing_rates(self, create_lls_container):
        """Test LLS with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_lls_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = impute_lls(
                container,
                assay_name="protein",
                source_layer="raw",
                k=10,
                max_iter=10,
            )
            X_imputed = result.assays["protein"].layers["imputed_lls"].X

            assert not np.any(np.isnan(X_imputed))
            # Imputed values should be finite
            assert np.all(np.isfinite(X_imputed))

    def test_lls_sparse_matrix(self, create_lls_container):
        """Test LLS with sparse input matrix."""
        container, X_true, missing_mask = create_lls_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_lls_different_k_values(self, lls_container):
        """Test LLS with different k values."""
        container, _, _ = lls_container

        for k_val in [1, 5, 10, 20]:
            result = impute_lls(
                container,
                assay_name="protein",
                source_layer="raw",
                k=k_val,
                max_iter=5,
            )
            X_imputed = result.assays["protein"].layers["imputed_lls"].X

            assert not np.any(np.isnan(X_imputed))

    def test_lls_custom_max_iter(self, lls_container):
        """Test LLS with different max_iter values."""
        container, _, _ = lls_container

        for max_iter in [1, 5, 20, 50]:
            result = impute_lls(
                container,
                assay_name="protein",
                source_layer="raw",
                k=10,
                max_iter=max_iter,
            )
            X_imputed = result.assays["protein"].layers["imputed_lls"].X

            assert not np.any(np.isnan(X_imputed))

    def test_lls_custom_tolerance(self, lls_container):
        """Test LLS with different tolerance values."""
        container, _, _ = lls_container

        for tol in [1e-4, 1e-6, 1e-8]:
            result = impute_lls(
                container,
                assay_name="protein",
                source_layer="raw",
                k=10,
                max_iter=50,
                tol=tol,
            )
            X_imputed = result.assays["protein"].layers["imputed_lls"].X

            assert not np.any(np.isnan(X_imputed))

    def test_lls_custom_layer_name(self, lls_container):
        """Test LLS with custom output layer name."""
        container, _, _ = lls_container

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="lls_result",
            k=10,
        )

        assert "lls_result" in result.assays["protein"].layers
        assert "imputed_lls" not in result.assays["protein"].layers

    def test_lls_no_missing_values(self, container_no_missing):
        """Test LLS with no missing values."""
        container = container_no_missing()

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
        )

        assert "imputed_lls" in result.assays["protein"].layers

    def test_lls_existing_mask_update(self, create_lls_container):
        """Test LLS with existing mask matrix."""
        container, X_true, missing_mask = create_lls_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=10,
        )
        result_matrix = result.assays["protein"].layers["imputed_lls"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_lls_sparse_mask_update(self, create_lls_container):
        """Test LLS with sparse mask matrix."""
        container, X_true, missing_mask = create_lls_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=False
        )

        # Create sparse mask
        M_sparse = sp_sparse.csr_matrix(
            container.assays["protein"].layers["raw"].X.shape, dtype=np.int8
        )
        container.assays["protein"].layers["raw"].M = M_sparse

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=10,
        )
        M_imputed = result.assays["protein"].layers["imputed_lls"].M

        # Check mask was created
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)

    def test_lls_parameter_validation(self, lls_container):
        """Test LLS parameter validation."""
        container, _, _ = lls_container

        # Invalid k
        with pytest.raises(ScpValueError, match="k.*must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", k=0)

        with pytest.raises(ScpValueError, match="k.*must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", k=-5)

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", max_iter=0)

        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", max_iter=-10)

        # Invalid tol
        with pytest.raises(ScpValueError, match="tol must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", tol=0)

        with pytest.raises(ScpValueError, match="tol must be positive"):
            impute_lls(container, assay_name="protein", source_layer="raw", tol=-1e-6)

    def test_lls_assay_not_found(self, lls_container):
        """Test LLS with non-existent assay."""
        container, _, _ = lls_container

        with pytest.raises(AssayNotFoundError):
            impute_lls(container, assay_name="nonexistent", source_layer="raw")

    def test_lls_layer_not_found(self, lls_container):
        """Test LLS with non-existent layer."""
        container, _, _ = lls_container

        with pytest.raises(LayerNotFoundError):
            impute_lls(container, assay_name="protein", source_layer="nonexistent")

    def test_lls_tiny_dataset(self, tiny_container):
        """Test LLS with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=2,
            max_iter=5,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        assert not np.any(np.isnan(X_imputed))

    def test_lls_logging(self, lls_container):
        """Test that LLS logs operation."""
        container, _, _ = lls_container
        initial_history_len = len(container.history)

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_lls"

    def test_lls_convergence(self, lls_container):
        """Test that LLS converges properly."""
        container, _, _ = lls_container

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=100,
            tol=1e-6,
        )

        X_imputed = result.assays["protein"].layers["imputed_lls"].X
        assert not np.any(np.isnan(X_imputed))

        # Check that iterations were logged
        assert result.history[-1].params["n_iterations"] > 0

    def test_lls_imputation_accuracy(self, create_lls_container):
        """Test LLS imputation accuracy on correlated data."""
        container, X_true, missing_mask = create_lls_container(
            n_samples=100, n_features=50, missing_rate=0.2
        )

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=20,
        )

        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        # Check correlation on imputed values
        imputed_values = X_imputed[missing_mask]
        true_values = X_true[missing_mask]
        correlation = np.corrcoef(imputed_values, true_values)[0, 1]

        # Should achieve reasonable correlation for correlated data
        assert correlation > 0.5, f"Expected correlation > 0.5, got {correlation:.3f}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestLLSEdgeCases:
    """Test edge cases for LLS imputation."""

    def test_all_missing_lls(self, container_all_missing):
        """Test LLS with all values missing."""
        container = container_all_missing()

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=2,
            max_iter=5,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        # Should fill with zeros (column means are all zero)
        assert not np.any(np.isnan(X_imputed))

    def test_single_row_imputation(self):
        """Test LLS with single row."""
        np.random.seed(42)
        X = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(5)]})
        obs = pl.DataFrame({"_index": ["cell_1"]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # LLS should handle single row gracefully
        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=1,
            max_iter=5,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        assert X_imputed.shape == (1, 5)

    def test_single_feature_imputation(self):
        """Test LLS with single feature."""
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])

        var = pl.DataFrame({"_index": ["prot_1"]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(5)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=2,
            max_iter=5,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == (5, 1)

    def test_preserves_original_data(self, create_lls_container):
        """Test that LLS doesn't modify original data."""
        container, X_true, missing_mask = create_lls_container()

        # Store original values
        original_X = container.assays["protein"].layers["raw"].X.copy()
        if hasattr(original_X, "toarray"):
            original_X = original_X.toarray()

        # Run imputation
        impute_lls(container, assay_name="protein", source_layer="raw", k=10)

        # Original should be unchanged
        X_after = container.assays["protein"].layers["raw"].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()

        np.testing.assert_array_equal(X_after, original_X)

    def test_high_missing_rate(self, create_lls_container):
        """Test LLS with very high missing rate (80%)."""
        container, X_true, missing_mask = create_lls_container(
            n_samples=50, n_features=20, missing_rate=0.8
        )

        result = impute_lls(
            container,
            assay_name="protein",
            source_layer="raw",
            k=10,
            max_iter=20,
        )
        X_imputed = result.assays["protein"].layers["imputed_lls"].X

        # Should still produce finite values
        assert not np.any(np.isnan(X_imputed))
        assert np.all(np.isfinite(X_imputed))
