"""
Comprehensive tests for impute module.

Tests cover:
- knn imputation
- missforest imputation
- ppca imputation
- svd imputation

Each method is tested with:
- Normal cases
- Different missing rates
- Sparse/dense matrices
- Parameter validation
- Edge cases (all missing, no missing, tiny data)
- Mask matrix updates
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
from scptensor.impute import impute_knn as knn
from scptensor.impute import impute_mf as missforest
from scptensor.impute import impute_ppca as ppca
from scptensor.impute import impute_svd as svd_impute

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_impute_container():
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

        # Create correlated data
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
def impute_container(create_impute_container):
    """Standard test container with missing data."""
    return create_impute_container(n_samples=50, n_features=20, missing_rate=0.2)


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
# KNN Imputation Tests
# =============================================================================


class TestKNNImputation:
    """Test KNN imputation functionality."""

    def test_knn_basic_imputation(self, impute_container):
        """Test basic KNN imputation."""
        container, X_true, missing_mask = impute_container

        result = knn(container, assay_name="protein", source_layer="raw", k=5)

        assert "imputed_knn" in result.assays["protein"].layers
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

        # Check original values preserved
        assert np.allclose(X_imputed[~missing_mask], X_true[~missing_mask])

    def test_knn_different_missing_rates(self, create_impute_container):
        """Test KNN with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5, 0.8]

        for rate in missing_rates:
            container, X_true, missing_mask = create_impute_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = knn(container, assay_name="protein", source_layer="raw", k=5)
            X_imputed = result.assays["protein"].layers["imputed_knn"].X

            assert not np.any(np.isnan(X_imputed))
            # Imputed values should be in reasonable range
            assert np.all(np.isfinite(X_imputed))

    @pytest.mark.xfail(
        reason="KNN with sparse matrices containing NaN is not supported - "
        "nan_euclidean_distances requires dense input"
    )
    def test_knn_sparse_matrix(self, create_impute_container):
        """Test KNN with sparse input matrix.

        Note: Sparse matrices with NaN values are problematic for sklearn.
        The nan_euclidean_distances function requires dense arrays.
        This test is marked as xfail to document this known limitation.
        """
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = knn(container, assay_name="protein", source_layer="raw", k=5)
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_knn_uniform_weights(self, impute_container):
        """Test KNN with uniform weights."""
        container, _, _ = impute_container

        result = knn(container, assay_name="protein", source_layer="raw", k=5, weights="uniform")
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        assert not np.any(np.isnan(X_imputed))

    def test_knn_distance_weights(self, impute_container):
        """Test KNN with distance weights."""
        container, _, _ = impute_container

        result = knn(container, assay_name="protein", source_layer="raw", k=5, weights="distance")
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        assert not np.any(np.isnan(X_imputed))

    def test_knn_different_k_values(self, impute_container):
        """Test KNN with different k values."""
        container, _, _ = impute_container

        for k_val in [1, 3, 5, 10, 20]:
            result = knn(container, assay_name="protein", source_layer="raw", k=k_val)
            X_imputed = result.assays["protein"].layers["imputed_knn"].X

            assert not np.any(np.isnan(X_imputed))

    def test_knn_custom_layer_name(self, impute_container):
        """Test KNN with custom output layer name."""
        container, _, _ = impute_container

        result = knn(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="my_imputed",
        )

        assert "my_imputed" in result.assays["protein"].layers
        assert "imputed_knn" not in result.assays["protein"].layers

    def test_knn_no_missing_values(self, container_no_missing):
        """Test KNN with no missing values."""
        container = container_no_missing()

        result = knn(container, assay_name="protein", source_layer="raw", k=5)

        assert "imputed_knn" in result.assays["protein"].layers

    def test_knn_mask_preserved(self, create_impute_container):
        """Test that KNN creates and updates mask matrix correctly."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        M_original = container.assays["protein"].layers["raw"].M

        result = knn(container, assay_name="protein", source_layer="raw", k=5)
        M_result = result.assays["protein"].layers["imputed_knn"].M

        # Mask should be created with IMPUTED (5) codes for missing values
        assert M_result is not None
        # All originally missing values should now have IMPUTED code
        assert np.all(M_result[missing_mask] == 5)  # IMPUTED code
        # Non-missing values should keep their original codes (VALID=0, MBR=1, LOD=2)
        assert np.array_equal(M_result[~missing_mask], M_original[~missing_mask])

    def test_knn_parameter_validation(self, impute_container):
        """Test KNN parameter validation."""
        container, _, _ = impute_container

        # Invalid k
        with pytest.raises(ScpValueError, match="k.*must be positive"):
            knn(container, assay_name="protein", source_layer="raw", k=0)

        with pytest.raises(ScpValueError, match="k.*must be positive"):
            knn(container, assay_name="protein", source_layer="raw", k=-5)

        # Invalid weights
        with pytest.raises(ScpValueError, match="Weights must be"):
            knn(container, assay_name="protein", source_layer="raw", weights="invalid")

        # Invalid batch_size
        with pytest.raises(ScpValueError, match="Batch size must be positive"):
            knn(container, assay_name="protein", source_layer="raw", batch_size=0)

        # Invalid oversample_factor
        with pytest.raises(ScpValueError, match="Oversample factor must be at least"):
            knn(container, assay_name="protein", source_layer="raw", oversample_factor=0)

    def test_knn_assay_not_found(self, impute_container):
        """Test KNN with non-existent assay."""
        container, _, _ = impute_container

        with pytest.raises(AssayNotFoundError):
            knn(container, assay_name="nonexistent", source_layer="raw")

    def test_knn_layer_not_found(self, impute_container):
        """Test KNN with non-existent layer."""
        container, _, _ = impute_container

        with pytest.raises(LayerNotFoundError):
            knn(container, assay_name="protein", source_layer="nonexistent")

    def test_knn_tiny_dataset(self, tiny_container):
        """Test KNN with very small dataset."""
        container = tiny_container(n_samples=3, n_features=3, missing_rate=0.3)

        result = knn(container, assay_name="protein", source_layer="raw", k=2)
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        assert not np.any(np.isnan(X_imputed))

    def test_knn_logging(self, impute_container):
        """Test that KNN logs operation."""
        container, _, _ = impute_container
        initial_history_len = len(container.history)

        result = knn(container, assay_name="protein", source_layer="raw", k=5)

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_knn"

    def test_knn_oversample_factor(self, impute_container):
        """Test KNN with different oversample factors."""
        container, _, _ = impute_container

        for factor in [1, 2, 5]:
            result = knn(
                container,
                assay_name="protein",
                source_layer="raw",
                k=5,
                oversample_factor=factor,
            )
            X_imputed = result.assays["protein"].layers["imputed_knn"].X

            assert not np.any(np.isnan(X_imputed))

    def test_knn_batch_processing(self, impute_container):
        """Test KNN with different batch sizes."""
        container, _, _ = impute_container

        for batch_size in [10, 50, 100]:
            result = knn(
                container,
                assay_name="protein",
                source_layer="raw",
                k=5,
                batch_size=batch_size,
            )
            X_imputed = result.assays["protein"].layers["imputed_knn"].X

            assert not np.any(np.isnan(X_imputed))


# =============================================================================
# MissForest Imputation Tests
# =============================================================================


class TestMissForestImputation:
    """Test MissForest imputation functionality."""

    def test_missforest_basic_imputation(self, impute_container):
        """Test basic MissForest imputation."""
        container, X_true, missing_mask = impute_container

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=5,
            n_estimators=50,
        )

        assert "imputed_missforest" in result.assays["protein"].layers
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        # Check no NaNs remain
        assert not np.any(np.isnan(X_imputed))

        # Check shape preserved
        assert X_imputed.shape == X_true.shape

    def test_missforest_different_missing_rates(self, create_impute_container):
        """Test MissForest with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, _ = create_impute_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=3,
                n_estimators=30,
            )
            X_imputed = result.assays["protein"].layers["imputed_missforest"].X

            assert not np.any(np.isnan(X_imputed))

    def test_missforest_custom_n_estimators(self, impute_container):
        """Test MissForest with different n_estimators."""
        container, _, _ = impute_container

        for n_est in [10, 50, 100]:
            result = missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=3,
                n_estimators=n_est,
            )
            X_imputed = result.assays["protein"].layers["imputed_missforest"].X

            assert not np.any(np.isnan(X_imputed))

    def test_missforest_custom_max_depth(self, impute_container):
        """Test MissForest with custom max_depth."""
        container, _, _ = impute_container

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=50,
            max_depth=5,
        )
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        assert not np.any(np.isnan(X_imputed))

    def test_missforest_max_depth_none(self, impute_container):
        """Test MissForest with max_depth=None (unlimited)."""
        container, _, _ = impute_container

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=50,
            max_depth=None,
        )
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        assert not np.any(np.isnan(X_imputed))

    def test_missforest_custom_max_iter(self, impute_container):
        """Test MissForest with different max_iter values."""
        container, _, _ = impute_container

        for max_iter in [1, 3, 10]:
            result = missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=max_iter,
                n_estimators=30,
            )
            X_imputed = result.assays["protein"].layers["imputed_missforest"].X

            assert not np.any(np.isnan(X_imputed))

    def test_missforest_custom_layer_name(self, impute_container):
        """Test MissForest with custom output layer name."""
        container, _, _ = impute_container

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="mf_imputed",
        )

        assert "mf_imputed" in result.assays["protein"].layers
        assert "imputed_missforest" not in result.assays["protein"].layers

    def test_missforest_no_missing_values(self, container_no_missing):
        """Test MissForest with no missing values."""
        container = container_no_missing()

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=30,
        )

        assert "imputed_missforest" in result.assays["protein"].layers

    def test_missforest_mask_preserved(self, create_impute_container):
        """Test that MissForest creates and updates mask matrix correctly."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        M_original = container.assays["protein"].layers["raw"].M

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=30,
        )
        M_result = result.assays["protein"].layers["imputed_missforest"].M

        # Mask should be created with IMPUTED (5) codes for missing values
        assert M_result is not None
        # All originally missing values should now have IMPUTED code
        assert np.all(M_result[missing_mask] == 5)  # IMPUTED code
        # Non-missing values should keep their original codes (VALID=0, MBR=1, LOD=2)
        assert np.array_equal(M_result[~missing_mask], M_original[~missing_mask])

    def test_missforest_parameter_validation(self, impute_container):
        """Test MissForest parameter validation."""
        container, _, _ = impute_container

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=0,
            )

        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=-5,
            )

        # Invalid n_estimators
        with pytest.raises(ScpValueError, match="n_estimators must be positive"):
            missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                n_estimators=0,
            )

        # Invalid max_depth
        with pytest.raises(ScpValueError, match="max_depth must be positive"):
            missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                max_depth=0,
            )

        # Invalid verbose
        with pytest.raises(ScpValueError, match="verbose must be 0, 1, or 2"):
            missforest(
                container,
                assay_name="protein",
                source_layer="raw",
                verbose=5,
            )

    def test_missforest_assay_not_found(self, impute_container):
        """Test MissForest with non-existent assay."""
        container, _, _ = impute_container

        with pytest.raises(AssayNotFoundError):
            missforest(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_missforest_layer_not_found(self, impute_container):
        """Test MissForest with non-existent layer."""
        container, _, _ = impute_container

        with pytest.raises(LayerNotFoundError):
            missforest(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_missforest_tiny_dataset(self, tiny_container):
        """Test MissForest with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=2,
            n_estimators=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        assert not np.any(np.isnan(X_imputed))

    def test_missforest_random_state(self, impute_container):
        """Test MissForest with random state for reproducibility."""
        container, _, _ = impute_container

        result1 = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=30,
            random_state=42,
        )

        result2 = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=30,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["imputed_missforest"].X
        X2 = result2.assays["protein"].layers["imputed_missforest"].X

        # Results should be identical with same random state
        np.testing.assert_array_almost_equal(X1, X2)

    def test_missforest_verbose_output(self, impute_container, capsys):
        """Test MissForest verbose output."""
        container, _, _ = impute_container

        missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=2,
            n_estimators=30,
            verbose=1,
        )

        captured = capsys.readouterr()
        assert "MissForest iteration" in captured.out

    def test_missforest_logging(self, impute_container):
        """Test that MissForest logs operation."""
        container, _, _ = impute_container
        initial_history_len = len(container.history)

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=3,
            n_estimators=30,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_missforest"

    def test_missforest_n_jobs(self, impute_container):
        """Test MissForest with different n_jobs values."""
        container, _, _ = impute_container

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=2,
            n_estimators=30,
            n_jobs=1,
        )
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        assert not np.any(np.isnan(X_imputed))


# =============================================================================
# PPCA Imputation Tests
# =============================================================================


class TestPPCAImputation:
    """Test PPCA imputation functionality."""

    def test_ppca_basic_imputation(self, impute_container):
        """Test basic PPCA imputation."""
        container, X_true, missing_mask = impute_container

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )

        assert "imputed_ppca" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_ppca"]
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

    def test_ppca_different_missing_rates(self, create_impute_container):
        """Test PPCA with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_impute_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_ppca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_ppca_sparse_matrix(self, create_impute_container):
        """Test PPCA with sparse input matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_ppca"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_ppca_different_n_components(self, impute_container):
        """Test PPCA with different n_components values."""
        container, _, _ = impute_container

        for n_comp in [2, 5, 10]:
            result = ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=n_comp,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_ppca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_ppca_custom_max_iter(self, impute_container):
        """Test PPCA with different max_iter values."""
        container, _, _ = impute_container

        for max_iter in [10, 50, 100]:
            result = ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=max_iter,
            )
            X_imputed = result.assays["protein"].layers["imputed_ppca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_ppca_custom_tolerance(self, impute_container):
        """Test PPCA with different tolerance values."""
        container, _, _ = impute_container

        for tol in [1e-4, 1e-6, 1e-8]:
            result = ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
                tol=tol,
            )
            X_imputed = result.assays["protein"].layers["imputed_ppca"].X

            assert not np.any(np.isnan(X_imputed))

    def test_ppca_custom_layer_name(self, impute_container):
        """Test PPCA with custom output layer name."""
        container, _, _ = impute_container

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="ppca_result",
        )

        assert "ppca_result" in result.assays["protein"].layers
        assert "imputed_ppca" not in result.assays["protein"].layers

    def test_ppca_no_missing_values(self, container_no_missing):
        """Test PPCA with no missing values."""
        container = container_no_missing()

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert "imputed_ppca" in result.assays["protein"].layers

    def test_ppca_existing_mask_update(self, create_impute_container):
        """Test PPCA with existing mask matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        result_matrix = result.assays["protein"].layers["imputed_ppca"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_ppca_sparse_mask_update(self, create_impute_container):
        """Test PPCA with sparse mask matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=False
        )

        # Create sparse mask
        M_sparse = sp_sparse.csr_matrix(
            container.assays["protein"].layers["raw"].X.shape, dtype=np.int8
        )
        container.assays["protein"].layers["raw"].M = M_sparse

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        M_imputed = result.assays["protein"].layers["imputed_ppca"].M

        # Check mask was created
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)

    def test_ppca_parameter_validation(self, impute_container):
        """Test PPCA parameter validation."""
        container, _, _ = impute_container

        # Invalid n_components
        with pytest.raises(ScpValueError, match="n_components must be positive"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=0,
            )

        with pytest.raises(ScpValueError, match="n_components must be positive"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=-5,
            )

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=0,
            )

        # Invalid tol
        with pytest.raises(ScpValueError, match="tol must be positive"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=0,
            )

        with pytest.raises(ScpValueError, match="tol must be positive"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=-1e-6,
            )

    def test_ppca_dimension_error(self, impute_container):
        """Test PPCA with n_components too large."""
        container, _, _ = impute_container

        # n_samples=50, n_features=20, so min is 20
        with pytest.raises(DimensionError, match="n_components.*must be less than"):
            ppca(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=25,
            )

    def test_ppca_assay_not_found(self, impute_container):
        """Test PPCA with non-existent assay."""
        container, _, _ = impute_container

        with pytest.raises(AssayNotFoundError):
            ppca(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_ppca_layer_not_found(self, impute_container):
        """Test PPCA with non-existent layer."""
        container, _, _ = impute_container

        with pytest.raises(LayerNotFoundError):
            ppca(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_ppca_tiny_dataset(self, tiny_container):
        """Test PPCA with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=20,
        )
        X_imputed = result.assays["protein"].layers["imputed_ppca"].X

        assert not np.any(np.isnan(X_imputed))

    def test_ppca_random_state(self, impute_container):
        """Test PPCA with random state for reproducibility."""
        container, _, _ = impute_container

        result1 = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        result2 = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            random_state=42,
        )

        X1 = result1.assays["protein"].layers["imputed_ppca"].X
        X2 = result2.assays["protein"].layers["imputed_ppca"].X

        # Results should be identical with same random state
        np.testing.assert_array_almost_equal(X1, X2)

    def test_ppca_logging(self, impute_container):
        """Test that PPCA logs operation."""
        container, _, _ = impute_container
        initial_history_len = len(container.history)

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_ppca"

    def test_ppca_convergence(self, impute_container):
        """Test that PPCA converges properly."""
        container, _, _ = impute_container

        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
            tol=1e-6,
        )

        X_imputed = result.assays["protein"].layers["imputed_ppca"].X
        assert not np.any(np.isnan(X_imputed))


# =============================================================================
# SVD Imputation Tests
# =============================================================================


class TestSVDImputation:
    """Test SVD imputation functionality."""

    def test_svd_basic_imputation(self, impute_container):
        """Test basic SVD imputation."""
        container, X_true, missing_mask = impute_container

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )

        assert "imputed_svd" in result.assays["protein"].layers
        result_matrix = result.assays["protein"].layers["imputed_svd"]
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

    def test_svd_different_missing_rates(self, create_impute_container):
        """Test SVD with different missing rates."""
        missing_rates = [0.05, 0.2, 0.5]

        for rate in missing_rates:
            container, X_true, missing_mask = create_impute_container(
                n_samples=50, n_features=20, missing_rate=rate
            )

            result = svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_svd"].X

            assert not np.any(np.isnan(X_imputed))

    def test_svd_sparse_matrix(self, create_impute_container):
        """Test SVD with sparse input matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, use_sparse=True
        )

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        X_imputed = result.assays["protein"].layers["imputed_svd"].X

        assert not np.any(np.isnan(X_imputed))
        assert X_imputed.shape == X_true.shape

    def test_svd_different_n_components(self, impute_container):
        """Test SVD with different n_components values."""
        container, _, _ = impute_container

        for n_comp in [2, 5, 10]:
            result = svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=n_comp,
                max_iter=50,
            )
            X_imputed = result.assays["protein"].layers["imputed_svd"].X

            assert not np.any(np.isnan(X_imputed))

    def test_svd_custom_max_iter(self, impute_container):
        """Test SVD with different max_iter values."""
        container, _, _ = impute_container

        for max_iter in [10, 50, 100]:
            result = svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=max_iter,
            )
            X_imputed = result.assays["protein"].layers["imputed_svd"].X

            assert not np.any(np.isnan(X_imputed))

    def test_svd_custom_tolerance(self, impute_container):
        """Test SVD with different tolerance values."""
        container, _, _ = impute_container

        for tol in [1e-4, 1e-6, 1e-8]:
            result = svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=5,
                max_iter=50,
                tol=tol,
            )
            X_imputed = result.assays["protein"].layers["imputed_svd"].X

            assert not np.any(np.isnan(X_imputed))

    def test_svd_mean_init(self, impute_container):
        """Test SVD with mean initialization (default)."""
        container, _, _ = impute_container

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            init_method="mean",
        )
        X_imputed = result.assays["protein"].layers["imputed_svd"].X

        assert not np.any(np.isnan(X_imputed))

    def test_svd_median_init(self, impute_container):
        """Test SVD with median initialization."""
        container, _, _ = impute_container

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            init_method="median",
        )
        X_imputed = result.assays["protein"].layers["imputed_svd"].X

        assert not np.any(np.isnan(X_imputed))

    def test_svd_custom_layer_name(self, impute_container):
        """Test SVD with custom output layer name."""
        container, _, _ = impute_container

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="svd_result",
        )

        assert "svd_result" in result.assays["protein"].layers
        assert "imputed_svd" not in result.assays["protein"].layers

    def test_svd_no_missing_values(self, container_no_missing):
        """Test SVD with no missing values."""
        container = container_no_missing()

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert "imputed_svd" in result.assays["protein"].layers

    def test_svd_existing_mask_update(self, create_impute_container):
        """Test SVD with existing mask matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=True
        )

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        result_matrix = result.assays["protein"].layers["imputed_svd"]
        M_imputed = result_matrix.M

        # Check that imputed values now have IMPUTED (5) code
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Check that valid values remain VALID (0)
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_svd_sparse_mask_update(self, create_impute_container):
        """Test SVD with sparse mask matrix."""
        container, X_true, missing_mask = create_impute_container(
            n_samples=50, n_features=20, missing_rate=0.2, with_mask=False
        )

        # Create sparse mask
        M_sparse = sp_sparse.csr_matrix(
            container.assays["protein"].layers["raw"].X.shape, dtype=np.int8
        )
        container.assays["protein"].layers["raw"].M = M_sparse

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=50,
        )
        M_imputed = result.assays["protein"].layers["imputed_svd"].M

        # Check mask was created
        assert M_imputed is not None
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)

    def test_svd_parameter_validation(self, impute_container):
        """Test SVD parameter validation."""
        container, _, _ = impute_container

        # Invalid n_components
        with pytest.raises(ScpValueError, match="n_components must be positive"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=0,
            )

        with pytest.raises(ScpValueError, match="n_components must be positive"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=-5,
            )

        # Invalid max_iter
        with pytest.raises(ScpValueError, match="max_iter must be positive"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                max_iter=0,
            )

        # Invalid tol
        with pytest.raises(ScpValueError, match="tol must be positive"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=0,
            )

        with pytest.raises(ScpValueError, match="tol must be positive"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                tol=-1e-6,
            )

        # Invalid init_method
        with pytest.raises(ScpValueError, match="init_method must be"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                init_method="invalid",
            )

    def test_svd_dimension_error(self, impute_container):
        """Test SVD with n_components too large."""
        container, _, _ = impute_container

        # n_samples=50, n_features=20, so min is 20
        with pytest.raises(DimensionError, match="n_components.*must be less than"):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="raw",
                n_components=25,
            )

    def test_svd_assay_not_found(self, impute_container):
        """Test SVD with non-existent assay."""
        container, _, _ = impute_container

        with pytest.raises(AssayNotFoundError):
            svd_impute(
                container,
                assay_name="nonexistent",
                source_layer="raw",
            )

    def test_svd_layer_not_found(self, impute_container):
        """Test SVD with non-existent layer."""
        container, _, _ = impute_container

        with pytest.raises(LayerNotFoundError):
            svd_impute(
                container,
                assay_name="protein",
                source_layer="nonexistent",
            )

    def test_svd_tiny_dataset(self, tiny_container):
        """Test SVD with very small dataset."""
        container = tiny_container(n_samples=5, n_features=5, missing_rate=0.3)

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=20,
        )
        X_imputed = result.assays["protein"].layers["imputed_svd"].X

        assert not np.any(np.isnan(X_imputed))

    def test_svd_logging(self, impute_container):
        """Test that SVD logs operation."""
        container, _, _ = impute_container
        initial_history_len = len(container.history)

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
        )

        assert len(result.history) == initial_history_len + 1
        assert result.history[-1].action == "impute_svd"

    def test_svd_convergence(self, impute_container):
        """Test that SVD converges properly."""
        container, _, _ = impute_container

        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=100,
            tol=1e-6,
        )

        X_imputed = result.assays["protein"].layers["imputed_svd"].X
        assert not np.any(np.isnan(X_imputed))


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestImputationEdgeCases:
    """Test edge cases across all imputation methods."""

    def test_all_missing_knn(self, container_all_missing):
        """Test KNN with all values missing."""
        container = container_all_missing()

        result = knn(container, assay_name="protein", source_layer="raw", k=2)
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        # Should fill with column means (zeros since all NaN)
        assert not np.any(np.isnan(X_imputed))

    def test_all_missing_missforest(self, container_all_missing):
        """Test MissForest with all values missing."""
        container = container_all_missing()

        result = missforest(
            container,
            assay_name="protein",
            source_layer="raw",
            max_iter=2,
            n_estimators=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_missforest"].X

        # Should handle all missing case
        assert not np.any(np.isnan(X_imputed))

    def test_all_missing_ppca(self, container_all_missing):
        """Test PPCA with all values missing."""
        container = container_all_missing()

        # This might fail or produce zeros
        result = ppca(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_ppca"].X

        # Should handle all missing case (filled with zeros/means)
        assert not np.any(np.isnan(X_imputed))

    def test_all_missing_svd(self, container_all_missing):
        """Test SVD with all values missing."""
        container = container_all_missing()

        # This might fail or produce zeros
        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=2,
            max_iter=10,
        )
        X_imputed = result.assays["protein"].layers["imputed_svd"].X

        # Should handle all missing case (filled with zeros/means)
        assert not np.any(np.isnan(X_imputed))

    def test_single_row_imputation(self):
        """Test imputation with single row."""
        np.random.seed(42)
        X = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(5)]})
        obs = pl.DataFrame({"_index": ["cell_1"]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # KNN should handle single row gracefully
        result = knn(container, assay_name="protein", source_layer="raw", k=1)
        X_imputed = result.assays["protein"].layers["imputed_knn"].X

        assert X_imputed.shape == (1, 5)

    def test_imputation_preserves_original_data(self, create_impute_container):
        """Test that imputation doesn't modify original data."""
        container, X_true, missing_mask = create_impute_container()

        # Store original values
        original_X = container.assays["protein"].layers["raw"].X.copy()
        if hasattr(original_X, "toarray"):
            original_X = original_X.toarray()

        # Run imputation
        knn(container, assay_name="protein", source_layer="raw", k=5)

        # Original should be unchanged
        X_after = container.assays["protein"].layers["raw"].X
        if hasattr(X_after, "toarray"):
            X_after = X_after.toarray()

        np.testing.assert_array_equal(X_after, original_X)

    def test_imputation_with_large_values(self):
        """Test imputation with very large values (but finite)."""
        np.random.seed(42)
        X = np.random.randn(20, 10) * 100  # Scale up
        X[5, 3] = np.nan
        X[10, 5] = 1e6  # Very large but finite

        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(10)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(20)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X, M=None))
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # SVD should handle large finite values
        result = svd_impute(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=5,
            max_iter=10,
        )

        # Should have imputed the NaN
        X_imputed = result.assays["protein"].layers["imputed_svd"].X
        assert not np.isnan(X_imputed[5, 3])
        assert np.isfinite(X_imputed).all()
