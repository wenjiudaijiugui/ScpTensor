"""
Comprehensive error handling tests for ScpTensor.

This module tests that all custom exceptions are raised correctly
across all modules with clear and actionable error messages.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
    ValidationError,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_container():
    """Create a sample container for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 20

    # Create data
    X = np.random.randn(n_samples, n_features) * 0.5 + 2.0
    X_missing = X.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.1
    X_missing[missing_mask] = np.nan

    # Create obs
    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(n_samples)],
            "batch": ["batch1"] * 25 + ["batch2"] * 25,
        }
    )

    # Create var
    var = pl.DataFrame(
        {
            "_index": [f"prot_{i}" for i in range(n_features)],
        }
    )

    # Create assay
    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def complete_container():
    """Create a container with complete (no missing values) data."""
    np.random.seed(42)
    n_samples = 50
    n_features = 20

    X = np.random.randn(n_samples, n_features) * 0.5 + 2.0

    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(n_samples)],
            "batch": ["batch1"] * 25 + ["batch2"] * 25,
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"prot_{i}" for i in range(n_features)],
        }
    )

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X, M=None))

    return ScpContainer(obs=obs, assays={"protein": assay})


# =============================================================================
# Normalization Module Tests
# =============================================================================


class TestNormalizationErrors:
    """Test error handling in normalization modules."""

    def test_log_normalize_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(AssayNotFoundError) as exc_info:
            log_normalize(sample_container, assay_name="invalid")
        assert "invalid" in str(exc_info.value)

    def test_log_normalize_layer_not_found(self, sample_container):
        """Test that LayerNotFoundError is raised for invalid layer."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(LayerNotFoundError) as exc_info:
            log_normalize(sample_container, source_layer="invalid")
        assert "invalid" in str(exc_info.value)

    def test_log_normalize_invalid_base(self, sample_container):
        """Test that ScpValueError is raised for invalid base."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(ScpValueError) as exc_info:
            log_normalize(sample_container, base=-1.0)
        assert "base" in str(exc_info.value).lower()
        assert "-1.0" in str(exc_info.value)

    def test_zscore_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.normalization.zscore import norm_zscore as zscore

        with pytest.raises(AssayNotFoundError):
            zscore(sample_container, assay_name="invalid")

    def test_zscore_invalid_axis(self, complete_container):
        """Test that ScpValueError is raised for invalid axis."""
        from scptensor.normalization.zscore import norm_zscore as zscore

        with pytest.raises(ScpValueError) as exc_info:
            zscore(complete_container, axis=5)
        assert "axis" in str(exc_info.value).lower()

    def test_zscore_requires_complete_data(self, sample_container):
        """Test that ValidationError is raised for data with NaNs."""
        from scptensor.normalization.zscore import norm_zscore as zscore

        with pytest.raises(ValidationError) as exc_info:
            zscore(sample_container, source_layer="raw")
        assert "complete" in str(exc_info.value).lower()


# =============================================================================
# Impute Module Tests
# =============================================================================


class TestImputeErrors:
    """Test error handling in impute modules."""

    def test_knn_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.impute.knn import impute_knn as knn

        with pytest.raises(AssayNotFoundError):
            knn(sample_container, assay_name="invalid", source_layer="raw")

    def test_knn_invalid_k(self, sample_container):
        """Test that ScpValueError is raised for invalid k."""
        from scptensor.impute.knn import impute_knn as knn

        with pytest.raises(ScpValueError) as exc_info:
            knn(sample_container, assay_name="protein", source_layer="raw", k=0)
        assert "k" in str(exc_info.value).lower()

    def test_knn_invalid_weights(self, sample_container):
        """Test that ScpValueError is raised for invalid weights."""
        from scptensor.impute.knn import impute_knn as knn

        with pytest.raises(ScpValueError) as exc_info:
            knn(sample_container, assay_name="protein", source_layer="raw", weights="invalid")
        assert "weights" in str(exc_info.value).lower()

    def test_missforest_invalid_max_iter(self, sample_container):
        """Test that ScpValueError is raised for invalid max_iter."""
        from scptensor.impute.missforest import impute_mf as missforest

        with pytest.raises(ScpValueError) as exc_info:
            missforest(sample_container, assay_name="protein", source_layer="raw", max_iter=0)
        assert "max_iter" in str(exc_info.value).lower()

    def test_ppca_invalid_n_components(self, sample_container):
        """Test that ScpValueError is raised for invalid n_components."""
        from scptensor.impute.ppca import impute_ppca as ppca

        with pytest.raises(ScpValueError) as exc_info:
            ppca(sample_container, assay_name="protein", source_layer="raw", n_components=0)
        assert "n_components" in str(exc_info.value).lower()

    def test_ppca_dimension_error(self, sample_container):
        """Test that DimensionError is raised for too large n_components."""
        from scptensor.impute.ppca import impute_ppca as ppca

        with pytest.raises(DimensionError) as exc_info:
            ppca(sample_container, assay_name="protein", source_layer="raw", n_components=1000)
        assert "n_components" in str(exc_info.value).lower()

    def test_svd_invalid_init_method(self, sample_container):
        """Test that ScpValueError is raised for invalid init_method."""
        from scptensor.impute.svd import impute_svd as svd_impute

        with pytest.raises(ScpValueError) as exc_info:
            svd_impute(
                sample_container, assay_name="protein", source_layer="raw", init_method="invalid"
            )
        assert "init_method" in str(exc_info.value).lower()


# =============================================================================
# Integration Module Tests
# =============================================================================


class TestIntegrationErrors:
    """Test error handling in integration modules."""

    def test_combat_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.integration.combat import integrate_combat as combat

        with pytest.raises(AssayNotFoundError):
            combat(sample_container, batch_key="batch", assay_name="invalid")

    def test_combat_invalid_batch_key(self, sample_container):
        """Test that ScpValueError is raised for invalid batch key."""
        from scptensor.integration.combat import integrate_combat as combat

        with pytest.raises(ScpValueError) as exc_info:
            combat(sample_container, batch_key="invalid")
        # The error message format is "Batch key 'invalid' not found..."
        error_str = str(exc_info.value).lower()
        assert "batch" in error_str
        assert "invalid" in error_str

    def test_mnn_invalid_k(self, sample_container):
        """Test that ScpValueError is raised for invalid k."""
        from scptensor.integration.mnn import integrate_mnn as mnn_correct

        with pytest.raises(ScpValueError) as exc_info:
            mnn_correct(sample_container, batch_key="batch", k=0)
        assert "k" in str(exc_info.value).lower()

    def test_mnn_invalid_sigma(self, sample_container):
        """Test that ScpValueError is raised for invalid sigma."""
        from scptensor.integration.mnn import integrate_mnn as mnn_correct

        with pytest.raises(ScpValueError) as exc_info:
            mnn_correct(sample_container, batch_key="batch", sigma=0)
        assert "sigma" in str(exc_info.value).lower()

    @pytest.mark.skip(reason="Requires optional dependency 'scanorama'")
    def test_scanorama_invalid_alpha(self, sample_container):
        """Test that ScpValueError is raised for invalid alpha."""
        from scptensor.integration.scanorama import integrate_scanorama as scanorama_integrate

        with pytest.raises(ScpValueError) as exc_info:
            scanorama_integrate(sample_container, batch_key="batch", alpha=2.0)
        assert "alpha" in str(exc_info.value).lower()


# =============================================================================
# QC Module Tests
# =============================================================================


class TestQCErrors:
    """Test error handling in QC modules."""

    def test_basic_qc_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.qc.basic import qc_basic as basic_qc

        with pytest.raises(AssayNotFoundError):
            basic_qc(sample_container, assay_name="invalid")

    def test_basic_qc_invalid_min_features(self, sample_container):
        """Test that ScpValueError is raised for invalid min_features."""
        from scptensor.qc.basic import qc_basic as basic_qc

        with pytest.raises(ScpValueError) as exc_info:
            basic_qc(sample_container, min_features=-1)
        assert "min_features" in str(exc_info.value).lower()

    def test_outlier_detection_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.qc.outlier import detect_outliers

        with pytest.raises(AssayNotFoundError):
            detect_outliers(sample_container, assay_name="invalid")

    def test_outlier_detection_invalid_contamination(self, sample_container):
        """Test that ScpValueError is raised for invalid contamination."""
        from scptensor.qc.outlier import detect_outliers

        with pytest.raises(ScpValueError) as exc_info:
            detect_outliers(sample_container, contamination=0.6)
        assert "contamination" in str(exc_info.value).lower()


# =============================================================================
# Cluster Module Tests
# =============================================================================


class TestClusterErrors:
    """Test error handling in cluster modules."""

    def test_kmeans_assay_not_found(self, sample_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.cluster.basic import cluster_kmeans as kmeans

        with pytest.raises(AssayNotFoundError):
            kmeans(sample_container, assay_name="invalid")

    def test_kmeans_invalid_n_clusters(self, sample_container):
        """Test that ScpValueError is raised for invalid n_clusters."""
        from scptensor.cluster.basic import cluster_kmeans as kmeans

        with pytest.raises(ScpValueError) as exc_info:
            kmeans(sample_container, n_clusters=0)
        assert "n_clusters" in str(exc_info.value).lower()

    def test_run_kmeans_layer_not_found(self, complete_container):
        """Test that LayerNotFoundError is raised for invalid layer."""
        from scptensor.cluster.kmeans import cluster_kmeans as run_kmeans

        with pytest.raises(LayerNotFoundError):
            run_kmeans(complete_container, assay_name="protein", base_layer="invalid")

    @pytest.mark.skip(reason="Requires optional dependency 'leidenalg'")
    def test_leiden_assay_not_found(self, complete_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.cluster.graph import cluster_leiden as leiden

        with pytest.raises(AssayNotFoundError):
            leiden(complete_container, assay_name="invalid", base_layer="raw")

    @pytest.mark.skip(reason="Requires optional dependency 'leidenalg'")
    def test_leiden_invalid_n_neighbors(self, complete_container):
        """Test that ScpValueError is raised for invalid n_neighbors."""
        from scptensor.cluster.graph import cluster_leiden as leiden

        with pytest.raises(ScpValueError) as exc_info:
            leiden(complete_container, assay_name="protein", base_layer="raw", n_neighbors=0)
        assert "n_neighbors" in str(exc_info.value).lower()


# =============================================================================
# Dim Reduction Module Tests
# =============================================================================


class TestDimReductionErrors:
    """Test error handling in dimensionality reduction modules."""

    def test_umap_assay_not_found(self, complete_container):
        """Test that AssayNotFoundError is raised for invalid assay."""
        from scptensor.dim_reduction.umap import reduce_umap as umap

        with pytest.raises(AssayNotFoundError):
            umap(complete_container, assay_name="invalid", base_layer="raw")

    def test_umap_layer_not_found(self, complete_container):
        """Test that LayerNotFoundError is raised for invalid layer."""
        from scptensor.dim_reduction.umap import reduce_umap as umap

        with pytest.raises(LayerNotFoundError):
            umap(complete_container, assay_name="protein", base_layer="invalid")

    def test_umap_invalid_n_neighbors(self, complete_container):
        """Test that ScpValueError is raised for invalid n_neighbors."""
        from scptensor.dim_reduction.umap import reduce_umap as umap

        with pytest.raises(ScpValueError) as exc_info:
            umap(complete_container, assay_name="protein", base_layer="raw", n_neighbors=0)
        assert "n_neighbors" in str(exc_info.value).lower()

    def test_umap_invalid_min_dist(self, complete_container):
        """Test that ScpValueError is raised for invalid min_dist."""
        from scptensor.dim_reduction.umap import reduce_umap as umap

        with pytest.raises(ScpValueError) as exc_info:
            umap(complete_container, assay_name="protein", base_layer="raw", min_dist=1.5)
        assert "min_dist" in str(exc_info.value).lower()

    def test_umap_requires_complete_data(self, sample_container):
        """Test that ValidationError is raised for data with NaNs."""
        from scptensor.dim_reduction.umap import reduce_umap as umap

        with pytest.raises(ValidationError) as exc_info:
            umap(sample_container, assay_name="protein", base_layer="raw")
        assert "NaN" in str(exc_info.value) or "complete" in str(exc_info.value).lower()


# =============================================================================
# Exception Message Quality Tests
# =============================================================================


class TestExceptionMessages:
    """Test that error messages are clear and actionable."""

    def test_assay_not_found_message(self, sample_container):
        """Test AssayNotFoundError message includes assay name."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(AssayNotFoundError) as exc_info:
            log_normalize(sample_container, assay_name="my_assay")
        # Message should include the assay name that wasn't found
        assert "my_assay" in str(exc_info.value)

    def test_layer_not_found_message(self, sample_container):
        """Test LayerNotFoundError message includes layer and assay names."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(LayerNotFoundError) as exc_info:
            log_normalize(sample_container, source_layer="my_layer")
        # Message should include the layer name
        assert "my_layer" in str(exc_info.value)

    def test_scvalue_error_includes_parameter_and_value(self, sample_container):
        """Test ScpValueError includes parameter name and value."""
        from scptensor.normalization.log import norm_log as log_normalize

        with pytest.raises(ScpValueError) as exc_info:
            log_normalize(sample_container, base=-5.0)
        error_str = str(exc_info.value).lower()
        assert "base" in error_str
        assert "-5.0" in error_str

    def test_batch_key_error_includes_available_columns(self, sample_container):
        """Test error message includes available columns for batch_key errors."""
        from scptensor.integration.combat import integrate_combat as combat

        with pytest.raises(ScpValueError) as exc_info:
            combat(sample_container, batch_key="wrong_key")
        error_str = str(exc_info.value)
        assert "wrong_key" in error_str
        # Should mention what columns are available
        assert "Available" in error_str or "available" in error_str


# =============================================================================
# Run tests if executed directly
# =============================================================================


if __name__ == "__main__":
    print("Running error handling tests...")
    pytest.main([__file__, "-v", "--tb=short"])
