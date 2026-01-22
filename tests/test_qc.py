"""
Comprehensive tests for QC module.

Tests cover the refactored QC module structure:
- qc_psm: PSM-level filtering (filter_contaminants, filter_psms_by_pif)
- qc_sample: Sample-level QC (calculate_sample_qc_metrics, filter_low_quality_samples,
             filter_doublets_mad, assess_batch_effects)
- qc_feature: Feature-level QC (calculate_feature_qc_metrics, filter_features_by_missingness,
               filter_features_by_cv)
- metrics: Utility functions (compute_mad, is_outlier_mad, compute_cv)

NOTE: This test file has been updated to match the refactored QC module structure.
Many functions from the old API (advanced, basic, outlier, batch, bivariate modules)
no longer exist. Tests for removed functions are marked with pytest.mark.skip.
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, ScpValueError

# New imports from refactored QC module
from scptensor.qc import qc_feature, qc_psm, qc_sample
from scptensor.qc.metrics import compute_cv, compute_mad, is_outlier_mad


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def qc_obs():
    """Create obs DataFrame for QC testing."""
    return pl.DataFrame(
        {
            "_index": [str(i) for i in range(20)],  # Use integer strings as IDs
            "batch": ["A"] * 10 + ["B"] * 10,
        }
    )


@pytest.fixture
def qc_var():
    """Create var DataFrame for QC testing."""
    # Include some contaminant proteins for testing
    protein_names = [
        "KRT1",
        "KRT2",  # Keratins (contaminants)
        "ALB001",  # Albumin (contaminant)
        "IGG1",  # Immunoglobulin (contaminant)
    ] + [f"PROT{i}" for i in range(4, 20)]
    return pl.DataFrame(
        {
            "_index": [
                str(i) for i in range(20)
            ],  # Use integer strings as IDs (matches positional indices)
            "name": protein_names,
        }
    )


@pytest.fixture
def qc_var_with_contaminants():
    """Create var DataFrame with known contaminants."""
    protein_names = [
        "KRT1",
        "KRT2",
        "KRT8",  # Keratins
        "Trypsin",  # Trypsin
        "Albumin",  # Albumin
        "IGHG1",
        "IGKC",  # Immunoglobulins
    ] + [f"PROT{i}" for i in range(7, 20)]
    # Use integer strings as IDs for compatibility with filter_features
    return pl.DataFrame(
        {
            "_index": [str(i) for i in range(20)],
            "name": protein_names,
        }
    )


@pytest.fixture
def qc_dense_X():
    """Create dense data matrix with some missing values."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(20, 20))
    # Introduce missing values (zeros)
    missing_mask = np.random.random((20, 20)) < 0.3
    X[missing_mask] = 0
    # Make some cells low quality (first 2 samples have few features)
    X[0, :] = 0
    X[0, :5] = 0.5  # Only 5 features detected (above default min_features=200 doesn't apply to 20)
    X[1, :] = 0
    X[1, :3] = 1.0  # Only 3 features detected
    # Make some features low prevalence (last 2 features rarely detected)
    X[:, 18] = 0
    X[:5, 18] = 0.5
    X[:, 19] = 0
    X[:3, 19] = 1.0
    return X


@pytest.fixture
def qc_sparse_X(qc_dense_X):
    """Create sparse data matrix."""
    return sparse.csr_matrix(qc_dense_X)


@pytest.fixture
def qc_container(qc_obs, qc_var, qc_dense_X):
    """Create a ScpContainer for QC testing."""
    matrix = ScpMatrix(X=qc_dense_X, M=None)
    assay = Assay(var=qc_var, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_sparse(qc_obs, qc_var, qc_sparse_X):
    """Create a ScpContainer with sparse data."""
    matrix = ScpMatrix(X=qc_sparse_X, M=None)
    assay = Assay(var=qc_var, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_multi_layer(qc_obs, qc_var, qc_dense_X):
    """Create a ScpContainer with multiple layers."""
    matrix_raw = ScpMatrix(X=qc_dense_X, M=None)
    matrix_norm = ScpMatrix(X=qc_dense_X * 2, M=None)
    assay = Assay(var=qc_var, layers={"raw": matrix_raw, "normalized": matrix_norm})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_with_contaminants(qc_obs, qc_var_with_contaminants, qc_dense_X):
    """Create a ScpContainer with contaminant proteins."""
    matrix = ScpMatrix(X=qc_dense_X, M=None)
    assay = Assay(var=qc_var_with_contaminants, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


# =============================================================================
# Tests for qc_sample module
# =============================================================================


class TestCalculateSampleQCMetrics:
    """Tests for calculate_sample_qc_metrics function."""

    def test_calculate_sample_qc_metrics_default(self, qc_container):
        """Test calculate_sample_qc_metrics with default parameters."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container)
        assert isinstance(result, ScpContainer)

        # Check that metrics were added to obs
        assert "n_features_protein" in result.obs.columns
        assert "total_intensity_protein" in result.obs.columns
        assert "log1p_total_intensity_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_sparse(self, qc_container_sparse):
        """Test calculate_sample_qc_metrics with sparse matrix."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container_sparse)
        assert isinstance(result, ScpContainer)
        assert "n_features_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        result = qc_sample.calculate_sample_qc_metrics(
            qc_container_multi_layer, layer_name="normalized"
        )
        assert "n_features_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.calculate_sample_qc_metrics(qc_container, assay_name="nonexistent")

    def test_calculate_sample_qc_metrics_invalid_layer(self, qc_container):
        """Test that invalid layer raises error."""
        with pytest.raises(ScpValueError):
            qc_sample.calculate_sample_qc_metrics(qc_container, layer_name="nonexistent")


class TestFilterLowQualitySamples:
    """Tests for filter_low_quality_samples function."""

    def test_filter_low_quality_samples_default(self, qc_container):
        """Test filter_low_quality_samples with default parameters."""
        # Use lower threshold since we only have 20 features
        result = qc_sample.filter_low_quality_samples(qc_container, min_features=5, use_mad=False)
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_low_quality_samples_with_mad(self, qc_container):
        """Test with MAD-based filtering."""
        result = qc_sample.filter_low_quality_samples(
            qc_container, min_features=2, nmads=2.0, use_mad=True
        )
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_low_quality_samples_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = qc_sample.filter_low_quality_samples(qc_container, min_features=0, use_mad=False)
        # Should keep all samples
        assert result.n_samples == qc_container.n_samples

    def test_filter_low_quality_samples_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.filter_low_quality_samples(qc_container, assay_name="nonexistent")


class TestFilterDoubletsMAD:
    """Tests for filter_doublets_mad function."""

    def test_filter_doublets_mad_default(self, qc_container):
        """Test doublet detection with default parameters."""
        result = qc_sample.filter_doublets_mad(qc_container, nmads=3.0)
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_doublets_mad_custom_nmads(self, qc_container):
        """Test with custom nmads threshold."""
        result = qc_sample.filter_doublets_mad(qc_container, nmads=2.0)
        assert isinstance(result, ScpContainer)

    def test_filter_doublets_mad_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.filter_doublets_mad(qc_container, assay_name="nonexistent")


class TestAssessBatchEffects:
    """Tests for assess_batch_effects function."""

    def test_assess_batch_effects_default(self, qc_container):
        """Test batch effect assessment."""
        result = qc_sample.assess_batch_effects(qc_container, batch_col="batch")
        assert isinstance(result, pl.DataFrame)
        assert "batch" in result.columns
        assert "n_cells" in result.columns
        assert "median_features" in result.columns
        assert "median_intensity" in result.columns

    def test_assess_batch_effects_invalid_batch_col(self, qc_container):
        """Test that invalid batch column raises error."""
        with pytest.raises(ScpValueError):
            qc_sample.assess_batch_effects(qc_container, batch_col="nonexistent")

    def test_assess_batch_effects_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.assess_batch_effects(
                qc_container, batch_col="batch", assay_name="nonexistent"
            )


# =============================================================================
# Tests for qc_feature module
# =============================================================================


class TestCalculateFeatureQCMetrics:
    """Tests for calculate_feature_qc_metrics function."""

    def test_calculate_feature_qc_metrics_default(self, qc_container):
        """Test calculate_feature_qc_metrics with default parameters."""
        result = qc_feature.calculate_feature_qc_metrics(qc_container)
        assert isinstance(result, ScpContainer)

        # Check that metrics were added to var
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns
        assert "detection_rate" in var.columns
        assert "mean_expression" in var.columns
        assert "cv" in var.columns

    def test_calculate_feature_qc_metrics_sparse(self, qc_container_sparse):
        """Test calculate_feature_qc_metrics with sparse matrix."""
        result = qc_feature.calculate_feature_qc_metrics(qc_container_sparse)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns
        assert "cv" in var.columns

    def test_calculate_feature_qc_metrics_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        result = qc_feature.calculate_feature_qc_metrics(
            qc_container_multi_layer, layer_name="normalized"
        )
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns

    def test_calculate_feature_qc_metrics_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.calculate_feature_qc_metrics(qc_container, assay_name="nonexistent")


class TestFilterFeaturesByMissingness:
    """Tests for filter_features_by_missingness function."""

    def test_filter_features_by_missingness_default(self, qc_container):
        """Test filter_features_by_missingness with default parameters."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=0.5)
        assert isinstance(result, ScpContainer)
        assert result.assays["protein"].n_features <= qc_container.assays["protein"].n_features

    def test_filter_features_by_missingness_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=1.0)
        # Should keep all features
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_filter_features_by_missingness_strict(self, qc_container):
        """Test with strict threshold."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=0.0)
        # Should keep only features with 0% missing
        assert result.assays["protein"].n_features >= 0

    def test_filter_features_by_missingness_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = qc_feature.filter_features_by_missingness(
            qc_container_sparse, max_missing_rate=0.5
        )
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_missingness_invalid_threshold(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError):
            qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=1.5)

    def test_filter_features_by_missingness_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.filter_features_by_missingness(qc_container, assay_name="nonexistent")


class TestFilterFeaturesByCV:
    """Tests for filter_features_by_cv function."""

    def test_filter_features_by_cv_default(self, qc_container):
        """Test filter_features_by_cv with default parameters."""
        result = qc_feature.filter_features_by_cv(qc_container, max_cv=1.0)
        assert isinstance(result, ScpContainer)
        assert result.assays["protein"].n_features <= qc_container.assays["protein"].n_features

    def test_filter_features_by_cv_custom_threshold(self, qc_container):
        """Test with custom max_cv threshold."""
        result = qc_feature.filter_features_by_cv(qc_container, max_cv=0.5)
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_cv_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = qc_feature.filter_features_by_cv(qc_container_sparse, max_cv=1.0)
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_cv_invalid_threshold(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError):
            qc_feature.filter_features_by_cv(qc_container, max_cv=0)

    def test_filter_features_by_cv_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.filter_features_by_cv(qc_container, assay_name="nonexistent")


# =============================================================================
# Tests for qc_psm module
# =============================================================================


class TestFilterContaminants:
    """Tests for filter_contaminants function."""

    def test_filter_contaminants_default(self, qc_container_with_contaminants):
        """Test contaminant filtering with default patterns."""
        result = qc_psm.filter_contaminants(qc_container_with_contaminants)
        assert isinstance(result, ScpContainer)
        # Should have removed some contaminant proteins
        assert (
            result.assays["protein"].n_features
            <= qc_container_with_contaminants.assays["protein"].n_features
        )

    def test_filter_contaminants_custom_patterns(self, qc_container_with_contaminants):
        """Test with custom contaminant patterns."""
        result = qc_psm.filter_contaminants(
            qc_container_with_contaminants, contaminant_patterns=[r"KRT\d+"]
        )
        assert isinstance(result, ScpContainer)

    def test_filter_contaminants_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_psm.filter_contaminants(qc_container, assay_name="nonexistent")


# =============================================================================
# Tests for metrics module
# =============================================================================


class TestComputeMAD:
    """Tests for compute_mad function."""

    def test_compute_mad_basic(self):
        """Test basic MAD computation."""
        data = np.array([1, 2, 3, 4, 5])
        mad = compute_mad(data)
        assert mad > 0

    def test_compute_mad_with_outliers(self):
        """Test MAD with outliers."""
        data = np.array([1, 2, 3, 4, 100])
        mad = compute_mad(data)
        # MAD should be robust to outlier
        assert mad > 0

    def test_compute_mad_empty(self):
        """Test MAD with empty array."""
        mad = compute_mad(np.array([]))
        assert np.isnan(mad)


class TestIsOutlierMAD:
    """Tests for is_outlier_mad function."""

    def test_is_outlier_mad_both_directions(self):
        """Test outlier detection in both directions."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="both")
        assert len(outliers) == len(data)
        assert outliers[-1]  # Last value should be outlier

    def test_is_outlier_mad_upper_only(self):
        """Test upper-tail outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="upper")
        assert outliers[-1]

    def test_is_outlier_mad_lower_only(self):
        """Test lower-tail outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="lower")
        # 1 might be detected as lower outlier depending on MAD
        assert len(outliers) == len(data)

    def test_is_outlier_mad_empty(self):
        """Test with empty array."""
        outliers = is_outlier_mad(np.array([]), nmads=3.0)
        assert len(outliers) == 0


class TestComputeCV:
    """Tests for compute_cv function."""

    def test_compute_cv_dense(self):
        """Test CV computation for dense array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=0)
        assert len(cv) == 3
        assert all(cv >= 0)

    def test_compute_cv_sparse(self):
        """Test CV computation for sparse matrix."""
        data = sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=0)
        assert len(cv) == 3

    def test_compute_cv_axis1(self):
        """Test CV computation along axis=1."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=1)
        assert len(cv) == 2


# =============================================================================
# Integration tests
# =============================================================================


class TestQCIntegration:
    """Integration tests for QC workflows."""

    def test_qc_pipeline_full(self, qc_container):
        """Test a complete QC pipeline."""
        # Step 1: Calculate sample metrics
        container = qc_sample.calculate_sample_qc_metrics(qc_container)
        assert "n_features_protein" in container.obs.columns

        # Step 2: Calculate feature metrics
        container = qc_feature.calculate_feature_qc_metrics(container)
        var = container.assays["protein"].var
        assert "missing_rate" in var.columns

        # Step 3: Filter samples
        container = qc_sample.filter_low_quality_samples(container, min_features=5, use_mad=False)

        # Step 4: Filter features
        container = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)

        assert isinstance(container, ScpContainer)

    def test_qc_with_sparse_matrix_pipeline(self, qc_container_sparse):
        """Test QC pipeline with sparse matrices."""
        container = qc_sample.calculate_sample_qc_metrics(qc_container_sparse)
        container = qc_feature.calculate_feature_qc_metrics(container)
        container = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)
        assert isinstance(container, ScpContainer)

    def test_batch_analysis_pipeline(self, qc_container):
        """Test a complete batch analysis pipeline."""
        # Step 1: Calculate sample metrics
        container = qc_sample.calculate_sample_qc_metrics(qc_container)

        # Step 2: Assess batch effects
        batch_summary = qc_sample.assess_batch_effects(container, batch_col="batch")
        assert isinstance(batch_summary, pl.DataFrame)
        assert "batch" in batch_summary.columns

        assert isinstance(container, ScpContainer)


# =============================================================================
# Skipped tests for removed functions
# =============================================================================


class TestRemovedFunctions:
    """Tests for functions that were removed during refactoring."""

    @pytest.mark.skip(
        reason="basic_qc function was removed during QC module refactoring. "
        "Use calculate_sample_qc_metrics + filter_low_quality_samples instead."
    )
    def test_basic_qc_default_parameters(self, qc_container):
        """Test basic_qc with default parameters."""
        pass

    @pytest.mark.skip(
        reason="detect_outliers function was removed. "
        "Use filter_low_quality_samples with MAD-based filtering instead."
    )
    def test_detect_outliers_default(self, qc_container):
        """Test detect_outliers with default parameters."""
        pass

    @pytest.mark.skip(
        reason="calculate_qc_metrics (old advanced version) was removed. "
        "Use calculate_sample_qc_metrics and calculate_feature_qc_metrics instead."
    )
    def test_calculate_qc_metrics_dense(self, qc_container):
        """Test calculate_qc_metrics with dense matrix."""
        pass

    @pytest.mark.skip(
        reason="filter_features_by_missing_rate was renamed to filter_features_by_missingness"
    )
    def test_filter_features_by_missing_rate_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        pass

    @pytest.mark.skip(
        reason="filter_features_by_variance was removed. "
        "Use filter_features_by_cv instead, which filters by coefficient of variation."
    )
    def test_filter_features_by_variance_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        pass

    @pytest.mark.skip(
        reason="filter_features_by_prevalence was removed. "
        "Use filter_features_by_missingness (inverse of prevalence) instead."
    )
    def test_filter_features_by_prevalence_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        pass

    @pytest.mark.skip(
        reason="filter_samples_by_total_count was removed. "
        "Use filter_low_quality_samples which filters by n_features instead."
    )
    def test_filter_samples_by_total_count_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        pass

    @pytest.mark.skip(
        reason="filter_samples_by_missing_rate was removed. "
        "Sample-level missing rate filtering is not implemented in the new API."
    )
    def test_filter_samples_by_missing_rate_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        pass

    @pytest.mark.skip(
        reason="detect_contaminant_proteins was removed. "
        "Use filter_contaminants from qc_psm module instead."
    )
    def test_detect_contaminant_default_patterns(self, qc_container_with_contaminants):
        """Test contaminant detection with default patterns."""
        pass

    @pytest.mark.skip(
        reason="detect_doublets (IsolationForest version) was removed. "
        "Use filter_doublets_mad which uses MAD-based detection instead."
    )
    def test_detect_doublets_knn_method(self, qc_container):
        """Test doublet detection with KNN method."""
        pass

    @pytest.mark.skip(
        reason="qc_score (compute_quality_score) from basic module was removed. "
        "Quality scoring is not implemented in the new API."
    )
    def test_compute_quality_score_default(self, qc_container):
        """Test quality score computation with default parameters."""
        pass

    @pytest.mark.skip(
        reason="compute_feature_variance from basic module was removed. "
        "Use calculate_feature_qc_metrics which computes CV instead."
    )
    def test_compute_feature_variance_default(self, qc_container):
        """Test feature variance computation with default parameters."""
        pass

    @pytest.mark.skip(
        reason="compute_feature_missing_rate from basic module was removed. "
        "Use calculate_feature_qc_metrics which computes missing_rate instead."
    )
    def test_compute_feature_missing_rate_default(self, qc_container):
        """Test feature missing rate computation with default parameters."""
        pass

    @pytest.mark.skip(
        reason="Bivariate analysis functions (compute_pairwise_correlation, "
        "detect_outlier_samples, compute_sample_similarity_network) were removed."
    )
    def test_compute_pairwise_correlation_default(self, qc_container):
        """Test pairwise correlation with default parameters."""
        pass

    @pytest.mark.skip(
        reason="Batch analysis functions (qc_batch_metrics, detect_batch_effects, "
        "compute_batch_pca) from batch module were removed. "
        "Use assess_batch_effects instead which provides batch summary statistics."
    )
    def test_compute_batch_metrics_default(self, qc_container):
        """Test batch metrics with default parameters."""
        pass
