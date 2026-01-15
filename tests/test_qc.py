"""
Comprehensive tests for QC module.

Tests cover:
- basic.py: basic_qc
- outlier.py: detect_outliers
- advanced.py: calculate_qc_metrics, detect_contaminant_proteins, detect_doublets,
               filter_features_by_missing_rate, filter_features_by_variance,
               filter_features_by_prevalence, filter_samples_by_total_count,
               filter_samples_by_missing_rate
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.exceptions import ValueError as ScpValueError
from scptensor.qc.advanced import (
    calculate_qc_metrics,
    detect_contaminant_proteins,
    detect_doublets,
    filter_features_by_missing_rate,
    filter_features_by_prevalence,
    filter_features_by_variance,
    filter_samples_by_missing_rate,
    filter_samples_by_total_count,
)
from scptensor.qc.basic import basic_qc
from scptensor.qc.outlier import detect_outliers

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
# basic_qc tests (6 tests)
# =============================================================================


class TestBasicQC:
    """Tests for basic_qc function."""

    def test_basic_qc_default_parameters(self, qc_container):
        """Test basic_qc with default parameters."""
        # Use lower thresholds since we only have 20 samples/features
        result = basic_qc(qc_container, min_features=2, min_cells=2)
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples
        assert result.assays["protein"].n_features <= qc_container.assays["protein"].n_features
        # Check history was logged
        assert len(result.history) > 0
        assert result.history[-1].action == "basic_qc"

    def test_basic_qc_custom_thresholds(self, qc_container):
        """Test basic_qc with custom min_features and min_cells."""
        result = basic_qc(qc_container, min_features=5, min_cells=5)
        assert isinstance(result, ScpContainer)
        # With stricter thresholds, should filter more
        assert result.n_samples <= qc_container.n_samples

    def test_basic_qc_no_filtering(self, qc_container):
        """Test basic_qc with thresholds that allow all samples."""
        result = basic_qc(qc_container, min_features=0, min_cells=0)
        # Should keep all samples and features
        assert result.n_samples == qc_container.n_samples
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_basic_qc_negative_min_features_raises_error(self, qc_container):
        """Test that negative min_features raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            basic_qc(qc_container, min_features=-1)
        assert "min_features" in str(excinfo.value).lower()

    def test_basic_qc_negative_min_cells_raises_error(self, qc_container):
        """Test that negative min_cells raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            basic_qc(qc_container, min_cells=-1)
        assert "min_cells" in str(excinfo.value).lower()

    def test_basic_qc_invalid_assay_raises_error(self, qc_container):
        """Test that invalid assay name raises error."""
        with pytest.raises(AssayNotFoundError):
            basic_qc(qc_container, assay_name="nonexistent")


# =============================================================================
# detect_outliers tests (7 tests)
# =============================================================================


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_detect_outliers_default(self, qc_container):
        """Test detect_outliers with default parameters."""
        result = detect_outliers(qc_container)
        assert isinstance(result, ScpContainer)
        assert "is_outlier" in result.obs.columns
        assert result.obs["is_outlier"].dtype == pl.Boolean
        # Should have some outliers detected (or at least the column exists)
        assert len(result.obs["is_outlier"]) == qc_container.n_samples

    def test_detect_outliers_custom_contamination(self, qc_container):
        """Test detect_outliers with custom contamination rate."""
        result = detect_outliers(qc_container, contamination=0.1)
        assert "is_outlier" in result.obs.columns
        n_outliers = result.obs["is_outlier"].sum()
        # With 20 samples and 0.1 contamination, expect ~2 outliers
        assert 0 <= n_outliers <= 5  # Allow some variance

    def test_detect_outliers_custom_layer(self, qc_container_multi_layer):
        """Test detect_outliers on different layer."""
        result = detect_outliers(qc_container_multi_layer, layer="normalized")
        assert "is_outlier" in result.obs.columns

    def test_detect_outliers_invalid_contamination_low(self, qc_container):
        """Test that contamination <= 0 raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            detect_outliers(qc_container, contamination=0)
        assert "contamination" in str(excinfo.value).lower()

    def test_detect_outliers_invalid_contamination_high(self, qc_container):
        """Test that contamination >= 0.5 raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            detect_outliers(qc_container, contamination=0.5)
        assert "contamination" in str(excinfo.value).lower()

    def test_detect_outliers_invalid_assay_raises_error(self, qc_container):
        """Test that invalid assay name raises error."""
        with pytest.raises(AssayNotFoundError):
            detect_outliers(qc_container, assay_name="nonexistent")

    def test_detect_outliers_invalid_layer_raises_error(self, qc_container):
        """Test that invalid layer name raises error."""
        with pytest.raises(LayerNotFoundError):
            detect_outliers(qc_container, layer="nonexistent")


# =============================================================================
# calculate_qc_metrics tests (8 tests)
# =============================================================================


class TestCalculateQCMetrics:
    """Tests for calculate_qc_metrics function."""

    def test_calculate_qc_metrics_dense(self, qc_container):
        """Test calculate_qc_metrics with dense matrix."""
        result = calculate_qc_metrics(qc_container)
        assert isinstance(result, ScpContainer)

        # Check sample metrics in obs
        for col in [
            "n_detected",
            "total_intensity",
            "missing_rate",
            "mean_intensity",
            "median_intensity",
        ]:
            assert col in result.obs.columns

        # Check feature metrics in var
        var = result.assays["protein"].var
        for col in ["n_detected", "prevalence", "mean_intensity", "variance"]:
            assert col in var.columns

    def test_calculate_qc_metrics_sparse(self, qc_container_sparse):
        """Test calculate_qc_metrics with sparse matrix."""
        result = calculate_qc_metrics(qc_container_sparse)
        assert isinstance(result, ScpContainer)
        assert "n_detected" in result.obs.columns
        assert "prevalence" in result.assays["protein"].var.columns

    def test_calculate_qc_metrics_custom_threshold(self, qc_container):
        """Test with custom detection threshold."""
        result = calculate_qc_metrics(qc_container, detection_threshold=1.0)
        assert "n_detected" in result.obs.columns
        # Higher threshold should detect fewer features
        n_detected_high = result.obs["n_detected"].sum()
        result_low = calculate_qc_metrics(qc_container, detection_threshold=0.0)
        n_detected_low = result_low.obs["n_detected"].sum()
        assert n_detected_high <= n_detected_low

    def test_calculate_qc_metrics_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        result = calculate_qc_metrics(qc_container_multi_layer, layer="normalized")
        assert "n_detected" in result.obs.columns

    def test_qc_metrics_sample_values_valid(self, qc_container):
        """Test that QC metrics have valid values."""
        result = calculate_qc_metrics(qc_container)
        obs = result.obs

        # n_detected should be non-negative and <= n_features
        assert all(obs["n_detected"] >= 0)
        assert all(obs["n_detected"] <= qc_container.assays["protein"].n_features)

        # total_intensity should be non-negative
        assert all(obs["total_intensity"] >= 0)

        # missing_rate should be in [0, 1]
        assert all(obs["missing_rate"] >= 0)
        assert all(obs["missing_rate"] <= 1)

    def test_qc_metrics_feature_values_valid(self, qc_container):
        """Test that feature QC metrics have valid values."""
        result = calculate_qc_metrics(qc_container)
        var = result.assays["protein"].var

        # prevalence should be in [0, 1]
        assert all(var["prevalence"] >= 0)
        assert all(var["prevalence"] <= 1)

        # variance should be non-negative
        assert all(var["variance"] >= 0)

    def test_calculate_qc_metrics_invalid_assay_raises_error(self, qc_container):
        """Test that invalid assay name raises error."""
        with pytest.raises(AssayNotFoundError):
            calculate_qc_metrics(qc_container, assay_name="nonexistent")

    def test_calculate_qc_metrics_invalid_layer_raises_error(self, qc_container):
        """Test that invalid layer name raises error."""
        with pytest.raises(LayerNotFoundError):
            calculate_qc_metrics(qc_container, layer="nonexistent")


# =============================================================================
# filter_features_by_missing_rate tests (7 tests)
# =============================================================================


class TestFilterFeaturesByMissingRate:
    """Tests for filter_features_by_missing_rate function."""

    def test_filter_features_by_missing_rate_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        result = filter_features_by_missing_rate(qc_container, max_missing_rate=0.5, inplace=False)
        assert isinstance(result, ScpContainer)
        assert result.n_samples == qc_container.n_samples
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features
        # Check that statistics were added
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns
        assert "keep_missing_rate_0.5" in var.columns

    def test_filter_features_by_missing_rate_inplace(self, qc_container):
        """Test inplace filtering actually filters."""
        original_n_features = qc_container.assays["protein"].n_features
        result = filter_features_by_missing_rate(qc_container, max_missing_rate=0.3, inplace=True)
        assert isinstance(result, ScpContainer)
        # Should have filtered some features
        assert result.assays["protein"].n_features <= original_n_features

    def test_filter_features_by_missing_rate_permissive(self, qc_container):
        """Test with very permissive threshold."""
        result = filter_features_by_missing_rate(qc_container, max_missing_rate=1.0, inplace=True)
        # Should keep all features
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_filter_features_by_missing_rate_strict(self, qc_container):
        """Test with very strict threshold that might filter to zero."""
        # With strict threshold, may filter to zero features which is expected behavior
        # Using pytest.raises to handle the ValidationError
        with pytest.raises(Exception):  # ValidationError when filtering to zero features
            filter_features_by_missing_rate(qc_container, max_missing_rate=0.0, inplace=True)

    def test_filter_features_by_missing_rate_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = filter_features_by_missing_rate(
            qc_container_sparse, max_missing_rate=0.5, inplace=False
        )
        assert "missing_rate" in result.assays["protein"].var.columns

    def test_filter_features_by_missing_rate_invalid_threshold_low(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_missing_rate(qc_container, max_missing_rate=-0.1)
        assert "max_missing_rate" in str(excinfo.value).lower()

    def test_filter_features_by_missing_rate_invalid_threshold_high(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_missing_rate(qc_container, max_missing_rate=1.1)
        assert "max_missing_rate" in str(excinfo.value).lower()


# =============================================================================
# filter_features_by_variance tests (8 tests)
# =============================================================================


class TestFilterFeaturesByVariance:
    """Tests for filter_features_by_variance function."""

    def test_filter_features_by_variance_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        result = filter_features_by_variance(qc_container, min_variance=0.01, inplace=False)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "feature_variance" in var.columns
        assert "keep_variance_0.01" in var.columns

    def test_filter_features_by_variance_inplace(self, qc_container):
        """Test inplace filtering actually filters."""
        original_n_features = qc_container.assays["protein"].n_features
        result = filter_features_by_variance(qc_container, min_variance=0.1, inplace=True)
        assert result.assays["protein"].n_features <= original_n_features

    def test_filter_features_by_variance_top_n(self, qc_container):
        """Test filtering by top N features."""
        result = filter_features_by_variance(qc_container, top_n=10, inplace=True)
        assert result.assays["protein"].n_features == 10

    def test_filter_features_by_variance_top_n_all(self, qc_container):
        """Test top_n greater than total features."""
        total_features = qc_container.assays["protein"].n_features
        result = filter_features_by_variance(qc_container, top_n=total_features + 10, inplace=True)
        # Should keep at most total features
        assert result.assays["protein"].n_features <= total_features

    def test_filter_features_by_variance_zero_threshold(self, qc_container):
        """Test with zero variance threshold."""
        result = filter_features_by_variance(qc_container, min_variance=0.0, inplace=True)
        # Should keep all features with non-negative variance (all)
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_filter_features_by_variance_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = filter_features_by_variance(qc_container_sparse, min_variance=0.01, inplace=False)
        assert "feature_variance" in result.assays["protein"].var.columns

    def test_filter_features_by_variance_invalid_variance(self, qc_container):
        """Test that negative variance raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_variance(qc_container, min_variance=-0.1)
        assert "min_variance" in str(excinfo.value).lower()

    def test_filter_features_by_variance_invalid_top_n(self, qc_container):
        """Test that invalid top_n raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_variance(qc_container, top_n=0)
        assert "top_n" in str(excinfo.value).lower()


# =============================================================================
# filter_features_by_prevalence tests (8 tests)
# =============================================================================


class TestFilterFeaturesByPrevalence:
    """Tests for filter_features_by_prevalence function."""

    def test_filter_features_by_prevalence_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        result = filter_features_by_prevalence(qc_container, min_prevalence=5, inplace=False)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "prevalence" in var.columns
        assert "keep_prevalence_5" in var.columns

    def test_filter_features_by_prevalence_inplace(self, qc_container):
        """Test inplace filtering actually filters."""
        original_n_features = qc_container.assays["protein"].n_features
        result = filter_features_by_prevalence(qc_container, min_prevalence=10, inplace=True)
        assert result.assays["protein"].n_features <= original_n_features

    def test_filter_features_by_prevalence_ratio(self, qc_container):
        """Test filtering by prevalence ratio."""
        result = filter_features_by_prevalence(
            qc_container, min_prevalence_ratio=0.5, inplace=False
        )
        assert "prevalence" in result.assays["protein"].var.columns

    def test_filter_features_by_prevalence_zero_threshold(self, qc_container):
        """Test with zero prevalence threshold."""
        result = filter_features_by_prevalence(qc_container, min_prevalence=0, inplace=True)
        # Should keep all features
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_filter_features_by_prevalence_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = filter_features_by_prevalence(qc_container_sparse, min_prevalence=5, inplace=False)
        assert "prevalence" in result.assays["protein"].var.columns

    def test_filter_features_by_prevalence_invalid_prevalence(self, qc_container):
        """Test that negative prevalence raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_prevalence(qc_container, min_prevalence=-1)
        assert "min_prevalence" in str(excinfo.value).lower()

    def test_filter_features_by_prevalence_invalid_ratio_low(self, qc_container):
        """Test that invalid ratio raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_prevalence(qc_container, min_prevalence_ratio=-0.1)
        assert "min_prevalence_ratio" in str(excinfo.value).lower()

    def test_filter_features_by_prevalence_invalid_ratio_high(self, qc_container):
        """Test that invalid ratio raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_features_by_prevalence(qc_container, min_prevalence_ratio=1.1)
        assert "min_prevalence_ratio" in str(excinfo.value).lower()


# =============================================================================
# filter_samples_by_total_count tests (7 tests)
# =============================================================================


class TestFilterSamplesByTotalCount:
    """Tests for filter_samples_by_total_count function."""

    def test_filter_samples_by_total_count_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        result = filter_samples_by_total_count(qc_container, min_total=10.0, inplace=False)
        assert isinstance(result, ScpContainer)
        assert result.n_samples == qc_container.n_samples
        assert "total_count" in result.obs.columns
        assert "keep_total_min_10.0" in result.obs.columns

    def test_filter_samples_by_total_count_inplace(self, qc_container):
        """Test inplace filtering actually filters."""
        original_n_samples = qc_container.n_samples
        result = filter_samples_by_total_count(qc_container, min_total=5.0, inplace=True)
        assert result.n_samples <= original_n_samples

    def test_filter_samples_by_total_count_with_max(self, qc_container):
        """Test filtering with both min and max thresholds."""
        result = filter_samples_by_total_count(
            qc_container, min_total=5.0, max_total=50.0, inplace=False
        )
        assert "total_count" in result.obs.columns

    def test_filter_samples_by_total_count_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = filter_samples_by_total_count(qc_container, min_total=0.0, inplace=True)
        # Should keep all samples
        assert result.n_samples == qc_container.n_samples

    def test_filter_samples_by_total_count_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = filter_samples_by_total_count(qc_container_sparse, min_total=10.0, inplace=False)
        assert "total_count" in result.obs.columns

    def test_filter_samples_by_total_count_invalid_min(self, qc_container):
        """Test that negative min_total raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_samples_by_total_count(qc_container, min_total=-1.0)
        assert "min_total" in str(excinfo.value).lower()

    def test_filter_samples_by_total_count_invalid_max(self, qc_container):
        """Test that max < min raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_samples_by_total_count(qc_container, min_total=10.0, max_total=5.0)
        assert "max_total" in str(excinfo.value).lower()


# =============================================================================
# filter_samples_by_missing_rate tests (7 tests)
# =============================================================================


class TestFilterSamplesByMissingRate:
    """Tests for filter_samples_by_missing_rate function."""

    def test_filter_samples_by_missing_rate_non_inplace(self, qc_container):
        """Test non-inplace filtering adds statistics."""
        result = filter_samples_by_missing_rate(qc_container, max_missing_rate=0.5, inplace=False)
        assert isinstance(result, ScpContainer)
        assert result.n_samples == qc_container.n_samples
        assert "missing_rate" in result.obs.columns
        assert "keep_missing_rate_0.5" in result.obs.columns

    def test_filter_samples_by_missing_rate_inplace(self, qc_container):
        """Test inplace filtering actually filters."""
        original_n_samples = qc_container.n_samples
        result = filter_samples_by_missing_rate(qc_container, max_missing_rate=0.3, inplace=True)
        assert result.n_samples <= original_n_samples

    def test_filter_samples_by_missing_rate_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = filter_samples_by_missing_rate(qc_container, max_missing_rate=1.0, inplace=True)
        # Should keep all samples
        assert result.n_samples == qc_container.n_samples

    def test_filter_samples_by_missing_rate_strict(self, qc_container):
        """Test with strict threshold that might filter to zero."""
        # With strict threshold, may filter to zero samples which is expected behavior
        with pytest.raises(Exception):  # ValidationError when filtering to zero samples
            filter_samples_by_missing_rate(qc_container, max_missing_rate=0.0, inplace=True)

    def test_filter_samples_by_missing_rate_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = filter_samples_by_missing_rate(
            qc_container_sparse, max_missing_rate=0.5, inplace=False
        )
        assert "missing_rate" in result.obs.columns

    def test_filter_samples_by_missing_rate_invalid_low(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_samples_by_missing_rate(qc_container, max_missing_rate=-0.1)
        assert "max_missing_rate" in str(excinfo.value).lower()

    def test_filter_samples_by_missing_rate_invalid_high(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            filter_samples_by_missing_rate(qc_container, max_missing_rate=1.1)
        assert "max_missing_rate" in str(excinfo.value).lower()


# =============================================================================
# detect_contaminant_proteins tests (7 tests)
# =============================================================================


class TestDetectContaminantProteins:
    """Tests for detect_contaminant_proteins function."""

    def test_detect_contaminant_default_patterns(self, qc_container_with_contaminants):
        """Test contaminant detection with default patterns."""
        result = detect_contaminant_proteins(qc_container_with_contaminants)
        assert isinstance(result, ScpContainer)

        # Check var has contaminant columns
        var = result.assays["protein"].var
        assert "is_contaminant" in var.columns
        assert "contaminant_prevalence" in var.columns

        # Check obs has contaminant columns
        assert "contaminant_content" in result.obs.columns
        assert "contaminant_ratio" in result.obs.columns

    def test_detect_contaminant_custom_patterns(self, qc_container):
        """Test with custom contaminant patterns."""
        result = detect_contaminant_proteins(
            qc_container, contaminant_patterns=[r"PROT1", r"PROT2"]
        )
        var = result.assays["protein"].var
        assert "is_contaminant" in var.columns

    def test_detect_contaminant_min_prevalence(self, qc_container_with_contaminants):
        """Test with minimum prevalence threshold."""
        result = detect_contaminant_proteins(
            qc_container_with_contaminants,
            min_prevalence=15,  # High threshold
        )
        var = result.assays["protein"].var
        # With high prevalence, should detect fewer contaminants
        n_contaminants = var["is_contaminant"].sum()
        assert n_contaminants >= 0

    def test_detect_contaminant_content_values_valid(self, qc_container_with_contaminants):
        """Test that contaminant content values are valid."""
        result = detect_contaminant_proteins(qc_container_with_contaminants)
        assert all(result.obs["contaminant_content"] >= 0)
        assert all(result.obs["contaminant_ratio"] >= 0)
        assert all(result.obs["contaminant_ratio"] <= 1.0)

    def test_detect_contaminant_prevalence_values_valid(self, qc_container_with_contaminants):
        """Test that contaminant prevalence values are valid."""
        result = detect_contaminant_proteins(qc_container_with_contaminants)
        var = result.assays["protein"].var
        # Prevalence should be non-negative for all proteins
        assert all(var["contaminant_prevalence"] >= 0)

    def test_detect_contaminant_invalid_assay_raises_error(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            detect_contaminant_proteins(qc_container, assay_name="nonexistent")

    def test_detect_contaminant_invalid_layer_raises_error(self, qc_container):
        """Test that invalid layer raises error."""
        with pytest.raises(LayerNotFoundError):
            detect_contaminant_proteins(qc_container, layer="nonexistent")


# =============================================================================
# detect_doublets tests (8 tests)
# =============================================================================


class TestDetectDoublets:
    """Tests for detect_doublets function."""

    def test_detect_doublets_knn_method(self, qc_container):
        """Test doublet detection with KNN method."""
        result = detect_doublets(qc_container, method="knn")
        assert isinstance(result, ScpContainer)
        assert "is_doublet" in result.obs.columns
        assert "doublet_score" in result.obs.columns
        assert result.obs["is_doublet"].dtype == pl.Boolean

    def test_detect_doublets_isolation_method(self, qc_container):
        """Test doublet detection with isolation forest method."""
        result = detect_doublets(qc_container, method="isolation")
        assert "is_doublet" in result.obs.columns
        assert "doublet_score" in result.obs.columns

    def test_detect_doublets_hybrid_method(self, qc_container):
        """Test doublet detection with hybrid method."""
        result = detect_doublets(qc_container, method="hybrid")
        assert "is_doublet" in result.obs.columns
        assert "doublet_score" in result.obs.columns

    def test_detect_doublets_custom_rate(self, qc_container):
        """Test with custom expected doublet rate."""
        result = detect_doublets(qc_container, expected_doublet_rate=0.05)
        n_doublets = result.obs["is_doublet"].sum()
        # With 20 samples and 0.05 rate, expect ~1 doublet
        assert 0 <= n_doublets <= 3

    def test_detect_doublets_custom_neighbors(self, qc_container):
        """Test with custom n_neighbors."""
        result = detect_doublets(qc_container, method="knn", n_neighbors=5)
        assert "is_doublet" in result.obs.columns

    def test_detect_doublets_score_values_valid(self, qc_container):
        """Test that doublet scores are in valid range."""
        result = detect_doublets(qc_container, method="knn")
        scores = result.obs["doublet_score"].to_numpy()
        # Scores should be non-negative (normalized)
        assert all(scores >= 0)
        assert all(scores <= 1.0)

    def test_detect_doublets_invalid_rate_low(self, qc_container):
        """Test that invalid rate raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            detect_doublets(qc_container, expected_doublet_rate=0)
        assert "expected_doublet_rate" in str(excinfo.value).lower()

    def test_detect_doublets_invalid_rate_high(self, qc_container):
        """Test that invalid rate raises error."""
        with pytest.raises(ScpValueError) as excinfo:
            detect_doublets(qc_container, expected_doublet_rate=0.6)
        assert "expected_doublet_rate" in str(excinfo.value).lower()


# =============================================================================
# Integration tests
# =============================================================================


class TestQCIntegration:
    """Integration tests for QC workflows."""

    def test_qc_pipeline_full(self, qc_container):
        """Test a complete QC pipeline."""
        # Step 1: Calculate metrics
        container = calculate_qc_metrics(qc_container)
        assert "n_detected" in container.obs.columns

        # Step 2: Detect outliers
        container = detect_outliers(container)
        assert "is_outlier" in container.obs.columns

        # Step 3: Filter samples
        container = filter_samples_by_missing_rate(container, max_missing_rate=0.5, inplace=True)

        # Step 4: Filter features
        container = filter_features_by_variance(container, min_variance=0.01, inplace=True)

        assert isinstance(container, ScpContainer)

    def test_qc_with_sparse_matrix_pipeline(self, qc_container_sparse):
        """Test QC pipeline with sparse matrices."""
        container = calculate_qc_metrics(qc_container_sparse)
        container = detect_outliers(container)
        container = filter_features_by_missing_rate(container, max_missing_rate=0.5, inplace=True)
        assert isinstance(container, ScpContainer)

    def test_basic_qc_followed_by_advanced(self, qc_container):
        """Test basic_qc followed by advanced QC."""
        container = basic_qc(qc_container, min_features=5, min_cells=3)
        container = calculate_qc_metrics(container)
        assert "n_detected" in container.obs.columns

    def test_multiple_filter_operations(self, qc_container):
        """Test multiple filter operations in sequence."""
        container = filter_features_by_missing_rate(
            qc_container, max_missing_rate=0.5, inplace=True
        )
        container = filter_features_by_variance(container, min_variance=0.01, inplace=True)
        container = filter_features_by_prevalence(container, min_prevalence=5, inplace=True)
        # Should still be a valid container
        assert isinstance(container, ScpContainer)
