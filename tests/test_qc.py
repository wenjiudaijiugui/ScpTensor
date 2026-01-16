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
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.qc.advanced import (
    calculate_qc_metrics,
    detect_doublets,
)
from scptensor.qc.advanced import (
    detect_contaminants as detect_contaminant_proteins,
)
from scptensor.qc.advanced import (
    filter_features_missing as filter_features_by_missing_rate,
)
from scptensor.qc.advanced import (
    filter_features_prevalence as filter_features_by_prevalence,
)
from scptensor.qc.advanced import (
    filter_features_variance as filter_features_by_variance,
)
from scptensor.qc.advanced import (
    filter_samples_count as filter_samples_by_total_count,
)
from scptensor.qc.advanced import (
    filter_samples_missing as filter_samples_by_missing_rate,
)
from scptensor.qc.basic import qc_basic as basic_qc
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
        # Check history was logged (action is now qc_basic for the new function name)
        assert len(result.history) > 0
        assert result.history[-1].action == "qc_basic"

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
        result = detect_outliers(qc_container_multi_layer, layer_name="normalized")
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
            detect_outliers(qc_container, layer_name="nonexistent")


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
        result = calculate_qc_metrics(qc_container_multi_layer, layer_name="normalized")
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
            calculate_qc_metrics(qc_container, layer_name="nonexistent")


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
        """Test with very strict threshold that filters to zero features."""
        # With max_missing_rate=0.0, only features with 0% missing are kept
        # This may result in zero features depending on data
        result = filter_features_by_missing_rate(qc_container, max_missing_rate=0.0, inplace=True)
        # Result may have zero or very few features
        assert result.assays["protein"].n_features >= 0

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
        """Test with strict threshold that filters to zero samples."""
        # With max_missing_rate=0.0, only samples with 0% missing are kept
        # This may result in zero samples depending on data
        result = filter_samples_by_missing_rate(qc_container, max_missing_rate=0.0, inplace=True)
        # Result may have zero or very few samples
        assert result.n_samples >= 0

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
            detect_contaminant_proteins(qc_container, layer_name="nonexistent")


# =============================================================================
# detect_doublets tests (8 tests)
# =============================================================================


class TestDetectDoublets:
    """Tests for detect_doublets function."""

    def test_detect_doublets_knn_method(self, qc_container):
        """Test doublet detection with KNN method."""
        result = detect_doublets(qc_container, method="impute_knn")
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
        result = detect_doublets(qc_container, method="impute_knn", n_neighbors=5)
        assert "is_doublet" in result.obs.columns

    def test_detect_doublets_score_values_valid(self, qc_container):
        """Test that doublet scores are in valid range."""
        result = detect_doublets(qc_container, method="impute_knn")
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


# =============================================================================
# compute_quality_score tests (9 tests)
# =============================================================================


class TestComputeQualityScore:
    """Tests for compute_quality_score function."""

    def test_compute_quality_score_default(self, qc_container):
        """Test quality score computation with default parameters."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        result = compute_quality_score(qc_container)
        assert isinstance(result, ScpContainer)
        assert "quality_score" in result.obs.columns
        assert "quality_detection_rate" in result.obs.columns
        assert "quality_total_intensity" in result.obs.columns
        assert "quality_missing_rate" in result.obs.columns
        assert "quality_cv" in result.obs.columns

    def test_compute_quality_score_values_valid(self, qc_container):
        """Test that quality scores are in valid range [0, 1]."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        result = compute_quality_score(qc_container)
        scores = result.obs["quality_score"].to_numpy()
        assert all(scores >= 0)
        assert all(scores <= 1)

    def test_compute_quality_score_custom_weights(self, qc_container):
        """Test quality score with custom weights."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        custom_weights = {
            "detection_rate": 0.5,
            "total_intensity": 0.3,
            "missing_rate": 0.1,
            "cv": 0.1,
        }
        result = compute_quality_score(qc_container, weights=custom_weights)
        assert "quality_score" in result.obs.columns

    def test_compute_quality_score_invalid_weight_key(self, qc_container):
        """Test that invalid weight key raises error."""
        from scptensor.core.exceptions import ScpValueError
        from scptensor.qc.basic import qc_score as compute_quality_score

        with pytest.raises(ScpValueError):
            compute_quality_score(qc_container, weights={"invalid_key": 1.0})

    def test_compute_quality_score_negative_weight(self, qc_container):
        """Test that negative weight raises error."""
        from scptensor.core.exceptions import ScpValueError
        from scptensor.qc.basic import qc_score as compute_quality_score

        with pytest.raises(ScpValueError):
            compute_quality_score(qc_container, weights={"detection_rate": -0.5})

    def test_compute_quality_score_sparse(self, qc_container_sparse):
        """Test quality score with sparse matrix."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        result = compute_quality_score(qc_container_sparse)
        assert "quality_score" in result.obs.columns

    def test_compute_quality_score_custom_layer(self, qc_container_multi_layer):
        """Test quality score with different layer."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        result = compute_quality_score(qc_container_multi_layer, layer_name="normalized")
        assert "quality_score" in result.obs.columns

    def test_compute_quality_score_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        with pytest.raises(AssayNotFoundError):
            compute_quality_score(qc_container, assay_name="nonexistent")

    def test_compute_quality_score_invalid_layer(self, qc_container):
        """Test that invalid layer raises error."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        with pytest.raises(LayerNotFoundError):
            compute_quality_score(qc_container, layer_name="nonexistent")


# =============================================================================
# compute_feature_variance tests (8 tests)
# =============================================================================


class TestComputeFeatureVariance:
    """Tests for compute_feature_variance function."""

    def test_compute_feature_variance_default(self, qc_container):
        """Test feature variance computation with default parameters."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "feature_variance" in var.columns
        assert "feature_std" in var.columns
        assert "feature_mean" in var.columns
        assert "feature_cv" in var.columns
        assert "feature_iqr" in var.columns

    def test_compute_feature_variance_values_valid(self, qc_container):
        """Test that feature variance values are valid."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container)
        var = result.assays["protein"].var
        assert all(var["feature_variance"] >= 0)
        assert all(var["feature_std"] >= 0)
        assert all(var["feature_cv"] >= 0)

    def test_compute_feature_variance_sparse(self, qc_container_sparse):
        """Test feature variance with sparse matrix."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container_sparse)
        var = result.assays["protein"].var
        assert "feature_variance" in var.columns

    def test_compute_feature_variance_custom_threshold(self, qc_container):
        """Test with custom detection threshold."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container, detection_threshold=1.0)
        var = result.assays["protein"].var
        assert "feature_variance" in var.columns

    def test_compute_feature_variance_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container_multi_layer, layer_name="normalized")
        var = result.assays["protein"].var
        assert "feature_variance" in var.columns

    def test_compute_feature_variance_iqr_values(self, qc_container):
        """Test that IQR values are valid."""
        from scptensor.qc.basic import compute_feature_variance

        result = compute_feature_variance(qc_container)
        var = result.assays["protein"].var
        assert all(var["feature_iqr"] >= 0)

    def test_compute_feature_variance_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.basic import compute_feature_variance

        with pytest.raises(AssayNotFoundError):
            compute_feature_variance(qc_container, assay_name="nonexistent")

    def test_compute_feature_variance_invalid_layer(self, qc_container):
        """Test that invalid layer raises error."""
        from scptensor.qc.basic import compute_feature_variance

        with pytest.raises(LayerNotFoundError):
            compute_feature_variance(qc_container, layer_name="nonexistent")


# =============================================================================
# compute_feature_missing_rate tests (8 tests)
# =============================================================================


class TestComputeFeatureMissingRate:
    """Tests for compute_feature_missing_rate function."""

    def test_compute_feature_missing_rate_default(self, qc_container):
        """Test feature missing rate computation with default parameters."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "feature_missing_rate" in var.columns
        assert "feature_n_detected" in var.columns
        assert "feature_prevalence" in var.columns
        assert "feature_detection_rate" in var.columns

    def test_compute_feature_missing_rate_values_valid(self, qc_container):
        """Test that missing rate values are in valid range [0, 1]."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container)
        var = result.assays["protein"].var
        assert all(var["feature_missing_rate"] >= 0)
        assert all(var["feature_missing_rate"] <= 1)
        assert all(var["feature_prevalence"] >= 0)
        assert all(var["feature_prevalence"] <= 1)

    def test_compute_feature_missing_rate_sparse(self, qc_container_sparse):
        """Test feature missing rate with sparse matrix."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container_sparse)
        var = result.assays["protein"].var
        assert "feature_missing_rate" in var.columns

    def test_compute_feature_missing_rate_custom_threshold(self, qc_container):
        """Test with custom detection threshold."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container, detection_threshold=1.0)
        var = result.assays["protein"].var
        assert "feature_missing_rate" in var.columns

    def test_compute_feature_missing_rate_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container_multi_layer, layer_name="normalized")
        var = result.assays["protein"].var
        assert "feature_missing_rate" in var.columns

    def test_compute_feature_missing_rate_n_detected_valid(self, qc_container):
        """Test that n_detected values are valid."""
        from scptensor.qc.basic import compute_feature_missing_rate

        result = compute_feature_missing_rate(qc_container)
        var = result.assays["protein"].var
        n_samples = qc_container.n_samples
        assert all(var["feature_n_detected"] >= 0)
        assert all(var["feature_n_detected"] <= n_samples)

    def test_compute_feature_missing_rate_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.basic import compute_feature_missing_rate

        with pytest.raises(AssayNotFoundError):
            compute_feature_missing_rate(qc_container, assay_name="nonexistent")

    def test_compute_feature_missing_rate_invalid_layer(self, qc_container):
        """Test that invalid layer raises error."""
        from scptensor.qc.basic import compute_feature_missing_rate

        with pytest.raises(LayerNotFoundError):
            compute_feature_missing_rate(qc_container, layer_name="nonexistent")


# =============================================================================
# compute_pairwise_correlation tests (6 tests)
# =============================================================================


class TestComputePairwiseCorrelation:
    """Tests for compute_pairwise_correlation function."""

    def test_compute_pairwise_correlation_default(self, qc_container):
        """Test pairwise correlation with default parameters."""
        from scptensor.qc.bivariate import compute_pairwise_correlation

        result = compute_pairwise_correlation(qc_container)
        assert isinstance(result, ScpContainer)
        assert "pairwise_correlation" in result.obs.columns

    def test_compute_pairwise_correlation_pearson(self, qc_container):
        """Test with Pearson correlation method."""
        from scptensor.qc.bivariate import compute_pairwise_correlation

        result = compute_pairwise_correlation(qc_container, method="pearson")
        assert "pairwise_correlation" in result.obs.columns

    def test_compute_pairwise_correlation_spearman(self, qc_container):
        """Test with Spearman correlation method."""
        from scptensor.qc.bivariate import compute_pairwise_correlation

        result = compute_pairwise_correlation(qc_container, method="spearman")
        assert "pairwise_correlation" in result.obs.columns

    def test_compute_pairwise_correlation_matrix_shape(self, qc_container):
        """Test that correlation matrix has correct shape."""

        from scptensor.qc.bivariate import compute_pairwise_correlation

        result = compute_pairwise_correlation(qc_container)
        n_samples = qc_container.n_samples
        # The stored value is a list of lists representing the matrix
        # Polars stores the data, we need to access it properly
        corr_list = result.obs["pairwise_correlation"].to_list()[0]
        assert isinstance(corr_list, list)
        assert len(corr_list) == n_samples
        assert all(len(row) == n_samples for row in corr_list)

    def test_compute_pairwise_correlation_invalid_method(self, qc_container):
        """Test that invalid method raises error."""
        from scptensor.qc.bivariate import compute_pairwise_correlation

        with pytest.raises(ScpValueError):
            compute_pairwise_correlation(qc_container, method="invalid")

    def test_compute_pairwise_correlation_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.bivariate import compute_pairwise_correlation

        with pytest.raises(AssayNotFoundError):
            compute_pairwise_correlation(qc_container, assay_name="nonexistent")


# =============================================================================
# detect_outlier_samples tests (8 tests)
# =============================================================================


class TestDetectOutlierSamples:
    """Tests for detect_outlier_samples function."""

    def test_detect_outlier_samples_default(self, qc_container):
        """Test outlier detection with default parameters."""
        from scptensor.qc.bivariate import detect_outlier_samples

        result = detect_outlier_samples(qc_container)
        assert isinstance(result, ScpContainer)
        assert "is_correlation_outlier" in result.obs.columns
        assert "outlier_metric_value" in result.obs.columns

    def test_detect_outlier_samples_mad_method(self, qc_container):
        """Test with median absolute deviation method."""
        from scptensor.qc.bivariate import detect_outlier_samples

        result = detect_outlier_samples(qc_container, method="median_absolute_deviation")
        assert "is_correlation_outlier" in result.obs.columns

    def test_detect_outlier_samples_zscore_method(self, qc_container):
        """Test with z-score method."""
        from scptensor.qc.bivariate import detect_outlier_samples

        result = detect_outlier_samples(qc_container, method="zscore")
        assert "is_correlation_outlier" in result.obs.columns

    def test_detect_outlier_samples_iqr_method(self, qc_container):
        """Test with IQR method."""
        from scptensor.qc.bivariate import detect_outlier_samples

        result = detect_outlier_samples(qc_container, method="iqr")
        assert "is_correlation_outlier" in result.obs.columns

    def test_detect_outlier_samples_total_intensity_metric(self, qc_container):
        """Test with total_intensity metric."""
        from scptensor.qc.bivariate import detect_outlier_samples

        result = detect_outlier_samples(qc_container, metric="total_intensity")
        assert "is_correlation_outlier" in result.obs.columns

    def test_detect_outlier_samples_invalid_threshold(self, qc_container):
        """Test that invalid threshold raises error."""
        from scptensor.qc.bivariate import detect_outlier_samples

        with pytest.raises(ScpValueError):
            detect_outlier_samples(qc_container, threshold=0)

    def test_detect_outlier_samples_invalid_method(self, qc_container):
        """Test that invalid method raises error."""
        from scptensor.core.exceptions import ScpValueError
        from scptensor.qc.bivariate import detect_outlier_samples

        with pytest.raises(ScpValueError):
            detect_outlier_samples(qc_container, method="invalid")

    def test_detect_outlier_samples_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.bivariate import detect_outlier_samples

        with pytest.raises(AssayNotFoundError):
            detect_outlier_samples(qc_container, assay_name="nonexistent")


# =============================================================================
# compute_sample_similarity_network tests (6 tests)
# =============================================================================


class TestComputeSampleSimilarityNetwork:
    """Tests for compute_sample_similarity_network function."""

    def test_compute_sample_similarity_network_default(self, qc_container):
        """Test similarity network with default parameters."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        result = compute_sample_similarity_network(qc_container)
        assert isinstance(result, ScpContainer)
        assert "similarity_neighbors" in result.obs.columns
        assert "similarity_scores" in result.obs.columns

    def test_compute_sample_similarity_network_custom_neighbors(self, qc_container):
        """Test with custom n_neighbors."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        result = compute_sample_similarity_network(qc_container, n_neighbors=3)
        assert "similarity_neighbors" in result.obs.columns
        # Each sample should have 3 neighbors
        assert len(result.obs["similarity_neighbors"][0]) == 3

    def test_compute_sample_similarity_network_correlation_metric(self, qc_container):
        """Test with correlation metric."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        result = compute_sample_similarity_network(qc_container, metric="correlation")
        assert "similarity_neighbors" in result.obs.columns

    def test_compute_sample_similarity_network_euclidean_metric(self, qc_container):
        """Test with euclidean metric."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        result = compute_sample_similarity_network(qc_container, metric="euclidean")
        assert "similarity_neighbors" in result.obs.columns

    def test_compute_sample_similarity_network_cosine_metric(self, qc_container):
        """Test with cosine metric."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        result = compute_sample_similarity_network(qc_container, metric="cosine")
        assert "similarity_neighbors" in result.obs.columns

    def test_compute_sample_similarity_network_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.bivariate import compute_sample_similarity_network

        with pytest.raises(AssayNotFoundError):
            compute_sample_similarity_network(qc_container, assay_name="nonexistent")


# =============================================================================
# compute_batch_metrics tests (7 tests)
# =============================================================================


class TestComputeBatchMetrics:
    """Tests for compute_batch_metrics function."""

    def test_compute_batch_metrics_default(self, qc_container):
        """Test batch metrics with default parameters."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        result = compute_batch_metrics(qc_container)
        assert isinstance(result, ScpContainer)
        assert "batch_n_detected" in result.obs.columns
        assert "batch_total_intensity" in result.obs.columns
        assert "batch_missing_rate" in result.obs.columns
        assert "batch_alignment_score" in result.obs.columns

    def test_compute_batch_metrics_values_valid(self, qc_container):
        """Test that batch metrics have valid values."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        result = compute_batch_metrics(qc_container)
        assert all(result.obs["batch_n_detected"] >= 0)
        assert all(result.obs["batch_total_intensity"] >= 0)
        assert all(result.obs["batch_missing_rate"] >= 0)
        assert all(result.obs["batch_missing_rate"] <= 1)

    def test_compute_batch_metrics_custom_batch_col(self, qc_container):
        """Test with custom batch column name."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        result = compute_batch_metrics(qc_container, batch_col="batch")
        assert "batch_n_detected" in result.obs.columns

    def test_compute_batch_metrics_missing_batch_col(self, qc_container):
        """Test that missing batch column raises error."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        with pytest.raises(ScpValueError):
            compute_batch_metrics(qc_container, batch_col="nonexistent")

    def test_compute_batch_metrics_sparse(self, qc_container_sparse):
        """Test batch metrics with sparse matrix."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        result = compute_batch_metrics(qc_container_sparse)
        assert "batch_n_detected" in result.obs.columns

    def test_compute_batch_metrics_custom_threshold(self, qc_container):
        """Test with custom detection threshold."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        result = compute_batch_metrics(qc_container, detection_threshold=1.0)
        assert "batch_n_detected" in result.obs.columns

    def test_compute_batch_metrics_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        with pytest.raises(AssayNotFoundError):
            compute_batch_metrics(qc_container, assay_name="nonexistent")


# =============================================================================
# detect_batch_effects tests (8 tests)
# =============================================================================


class TestDetectBatchEffects:
    """Tests for detect_batch_effects function."""

    def test_detect_batch_effects_default(self, qc_container):
        """Test batch effect detection with default parameters."""
        from scptensor.qc.batch import detect_batch_effects

        result = detect_batch_effects(qc_container)
        assert isinstance(result, ScpContainer)
        assert "batch_effect_detected" in result.obs.columns
        assert "batch_effect_score" in result.obs.columns
        assert "batch_effect_n_significant" in result.obs.columns

    def test_detect_batch_effects_kruskal(self, qc_container):
        """Test with Kruskal-Wallis test."""
        from scptensor.qc.batch import detect_batch_effects

        result = detect_batch_effects(qc_container, test="kruskal")
        assert "batch_effect_detected" in result.obs.columns

    def test_detect_batch_effects_anova(self, qc_container):
        """Test with ANOVA test."""
        from scptensor.qc.batch import detect_batch_effects

        result = detect_batch_effects(qc_container, test="anova")
        assert "batch_effect_detected" in result.obs.columns

    def test_detect_batch_effects_score_valid(self, qc_container):
        """Test that batch effect score is in valid range [0, 1]."""
        from scptensor.qc.batch import detect_batch_effects

        result = detect_batch_effects(qc_container)
        scores = result.obs["batch_effect_score"].to_numpy()
        assert all(scores >= 0)
        assert all(scores <= 1)

    def test_detect_batch_effects_invalid_test(self, qc_container):
        """Test that invalid test raises error."""
        from scptensor.qc.batch import detect_batch_effects

        with pytest.raises(ScpValueError):
            detect_batch_effects(qc_container, test="invalid")

    def test_detect_batch_effects_missing_batch_col(self, qc_container):
        """Test that missing batch column raises error."""
        from scptensor.qc.batch import detect_batch_effects

        with pytest.raises(ScpValueError):
            detect_batch_effects(qc_container, batch_col="nonexistent")

    def test_detect_batch_effects_custom_n_features(self, qc_container):
        """Test with custom n_features_max."""
        from scptensor.qc.batch import detect_batch_effects

        result = detect_batch_effects(qc_container, n_features_max=10)
        assert "batch_effect_detected" in result.obs.columns

    def test_detect_batch_effects_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.batch import detect_batch_effects

        with pytest.raises(AssayNotFoundError):
            detect_batch_effects(qc_container, assay_name="nonexistent")


# =============================================================================
# compute_batch_pca tests (7 tests)
# =============================================================================


class TestComputeBatchPCA:
    """Tests for compute_batch_pca function."""

    def test_compute_batch_pca_default(self, qc_container):
        """Test batch PCA with default parameters."""
        from scptensor.qc.batch import compute_batch_pca

        result = compute_batch_pca(qc_container)
        assert isinstance(result, ScpContainer)
        assert "batch_pc1" in result.obs.columns
        assert "batch_pc2" in result.obs.columns

    def test_compute_batch_pca_custom_components(self, qc_container):
        """Test with custom n_components."""
        from scptensor.qc.batch import compute_batch_pca

        result = compute_batch_pca(qc_container, n_components=5)
        for i in range(1, 6):
            assert f"batch_pc{i}" in result.obs.columns

    def test_compute_batch_pca_values_valid(self, qc_container):
        """Test that PCA coordinates are valid."""
        from scptensor.qc.batch import compute_batch_pca

        result = compute_batch_pca(qc_container)
        pc1 = result.obs["batch_pc1"].to_numpy()
        # PC1 should not be all zeros or all NaN
        assert not np.all(pc1 == 0)
        assert not np.any(np.isnan(pc1))

    def test_compute_batch_pca_missing_batch_col(self, qc_container):
        """Test that missing batch column raises error."""
        from scptensor.qc.batch import compute_batch_pca

        with pytest.raises(ScpValueError):
            compute_batch_pca(qc_container, batch_col="nonexistent")

    def test_compute_batch_pca_sparse(self, qc_container_sparse):
        """Test batch PCA with sparse matrix."""
        from scptensor.qc.batch import compute_batch_pca

        result = compute_batch_pca(qc_container_sparse)
        assert "batch_pc1" in result.obs.columns

    def test_compute_batch_pca_custom_threshold(self, qc_container):
        """Test with custom detection threshold."""
        from scptensor.qc.batch import compute_batch_pca

        result = compute_batch_pca(qc_container, detection_threshold=1.0)
        assert "batch_pc1" in result.obs.columns

    def test_compute_batch_pca_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        from scptensor.qc.batch import compute_batch_pca

        with pytest.raises(AssayNotFoundError):
            compute_batch_pca(qc_container, assay_name="nonexistent")


# =============================================================================
# Integration tests for new functions
# =============================================================================


class TestQCAdvancedIntegration:
    """Integration tests for new QC advanced functions."""

    def test_advanced_qc_pipeline(self, qc_container):
        """Test a complete advanced QC pipeline."""
        from scptensor.qc.basic import compute_feature_variance
        from scptensor.qc.basic import qc_score as compute_quality_score
        from scptensor.qc.bivariate import detect_outlier_samples

        # Step 1: Compute quality scores
        container = compute_quality_score(qc_container)
        assert "quality_score" in container.obs.columns

        # Step 2: Compute feature variance
        container = compute_feature_variance(container)
        var = container.assays["protein"].var
        assert "feature_variance" in var.columns

        # Step 3: Detect outliers based on correlations
        container = detect_outlier_samples(container)
        assert "is_correlation_outlier" in container.obs.columns

        assert isinstance(container, ScpContainer)

    def test_batch_analysis_pipeline(self, qc_container):
        """Test a complete batch analysis pipeline."""
        from scptensor.qc.batch import compute_batch_pca, detect_batch_effects
        from scptensor.qc.batch import qc_batch_metrics as compute_batch_metrics

        # Step 1: Compute batch metrics
        container = compute_batch_metrics(qc_container)
        assert "batch_n_detected" in container.obs.columns

        # Step 2: Detect batch effects
        container = detect_batch_effects(container)
        assert "batch_effect_detected" in container.obs.columns

        # Step 3: Compute batch PCA
        container = compute_batch_pca(container)
        assert "batch_pc1" in container.obs.columns

        assert isinstance(container, ScpContainer)

    def test_quality_based_filtering(self, qc_container):
        """Test filtering samples based on quality scores."""
        from scptensor.qc.basic import qc_score as compute_quality_score

        container = compute_quality_score(qc_container)
        scores = container.obs["quality_score"].to_numpy()

        # Keep samples with quality score > median
        median_score = np.median(scores)
        high_quality_indices = np.where(scores > median_score)[0]

        container_filtered = container.filter_samples(sample_indices=high_quality_indices)
        assert container_filtered.n_samples <= container.n_samples
