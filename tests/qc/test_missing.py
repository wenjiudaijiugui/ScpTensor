"""Comprehensive tests for missing value analysis module.

Tests cover:
- analyze_missing_types: Mask type distribution analysis
- compute_missing_stats: Comprehensive missing value statistics
- report_missing_values: scp-style missing value reports

Test categories:
1. Normal functionality tests
2. Mask type recognition tests (VALID, MBR, LOD, FILTERED, IMPUTED)
3. Structural missing detection tests
4. High missing rate sample detection tests
5. Grouped reporting tests
6. Edge cases (all missing, all valid, empty mask, sparse matrices)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.qc.missing import (
    MissingValueReport,
    analyze_missing_types,
    compute_missing_stats,
    report_missing_values,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def missing_obs():
    """Create obs DataFrame for missing value testing."""
    return pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(10)],
            "batch": ["A"] * 5 + ["B"] * 5,
            "condition": ["ctrl"] * 3 + ["treat"] * 4 + ["ctrl"] * 3,
        }
    )


@pytest.fixture
def missing_var():
    """Create var DataFrame for missing value testing."""
    return pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(20)],
            "name": [f"PROT{i}" for i in range(20)],
        }
    )


@pytest.fixture
def missing_container_with_mask(missing_obs, missing_var):
    """Create container with various mask codes for testing."""
    np.random.seed(42)

    # Create data matrix
    X = np.random.exponential(1.0, size=(10, 20))
    X[X < 0.3] = 0  # Set some values to zero

    # Create mask matrix with different codes
    M = np.zeros((10, 20), dtype=np.int8)

    # Set some MBR (1) codes
    M[0, 0:5] = MaskCode.MBR
    M[1, 5:10] = MaskCode.MBR

    # Set some LOD (2) codes
    M[2, 0:3] = MaskCode.LOD
    M[3, 10:15] = MaskCode.LOD

    # Set some FILTERED (3) codes
    M[4, 5:8] = MaskCode.FILTERED

    # Set some IMPUTED (5) codes
    M[5, 15:18] = MaskCode.IMPUTED

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def missing_container_no_mask(missing_obs, missing_var):
    """Create container without mask matrix."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(10, 20))
    X[X < 0.3] = 0

    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def missing_container_sparse(missing_obs, missing_var):
    """Create container with sparse matrix."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(10, 20))
    X[X < 0.5] = 0
    X_sparse = sparse.csr_matrix(X)

    # Create sparse mask
    M = np.zeros((10, 20), dtype=np.int8)
    M[X == 0] = MaskCode.MBR
    M_sparse = sparse.csr_matrix(M)

    matrix = ScpMatrix(X=X_sparse, M=M_sparse)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def container_all_missing(missing_obs, missing_var):
    """Create container with all missing values."""
    X = np.zeros((10, 20))  # All zeros
    M = np.ones((10, 20), dtype=np.int8) * MaskCode.MBR  # All MBR

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def container_all_valid(missing_obs, missing_var):
    """Create container with all valid values."""
    np.random.seed(42)
    X = np.random.exponential(5.0, size=(10, 20))  # All positive
    M = np.zeros((10, 20), dtype=np.int8)  # All VALID

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def container_structural_missing(missing_obs, missing_var):
    """Create container with structural missing features."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(10, 20))

    # Make first 3 features completely missing (structural)
    X[:, 0:3] = 0

    M = np.zeros((10, 20), dtype=np.int8)
    M[:, 0:3] = MaskCode.LOD  # Structural missing

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def container_high_missing_samples(missing_obs, missing_var):
    """Create container with high missing rate samples."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(10, 20))

    # Make last 3 samples have >50% missing
    X[7:, :] = 0
    X[7:, :5] = np.random.exponential(1.0, size=(3, 5))  # Only 5/20 features

    M = np.zeros((10, 20), dtype=np.int8)
    M[7:, :] = MaskCode.MBR
    M[7:, :5] = MaskCode.VALID

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=missing_var, layers={"raw": matrix})
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


@pytest.fixture
def missing_container_multi_layer(missing_obs, missing_var):
    """Create container with multiple layers."""
    np.random.seed(42)
    X_raw = np.random.exponential(1.0, size=(10, 20))
    X_norm = np.random.exponential(2.0, size=(10, 20))

    M_raw = np.zeros((10, 20), dtype=np.int8)
    M_raw[0:3, 0:5] = MaskCode.MBR

    M_norm = np.zeros((10, 20), dtype=np.int8)
    M_norm[3:5, 5:10] = MaskCode.LOD

    assay = Assay(
        var=missing_var,
        layers={
            "raw": ScpMatrix(X=X_raw, M=M_raw),
            "normalized": ScpMatrix(X=X_norm, M=M_norm),
        },
    )
    return ScpContainer(obs=missing_obs, assays={"protein": assay})


# =============================================================================
# analyze_missing_types tests
# =============================================================================


class TestAnalyzeMissingTypes:
    """Tests for analyze_missing_types function."""

    def test_analyze_missing_types_returns_container(self, missing_container_with_mask):
        """Test that analyze_missing_types returns ScpContainer."""
        result = analyze_missing_types(missing_container_with_mask)
        assert isinstance(result, ScpContainer)

    def test_analyze_missing_types_adds_mask_valid_count(self, missing_container_with_mask):
        """Test that mask_valid_count column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_valid_count" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_mbr_count(self, missing_container_with_mask):
        """Test that mask_mbr_count column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_mbr_count" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_lod_count(self, missing_container_with_mask):
        """Test that mask_lod_count column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_lod_count" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_filtered_count(self, missing_container_with_mask):
        """Test that mask_filtered_count column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_filtered_count" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_imputed_count(self, missing_container_with_mask):
        """Test that mask_imputed_count column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_imputed_count" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_valid_rate(self, missing_container_with_mask):
        """Test that mask_valid_rate column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_valid_rate" in result.assays["protein"].var.columns

    def test_analyze_missing_types_adds_mask_missing_rate(self, missing_container_with_mask):
        """Test that mask_missing_rate column is added to var."""
        result = analyze_missing_types(missing_container_with_mask)
        assert "mask_missing_rate" in result.assays["protein"].var.columns

    def test_analyze_missing_types_count_values(self, missing_container_with_mask):
        """Test that mask counts are accurate."""
        result = analyze_missing_types(missing_container_with_mask)
        var = result.assays["protein"].var

        # First 5 columns should have MBR in row 0
        mbr_count = var["mask_mbr_count"].to_numpy()
        assert mbr_count[0] == 1  # Row 0 has MBR

    def test_analyze_missing_types_rate_range(self, missing_container_with_mask):
        """Test that rates are in valid range [0, 1]."""
        result = analyze_missing_types(missing_container_with_mask)
        var = result.assays["protein"].var

        valid_rate = var["mask_valid_rate"].to_numpy()
        missing_rate = var["mask_missing_rate"].to_numpy()

        assert np.all(valid_rate >= 0)
        assert np.all(valid_rate <= 1)
        assert np.all(missing_rate >= 0)
        assert np.all(missing_rate <= 1)

    def test_analyze_missing_types_rates_sum_to_one(self, missing_container_with_mask):
        """Test that valid_rate + missing_rate = 1."""
        result = analyze_missing_types(missing_container_with_mask)
        var = result.assays["protein"].var

        valid_rate = var["mask_valid_rate"].to_numpy()
        missing_rate = var["mask_missing_rate"].to_numpy()

        assert np.allclose(valid_rate + missing_rate, 1.0)

    def test_analyze_missing_types_no_mask(self, missing_container_no_mask):
        """Test behavior when no mask matrix exists."""
        result = analyze_missing_types(missing_container_no_mask)
        var = result.assays["protein"].var

        # Should create default mask with all zeros (VALID)
        valid_count = var["mask_valid_count"].to_numpy()
        n_samples = missing_container_no_mask.n_samples

        assert np.all(valid_count == n_samples)

    def test_analyze_missing_types_sparse_matrix(self, missing_container_sparse):
        """Test with sparse matrix."""
        result = analyze_missing_types(missing_container_sparse)
        var = result.assays["protein"].var

        assert "mask_valid_count" in var.columns
        assert "mask_missing_rate" in var.columns

    def test_analyze_missing_types_custom_layer(self, missing_container_multi_layer):
        """Test with custom layer."""
        result = analyze_missing_types(missing_container_multi_layer, layer_name="normalized")
        var = result.assays["protein"].var

        assert "mask_valid_count" in var.columns

    def test_analyze_missing_types_logs_history(self, missing_container_with_mask):
        """Test that operation is logged to history."""
        result = analyze_missing_types(missing_container_with_mask)
        assert len(result.history) > 0
        assert result.history[-1].action == "analyze_missing_types"

    def test_analyze_missing_types_invalid_assay_raises_error(self, missing_container_with_mask):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            analyze_missing_types(missing_container_with_mask, assay_name="nonexistent")

    def test_analyze_missing_types_invalid_layer_raises_error(self, missing_container_with_mask):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            analyze_missing_types(missing_container_with_mask, layer_name="nonexistent")

    def test_analyze_missing_types_all_missing(self, container_all_missing):
        """Test with all missing data."""
        result = analyze_missing_types(container_all_missing)
        var = result.assays["protein"].var

        valid_count = var["mask_valid_count"].to_numpy()
        mbr_count = var["mask_mbr_count"].to_numpy()

        assert np.all(valid_count == 0)
        assert np.all(mbr_count == 10)  # All 10 samples

    def test_analyze_missing_types_all_valid(self, container_all_valid):
        """Test with all valid data."""
        result = analyze_missing_types(container_all_valid)
        var = result.assays["protein"].var

        valid_count = var["mask_valid_count"].to_numpy()
        mbr_count = var["mask_mbr_count"].to_numpy()

        assert np.all(valid_count == 10)  # All 10 samples
        assert np.all(mbr_count == 0)

    def test_analyze_missing_types_preserves_other_columns(self, missing_container_with_mask):
        """Test that other var columns are preserved."""
        original_var = missing_container_with_mask.assays["protein"].var
        original_cols = set(original_var.columns)

        result = analyze_missing_types(missing_container_with_mask)
        result_var = result.assays["protein"].var
        result_cols = set(result_var.columns)

        # All original columns should still be present
        assert original_cols.issubset(result_cols)


# =============================================================================
# compute_missing_stats tests
# =============================================================================


class TestComputeMissingStats:
    """Tests for compute_missing_stats function."""

    def test_compute_missing_stats_returns_report(self, missing_container_with_mask):
        """Test that compute_missing_stats returns MissingValueReport."""
        result = compute_missing_stats(missing_container_with_mask)
        assert isinstance(result, MissingValueReport)

    def test_compute_missing_stats_has_total_missing_rate(self, missing_container_with_mask):
        """Test that report has total_missing_rate."""
        result = compute_missing_stats(missing_container_with_mask)
        assert hasattr(result, "total_missing_rate")
        assert isinstance(result.total_missing_rate, float)

    def test_compute_missing_stats_has_valid_rate(self, missing_container_with_mask):
        """Test that report has valid_rate."""
        result = compute_missing_stats(missing_container_with_mask)
        assert hasattr(result, "valid_rate")
        assert isinstance(result.valid_rate, float)

    def test_compute_missing_stats_has_mbr_rate(self, missing_container_with_mask):
        """Test that report has mbr_rate."""
        result = compute_missing_stats(missing_container_with_mask)
        assert hasattr(result, "mbr_rate")
        assert isinstance(result.mbr_rate, float)

    def test_compute_missing_stats_has_lod_rate(self, missing_container_with_mask):
        """Test that report has lod_rate."""
        result = compute_missing_stats(missing_container_with_mask)
        assert hasattr(result, "lod_rate")
        assert isinstance(result.lod_rate, float)

    def test_compute_missing_stats_has_imputed_rate(self, missing_container_with_mask):
        """Test that report has imputed_rate."""
        result = compute_missing_stats(missing_container_with_mask)
        assert hasattr(result, "imputed_rate")
        assert isinstance(result.imputed_rate, float)

    def test_compute_missing_stats_rates_sum_valid(self, missing_container_with_mask):
        """Test that valid_rate + total_missing_rate = 1."""
        result = compute_missing_stats(missing_container_with_mask)
        assert np.isclose(result.valid_rate + result.total_missing_rate, 1.0)

    def test_compute_missing_stats_feature_missing_rate_shape(self, missing_container_with_mask):
        """Test that feature_missing_rate has correct shape."""
        result = compute_missing_stats(missing_container_with_mask)
        n_features = missing_container_with_mask.assays["protein"].n_features
        assert len(result.feature_missing_rate) == n_features

    def test_compute_missing_stats_sample_missing_rate_shape(self, missing_container_with_mask):
        """Test that sample_missing_rate has correct shape."""
        result = compute_missing_stats(missing_container_with_mask)
        n_samples = missing_container_with_mask.n_samples
        assert len(result.sample_missing_rate) == n_samples

    def test_compute_missing_stats_rates_in_range(self, missing_container_with_mask):
        """Test that all rates are in [0, 1]."""
        result = compute_missing_stats(missing_container_with_mask)

        assert 0 <= result.total_missing_rate <= 1
        assert 0 <= result.valid_rate <= 1
        assert 0 <= result.mbr_rate <= 1
        assert 0 <= result.lod_rate <= 1
        assert 0 <= result.imputed_rate <= 1

        assert np.all(result.feature_missing_rate >= 0)
        assert np.all(result.feature_missing_rate <= 1)

        assert np.all(result.sample_missing_rate >= 0)
        assert np.all(result.sample_missing_rate <= 1)

    def test_compute_missing_stats_identifies_structural_missing(
        self, container_structural_missing
    ):
        """Test that structural missing features are identified."""
        result = compute_missing_stats(container_structural_missing)
        assert len(result.structural_missing_features) == 3
        assert "protein_0" in result.structural_missing_features
        assert "protein_1" in result.structural_missing_features
        assert "protein_2" in result.structural_missing_features

    def test_compute_missing_stats_structural_missing_empty_when_none(
        self, missing_container_with_mask
    ):
        """Test structural_missing_features is empty when no structural missing."""
        result = compute_missing_stats(missing_container_with_mask)
        assert len(result.structural_missing_features) == 0

    def test_compute_missing_stats_identifies_high_missing_samples(
        self, container_high_missing_samples
    ):
        """Test that high missing samples are identified."""
        result = compute_missing_stats(container_high_missing_samples)
        # Samples 7, 8, 9 have >50% missing
        assert len(result.samples_with_high_missing) == 3
        assert "sample_7" in result.samples_with_high_missing

    def test_compute_missing_stats_custom_threshold(self, container_high_missing_samples):
        """Test with custom high_missing_threshold."""
        # Lower threshold should catch more samples
        result_low = compute_missing_stats(
            container_high_missing_samples, high_missing_threshold=0.3
        )
        result_default = compute_missing_stats(container_high_missing_samples)

        assert len(result_low.samples_with_high_missing) >= len(
            result_default.samples_with_high_missing
        )

    def test_compute_missing_stats_threshold_out_of_range_low(self, missing_container_with_mask):
        """Test that threshold < 0 raises ScpValueError."""
        with pytest.raises(ScpValueError) as excinfo:
            compute_missing_stats(missing_container_with_mask, high_missing_threshold=-0.1)
        assert "high_missing_threshold" in str(excinfo.value).lower()

    def test_compute_missing_stats_threshold_out_of_range_high(self, missing_container_with_mask):
        """Test that threshold > 1 raises ScpValueError."""
        with pytest.raises(ScpValueError) as excinfo:
            compute_missing_stats(missing_container_with_mask, high_missing_threshold=1.5)
        assert "high_missing_threshold" in str(excinfo.value).lower()

    def test_compute_missing_stats_no_mask(self, missing_container_no_mask):
        """Test behavior when no mask matrix exists."""
        result = compute_missing_stats(missing_container_no_mask)
        # Should create default mask (all valid for non-zero values)
        assert result.total_missing_rate >= 0

    def test_compute_missing_stats_sparse_matrix(self, missing_container_sparse):
        """Test with sparse matrix."""
        result = compute_missing_stats(missing_container_sparse)
        assert isinstance(result, MissingValueReport)
        assert result.total_missing_rate >= 0

    def test_compute_missing_stats_custom_layer(self, missing_container_multi_layer):
        """Test with custom layer."""
        result = compute_missing_stats(missing_container_multi_layer, layer_name="normalized")
        assert isinstance(result, MissingValueReport)

    def test_compute_missing_stats_all_missing(self, container_all_missing):
        """Test with all missing data."""
        result = compute_missing_stats(container_all_missing)
        assert result.valid_rate == 0.0
        assert result.total_missing_rate == 1.0
        assert len(result.structural_missing_features) == 20  # All features

    def test_compute_missing_stats_all_valid(self, container_all_valid):
        """Test with all valid data."""
        result = compute_missing_stats(container_all_valid)
        assert result.valid_rate == 1.0
        assert result.total_missing_rate == 0.0
        assert len(result.samples_with_high_missing) == 0

    def test_compute_missing_stats_invalid_assay_raises_error(self, missing_container_with_mask):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_missing_stats(missing_container_with_mask, assay_name="nonexistent")

    def test_compute_missing_stats_invalid_layer_raises_error(self, missing_container_with_mask):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_missing_stats(missing_container_with_mask, layer_name="nonexistent")


# =============================================================================
# report_missing_values tests
# =============================================================================


class TestReportMissingValues:
    """Tests for report_missing_values function."""

    def test_report_missing_values_returns_dataframe(self, missing_container_with_mask):
        """Test that report_missing_values returns pl.DataFrame."""
        result = report_missing_values(missing_container_with_mask)
        assert isinstance(result, pl.DataFrame)

    def test_report_missing_values_has_group_column(self, missing_container_with_mask):
        """Test that report has 'group' column when by is None."""
        result = report_missing_values(missing_container_with_mask, by=None)
        # When by is None, group column should be 'all'
        if "group" in result.columns:
            assert result["group"][0] == "all"

    def test_report_missing_values_has_local_sensitivity_mean(self, missing_container_with_mask):
        """Test that report has LocalSensitivityMean column."""
        result = report_missing_values(missing_container_with_mask)
        assert "LocalSensitivityMean" in result.columns

    def test_report_missing_values_has_local_sensitivity_sd(self, missing_container_with_mask):
        """Test that report has LocalSensitivitySd column."""
        result = report_missing_values(missing_container_with_mask)
        assert "LocalSensitivitySd" in result.columns

    def test_report_missing_values_has_total_sensitivity(self, missing_container_with_mask):
        """Test that report has TotalSensitivity column."""
        result = report_missing_values(missing_container_with_mask)
        assert "TotalSensitivity" in result.columns

    def test_report_missing_values_has_completeness(self, missing_container_with_mask):
        """Test that report has Completeness column."""
        result = report_missing_values(missing_container_with_mask)
        assert "Completeness" in result.columns

    def test_report_missing_values_has_number_cells(self, missing_container_with_mask):
        """Test that report has NumberCells column."""
        result = report_missing_values(missing_container_with_mask)
        assert "NumberCells" in result.columns

    def test_report_missing_values_local_sensitivity_positive(self, missing_container_with_mask):
        """Test that LocalSensitivityMean is non-negative."""
        result = report_missing_values(missing_container_with_mask)
        assert all(result["LocalSensitivityMean"] >= 0)

    def test_report_missing_values_completeness_in_range(self, missing_container_with_mask):
        """Test that Completeness is in [0, 1]."""
        result = report_missing_values(missing_container_with_mask)
        assert all(result["Completeness"] >= 0)
        assert all(result["Completeness"] <= 1)

    def test_report_missing_values_number_cells_matches_total(self, missing_container_with_mask):
        """Test that NumberCells matches total samples when no grouping."""
        result = report_missing_values(missing_container_with_mask)
        assert result["NumberCells"][0] == missing_container_with_mask.n_samples

    def test_report_missing_values_grouped_by_batch(self, missing_container_with_mask):
        """Test grouping by batch column."""
        result = report_missing_values(missing_container_with_mask, by="batch")
        assert len(result) == 2  # Two batches: A and B
        assert "group" in result.columns

    def test_report_missing_values_grouped_number_cells_sum(self, missing_container_with_mask):
        """Test that grouped NumberCells sum to total."""
        result = report_missing_values(missing_container_with_mask, by="batch")
        total_cells = result["NumberCells"].sum()
        assert total_cells == missing_container_with_mask.n_samples

    def test_report_missing_values_group_names(self, missing_container_with_mask):
        """Test that group names match obs column values."""
        result = report_missing_values(missing_container_with_mask, by="batch")
        groups = set(result["group"].to_list())
        assert groups == {"A", "B"}

    def test_report_missing_values_invalid_group_column_raises_error(
        self, missing_container_with_mask
    ):
        """Test that invalid group column raises ScpValueError."""
        with pytest.raises(ScpValueError) as excinfo:
            report_missing_values(missing_container_with_mask, by="nonexistent")
        assert "nonexistent" in str(excinfo.value)

    def test_report_missing_values_custom_threshold(self, missing_container_with_mask):
        """Test with custom detection_threshold."""
        result = report_missing_values(missing_container_with_mask, detection_threshold=1.0)
        # Higher threshold should detect fewer features
        assert all(result["LocalSensitivityMean"] >= 0)

    def test_report_missing_values_no_mask(self, missing_container_no_mask):
        """Test behavior when no mask matrix exists."""
        result = report_missing_values(missing_container_no_mask)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_report_missing_values_sparse_matrix(self, missing_container_sparse):
        """Test with sparse matrix."""
        result = report_missing_values(missing_container_sparse)
        assert isinstance(result, pl.DataFrame)

    def test_report_missing_values_custom_layer(self, missing_container_multi_layer):
        """Test with custom layer."""
        result = report_missing_values(missing_container_multi_layer, layer_name="normalized")
        assert isinstance(result, pl.DataFrame)

    def test_report_missing_values_all_missing(self, container_all_missing):
        """Test with all missing data."""
        result = report_missing_values(container_all_missing)
        assert result["LocalSensitivityMean"][0] == 0
        assert result["Completeness"][0] == 0

    def test_report_missing_values_all_valid(self, container_all_valid):
        """Test with all valid data."""
        result = report_missing_values(container_all_valid)
        assert result["LocalSensitivityMean"][0] > 0
        assert result["Completeness"][0] == 1.0

    def test_report_missing_values_invalid_assay_raises_error(self, missing_container_with_mask):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            report_missing_values(missing_container_with_mask, assay_name="nonexistent")

    def test_report_missing_values_invalid_layer_raises_error(self, missing_container_with_mask):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            report_missing_values(missing_container_with_mask, layer_name="nonexistent")

    def test_report_missing_values_total_sensitivity_leq_features(
        self, missing_container_with_mask
    ):
        """Test that TotalSensitivity <= total features."""
        result = report_missing_values(missing_container_with_mask)
        n_features = missing_container_with_mask.assays["protein"].n_features
        assert all(result["TotalSensitivity"] <= n_features)

    def test_report_missing_values_grouped_by_condition(self, missing_container_with_mask):
        """Test grouping by condition column."""
        result = report_missing_values(missing_container_with_mask, by="condition")
        assert len(result) == 2  # ctrl and treat
        groups = set(result["group"].to_list())
        assert groups == {"ctrl", "treat"}


# =============================================================================
# Integration tests
# =============================================================================


class TestMissingValueIntegration:
    """Integration tests for missing value analysis."""

    def test_full_workflow_with_mask(self, missing_container_with_mask):
        """Test full workflow: analyze -> stats -> report."""
        # Step 1: Analyze missing types
        container = analyze_missing_types(missing_container_with_mask)
        assert "mask_missing_rate" in container.assays["protein"].var.columns

        # Step 2: Compute missing stats
        stats = compute_missing_stats(container)
        assert isinstance(stats, MissingValueReport)

        # Step 3: Generate report
        report = report_missing_values(container)
        assert isinstance(report, pl.DataFrame)

    def test_full_workflow_without_mask(self, missing_container_no_mask):
        """Test full workflow with container lacking mask matrix."""
        container = analyze_missing_types(missing_container_no_mask)
        stats = compute_missing_stats(container)
        report = report_missing_values(container)

        assert "mask_missing_rate" in container.assays["protein"].var.columns
        assert isinstance(stats, MissingValueReport)
        assert isinstance(report, pl.DataFrame)

    def test_consistency_between_functions(self, missing_container_with_mask):
        """Test that functions produce internally consistent results."""
        stats = compute_missing_stats(missing_container_with_mask)
        report = report_missing_values(missing_container_with_mask)

        # Both functions should report the same number of samples analyzed
        assert report["NumberCells"][0] == missing_container_with_mask.n_samples

        # Report's total sensitivity should not exceed total features
        n_features = missing_container_with_mask.assays["protein"].n_features
        assert report["TotalSensitivity"][0] <= n_features

        # Stats total_missing_rate should be in valid range
        assert 0 <= stats.total_missing_rate <= 1

    def test_grouped_report_consistency(self, missing_container_with_mask):
        """Test that grouped report totals match overall report."""
        overall = report_missing_values(missing_container_with_mask)
        grouped = report_missing_values(missing_container_with_mask, by="batch")

        # Sum of grouped NumberCells should equal overall
        assert grouped["NumberCells"].sum() == overall["NumberCells"][0]
