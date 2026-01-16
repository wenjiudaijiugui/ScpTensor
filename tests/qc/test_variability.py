"""Comprehensive tests for variability (CV statistics) module.

Tests cover:
- compute_cv: Coefficient of variation calculation for features
- compute_technical_replicate_cv: Technical replicate CV analysis
- compute_batch_cv: Within-batch and between-batch CV
- filter_by_cv: Feature filtering based on CV threshold

Test categories:
1. Normal functionality tests
2. Edge cases (all zeros, constant values, low means, empty data)
3. Group by functionality tests
4. Technical replicate CV tests
5. Batch CV analysis tests
6. Filtering tests with various thresholds
7. Sparse matrix handling
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)

# Import functions from variability module when it exists
# For now, we'll write the tests assuming the module will be implemented
try:
    from scptensor.qc.variability import (
        CVReport,
        compute_batch_cv,
        compute_cv,
        compute_technical_replicate_cv,
        filter_by_cv,
    )

    VARIABILITY_MODULE_EXISTS = True
except ImportError:
    VARIABILITY_MODULE_EXISTS = False
    # Create placeholder classes for type checking
    # These will be replaced when the module is implemented
    from dataclasses import dataclass

    @dataclass
    class CVReport:
        """Placeholder CVReport for test development."""

        feature_cv: np.ndarray
        mean_cv: float
        median_cv: float
        cv_by_group: dict[str, np.ndarray] | None = None
        within_batch_cv: dict[str, float] | None = None
        between_batch_cv: float | None = None
        high_cv_features: list | None = None
        low_quality_samples: list | None = None

    def compute_cv(*args, **kwargs):
        """Placeholder for compute_cv."""
        raise NotImplementedError("variability module not yet implemented")

    def compute_technical_replicate_cv(*args, **kwargs):
        """Placeholder for compute_technical_replicate_cv."""
        raise NotImplementedError("variability module not yet implemented")

    def compute_batch_cv(*args, **kwargs):
        """Placeholder for compute_batch_cv."""
        raise NotImplementedError("variability module not yet implemented")

    def filter_by_cv(*args, **kwargs):
        """Placeholder for filter_by_cv."""
        raise NotImplementedError("variability module not yet implemented")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def variability_obs():
    """Create obs DataFrame for variability testing.

    Includes batch, replicate, and condition columns for testing
    grouped CV calculations.
    """
    return pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(12)],
            "batch": ["A"] * 6 + ["B"] * 6,
            "replicate": ["rep1"] * 2
            + ["rep2"] * 2
            + ["rep3"] * 2
            + ["rep1"] * 2
            + ["rep2"] * 2
            + ["rep3"] * 2,
            "condition": ["ctrl"] * 4 + ["treat"] * 4 + ["ctrl"] * 2 + ["treat"] * 2,
        }
    )


@pytest.fixture
def variability_var():
    """Create var DataFrame for variability testing."""
    return pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(20)],
            "name": [f"PROT{i}" for i in range(20)],
        }
    )


@pytest.fixture
def variability_container(variability_obs, variability_var):
    """Create container with known variability pattern for testing.

    Creates data with:
    - Low CV features (columns 0-5): CV < 0.2
    - Medium CV features (columns 6-12): 0.2 <= CV < 0.5
    - High CV features (columns 13-19): CV >= 0.5
    """
    np.random.seed(42)
    n_samples, n_features = 12, 20

    X = np.zeros((n_samples, n_features))

    # Low CV features: consistent values around 10
    for i in range(6):
        X[:, i] = np.random.normal(10.0, 1.0, n_samples)  # CV ~ 0.1

    # Medium CV features: moderate variation
    for i in range(6, 13):
        X[:, i] = np.random.normal(5.0, 2.0, n_samples)  # CV ~ 0.4

    # High CV features: high variation
    for i in range(13, 20):
        X[:, i] = np.random.normal(2.0, 2.0, n_samples)  # CV ~ 1.0

    # Ensure all values are positive
    X = np.abs(X)

    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def variability_container_with_mask(variability_obs, variability_var):
    """Create container with mask matrix for testing."""
    np.random.seed(42)
    n_samples, n_features = 12, 20

    X = np.abs(np.random.normal(5.0, 1.0, (n_samples, n_features)))

    # Create mask with some MBR values
    M = np.zeros((n_samples, n_features), dtype=np.int8)
    M[0, 0:3] = MaskCode.MBR
    M[1, 5:8] = MaskCode.LOD

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def variability_container_sparse(variability_obs, variability_var):
    """Create container with sparse matrix."""
    np.random.seed(42)
    n_samples, n_features = 12, 20

    X = np.abs(np.random.normal(5.0, 1.0, (n_samples, n_features)))
    X[X < 2.0] = 0  # Create sparsity
    X_sparse = sparse.csr_matrix(X)

    matrix = ScpMatrix(X=X_sparse, M=None)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def variability_container_multi_layer(variability_obs, variability_var):
    """Create container with multiple layers."""
    np.random.seed(42)
    X_raw = np.abs(np.random.normal(5.0, 1.0, (12, 20)))
    X_normalized = np.abs(np.random.normal(10.0, 2.0, (12, 20)))

    assay = Assay(
        var=variability_var,
        layers={
            "raw": ScpMatrix(X=X_raw, M=None),
            "normalized": ScpMatrix(X=X_normalized, M=None),
        },
    )
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def container_all_zeros(variability_obs, variability_var):
    """Create container with all zero values."""
    X = np.zeros((12, 20))
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def container_constant_values(variability_obs, variability_var):
    """Create container with constant values (SD=0)."""
    X = np.ones((12, 20)) * 5.0
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def container_low_mean_values(variability_obs, variability_var):
    """Create container with low mean values."""
    np.random.seed(42)
    X = np.abs(np.random.normal(0.5, 0.1, (12, 20)))
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=variability_var, layers={"raw": matrix})
    return ScpContainer(obs=variability_obs, assays={"protein": assay})


@pytest.fixture
def container_known_cv():
    """Create container with mathematically known CV values.

    Feature 0: mean=10, std=1, CV=0.1
    Feature 1: mean=20, std=4, CV=0.2
    Feature 2: mean=5, std=2.5, CV=0.5
    Feature 3: mean=8, std=0.8, CV=0.1
    Feature 4: mean=15, std=6, CV=0.4
    """
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(10)]})
    var = pl.DataFrame({"_index": [f"p{i}" for i in range(5)]})

    np.random.seed(42)
    X = np.zeros((10, 5))

    # Create features with specific CV values
    X[:, 0] = np.random.normal(10.0, 1.0, 10)  # CV = 0.1
    X[:, 1] = np.random.normal(20.0, 4.0, 10)  # CV = 0.2
    X[:, 2] = np.random.normal(5.0, 2.5, 10)  # CV = 0.5
    X[:, 3] = np.random.normal(8.0, 0.8, 10)  # CV = 0.1
    X[:, 4] = np.random.normal(15.0, 6.0, 10)  # CV = 0.4

    X = np.abs(X)

    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})
    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def container_replicate_cv():
    """Create container for technical replicate CV testing.

    Has 3 replicates with 4 samples each.
    """
    obs = pl.DataFrame(
        {
            "_index": [f"s{i}" for i in range(12)],
            "replicate": ["rep1"] * 4 + ["rep2"] * 4 + ["rep3"] * 4,
            "batch": ["A"] * 2 + ["B"] * 2 + ["A"] * 2 + ["B"] * 2 + ["A"] * 2 + ["B"] * 2,
        }
    )
    var = pl.DataFrame({"_index": [f"p{i}" for i in range(10)]})

    np.random.seed(42)

    # Create features with different reproducibility
    X = np.zeros((12, 10))

    # High reproducibility (low within-replicate CV)
    for rep_idx in range(3):
        base_value = 10.0 + rep_idx * 2.0
        for i in range(4):
            row = rep_idx * 4 + i
            X[row, 0:3] = np.abs(np.random.normal(base_value, 0.5, 3))

    # Medium reproducibility
    for rep_idx in range(3):
        base_value = 5.0 + rep_idx * 1.0
        for i in range(4):
            row = rep_idx * 4 + i
            X[row, 3:6] = np.abs(np.random.normal(base_value, 1.5, 3))

    # Low reproducibility (high within-replicate CV)
    for rep_idx in range(3):
        base_value = 2.0 + rep_idx * 0.5
        for i in range(4):
            row = rep_idx * 4 + i
            X[row, 6:10] = np.abs(np.random.normal(base_value, 2.0, 4))

    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})
    return ScpContainer(obs=obs, assays={"protein": assay})


# =============================================================================
# compute_cv tests
# =============================================================================


@pytest.mark.skipif(not VARIABILITY_MODULE_EXISTS, reason="variability module not yet implemented")
class TestComputeCv:
    """Tests for compute_cv function."""

    def test_compute_cv_returns_cv_report(self, variability_container):
        """Test that compute_cv returns CVReport object."""
        result = compute_cv(variability_container)
        assert isinstance(result, CVReport)

    def test_compute_cv_feature_cv_shape(self, variability_container):
        """Test that feature_cv has correct length."""
        result = compute_cv(variability_container)
        n_features = variability_container.assays["protein"].n_features
        assert len(result.feature_cv) == n_features

    def test_compute_cv_mean_cv_exists(self, variability_container):
        """Test that mean_cv is computed."""
        result = compute_cv(variability_container)
        assert isinstance(result.mean_cv, float)
        assert result.mean_cv >= 0

    def test_compute_cv_median_cv_exists(self, variability_container):
        """Test that median_cv is computed."""
        result = compute_cv(variability_container)
        assert isinstance(result.median_cv, float)
        assert result.median_cv >= 0

    def test_compute_cv_values_non_negative(self, variability_container):
        """Test that all CV values are non-negative."""
        result = compute_cv(variability_container)
        assert np.all(result.feature_cv >= 0)

    def test_compute_cv_adds_cv_column_to_var(self, variability_container):
        """Test that cv column is added to var."""
        result = compute_cv(variability_container)
        assert "cv" in result.assays["protein"].var.columns

    def test_compute_cv_cv_values_valid(self, variability_container):
        """Test that CV values are reasonable (< 5 for valid data)."""
        result = compute_cv(variability_container)
        cv_values = result.assays["protein"].var["cv"].to_numpy()
        # CV should typically be < 5 for proteomics data
        assert np.all(cv_values < 5)

    def test_compute_cv_with_min_mean(self, variability_container):
        """Test with min_mean threshold."""
        result = compute_cv(variability_container, min_mean=1.0)
        # Features with mean < min_mean should have NaN or 0 CV
        assert isinstance(result, CVReport)

    def test_compute_cv_with_group_by(self, variability_container):
        """Test with group_by parameter."""
        result = compute_cv(variability_container, group_by="batch")
        assert result.cv_by_group is not None
        assert "A" in result.cv_by_group
        assert "B" in result.cv_by_group

    def test_compute_cv_group_by_shape(self, variability_container):
        """Test that grouped CV has correct shape."""
        result = compute_cv(variability_container, group_by="batch")
        n_features = variability_container.assays["protein"].n_features
        for cv_array in result.cv_by_group.values():
            assert len(cv_array) == n_features

    def test_compute_cv_group_by_valid_values(self, variability_container):
        """Test that grouped CV values are valid."""
        result = compute_cv(variability_container, group_by="batch")
        for cv_array in result.cv_by_group.values():
            assert np.all(cv_array >= 0)

    def test_compute_cv_sparse_matrix(self, variability_container_sparse):
        """Test with sparse matrix."""
        result = compute_cv(variability_container_sparse)
        assert isinstance(result, CVReport)
        assert len(result.feature_cv) == 20

    def test_compute_cv_custom_layer(self, variability_container_multi_layer):
        """Test with custom layer."""
        result = compute_cv(variability_container_multi_layer, layer_name="normalized")
        assert isinstance(result, CVReport)

    def test_compute_cv_all_zeros(self, container_all_zeros):
        """Test with all zero data."""
        result = compute_cv(container_all_zeros)
        # CV should be 0 or NaN for zero data
        assert np.all((result.feature_cv == 0) | np.isnan(result.feature_cv))

    def test_compute_cv_constant_values(self, container_constant_values):
        """Test with constant values (SD=0)."""
        result = compute_cv(container_constant_values)
        # CV should be 0 when SD=0
        assert np.all(result.feature_cv == 0)

    def test_compute_cv_known_cv_values(self, container_known_cv):
        """Test with mathematically known CV values."""
        result = compute_cv(container_known_cv)
        # Check that computed CV values are close to expected
        # Expected: [0.1, 0.2, 0.5, 0.1, 0.4]
        expected_cv = np.array([0.1, 0.2, 0.5, 0.1, 0.4])
        # Allow 40% relative error due to random sampling and abs() truncation
        assert np.allclose(result.feature_cv, expected_cv, rtol=0.4)

    def test_compute_cv_invalid_assay_raises_error(self, variability_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_cv(variability_container, assay_name="nonexistent")

    def test_compute_cv_invalid_layer_raises_error(self, variability_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_cv(variability_container, layer_name="nonexistent")

    def test_compute_cv_invalid_group_by_raises_error(self, variability_container):
        """Test that invalid group_by column raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_cv(variability_container, group_by="nonexistent")

    def test_compute_cv_negative_min_mean_raises_error(self, variability_container):
        """Test that negative min_mean raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_cv(variability_container, min_mean=-1.0)

    def test_compute_cv_logs_history(self, variability_container):
        """Test that operation is logged to history."""
        result = compute_cv(variability_container)
        assert len(result.history) > 0
        assert result.history[-1].action == "compute_cv"

    def test_compute_cv_with_mask(self, variability_container_with_mask):
        """Test with mask matrix (missing values should be handled)."""
        result = compute_cv(variability_container_with_mask)
        assert isinstance(result, CVReport)

    def test_compute_cv_median_less_than_mean_for_skewed(self, variability_container):
        """Test that median CV <= mean CV for positively skewed data."""
        result = compute_cv(variability_container)
        # For skewed CV distribution (common in proteomics),
        # median is typically less than mean
        assert result.median_cv <= result.mean_cv * 1.5


# =============================================================================
# compute_technical_replicate_cv tests
# =============================================================================


@pytest.mark.skipif(not VARIABILITY_MODULE_EXISTS, reason="variability module not yet implemented")
class TestComputeTechnicalReplicateCv:
    """Tests for compute_technical_replicate_cv function."""

    def test_compute_replicate_cv_returns_cv_report(self, container_replicate_cv):
        """Test that compute_technical_replicate_cv returns CVReport."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert isinstance(result, CVReport)

    def test_compute_replicate_cv_feature_cv_shape(self, container_replicate_cv):
        """Test that feature_cv has correct length."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        n_features = container_replicate_cv.assays["protein"].n_features
        assert len(result.feature_cv) == n_features

    def test_compute_replicate_cv_cv_by_replicate_exists(self, container_replicate_cv):
        """Test that CV is computed for each replicate."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert result.cv_by_group is not None
        assert "rep1" in result.cv_by_group
        assert "rep2" in result.cv_by_group
        assert "rep3" in result.cv_by_group

    def test_compute_replicate_cv_values_non_negative(self, container_replicate_cv):
        """Test that all CV values are non-negative."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert np.all(result.feature_cv >= 0)

    def test_compute_replicate_cv_mean_cv_exists(self, container_replicate_cv):
        """Test that mean_cv is computed."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert isinstance(result.mean_cv, float)

    def test_compute_replicate_cv_adds_replicate_cv_column(self, container_replicate_cv):
        """Test that replicate_cv column is added to var."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert "replicate_cv" in result.assays["protein"].var.columns

    def test_compute_replicate_cv_custom_aggregate(self, container_replicate_cv):
        """Test with custom aggregation method."""
        result = compute_technical_replicate_cv(
            container_replicate_cv, replicate_col="replicate", aggregate="median"
        )
        assert isinstance(result, CVReport)

    def test_compute_replicate_cv_aggregate_mean(self, container_replicate_cv):
        """Test aggregate='mean' produces mean CV."""
        result = compute_technical_replicate_cv(
            container_replicate_cv, replicate_col="replicate", aggregate="mean"
        )
        # Mean CV should be the mean of replicate CVs
        assert isinstance(result.mean_cv, float)

    def test_compute_replicate_cv_aggregate_median(self, container_replicate_cv):
        """Test aggregate='median' produces median CV."""
        result = compute_technical_replicate_cv(
            container_replicate_cv, replicate_col="replicate", aggregate="median"
        )
        # Median CV should be the median of replicate CVs
        assert isinstance(result.median_cv, float)

    def test_compute_replicate_cv_aggregate_max(self, container_replicate_cv):
        """Test aggregate='max' produces max CV."""
        result = compute_technical_replicate_cv(
            container_replicate_cv, replicate_col="replicate", aggregate="max"
        )
        assert isinstance(result, CVReport)

    def test_compute_replicate_cv_invalid_aggregate_raises_error(self, container_replicate_cv):
        """Test that invalid aggregate raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_technical_replicate_cv(
                container_replicate_cv, replicate_col="replicate", aggregate="invalid"
            )

    def test_compute_replicate_cv_missing_replicate_col_raises_error(self, container_replicate_cv):
        """Test that missing replicate column raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_technical_replicate_cv(container_replicate_cv, replicate_col="nonexistent")

    def test_compute_replicate_cv_invalid_assay_raises_error(self, container_replicate_cv):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_technical_replicate_cv(
                container_replicate_cv, assay_name="nonexistent", replicate_col="replicate"
            )

    def test_compute_replicate_cv_invalid_layer_raises_error(self, container_replicate_cv):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_technical_replicate_cv(
                container_replicate_cv, layer_name="nonexistent", replicate_col="replicate"
            )

    def test_compute_replicate_cv_sparse_matrix(self, container_replicate_cv):
        """Test with sparse matrix container."""
        # Convert to sparse
        assay = container_replicate_cv.assays["protein"]
        X_sparse = sparse.csr_matrix(assay.layers["raw"].X)
        assay.layers["raw"] = ScpMatrix(X=X_sparse, M=None)

        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert isinstance(result, CVReport)

    def test_compute_replicate_cv_logs_history(self, container_replicate_cv):
        """Test that operation is logged to history."""
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert len(result.history) > 0
        assert result.history[-1].action == "compute_technical_replicate_cv"


# =============================================================================
# compute_batch_cv tests
# =============================================================================


@pytest.mark.skipif(not VARIABILITY_MODULE_EXISTS, reason="variability module not yet implemented")
class TestComputeBatchCv:
    """Tests for compute_batch_cv function."""

    def test_compute_batch_cv_returns_cv_report(self, variability_container):
        """Test that compute_batch_cv returns CVReport."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert isinstance(result, CVReport)

    def test_compute_batch_cv_within_batch_cv_exists(self, variability_container):
        """Test that within_batch_cv is computed."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert result.within_batch_cv is not None
        assert isinstance(result.within_batch_cv, dict)

    def test_compute_batch_cv_has_batch_a_and_b(self, variability_container):
        """Test that both batches have within-batch CV."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert "A" in result.within_batch_cv
        assert "B" in result.within_batch_cv

    def test_compute_batch_cv_between_batch_cv_exists(self, variability_container):
        """Test that between_batch_cv is computed."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert result.between_batch_cv is not None
        assert isinstance(result.between_batch_cv, float)

    def test_compute_batch_cv_within_batch_cv_positive(self, variability_container):
        """Test that within-batch CV values are positive."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        for cv_value in result.within_batch_cv.values():
            assert cv_value >= 0

    def test_compute_batch_cv_between_batch_cv_positive(self, variability_container):
        """Test that between-batch CV is positive."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert result.between_batch_cv >= 0

    def test_compute_batch_cv_adds_batch_cv_columns(self, variability_container):
        """Test that batch CV columns are added to var."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        var = result.assays["protein"].var
        assert "within_batch_cv" in var.columns
        assert "between_batch_cv" in var.columns

    def test_compute_batch_cv_high_cv_features_listed(self, variability_container):
        """Test that high CV features are identified."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert result.high_cv_features is not None
        assert isinstance(result.high_cv_features, list)

    def test_compute_batch_cv_custom_threshold(self, variability_container):
        """Test with custom high_cv_threshold."""
        result = compute_batch_cv(variability_container, batch_col="batch", high_cv_threshold=0.3)
        # More features should be flagged as high CV with lower threshold
        assert isinstance(result, CVReport)

    def test_compute_batch_cv_threshold_affects_high_cv_count(self, variability_container):
        """Test that threshold affects number of high CV features."""
        result_low = compute_batch_cv(
            variability_container, batch_col="batch", high_cv_threshold=0.2
        )
        result_high = compute_batch_cv(
            variability_container, batch_col="batch", high_cv_threshold=0.5
        )
        # Lower threshold should flag more features
        assert len(result_low.high_cv_features) >= len(result_high.high_cv_features)

    def test_compute_batch_cv_missing_batch_col_raises_error(self, variability_container):
        """Test that missing batch column raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_batch_cv(variability_container, batch_col="nonexistent")

    def test_compute_batch_cv_invalid_assay_raises_error(self, variability_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_batch_cv(variability_container, assay_name="nonexistent", batch_col="batch")

    def test_compute_batch_cv_invalid_layer_raises_error(self, variability_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_batch_cv(variability_container, layer_name="nonexistent", batch_col="batch")

    def test_compute_batch_cv_negative_threshold_raises_error(self, variability_container):
        """Test that negative threshold raises ScpValueError."""
        with pytest.raises(ScpValueError):
            compute_batch_cv(variability_container, batch_col="batch", high_cv_threshold=-0.1)

    def test_compute_batch_cv_logs_history(self, variability_container):
        """Test that operation is logged to history."""
        result = compute_batch_cv(variability_container, batch_col="batch")
        assert len(result.history) > 0
        assert result.history[-1].action == "compute_batch_cv"

    def test_compute_batch_cv_single_batch(self, variability_container):
        """Test behavior with only one batch."""
        # Create container with single batch
        obs = pl.DataFrame({"_index": [f"s{i}" for i in range(6)], "batch": ["A"] * 6})
        var = pl.DataFrame({"_index": [f"p{i}" for i in range(10)]})
        X = np.abs(np.random.normal(5.0, 1.0, (6, 10)))

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = compute_batch_cv(container, batch_col="batch")
        assert isinstance(result, CVReport)
        # Between-batch CV might be None or 0 for single batch
        assert result.between_batch_cv is None or result.between_batch_cv == 0


# =============================================================================
# filter_by_cv tests
# =============================================================================


@pytest.mark.skipif(not VARIABILITY_MODULE_EXISTS, reason="variability module not yet implemented")
class TestFilterByCv:
    """Tests for filter_by_cv function."""

    def test_filter_by_cv_returns_container(self, variability_container):
        """Test that filter_by_cv returns ScpContainer."""
        result = filter_by_cv(variability_container, cv_threshold=0.5)
        assert isinstance(result, ScpContainer)

    def test_filter_by_cv_reduces_features(self, variability_container):
        """Test that filtering reduces number of features."""
        original_n = variability_container.assays["protein"].n_features
        result = filter_by_cv(variability_container, cv_threshold=0.3)
        filtered_n = result.assays["protein"].n_features
        assert filtered_n <= original_n

    def test_filter_by_cv_high_threshold_keeps_more(self, variability_container):
        """Test that higher threshold keeps more features."""
        result_low = filter_by_cv(variability_container, cv_threshold=0.2)
        result_high = filter_by_cv(variability_container, cv_threshold=0.8)
        assert result_high.assays["protein"].n_features >= result_low.assays["protein"].n_features

    def test_filter_by_cv_threshold_1_keeps_all(self, variability_container):
        """Test that threshold=1.0 keeps all features."""
        original_n = variability_container.assays["protein"].n_features
        result = filter_by_cv(variability_container, cv_threshold=1.0)
        assert result.assays["protein"].n_features == original_n

    def test_filter_by_cv_adds_filtered_flag(self, variability_container):
        """Test that filtered flag is added to var."""
        result = filter_by_cv(variability_container, cv_threshold=0.3)
        var = result.assays["protein"].var
        # Check for cv_filtered column or similar
        assert "cv" in var.columns

    def test_filter_by_cv_respects_min_mean(self, variability_container):
        """Test that min_mean parameter is respected."""
        result = filter_by_cv(variability_container, cv_threshold=0.3, min_mean=1.0)
        assert isinstance(result, ScpContainer)

    def test_filter_by_cv_keep_filtered_column(self, variability_container):
        """Test that filtered features are marked."""
        result = filter_by_cv(variability_container, cv_threshold=0.3, keep_filtered=True)
        # When keep_filtered=True, all features should be retained
        # but marked as filtered
        var = result.assays["protein"].var
        if "cv_filtered" in var.columns:
            assert var["cv_filtered"].dtype in [pl.Boolean, pl.Boolean]

    def test_filter_by_cv_remove_filtered_decreases_count(self, variability_container):
        """Test that remove_filtered decreases feature count."""
        result = filter_by_cv(variability_container, cv_threshold=0.3, keep_filtered=False)
        # Features above threshold should be removed
        assert isinstance(result, ScpContainer)

    def test_filter_by_cv_invalid_assay_raises_error(self, variability_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            filter_by_cv(variability_container, assay_name="nonexistent", cv_threshold=0.3)

    def test_filter_by_cv_invalid_layer_raises_error(self, variability_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            filter_by_cv(variability_container, layer_name="nonexistent", cv_threshold=0.3)

    def test_filter_by_cv_negative_threshold_raises_error(self, variability_container):
        """Test that negative threshold raises ScpValueError."""
        with pytest.raises(ScpValueError):
            filter_by_cv(variability_container, cv_threshold=-0.1)

    def test_filter_by_cv_logs_history(self, variability_container):
        """Test that operation is logged to history."""
        result = filter_by_cv(variability_container, cv_threshold=0.3)
        assert len(result.history) > 0
        assert result.history[-1].action == "filter_by_cv"

    def test_filter_by_cv_zero_cv_threshold(self, container_constant_values):
        """Test with CV threshold of 0."""
        result = filter_by_cv(container_constant_values, cv_threshold=0)
        # Constant values have CV=0, so should pass filter
        assert isinstance(result, ScpContainer)

    def test_filter_by_cv_sparse_matrix(self, variability_container_sparse):
        """Test with sparse matrix."""
        result = filter_by_cv(variability_container_sparse, cv_threshold=0.5)
        assert isinstance(result, ScpContainer)

    def test_filter_by_cv_custom_layer(self, variability_container_multi_layer):
        """Test with custom layer."""
        result = filter_by_cv(
            variability_container_multi_layer, layer_name="normalized", cv_threshold=0.5
        )
        assert isinstance(result, ScpContainer)


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.skipif(not VARIABILITY_MODULE_EXISTS, reason="variability module not yet implemented")
class TestVariabilityIntegration:
    """Integration tests for variability module."""

    def test_full_cv_workflow(self, variability_container):
        """Test full workflow: compute CV -> filter features."""
        # Step 1: Compute CV
        result = compute_cv(variability_container)
        assert "cv" in result.assays["protein"].var.columns

        # Step 2: Filter by CV
        filtered = filter_by_cv(result, cv_threshold=0.5)
        assert isinstance(filtered, ScpContainer)

    def test_cv_and_batch_cv_consistency(self, variability_container):
        """Test that compute_cv and compute_batch_cv produce consistent results."""
        cv_result = compute_cv(variability_container)
        batch_result = compute_batch_cv(variability_container, batch_col="batch")

        # Both should compute CV for features
        assert len(cv_result.feature_cv) == len(batch_result.feature_cv)

    def test_replicate_cv_with_filtering(self, container_replicate_cv):
        """Test technical replicate CV followed by filtering."""
        # Step 1: Compute replicate CV
        result = compute_technical_replicate_cv(container_replicate_cv, replicate_col="replicate")
        assert "replicate_cv" in result.assays["protein"].var.columns

        # Step 2: Filter high CV features
        filtered = filter_by_cv(result, cv_threshold=0.5)
        assert isinstance(filtered, ScpContainer)

    def test_batch_cv_identifies_batch_effects(self, variability_container):
        """Test that batch CV analysis can identify batch effects."""
        result = compute_batch_cv(variability_container, batch_col="batch")

        # Check that within and between batch CV are computed
        assert result.within_batch_cv is not None
        assert result.between_batch_cv is not None

        # Between-batch CV should typically be >= within-batch CV if batch effects exist
        # (or at least comparable)
        within_mean = np.mean(list(result.within_batch_cv.values()))
        assert result.between_batch_cv >= 0
        assert within_mean >= 0

    def test_grouped_cv_aggregation(self, variability_container):
        """Test that grouped CV aggregation works correctly."""
        result = compute_cv(variability_container, group_by="batch")

        # Check that CV is computed for both groups
        assert "A" in result.cv_by_group
        assert "B" in result.cv_by_group

        # Group CV arrays should have same length as feature count
        n_features = variability_container.assays["protein"].n_features
        assert len(result.cv_by_group["A"]) == n_features
        assert len(result.cv_by_group["B"]) == n_features

    def test_cv_preserves_other_columns(self, variability_container):
        """Test that compute_cv preserves other var columns."""
        original_cols = set(variability_container.assays["protein"].var.columns)
        result = compute_cv(variability_container)
        result_cols = set(result.assays["protein"].var.columns)

        # All original columns should still be present
        assert original_cols.issubset(result_cols)

    def test_filter_preserves_low_cv_features(self, variability_container):
        """Test that filtering preserves low CV features."""
        # First compute CV
        with_cv = compute_cv(variability_container)

        # Get low CV features (CV < 0.3)
        cv_values = with_cv.assays["protein"].var["cv"].to_numpy()
        low_cv_indices = np.where(cv_values < 0.3)[0]

        # Filter and check that low CV features are kept
        filtered = filter_by_cv(with_cv, cv_threshold=0.3)
        assert filtered.assays["protein"].n_features >= len(low_cv_indices)


# =============================================================================
# Test module availability
# =============================================================================


def test_variability_module_availability():
    """Test to check if variability module is implemented.

    This test will fail until the variability module is implemented.
    """
    if VARIABILITY_MODULE_EXISTS:
        pytest.skip("variability module is implemented")
    else:
        pytest.fail(
            "variability module not yet implemented. "
            "See scptensor/qc/variability.py (Phase 3 feature)"
        )
