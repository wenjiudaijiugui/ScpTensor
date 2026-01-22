"""Comprehensive tests for sensitivity QC module.

Tests cover:
- compute_sensitivity: Total and local sensitivity metrics
- compute_completeness: Data completeness calculation
- compute_jaccard_index: Feature similarity between samples
- compute_cumulative_sensitivity: Feature saturation analysis
- qc_report_metrics: Main entry point for QC metrics

Test categories:
1. Normal functionality tests
2. Edge cases (all missing, all valid, empty data)
3. Sparse matrix handling
4. Group by functionality
5. Jaccard special cases (identical, completely different)
6. Cumulative sensitivity properties

SKIPPED: The sensitivity module (scptensor.qc.sensitivity) has been removed during
the QC module refactoring. The new QC architecture focuses on PSM, sample, and
feature-level QC through qc_psm, qc_sample, and qc_feature modules respectively.
Sensitivity metrics may be reintroduced in future updates as part of the core
QC metrics framework.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "The sensitivity module has been removed during QC refactoring. "
    "New QC architecture uses qc_psm, qc_sample, and qc_feature modules. "
    "Sensitivity metrics may be reintroduced in future updates.",
    allow_module_level=True,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sensitivity_obs():
    """Create obs DataFrame for sensitivity testing."""
    return pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(10)],
            "batch": ["A"] * 5 + ["B"] * 5,
            "condition": ["ctrl"] * 3 + ["treat"] * 4 + ["ctrl"] * 3,
        }
    )


@pytest.fixture
def sensitivity_var():
    """Create var DataFrame for sensitivity testing."""
    return pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(20)],
            "name": [f"PROT{i}" for i in range(20)],
        }
    )


@pytest.fixture
def sensitivity_dense_X():
    """Create dense data matrix with known missing pattern."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(10, 20))

    # Create pattern: first 5 samples detect more features
    X[0, :] = 0
    X[0, :15] = np.random.uniform(5, 10, 15)  # Sample 0 has 15 features
    X[1, :] = 0
    X[1, :12] = np.random.uniform(5, 10, 12)  # Sample 1 has 12 features
    X[2, :] = 0
    X[2, :10] = np.random.uniform(5, 10, 10)  # Sample 2 has 10 features

    # Last 5 samples have fewer features
    for i in range(5, 10):
        X[i, :] = 0
        X[i, :5] = np.random.uniform(5, 10, 5)

    return X


@pytest.fixture
def sensitivity_container(sensitivity_obs, sensitivity_var, sensitivity_dense_X):
    """Create container for sensitivity testing."""
    matrix = ScpMatrix(X=sensitivity_dense_X, M=None)
    assay = Assay(var=sensitivity_var, layers={"raw": matrix})
    return ScpContainer(obs=sensitivity_obs, assays={"protein": assay})


@pytest.fixture
def sensitivity_container_sparse(sensitivity_obs, sensitivity_var, sensitivity_dense_X):
    """Create container with sparse matrix."""
    X_sparse = sparse.csr_matrix(sensitivity_dense_X)
    matrix = ScpMatrix(X=X_sparse, M=None)
    assay = Assay(var=sensitivity_var, layers={"raw": matrix})
    return ScpContainer(obs=sensitivity_obs, assays={"protein": assay})


@pytest.fixture
def sensitivity_container_multi_layer(sensitivity_obs, sensitivity_var, sensitivity_dense_X):
    """Create container with multiple layers."""
    matrix_raw = ScpMatrix(X=sensitivity_dense_X, M=None)
    matrix_norm = ScpMatrix(X=sensitivity_dense_X * 2, M=None)
    assay = Assay(var=sensitivity_var, layers={"raw": matrix_raw, "normalized": matrix_norm})
    return ScpContainer(obs=sensitivity_obs, assays={"protein": assay})


@pytest.fixture
def container_all_missing():
    """Create container with all missing values."""
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(5)]})
    var = pl.DataFrame({"_index": [f"protein_{i}" for i in range(10)]})
    X = np.zeros((5, 10))
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})
    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def container_all_valid():
    """Create container with all valid values."""
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(5)]})
    var = pl.DataFrame({"_index": [f"protein_{i}" for i in range(10)]})
    X = np.ones((5, 10)) * 10.0
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})
    return ScpContainer(obs=obs, assays={"protein": assay})


# =============================================================================
# compute_sensitivity tests
# =============================================================================


class TestComputeSensitivity:
    """Tests for compute_sensitivity function."""

    def test_compute_sensitivity_returns_metrics_object(self, sensitivity_container):
        """Test that compute_sensitivity returns QCMetrics object."""
        result = compute_sensitivity(sensitivity_container)
        assert isinstance(result, QCMetrics)

    def test_compute_sensitivity_local_sensitivity_shape(self, sensitivity_container):
        """Test local sensitivity has correct shape."""
        result = compute_sensitivity(sensitivity_container)
        assert len(result.n_features_per_sample) == sensitivity_container.n_samples

    def test_compute_sensitivity_total_features_positive(self, sensitivity_container):
        """Test total features is positive."""
        result = compute_sensitivity(sensitivity_container)
        assert result.total_features > 0

    def test_compute_sensitivity_total_features_leq_total(self, sensitivity_container):
        """Test total features <= total features in dataset."""
        result = compute_sensitivity(sensitivity_container)
        assert result.total_features <= sensitivity_container.assays["protein"].n_features

    def test_compute_sensitivity_mean_sensitivity_valid(self, sensitivity_container):
        """Test mean sensitivity is within valid range."""
        result = compute_sensitivity(sensitivity_container)
        n_features = sensitivity_container.assays["protein"].n_features
        assert 0 <= result.mean_sensitivity <= n_features

    def test_compute_sensitivity_completeness_values_valid(self, sensitivity_container):
        """Test completeness values are in [0, 1]."""
        result = compute_sensitivity(sensitivity_container)
        assert np.all(result.completeness_per_sample >= 0)
        assert np.all(result.completeness_per_sample <= 1)

    def test_compute_sensitivity_with_threshold(self, sensitivity_container):
        """Test with custom detection threshold."""
        result = compute_sensitivity(sensitivity_container, detection_threshold=5.0)
        # Higher threshold should detect fewer features
        assert result.total_features <= sensitivity_container.assays["protein"].n_features

    def test_compute_sensitivity_sparse_matrix(self, sensitivity_container_sparse):
        """Test with sparse matrix."""
        result = compute_sensitivity(sensitivity_container_sparse)
        assert isinstance(result, QCMetrics)
        assert result.total_features > 0

    def test_compute_sensitivity_custom_layer(self, sensitivity_container_multi_layer):
        """Test with custom layer."""
        result = compute_sensitivity(sensitivity_container_multi_layer, layer_name="normalized")
        assert isinstance(result, QCMetrics)

    def test_compute_sensitivity_all_missing(self, container_all_missing):
        """Test with all missing data."""
        result = compute_sensitivity(container_all_missing)
        assert result.total_features == 0
        assert np.all(result.n_features_per_sample == 0)
        assert np.all(result.completeness_per_sample == 0)

    def test_compute_sensitivity_all_valid(self, container_all_valid):
        """Test with all valid data."""
        result = compute_sensitivity(container_all_valid)
        assert result.total_features == 10
        assert np.all(result.n_features_per_sample == 10)
        assert np.all(result.completeness_per_sample == 1.0)

    def test_compute_sensitivity_invalid_assay_raises_error(self, sensitivity_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_sensitivity(sensitivity_container, assay_name="nonexistent")

    def test_compute_sensitivity_invalid_layer_raises_error(self, sensitivity_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_sensitivity(sensitivity_container, layer_name="nonexistent")


# =============================================================================
# compute_completeness tests
# =============================================================================


class TestComputeCompleteness:
    """Tests for compute_completeness function."""

    def test_compute_completeness_returns_array(self, sensitivity_container):
        """Test that compute_completeness returns numpy array."""
        result = compute_completeness(sensitivity_container)
        assert isinstance(result, np.ndarray)

    def test_compute_completeness_shape_matches_samples(self, sensitivity_container):
        """Test completeness array has same length as samples."""
        result = compute_completeness(sensitivity_container)
        assert len(result) == sensitivity_container.n_samples

    def test_compute_completeness_values_in_range(self, sensitivity_container):
        """Test completeness values are in [0, 1]."""
        result = compute_completeness(sensitivity_container)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_compute_completeness_all_missing(self, container_all_missing):
        """Test completeness with all missing data."""
        result = compute_completeness(container_all_missing)
        assert np.all(result == 0)

    def test_compute_completeness_all_valid(self, container_all_valid):
        """Test completeness with all valid data."""
        result = compute_completeness(container_all_valid)
        assert np.all(result == 1.0)

    def test_compute_completeness_sparse_matrix(self, sensitivity_container_sparse):
        """Test completeness with sparse matrix."""
        result = compute_completeness(sensitivity_container_sparse)
        assert isinstance(result, np.ndarray)
        assert len(result) == sensitivity_container_sparse.n_samples

    def test_compute_completeness_custom_layer(self, sensitivity_container_multi_layer):
        """Test completeness with custom layer."""
        result = compute_completeness(sensitivity_container_multi_layer, layer_name="normalized")
        assert isinstance(result, np.ndarray)

    def test_compute_completeness_with_threshold(self, sensitivity_container):
        """Test completeness with custom detection threshold."""
        result = compute_completeness(sensitivity_container, detection_threshold=5.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_compute_completeness_invalid_assay_raises_error(self, sensitivity_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_completeness(sensitivity_container, assay_name="nonexistent")

    def test_compute_completeness_invalid_layer_raises_error(self, sensitivity_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_completeness(sensitivity_container, layer_name="nonexistent")


# =============================================================================
# compute_jaccard_index tests
# =============================================================================


class TestComputeJaccardIndex:
    """Tests for compute_jaccard_index function."""

    def test_compute_jaccard_returns_matrix(self, sensitivity_container):
        """Test that compute_jaccard_index returns square matrix."""
        result = compute_jaccard_index(sensitivity_container)
        assert isinstance(result, np.ndarray)
        n_samples = sensitivity_container.n_samples
        assert result.shape == (n_samples, n_samples)

    def test_compute_jaccard_diagonal_is_one(self, sensitivity_container):
        """Test that diagonal elements are 1 (self-similarity)."""
        result = compute_jaccard_index(sensitivity_container)
        assert np.allclose(np.diag(result), 1.0)

    def test_compute_jaccard_values_in_range(self, sensitivity_container):
        """Test Jaccard values are in [0, 1]."""
        result = compute_jaccard_index(sensitivity_container)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_compute_jaccard_symmetric(self, sensitivity_container):
        """Test Jaccard matrix is symmetric."""
        result = compute_jaccard_index(sensitivity_container)
        assert np.allclose(result, result.T)

    def test_compute_jaccard_identical_samples(self):
        """Test Jaccard index for identical samples is 1."""
        obs = pl.DataFrame({"_index": ["s1", "s2"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
        X = np.array([[1, 2, 3], [1, 2, 3]])  # Identical samples
        matrix = ScpMatrix(X=X, M=None)
        assay = Assay(var=var, layers={"raw": matrix})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = compute_jaccard_index(container)
        assert result[0, 1] == 1.0

    def test_compute_jaccard_completely_different_samples(self):
        """Test Jaccard index for completely different samples is 0."""
        obs = pl.DataFrame({"_index": ["s1", "s2"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
        X = np.array([[1, 2, 3], [0, 0, 0]])  # No overlap
        matrix = ScpMatrix(X=X, M=None)
        assay = Assay(var=var, layers={"raw": matrix})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = compute_jaccard_index(container)
        assert result[0, 1] == 0.0

    def test_compute_jaccard_partial_overlap(self):
        """Test Jaccard index with partial feature overlap."""
        obs = pl.DataFrame({"_index": ["s1", "s2"]})
        var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})
        # s1: p1, p2; s2: p2, p3; intersection: p2; union: p1, p2, p3
        X = np.array([[1, 2, 0, 0], [0, 2, 3, 0]])
        matrix = ScpMatrix(X=X, M=None)
        assay = Assay(var=var, layers={"raw": matrix})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = compute_jaccard_index(container)
        # Jaccard = |intersection| / |union| = 1 / 3
        assert np.isclose(result[0, 1], 1.0 / 3.0)

    def test_compute_jaccard_sparse_matrix(self, sensitivity_container_sparse):
        """Test Jaccard with sparse matrix."""
        result = compute_jaccard_index(sensitivity_container_sparse)
        n_samples = sensitivity_container_sparse.n_samples
        assert result.shape == (n_samples, n_samples)

    def test_compute_jaccard_custom_layer(self, sensitivity_container_multi_layer):
        """Test Jaccard with custom layer."""
        result = compute_jaccard_index(sensitivity_container_multi_layer, layer_name="normalized")
        n_samples = sensitivity_container_multi_layer.n_samples
        assert result.shape == (n_samples, n_samples)

    def test_compute_jaccard_invalid_assay_raises_error(self, sensitivity_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_jaccard_index(sensitivity_container, assay_name="nonexistent")

    def test_compute_jaccard_invalid_layer_raises_error(self, sensitivity_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_jaccard_index(sensitivity_container, layer_name="nonexistent")


# =============================================================================
# compute_cumulative_sensitivity tests
# =============================================================================


class TestComputeCumulativeSensitivity:
    """Tests for compute_cumulative_sensitivity function."""

    def test_compute_cumulative_returns_result_object(self, sensitivity_container):
        """Test that compute_cumulative_sensitivity returns CumulativeSensitivityResult."""
        result = compute_cumulative_sensitivity(sensitivity_container)
        assert isinstance(result, CumulativeSensitivityResult)

    def test_compute_cumulative_sample_sizes_shape(self, sensitivity_container):
        """Test sample_sizes has correct length."""
        result = compute_cumulative_sensitivity(sensitivity_container, n_steps=10)
        assert len(result.sample_sizes) == 10

    def test_compute_cumulative_features_shape(self, sensitivity_container):
        """Test cumulative_features has same length as sample_sizes."""
        result = compute_cumulative_sensitivity(sensitivity_container, n_steps=10)
        assert len(result.cumulative_features) == len(result.sample_sizes)

    def test_compute_cumulative_monotonic_increasing(self, sensitivity_container):
        """Test cumulative features is monotonic increasing."""
        result = compute_cumulative_sensitivity(sensitivity_container)
        assert np.all(np.diff(result.cumulative_features) >= 0)

    def test_compute_cumulative_max_equals_total(self, sensitivity_container):
        """Test max cumulative features equals or exceeds total sensitivity."""
        result = compute_cumulative_sensitivity(sensitivity_container)
        sensitivity = compute_sensitivity(sensitivity_container)
        # Cumulative should reach at least the total features
        assert result.cumulative_features[-1] >= sensitivity.total_features

    def test_compute_cumulative_with_seed_reproducible(self, sensitivity_container):
        """Test that seed produces reproducible results."""
        result1 = compute_cumulative_sensitivity(sensitivity_container, seed=42, n_steps=5)
        result2 = compute_cumulative_sensitivity(sensitivity_container, seed=42, n_steps=5)
        assert np.array_equal(result1.cumulative_features, result2.cumulative_features)

    def test_compute_cumulative_custom_n_steps(self, sensitivity_container):
        """Test with custom n_steps."""
        n_steps = 5
        result = compute_cumulative_sensitivity(sensitivity_container, n_steps=n_steps)
        assert len(result.sample_sizes) == n_steps

    def test_compute_cumulative_n_steps_too_large(self, sensitivity_container):
        """Test that n_steps > n_samples is adjusted."""
        result = compute_cumulative_sensitivity(sensitivity_container, n_steps=100)
        # Should not exceed number of samples
        assert len(result.sample_sizes) <= sensitivity_container.n_samples

    def test_compute_cumulative_n_steps_too_small_raises_error(self, sensitivity_container):
        """Test that n_steps < 2 raises ScpValueError."""
        with pytest.raises(ScpValueError) as excinfo:
            compute_cumulative_sensitivity(sensitivity_container, n_steps=1)
        assert "n_steps" in str(excinfo.value).lower()

    def test_compute_cumulative_sparse_matrix(self, sensitivity_container_sparse):
        """Test cumulative sensitivity with sparse matrix."""
        result = compute_cumulative_sensitivity(sensitivity_container_sparse)
        assert isinstance(result, CumulativeSensitivityResult)

    def test_compute_cumulative_custom_layer(self, sensitivity_container_multi_layer):
        """Test cumulative sensitivity with custom layer."""
        result = compute_cumulative_sensitivity(
            sensitivity_container_multi_layer, layer_name="normalized"
        )
        assert isinstance(result, CumulativeSensitivityResult)

    def test_compute_cumulative_saturation_point_type(self, sensitivity_container):
        """Test saturation_point is either int or None."""
        result = compute_cumulative_sensitivity(sensitivity_container)
        assert result.saturation_point is None or isinstance(result.saturation_point, int)

    def test_compute_cumulative_invalid_assay_raises_error(self, sensitivity_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            compute_cumulative_sensitivity(sensitivity_container, assay_name="nonexistent")

    def test_compute_cumulative_invalid_layer_raises_error(self, sensitivity_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            compute_cumulative_sensitivity(sensitivity_container, layer_name="nonexistent")


# =============================================================================
# qc_report_metrics tests
# =============================================================================


class TestQcReportMetrics:
    """Tests for qc_report_metrics function."""

    def test_qc_report_metrics_returns_container(self, sensitivity_container):
        """Test that qc_report_metrics returns ScpContainer."""
        result = qc_report_metrics(sensitivity_container)
        assert isinstance(result, ScpContainer)

    def test_qc_report_metrics_adds_n_detected_features(self, sensitivity_container):
        """Test that n_detected_features column is added."""
        result = qc_report_metrics(sensitivity_container)
        assert "n_detected_features" in result.obs.columns

    def test_qc_report_metrics_adds_total_features(self, sensitivity_container):
        """Test that total_features column is added."""
        result = qc_report_metrics(sensitivity_container)
        assert "total_features" in result.obs.columns

    def test_qc_report_metrics_adds_completeness(self, sensitivity_container):
        """Test that completeness column is added."""
        result = qc_report_metrics(sensitivity_container)
        assert "completeness" in result.obs.columns

    def test_qc_report_metrics_adds_local_sensitivity(self, sensitivity_container):
        """Test that local_sensitivity column is added."""
        result = qc_report_metrics(sensitivity_container)
        assert "local_sensitivity" in result.obs.columns

    def test_qc_report_metrics_n_detected_features_values(self, sensitivity_container):
        """Test n_detected_features values are valid."""
        result = qc_report_metrics(sensitivity_container)
        n_detected = result.obs["n_detected_features"].to_numpy()
        assert np.all(n_detected >= 0)
        n_features = sensitivity_container.assays["protein"].n_features
        assert np.all(n_detected <= n_features)

    def test_qc_report_metrics_completeness_values(self, sensitivity_container):
        """Test completeness values are in [0, 1]."""
        result = qc_report_metrics(sensitivity_container)
        completeness = result.obs["completeness"].to_numpy()
        assert np.all(completeness >= 0)
        assert np.all(completeness <= 1)

    def test_qc_report_metrics_local_sensitivity_matches_n_detected(self, sensitivity_container):
        """Test local_sensitivity matches n_detected_features."""
        result = qc_report_metrics(sensitivity_container)
        n_detected = result.obs["n_detected_features"].to_numpy()
        local_sens = result.obs["local_sensitivity"].to_numpy()
        assert np.array_equal(n_detected, local_sens)

    def test_qc_report_metrics_total_features_consistent(self, sensitivity_container):
        """Test total_features is consistent across samples."""
        result = qc_report_metrics(sensitivity_container)
        total_features = result.obs["total_features"].to_numpy()
        # All samples should have same total_features value
        assert len(np.unique(total_features)) == 1

    def test_qc_report_metrics_with_group_by(self, sensitivity_container):
        """Test qc_report_metrics with group_by parameter."""
        result = qc_report_metrics(sensitivity_container, group_by="batch")
        assert "batch_mean_features" in result.obs.columns
        assert "batch_total_features" in result.obs.columns

    def test_qc_report_metrics_group_by_invalid_column_raises_error(self, sensitivity_container):
        """Test that invalid group_by column raises ScpValueError."""
        with pytest.raises(ScpValueError) as excinfo:
            qc_report_metrics(sensitivity_container, group_by="nonexistent")
        assert "nonexistent" in str(excinfo.value)

    def test_qc_report_metrics_sparse_matrix(self, sensitivity_container_sparse):
        """Test qc_report_metrics with sparse matrix."""
        result = qc_report_metrics(sensitivity_container_sparse)
        assert "n_detected_features" in result.obs.columns
        assert "completeness" in result.obs.columns

    def test_qc_report_metrics_custom_layer(self, sensitivity_container_multi_layer):
        """Test qc_report_metrics with custom layer."""
        result = qc_report_metrics(sensitivity_container_multi_layer, layer_name="normalized")
        assert "n_detected_features" in result.obs.columns

    def test_qc_report_metrics_with_detection_threshold(self, sensitivity_container):
        """Test qc_report_metrics with custom detection threshold."""
        result = qc_report_metrics(sensitivity_container, detection_threshold=5.0)
        assert "n_detected_features" in result.obs.columns

    def test_qc_report_metrics_logs_history(self, sensitivity_container):
        """Test that qc_report_metrics logs operation to history."""
        result = qc_report_metrics(sensitivity_container)
        assert len(result.history) > 0
        assert result.history[-1].action == "qc_report_metrics"

    def test_qc_report_metrics_invalid_assay_raises_error(self, sensitivity_container):
        """Test that invalid assay raises AssayNotFoundError."""
        with pytest.raises(AssayNotFoundError):
            qc_report_metrics(sensitivity_container, assay_name="nonexistent")

    def test_qc_report_metrics_invalid_layer_raises_error(self, sensitivity_container):
        """Test that invalid layer raises LayerNotFoundError."""
        with pytest.raises(LayerNotFoundError):
            qc_report_metrics(sensitivity_container, layer_name="nonexistent")

    def test_qc_report_metrics_all_missing(self, container_all_missing):
        """Test qc_report_metrics with all missing data."""
        result = qc_report_metrics(container_all_missing)
        assert np.all(result.obs["n_detected_features"].to_numpy() == 0)
        assert np.all(result.obs["completeness"].to_numpy() == 0)

    def test_qc_report_metrics_all_valid(self, container_all_valid):
        """Test qc_report_metrics with all valid data."""
        result = qc_report_metrics(container_all_valid)
        assert np.all(result.obs["n_detected_features"].to_numpy() == 10)
        assert np.all(result.obs["completeness"].to_numpy() == 1.0)
