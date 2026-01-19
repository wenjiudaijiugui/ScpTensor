"""Tests for parameter sensitivity evaluator."""

import numpy as np
import pytest

from scptensor.benchmark.evaluators import (
    ParameterSensitivityEvaluator,
    SensitivityResult,
    evaluate_parameter_sensitivity,
    get_parameter_spec,
    get_supported_parameters,
)


# =============================================================================
# Test Data Generation
# =============================================================================


@pytest.fixture
def synthetic_clustering_data():
    """Create synthetic data for clustering sensitivity tests."""
    np.random.seed(42)
    # Create 3 distinct clusters
    n_samples = 150
    n_features = 10

    # Cluster 1
    X1 = np.random.randn(50, n_features) + 2.0
    # Cluster 2
    X2 = np.random.randn(50, n_features) - 2.0
    # Cluster 3
    X3 = np.random.randn(50, n_features)
    X3[:, 0] += 4.0

    X = np.vstack([X1, X2, X3])
    true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

    return X, true_labels


@pytest.fixture
def synthetic_pca_data():
    """Create synthetic data for PCA sensitivity tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_components = 5

    # Low-rank data
    U = np.random.randn(n_samples, n_components)
    V = np.random.randn(n_features, n_components)
    X = U @ V.T + np.random.randn(n_samples, n_features) * 0.1

    return X


@pytest.fixture
def simple_2d_data():
    """Create simple 2D data for quick tests."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    return X


@pytest.fixture
def evaluator_n_clusters():
    """Create evaluator for n_clusters parameter."""
    return ParameterSensitivityEvaluator("n_clusters", [2, 3, 4, 5, 8])


@pytest.fixture
def evaluator_n_neighbors():
    """Create evaluator for n_neighbors parameter."""
    return ParameterSensitivityEvaluator("n_neighbors", [5, 10, 15, 20])


@pytest.fixture
def evaluator_n_pcs():
    """Create evaluator for n_pcs parameter."""
    return ParameterSensitivityEvaluator("n_pcs", [3, 5, 10, 15, 20])


@pytest.fixture
def evaluator_resolution():
    """Create evaluator for resolution parameter."""
    return ParameterSensitivityEvaluator("resolution", [0.5, 1.0, 1.5, 2.0])


# =============================================================================
# Initialization Tests
# =============================================================================


def test_evaluator_initialization_n_clusters():
    """Test evaluator initialization for n_clusters parameter."""
    evaluator = ParameterSensitivityEvaluator("n_clusters")

    assert evaluator.param_name == "n_clusters"
    assert evaluator.param_spec is not None
    assert evaluator.param_spec.name == "n_clusters"
    assert evaluator.param_spec.default_value == 5.0
    assert evaluator.param_spec.is_integer is True


def test_evaluator_initialization_with_custom_values():
    """Test evaluator initialization with custom parameter values."""
    custom_values = [3, 5, 10, 15]
    evaluator = ParameterSensitivityEvaluator("n_clusters", custom_values)

    assert evaluator.param_values == custom_values


def test_evaluator_initialization_default_values():
    """Test evaluator initialization uses default values when None provided."""
    evaluator = ParameterSensitivityEvaluator("n_neighbors", None)

    assert evaluator.param_values is not None
    assert len(evaluator.param_values) > 0


def test_evaluator_initialization_invalid_parameter():
    """Test evaluator initialization with invalid parameter name."""
    with pytest.raises(ValueError, match="Unsupported parameter"):
        ParameterSensitivityEvaluator("invalid_param")


def test_evaluator_n_jobs_validation():
    """Test evaluator n_jobs parameter validation."""
    evaluator = ParameterSensitivityEvaluator("n_clusters", n_jobs=0)
    assert evaluator.n_jobs == 1  # Should be corrected to 1

    evaluator = ParameterSensitivityEvaluator("n_clusters", n_jobs=4)
    assert evaluator.n_jobs == 4


# =============================================================================
# Supported Parameters Tests
# =============================================================================


def test_get_supported_parameters():
    """Test getting list of supported parameters."""
    params = get_supported_parameters()

    assert isinstance(params, list)
    assert "n_clusters" in params
    assert "n_neighbors" in params
    assert "n_pcs" in params
    assert "resolution" in params


def test_get_parameter_spec():
    """Test getting specification for specific parameter."""
    spec = get_parameter_spec("n_clusters")

    assert spec is not None
    assert spec.name == "n_clusters"
    assert spec.default_value == 5.0
    assert spec.is_integer is True


def test_get_parameter_spec_not_found():
    """Test getting specification for non-existent parameter."""
    spec = get_parameter_spec("nonexistent")
    assert spec is None


# =============================================================================
# n_clusters Evaluation Tests
# =============================================================================


def test_evaluate_n_clusters_basic(synthetic_clustering_data, evaluator_n_clusters):
    """Test basic n_clusters evaluation."""
    X, _ = synthetic_clustering_data

    result = evaluator_n_clusters.evaluate(X)

    assert isinstance(result, dict)
    assert "sensitivity_score" in result
    assert "stability_score" in result
    assert "optimal_value" in result
    assert "metric_variance" in result
    assert "metric_mean" in result

    # Check value ranges
    assert 0.0 <= result["sensitivity_score"] <= 1.0
    assert 0.0 <= result["stability_score"] <= 1.0


def test_evaluate_n_clusters_all_scores_recorded(synthetic_clustering_data, evaluator_n_clusters):
    """Test that all parameter values are scored."""
    X, _ = synthetic_clustering_data

    evaluator_n_clusters.evaluate(X)
    all_scores = evaluator_n_clusters._all_scores

    assert len(all_scores) == len(evaluator_n_clusters.param_values)
    for value in evaluator_n_clusters.param_values:
        assert value in all_scores


def test_evaluate_n_clusters_with_small_data(simple_2d_data):
    """Test n_clusters evaluation with small dataset."""
    evaluator = ParameterSensitivityEvaluator("n_clusters", [2, 3])
    result = evaluator.evaluate(simple_2d_data)

    assert result["optimal_value"] in [2.0, 3.0]


def test_evaluate_n_clusters_optimal_near_true(synthetic_clustering_data):
    """Test that optimal n_clusters is near true value."""
    X, true_labels = synthetic_clustering_data
    n_true = len(np.unique(true_labels))

    evaluator = ParameterSensitivityEvaluator("n_clusters", [2, 3, 4, 5, 6, 8])
    result = evaluator.evaluate(X)

    # Optimal should be close to true value (3 clusters)
    assert abs(result["optimal_value"] - n_true) <= 2


# =============================================================================
# n_neighbors Evaluation Tests
# =============================================================================


def test_evaluate_n_neighbors_basic(synthetic_pca_data, evaluator_n_neighbors):
    """Test basic n_neighbors evaluation."""
    X = synthetic_pca_data

    result = evaluator_n_neighbors.evaluate(X)

    assert isinstance(result, dict)
    assert "sensitivity_score" in result
    assert "optimal_value" in result


def test_evaluate_n_neighbors_scores_negative(synthetic_pca_data, evaluator_n_neighbors):
    """Test that n_neighbors scores are negative (reconstruction error)."""
    X = synthetic_pca_data

    evaluator_n_neighbors.evaluate(X)
    all_scores = evaluator_n_neighbors._all_scores

    # All scores should be negative or zero (reconstruction error)
    for score in all_scores.values():
        assert score <= 0


# =============================================================================
# n_pcs Evaluation Tests
# =============================================================================


def test_evaluate_n_pcs_basic(synthetic_pca_data, evaluator_n_pcs):
    """Test basic n_pcs evaluation."""
    X = synthetic_pca_data

    result = evaluator_n_pcs.evaluate(X)

    assert isinstance(result, dict)
    assert "sensitivity_score" in result
    assert "optimal_value" in result


def test_evaluate_n_pcs_scores_positive(synthetic_pca_data, evaluator_n_pcs):
    """Test that n_pcs scores are positive (explained variance)."""
    X = synthetic_pca_data

    evaluator_n_pcs.evaluate(X)
    all_scores = evaluator_n_pcs._all_scores

    # All scores should be positive (explained variance ratio)
    for score in all_scores.values():
        assert 0.0 <= score <= 1.0


def test_evaluate_n_pcs_more_components_better(synthetic_pca_data):
    """Test that more PCs generally give better explained variance."""
    X = synthetic_pca_data

    evaluator = ParameterSensitivityEvaluator("n_pcs", [2, 5, 10, 15])
    evaluator.evaluate(X)
    all_scores = evaluator._all_scores

    # Scores should generally increase with more PCs
    scores_list = list(all_scores.values())
    assert scores_list[-1] >= scores_list[0]  # Last (most PCs) >= First (fewest PCs)


# =============================================================================
# Resolution Evaluation Tests
# =============================================================================


def test_evaluate_resolution_basic(synthetic_clustering_data, evaluator_resolution):
    """Test basic resolution evaluation."""
    X, _ = synthetic_clustering_data

    result = evaluator_resolution.evaluate(X)

    assert isinstance(result, dict)
    assert "sensitivity_score" in result
    assert "optimal_value" in result


def test_evaluate_resolution_with_knn_graph(synthetic_clustering_data):
    """Test resolution evaluation with custom n_neighbors."""
    X, _ = synthetic_clustering_data

    evaluator = ParameterSensitivityEvaluator("resolution", [0.5, 1.0, 1.5])
    result = evaluator.evaluate(X, n_neighbors=10)

    assert "optimal_value" in result
    assert 0.0 <= result["sensitivity_score"] <= 1.0


# =============================================================================
# Sensitivity and Stability Metrics Tests
# =============================================================================


def test_sensitivity_score_computation():
    """Test sensitivity score is computed correctly."""
    np.random.seed(42)
    # Create data where clustering is stable
    X = np.random.randn(100, 5)

    evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 4, 5])
    result = evaluator.evaluate(X)

    assert "sensitivity_score" in result
    assert isinstance(result["sensitivity_score"], float)


def test_stability_score_computation():
    """Test stability score is computed correctly."""
    np.random.seed(42)
    X = np.random.randn(100, 5)

    evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 4, 5])
    result = evaluator.evaluate(X)

    assert "stability_score" in result
    assert isinstance(result["stability_score"], float)
    # Stability should be between 0 and 1
    assert 0.0 <= result["stability_score"] <= 1.0


def test_metric_variance_computation():
    """Test metric variance is computed correctly."""
    np.random.seed(42)
    X = np.random.randn(100, 5)

    evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 4, 5, 8])
    result = evaluator.evaluate(X)

    assert "metric_variance" in result
    assert isinstance(result["metric_variance"], float)
    assert result["metric_variance"] >= 0.0


def test_metric_mean_computation():
    """Test metric mean is computed correctly."""
    np.random.seed(42)
    X = np.random.randn(100, 5)

    evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 4, 5])
    result = evaluator.evaluate(X)

    assert "metric_mean" in result
    assert isinstance(result["metric_mean"], float)


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_evaluate_with_edge_case_n_clusters_too_large():
    """Test evaluation when n_clusters is larger than samples."""
    X = np.random.randn(10, 5)
    evaluator = ParameterSensitivityEvaluator("n_clusters", [5, 15, 20])

    result = evaluator.evaluate(X)

    # Should handle gracefully - extreme values should return 0 score
    assert isinstance(result, dict)
    assert "optimal_value" in result


def test_evaluate_with_edge_case_single_value():
    """Test evaluation with only one parameter value."""
    X = np.random.randn(50, 5)
    evaluator = ParameterSensitivityEvaluator("n_clusters", [5])

    result = evaluator.evaluate(X)

    # With single value, variance should be 0
    assert result["metric_variance"] == 0.0


def test_evaluate_with_edge_case_n_pcs_exceeds_features():
    """Test evaluation when n_pcs exceeds data dimensions."""
    X = np.random.randn(50, 5)
    evaluator = ParameterSensitivityEvaluator("n_pcs", [3, 5, 10, 20])

    result = evaluator.evaluate(X)

    # Should handle by capping at max dimension
    assert isinstance(result, dict)


# =============================================================================
# Detailed Result Tests
# =============================================================================


def test_get_detailed_result(synthetic_clustering_data):
    """Test getting detailed SensitivityResult."""
    X, _ = synthetic_clustering_data

    evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 4, 5])
    evaluator.evaluate(X)

    detailed = evaluator.get_detailed_result()

    assert isinstance(detailed, SensitivityResult)
    assert detailed.param_name == "n_clusters"
    assert isinstance(detailed.sensitivity_score, float)
    assert isinstance(detailed.stability_score, float)
    assert isinstance(detailed.optimal_value, float)
    assert isinstance(detailed.metric_variance, float)
    assert isinstance(detailed.metric_mean, float)
    assert isinstance(detailed.all_scores, dict)
    assert len(detailed.all_scores) == 3


# =============================================================================
# Convenience Function Tests
# =============================================================================


def test_convenience_function_basic():
    """Test the evaluate_parameter_sensitivity convenience function."""
    np.random.seed(42)
    X = np.random.randn(50, 5)

    result = evaluate_parameter_sensitivity(X, "n_clusters", [3, 5, 8])

    assert isinstance(result, dict)
    assert "sensitivity_score" in result
    assert "optimal_value" in result


def test_convenience_function_with_labels(synthetic_clustering_data):
    """Test convenience function with labels."""
    X, labels = synthetic_clustering_data

    result = evaluate_parameter_sensitivity(
        X, "n_clusters", [2, 3, 4, 5], labels=labels
    )

    assert isinstance(result, dict)


def test_convenience_function_kwargs():
    """Test convenience function with additional kwargs."""
    X = np.random.randn(50, 5)

    result = evaluate_parameter_sensitivity(
        X, "n_clusters", [3, 5], random_state=123
    )

    assert isinstance(result, dict)


# =============================================================================
# ParameterSpec Tests
# =============================================================================


def test_parameter_spec_attributes():
    """Test ParameterSpec dataclass attributes."""
    from scptensor.benchmark.evaluators.parameter_sensitivity import ParameterSpec

    spec = ParameterSpec(
        name="test_param",
        default_value=10.0,
        value_range=(1.0, 100.0),
        is_integer=True,
        recommended_values=[5.0, 10.0, 20.0],
    )

    assert spec.name == "test_param"
    assert spec.default_value == 10.0
    assert spec.value_range == (1.0, 100.0)
    assert spec.is_integer is True
    assert spec.recommended_values == [5.0, 10.0, 20.0]


def test_parameter_spec_frozen():
    """Test that ParameterSpec is frozen (immutable)."""
    from scptensor.benchmark.evaluators.parameter_sensitivity import ParameterSpec

    spec = ParameterSpec(
        name="test",
        default_value=5.0,
        value_range=(1.0, 10.0),
    )

    with pytest.raises(AttributeError):  # frozen dataclass
        spec.name = "new_name"


# =============================================================================
# SensitivityResult Tests
# =============================================================================


def test_sensitivity_result_creation():
    """Test SensitivityResult dataclass creation."""
    result = SensitivityResult(
        param_name="n_clusters",
        sensitivity_score=0.5,
        stability_score=0.7,
        optimal_value=5.0,
        metric_variance=0.1,
        metric_mean=0.6,
        all_scores={3.0: 0.5, 5.0: 0.7, 8.0: 0.6},
    )

    assert result.param_name == "n_clusters"
    assert result.sensitivity_score == 0.5
    assert result.stability_score == 0.7
    assert result.optimal_value == 5.0
    assert result.metric_variance == 0.1
    assert result.metric_mean == 0.6
    assert len(result.all_scores) == 3


def test_sensitivity_result_default_all_scores():
    """Test SensitivityResult default empty all_scores."""
    result = SensitivityResult(
        param_name="test",
        sensitivity_score=0.0,
        stability_score=0.0,
        optimal_value=1.0,
        metric_variance=0.0,
        metric_mean=0.0,
    )

    assert result.all_scores == {}


# =============================================================================
# BaseEvaluator Inheritance Tests
# =============================================================================


def test_parameter_sensitivity_evaluator_is_base_evaluator():
    """Test that ParameterSensitivityEvaluator is a BaseEvaluator."""
    from scptensor.benchmark.evaluators.biological import BaseEvaluator

    evaluator = ParameterSensitivityEvaluator("n_clusters")

    assert isinstance(evaluator, BaseEvaluator)


def test_parameter_sensitivity_evaluator_has_evaluate_method():
    """Test that ParameterSensitivityEvaluator has evaluate method."""
    evaluator = ParameterSensitivityEvaluator("n_clusters")

    assert hasattr(evaluator, "evaluate")
    assert callable(evaluator.evaluate)
