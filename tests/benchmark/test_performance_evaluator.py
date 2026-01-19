"""Tests for performance evaluator."""

import time as time_module

import numpy as np
import pytest

from scptensor.benchmark.evaluators import (
    PerformanceEvaluator,
    PerformanceResult,
    benchmark_scalability,
    evaluate_performance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def perf_evaluator():
    """Create performance evaluator instance."""
    return PerformanceEvaluator(n_runs=3, warmup_runs=1)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return np.random.randn(100, 50)


@pytest.fixture
def simple_function():
    """Create a simple function for benchmarking."""
    def func(X):
        return X * 2
    return func


@pytest.fixture
def slow_function():
    """Create a slower function for benchmarking."""
    def func(X):
        time_module.sleep(0.01)
        return X * 2
    return func


# =============================================================================
# Test Initialization
# =============================================================================


def test_performance_evaluator_initialization():
    """Test evaluator initialization with default parameters."""
    evaluator = PerformanceEvaluator()
    assert evaluator.n_runs == 3
    assert evaluator.warmup_runs == 1
    assert evaluator.track_memory is True
    assert evaluator.track_cpu is True


def test_performance_evaluator_custom_initialization():
    """Test evaluator initialization with custom parameters."""
    evaluator = PerformanceEvaluator(n_runs=5, warmup_runs=2, track_memory=False, track_cpu=False)
    assert evaluator.n_runs == 5
    assert evaluator.warmup_runs == 2
    assert evaluator.track_memory is False
    assert evaluator.track_cpu is False


def test_performance_evaluator_negative_n_runs():
    """Test evaluator handles negative n_runs."""
    evaluator = PerformanceEvaluator(n_runs=-1)
    assert evaluator.n_runs == 1  # Should be clamped to minimum


def test_performance_evaluator_negative_warmup_runs():
    """Test evaluator handles negative warmup_runs."""
    evaluator = PerformanceEvaluator(warmup_runs=-1)
    assert evaluator.warmup_runs == 0  # Should be clamped to minimum


# =============================================================================
# Test evaluate() method
# =============================================================================


def test_evaluate_returns_dict(perf_evaluator, simple_function, sample_data):
    """Test that evaluate returns a dictionary."""
    result = perf_evaluator.evaluate(simple_function, sample_data)
    assert isinstance(result, dict)


def test_evaluate_has_all_metrics(perf_evaluator, simple_function, sample_data):
    """Test that evaluate returns all expected metrics."""
    result = perf_evaluator.evaluate(simple_function, sample_data)

    expected_keys = [
        "runtime", "runtime_std",
        "memory_usage", "memory_usage_std",
        "throughput", "throughput_std",
        "cpu_time", "cpu_time_std",
    ]
    for key in expected_keys:
        assert key in result


def test_evaluate_runtime_positive(perf_evaluator, simple_function, sample_data):
    """Test that runtime is positive."""
    result = perf_evaluator.evaluate(simple_function, sample_data)
    assert result["runtime"] > 0


def test_evaluate_throughput_positive(perf_evaluator, simple_function, sample_data):
    """Test that throughput is positive."""
    result = perf_evaluator.evaluate(simple_function, sample_data)
    assert result["throughput"] > 0


def test_evaluate_with_slow_function(perf_evaluator, slow_function, sample_data):
    """Test evaluation with a slower function."""
    result = perf_evaluator.evaluate(slow_function, sample_data)
    # Should take at least 0.01 seconds per run
    assert result["runtime"] >= 0.01


def test_evaluate_with_function_args(perf_evaluator, sample_data):
    """Test evaluation with additional function arguments."""
    def func_with_args(X, multiplier=1.0):
        return X * multiplier

    result = perf_evaluator.evaluate(func_with_args, sample_data, multiplier=3.0)
    assert result["runtime"] > 0


def test_evaluate_with_function_kwargs(perf_evaluator, sample_data):
    """Test evaluation with keyword arguments."""
    def func_with_kwargs(X, offset=0.0):
        return X + offset

    result = perf_evaluator.evaluate(func_with_kwargs, sample_data, offset=5.0)
    assert result["runtime"] > 0


def test_evaluate_without_memory_tracking(sample_data):
    """Test evaluation without memory tracking."""
    evaluator = PerformanceEvaluator(track_memory=False)
    result = evaluator.evaluate(lambda X: X * 2, sample_data)

    assert result["memory_usage"] == 0.0
    assert result["memory_usage_std"] == 0.0


def test_evaluate_without_cpu_tracking(sample_data):
    """Test evaluation without CPU tracking."""
    evaluator = PerformanceEvaluator(track_cpu=False)
    result = evaluator.evaluate(lambda X: X * 2, sample_data)

    assert result["cpu_time"] == 0.0
    assert result["cpu_time_std"] == 0.0


def test_evaluate_runtime_std_with_single_run(sample_data):
    """Test that runtime_std is zero for single run."""
    evaluator = PerformanceEvaluator(n_runs=1, warmup_runs=0)
    result = evaluator.evaluate(lambda X: X * 2, sample_data)

    assert result["runtime_std"] == 0.0


def test_evaluate_function_exception(perf_evaluator, sample_data):
    """Test that evaluate raises exception when function fails."""
    def failing_func(X):
        raise ValueError("Intentional error")

    with pytest.raises(RuntimeError, match="Function execution failed"):
        perf_evaluator.evaluate(failing_func, sample_data)


# =============================================================================
# Test evaluate_scalability() method
# =============================================================================


def test_evaluate_scalability_returns_dict(perf_evaluator):
    """Test that evaluate_scalability returns a dictionary."""
    sizes = [50, 100, 150]
    result = perf_evaluator.evaluate_scalability(lambda X: X * 2, sizes, n_features=20)

    assert isinstance(result, dict)
    assert len(result) == len(sizes)


def test_evaluate_scalability_all_sizes(perf_evaluator):
    """Test that evaluate_scalability tests all requested sizes."""
    sizes = [50, 100, 150]
    result = perf_evaluator.evaluate_scalability(lambda X: X * 2, sizes, n_features=20)

    for size in sizes:
        assert size in result
        assert "runtime" in result[size]


def test_evaluate_scalability_runtime_increases_with_size(perf_evaluator):
    """Test that runtime generally increases with data size."""
    sizes = [50, 100, 200]
    result = perf_evaluator.evaluate_scalability(
        lambda X: X @ X.T,  # Matrix multiplication scales O(n^3)
        sizes,
        n_features=20
    )

    # Larger sizes should generally take longer
    assert result[100]["runtime"] >= result[50]["runtime"] * 0.5  # Allow some variance
    assert result[200]["runtime"] >= result[100]["runtime"] * 0.5


# =============================================================================
# Test compare() method
# =============================================================================


def test_compare_returns_dict(perf_evaluator, sample_data):
    """Test that compare returns a dictionary."""
    funcs = {
        "fast": lambda X: X * 2,
        "slow": lambda X: X @ X.T,
    }

    result = perf_evaluator.compare(funcs, sample_data)

    assert isinstance(result, dict)
    assert len(result) == 2


def test_compare_all_functions(perf_evaluator, sample_data):
    """Test that compare evaluates all functions."""
    funcs = {
        "fast": lambda X: X * 2,
        "slow": lambda X: X @ X.T,
    }

    result = perf_evaluator.compare(funcs, sample_data)

    assert "fast" in result
    assert "slow" in result
    assert "runtime" in result["fast"]
    assert "runtime" in result["slow"]


def test_compare_fast_vs_slow(perf_evaluator, sample_data):
    """Test that compare correctly identifies faster function."""
    funcs = {
        "fast": lambda X: X * 2,
        "slow": lambda X: X @ X.T,
    }

    result = perf_evaluator.compare(funcs, sample_data)

    # Fast function should be faster than slow
    assert result["fast"]["runtime"] < result["slow"]["runtime"]


def test_compare_with_failing_function(perf_evaluator, sample_data):
    """Test that compare handles failing functions gracefully."""
    funcs = {
        "good": lambda X: X * 2,
        "bad": lambda X: (_ for _ in ()).throw(ValueError("error")),
    }

    result = perf_evaluator.compare(funcs, sample_data)

    assert "good" in result
    assert "bad" in result
    assert "runtime" in result["good"]
    # Bad function should have error marker
    assert "error" in result["bad"] or np.isnan(result["bad"]["runtime"])


# =============================================================================
# Test PerformanceResult dataclass
# =============================================================================


def test_performance_result_creation():
    """Test PerformanceResult dataclass creation."""
    result = PerformanceResult(
        runtime=0.1,
        runtime_std=0.01,
        memory_usage=10.0,
        memory_usage_std=1.0,
        throughput=1000.0,
        throughput_std=50.0,
        cpu_time=0.09,
        cpu_time_std=0.01,
        n_runs=3,
        n_samples=100,
    )

    assert result.runtime == 0.1
    assert result.n_runs == 3
    assert result.n_samples == 100


def test_performance_result_is_frozen():
    """Test that PerformanceResult is frozen."""
    result = PerformanceResult(
        runtime=0.1,
        runtime_std=0.01,
        memory_usage=10.0,
        memory_usage_std=1.0,
        throughput=1000.0,
        throughput_std=50.0,
        cpu_time=0.09,
        cpu_time_std=0.01,
        n_runs=3,
        n_samples=100,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.runtime = 0.2


def test_performance_result_to_dict():
    """Test PerformanceResult.to_dict() method."""
    result = PerformanceResult(
        runtime=0.1,
        runtime_std=0.01,
        memory_usage=10.0,
        memory_usage_std=1.0,
        throughput=1000.0,
        throughput_std=50.0,
        cpu_time=0.09,
        cpu_time_std=0.01,
        n_runs=3,
        n_samples=100,
    )

    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["runtime"] == 0.1
    assert "n_runs" not in d  # to_dict only returns metrics, not metadata


def test_performance_result_has_slots():
    """Test that PerformanceResult uses slots."""
    result = PerformanceResult(
        runtime=0.1,
        runtime_std=0.01,
        memory_usage=10.0,
        memory_usage_std=1.0,
        throughput=1000.0,
        throughput_std=50.0,
        cpu_time=0.09,
        cpu_time_std=0.01,
        n_runs=3,
        n_samples=100,
    )

    assert hasattr(result, "__slots__")


# =============================================================================
# Test to_performance_result() method
# =============================================================================


def test_to_performance_result(perf_evaluator, simple_function, sample_data):
    """Test conversion of metrics to PerformanceResult."""
    metrics = perf_evaluator.evaluate(simple_function, sample_data)
    result = perf_evaluator.to_performance_result(metrics, n_samples=100)

    assert isinstance(result, PerformanceResult)
    assert result.n_samples == 100
    assert result.n_runs == perf_evaluator.n_runs


def test_to_performance_result_without_n_samples(perf_evaluator, simple_function, sample_data):
    """Test conversion without specifying n_samples."""
    metrics = perf_evaluator.evaluate(simple_function, sample_data)
    result = perf_evaluator.to_performance_result(metrics)

    assert isinstance(result, PerformanceResult)
    assert result.n_samples is None


# =============================================================================
# Test _get_n_samples() method
# =============================================================================


def test_get_n_samples_with_2d_array():
    """Test _get_n_samples with 2D numpy array."""
    X = np.random.randn(100, 50)
    n = PerformanceEvaluator._get_n_samples(X)
    assert n == 100


def test_get_n_samples_with_1d_array():
    """Test _get_n_samples with 1D numpy array."""
    X = np.random.randn(100)
    n = PerformanceEvaluator._get_n_samples(X)
    assert n == 100


def test_get_n_samples_with_list():
    """Test _get_n_samples with list."""
    X = list(range(100))
    n = PerformanceEvaluator._get_n_samples(X)
    assert n == 100


def test_get_n_samples_with_non_sequence():
    """Test _get_n_samples with non-sequence object."""
    X = 42  # Not a sequence
    n = PerformanceEvaluator._get_n_samples(X)
    assert n is None


# =============================================================================
# Test Convenience Functions
# =============================================================================


def test_evaluate_performance_convenience(simple_function, sample_data):
    """Test evaluate_performance convenience function."""
    result = evaluate_performance(simple_function, sample_data, n_runs=2)

    assert isinstance(result, dict)
    assert "runtime" in result
    assert result["runtime"] > 0


def test_evaluate_performance_with_options(simple_function, sample_data):
    """Test evaluate_performance with custom options."""
    result = evaluate_performance(
        simple_function,
        sample_data,
        n_runs=2,
        warmup_runs=1,
        track_memory=False,
        track_cpu=False,
    )

    assert result["runtime"] > 0
    assert result["memory_usage"] == 0.0
    assert result["cpu_time"] == 0.0


def test_benchmark_scalability_convenience():
    """Test benchmark_scalability convenience function."""
    sizes = [50, 100]
    result = benchmark_scalability(lambda X: X * 2, sizes, n_features=20, n_runs=2)

    assert isinstance(result, dict)
    assert 50 in result
    assert 100 in result


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow_with_real_algorithm():
    """Test full workflow with a realistic algorithm."""
    # Simulate PCA-like computation
    def mock_pca(X):
        # Center the data
        X_centered = X - X.mean(axis=0)
        # Compute covariance matrix
        cov = X_centered.T @ X_centered / (X.shape[0] - 1)
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(cov)
        return eigenvalues

    X = np.random.randn(100, 20)
    evaluator = PerformanceEvaluator(n_runs=3)

    result = evaluator.evaluate(mock_pca, X)

    assert result["runtime"] > 0
    assert result["throughput"] > 0
    assert result["memory_usage"] >= 0


def test_different_data_sizes_performance():
    """Test performance with different data sizes."""
    evaluator = PerformanceEvaluator(n_runs=2)

    small = np.random.randn(50, 20)
    large = np.random.randn(500, 20)

    small_result = evaluator.evaluate(lambda X: X @ X.T, small)
    large_result = evaluator.evaluate(lambda X: X @ X.T, large)

    # Large should take longer
    assert large_result["runtime"] > small_result["runtime"]


def test_comparison_matrix_operations():
    """Test comparison of different matrix operations."""
    evaluator = PerformanceEvaluator(n_runs=2)

    X = np.random.randn(100, 50)

    funcs = {
        "element_wise": lambda X: X * 2,
        "matrix_mult": lambda X: X @ X.T,
        "transpose": lambda X: X.T,
    }

    results = evaluator.compare(funcs, X)

    # All functions should complete successfully
    for name in funcs:
        assert "runtime" in results[name]
        assert results[name]["runtime"] > 0
        assert not np.isnan(results[name]["runtime"])


# Need to import dataclasses for the frozen test
import dataclasses
