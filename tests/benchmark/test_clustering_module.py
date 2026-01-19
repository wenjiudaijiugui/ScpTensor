"""Tests for clustering test module."""

import numpy as np
import pytest

from scptensor.benchmark.modules import ClusteringTestModule, ModuleConfig


@pytest.fixture
def clustering_config():
    """Create test configuration for clustering module."""
    return ModuleConfig(
        name="clustering_test",
        datasets=["synthetic_small"],
        params={
            "n_clusters_list": [3, 5],
            "methods": ["kmeans"],
            "n_neighbors": 10,
            "random_state": 42,
        },
    )


@pytest.fixture
def clustering_module(clustering_config):
    """Create clustering test module instance."""
    return ClusteringTestModule(clustering_config)


def test_clustering_module_initialization(clustering_module):
    """Test module initialization."""
    assert clustering_module is not None
    assert clustering_module.config.name == "clustering_test"
    assert clustering_module.is_enabled()


def test_clustering_module_params(clustering_module):
    """Test parameter extraction."""
    assert clustering_module._n_clusters_list == [3, 5]
    assert clustering_module._methods == ["kmeans"]
    assert clustering_module._n_neighbors == 10
    assert clustering_module._random_state == 42


def test_clustering_module_run(clustering_module):
    """Test running clustering on synthetic_small dataset."""
    results = clustering_module.run("synthetic_small")

    # Should have results for each n_clusters value
    assert len(results) >= 2

    # Check first result
    first_result = results[0]
    assert first_result.module_name == "clustering_test"
    assert first_result.dataset_name == "synthetic_small"
    assert "kmeans" in first_result.method_name
    assert first_result.success is True


def test_clustering_module_metrics(clustering_module):
    """Test that clustering metrics are computed."""
    results = clustering_module.run("synthetic_small")

    for result in results:
        if result.success:
            assert result.metrics is not None
            # Should have at least some metrics
            assert "n_clusters_found" in result.metrics or len(result.metrics) > 0
            # Runtime should be positive
            assert result.runtime_seconds >= 0


def test_clustering_module_output(clustering_module):
    """Test that clustering outputs labels."""
    results = clustering_module.run("synthetic_small")

    for result in results:
        if result.success:
            assert result.output is not None
            assert isinstance(result.output, np.ndarray)
            # Labels should be integers
            assert result.output.dtype in [np.int32, np.int64, int]


def test_clustering_module_should_process_dataset(clustering_module):
    """Test dataset filtering."""
    # synthetic_small is in the config
    assert clustering_module.should_process_dataset("synthetic_small") is True
    # other datasets are not
    assert clustering_module.should_process_dataset("synthetic_large") is False


def test_clustering_module_should_process_method(clustering_module):
    """Test method filtering."""
    # When config.methods is empty, all methods are allowed
    # The clustering module uses params["methods"] for its own filtering
    assert clustering_module.config.methods == []
    # Empty methods list means process all methods
    assert clustering_module.should_process_method("kmeans") is True
    assert clustering_module.should_process_method("leiden") is True
    assert clustering_module.should_process_method("any_method") is True


def test_clustering_module_unknown_method(clustering_config):
    """Test handling of unknown method."""
    clustering_config.params["methods"] = ["unknown_method"]
    module = ClusteringTestModule(clustering_config)
    results = module.run("synthetic_small")

    # Should return error result
    assert len(results) == 1
    assert results[0].success is False
    assert "Unknown method" in results[0].error_message


def test_clustering_module_unknown_dataset(clustering_config):
    """Test handling of unknown dataset."""
    # Clear datasets filter to allow any dataset name
    clustering_config.datasets = []
    module = ClusteringTestModule(clustering_config)
    results = module.run("nonexistent_dataset")

    # Should return error result
    assert len(results) == 1
    assert results[0].success is False
    assert "not found" in results[0].error_message


def test_clustering_module_multiple_n_clusters(clustering_config):
    """Test with multiple cluster counts."""
    clustering_config.params["n_clusters_list"] = [2, 4, 6, 8]
    module = ClusteringTestModule(clustering_config)
    results = module.run("synthetic_small")

    # Should have results for scptensor and scanpy for each n_clusters
    # Each n_clusters produces 2 results (scptensor + scanpy)
    # But only if n_clusters < n_samples
    valid_clusters = [n for n in [2, 4, 6, 8] if n < 100]  # Assuming >100 samples
    assert len(results) >= len(valid_clusters)


def test_clustering_module_get_results(clustering_module):
    """Test get_results method."""
    clustering_module.run("synthetic_small")
    results = clustering_module.get_results()

    assert len(results) > 0


def test_clustering_module_clear_results(clustering_module):
    """Test clear_results method."""
    clustering_module.run("synthetic_small")
    assert len(clustering_module.get_results()) > 0

    clustering_module.clear_results()
    assert len(clustering_module.get_results()) == 0


def test_clustering_module_n_clusters_too_large(clustering_config):
    """Test handling of n_clusters larger than dataset size."""
    # Set n_clusters larger than typical dataset
    clustering_config.params["n_clusters_list"] = [10000]
    module = ClusteringTestModule(clustering_config)
    results = module.run("synthetic_small")

    # Should skip the too-large n_clusters
    assert len(results) == 0


def test_clustering_module_with_empty_methods(clustering_config):
    """Test with empty methods list."""
    clustering_config.params["methods"] = []
    module = ClusteringTestModule(clustering_config)
    results = module.run("synthetic_small")

    # Should return empty results
    assert len(results) == 0


def test_clustering_module_is_enabled_check():
    """Test is_enabled method."""
    config = ModuleConfig(name="clustering_test", enabled=True)
    module_enabled = ClusteringTestModule(config)
    assert module_enabled.is_enabled() is True

    config_disabled = ModuleConfig(name="clustering_test", enabled=False)
    module_disabled = ClusteringTestModule(config_disabled)
    assert module_disabled.is_enabled() is False


def test_clustering_metrics_internal():
    """Test internal _metrics method."""
    from scptensor.benchmark.modules.clustering_test import ClusteringTestModule

    config = ModuleConfig(name="clustering_test")
    module = ClusteringTestModule(config)

    # Create synthetic data
    X = np.random.randn(50, 10)
    labels = np.array([0] * 25 + [1] * 25)
    true_labels = labels.copy()

    metrics = module._metrics(X, labels, true_labels, inertia=100.0)

    # Check metrics exist
    assert "ari" in metrics
    assert "nmi" in metrics
    assert "n_clusters_found" in metrics
    assert "inertia" in metrics
    assert metrics["n_clusters_found"] == 2
    assert metrics["inertia"] == 100.0
