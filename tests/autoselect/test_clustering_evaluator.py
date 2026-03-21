"""Tests for ClusteringEvaluator.

This module contains tests for the ClusteringEvaluator class.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect.evaluators.clustering import ClusteringEvaluator
from scptensor.core import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def container_for_clustering() -> ScpContainer:
    """Create a container with PCA data for clustering testing."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 10

    # Create data with some cluster structure
    X = np.zeros((n_samples, n_features))
    # Create 3 clusters
    X[:33] = rng.random((33, n_features)) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    X[33:66] = rng.random((33, n_features)) + np.array([3, 3, 3, 3, 3, 0, 0, 0, 0, 0])
    X[66:] = rng.random((34, n_features)) + np.array([6, 6, 6, 6, 6, 0, 0, 0, 0, 0])

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"PC{i}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("X", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("pca", assay)

    return container


@pytest.fixture
def container_with_clusters() -> ScpContainer:
    """Create a container with clustering results for testing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 10

    X = rng.random((n_samples, n_features))

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
            "kmeans_k5": np.repeat([0, 1, 2, 3, 4], 10),
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"PC{i}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("X", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("pca", assay)

    return container


class TestClusteringEvaluatorInit:
    """Test ClusteringEvaluator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        evaluator = ClusteringEvaluator()
        assert evaluator._n_clusters == 5
        assert evaluator._resolution == 1.0
        assert evaluator._n_neighbors == 15
        assert evaluator._random_state == 42

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        evaluator = ClusteringEvaluator(
            n_clusters=10, resolution=0.8, n_neighbors=20, random_state=123
        )
        assert evaluator._n_clusters == 10
        assert evaluator._resolution == 0.8
        assert evaluator._n_neighbors == 20
        assert evaluator._random_state == 123


class TestClusteringEvaluatorProperties:
    """Test ClusteringEvaluator properties."""

    def test_stage_name(self):
        """Test stage_name property."""
        evaluator = ClusteringEvaluator()
        assert evaluator.stage_name == "cluster"

    def test_methods_returns_dict(self):
        """Test methods property returns a dictionary."""
        evaluator = ClusteringEvaluator()
        methods = evaluator.methods
        assert isinstance(methods, dict)
        # K-means should always be available
        assert "kmeans" in methods

    def test_metric_weights(self):
        """Test metric_weights property."""
        evaluator = ClusteringEvaluator()
        weights = evaluator.metric_weights
        assert isinstance(weights, dict)
        assert "silhouette" in weights
        assert "calinski_harabasz" in weights
        assert "davies_bouldin" in weights
        assert "stability" in weights
        # Check weights sum to approximately 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)


class TestClusteringEvaluatorComputeMetrics:
    """Test ClusteringEvaluator compute_metrics method."""

    def test_compute_metrics_returns_dict(self, container_with_clusters):
        """Test that compute_metrics returns a dictionary."""
        evaluator = ClusteringEvaluator()
        scores = evaluator.compute_metrics(
            container=container_with_clusters,
            original_container=container_with_clusters,
            layer_name="kmeans_k5",
        )

        assert isinstance(scores, dict)
        for key in evaluator.metric_weights:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_compute_metrics_missing_column(self, container_for_clustering):
        """Test compute_metrics with missing clustering column."""
        evaluator = ClusteringEvaluator()
        scores = evaluator.compute_metrics(
            container=container_for_clustering,
            original_container=container_for_clustering,
            layer_name="nonexistent",
        )

        for key in evaluator.metric_weights:
            assert scores[key] == 0.0


class TestClusteringEvaluatorHelpers:
    """Test ClusteringEvaluator helper methods."""

    def test_compute_silhouette(self, container_with_clusters):
        """Test _compute_silhouette method."""
        evaluator = ClusteringEvaluator()
        assay = container_with_clusters.assays["pca"]
        X = assay.layers["X"].X
        labels = container_with_clusters.obs["kmeans_k5"].to_numpy()

        score = evaluator._compute_silhouette(X, labels)
        assert 0.0 <= score <= 1.0

    def test_compute_calinski_harabasz(self, container_with_clusters):
        """Test _compute_calinski_harabasz method."""
        evaluator = ClusteringEvaluator()
        assay = container_with_clusters.assays["pca"]
        X = assay.layers["X"].X
        labels = container_with_clusters.obs["kmeans_k5"].to_numpy()

        score = evaluator._compute_calinski_harabasz(X, labels)
        assert 0.0 <= score <= 1.0

    def test_compute_davies_bouldin(self, container_with_clusters):
        """Test _compute_davies_bouldin method."""
        evaluator = ClusteringEvaluator()
        assay = container_with_clusters.assays["pca"]
        X = assay.layers["X"].X
        labels = container_with_clusters.obs["kmeans_k5"].to_numpy()

        score = evaluator._compute_davies_bouldin(X, labels)
        assert 0.0 <= score <= 1.0

    def test_compute_stability(self, container_with_clusters):
        """Test _compute_stability method."""
        evaluator = ClusteringEvaluator()
        assay = container_with_clusters.assays["pca"]
        X = assay.layers["X"].X
        labels = container_with_clusters.obs["kmeans_k5"].to_numpy()

        score = evaluator._compute_stability(X, labels)
        assert 0.0 <= score <= 1.0

    def test_compute_silhouette_single_cluster(self):
        """Test _compute_silhouette with single cluster."""
        evaluator = ClusteringEvaluator()
        rng = np.random.default_rng(42)
        X = rng.random((50, 10))
        labels = np.zeros(50)  # All same cluster

        score = evaluator._compute_silhouette(X, labels)
        assert score == 0.0


class TestClusteringEvaluatorRunAll:
    """Test ClusteringEvaluator run_all method."""

    def test_run_all_returns_container_and_report(self, container_for_clustering):
        """Test that run_all returns container and report."""
        evaluator = ClusteringEvaluator(n_clusters=3)
        result_container, report = evaluator.run_all(
            container=container_for_clustering,
            assay_name="pca",
            source_layer="X",
        )

        assert isinstance(result_container, ScpContainer)
        assert report.stage_name == "cluster"
        assert len(report.results) > 0

    def test_run_all_adds_clustering_to_obs(self, container_for_clustering):
        """Test that run_all adds clustering results to obs."""
        evaluator = ClusteringEvaluator(n_clusters=3)
        result_container, report = evaluator.run_all(
            container=container_for_clustering,
            assay_name="pca",
            source_layer="X",
        )

        # K-means should succeed and add a column
        if report.best_method == "kmeans":
            assert "kmeans_k3" in result_container.obs.columns

    def test_run_all_identifies_best_method(self, container_for_clustering):
        """Test that run_all identifies best method."""
        evaluator = ClusteringEvaluator(n_clusters=3)
        result_container, report = evaluator.run_all(
            container=container_for_clustering,
            assay_name="pca",
            source_layer="X",
        )

        # At least one method should succeed
        if report.best_method:
            assert report.best_result is not None
            assert report.best_result.error is None
            assert report.best_result.overall_score >= 0

    def test_run_all_keep_all_uses_unified_obs_attachment(self, container_for_clustering):
        """Obs outputs should be preserved for every successful method when keep_all=True."""
        evaluator = ClusteringEvaluator(n_clusters=3)

        def method_a(container, assay_name, source_layer, **kwargs):
            del assay_name, source_layer, kwargs
            labels = np.tile(np.array([0, 1, 2]), len(container.obs) // 3 + 1)[: len(container.obs)]
            container.obs = container.obs.with_columns(
                pl.Series(name="method_a_result", values=labels)
            )
            return container

        def method_b(container, assay_name, source_layer, **kwargs):
            del assay_name, source_layer, kwargs
            labels = np.tile(np.array([0, 0, 1, 1]), len(container.obs) // 4 + 1)[
                : len(container.obs)
            ]
            container.obs = container.obs.with_columns(
                pl.Series(name="method_b_result", values=labels)
            )
            return container

        evaluator._available_methods = {"method_a": method_a, "method_b": method_b}

        def fake_compute_metrics(self, container, original_container, layer_name):
            del container, original_container
            score = 0.9 if layer_name == "method_a_result" else 0.8
            return {
                "silhouette": score,
                "calinski_harabasz": score,
                "davies_bouldin": score,
                "stability": score,
            }

        evaluator.compute_metrics = fake_compute_metrics.__get__(evaluator, ClusteringEvaluator)

        result_container, report = evaluator.run_all(
            container=container_for_clustering,
            assay_name="pca",
            source_layer="X",
            keep_all=True,
            selection_strategy="quality",
        )

        assert report.best_method == "method_a"
        assert "method_a_result" in result_container.obs.columns
        assert "method_b_result" in result_container.obs.columns
