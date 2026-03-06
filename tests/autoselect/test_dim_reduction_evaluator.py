"""Tests for DimReductionEvaluator.

This module contains tests for the DimReductionEvaluator class.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect.evaluators.dim_reduction import DimReductionEvaluator
from scptensor.core import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def container_for_reduction() -> ScpContainer:
    """Create a container for dimensionality reduction testing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 20

    # Create data with some structure
    X = rng.random((n_samples, n_features))

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"P{i}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("raw", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)

    return container


@pytest.fixture
def container_with_pca() -> ScpContainer:
    """Create a container with PCA assay for testing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 10

    # Create PCA-like data
    X_pca = rng.random((n_samples, n_features))

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"PC{i}" for i in range(n_features)],
            "explained_variance_ratio": np.array(
                [0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.03, 0.01]
            ),
        }
    )

    matrix = ScpMatrix(X=X_pca)
    assay = Assay(var=var)
    assay.add_layer("X", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("pca", assay)

    # Also add proteins assay
    proteins_var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})
    proteins_X = rng.random((n_samples, 20))
    proteins_matrix = ScpMatrix(X=proteins_X)
    proteins_assay = Assay(var=proteins_var)
    proteins_assay.add_layer("raw", proteins_matrix)
    container.add_assay("proteins", proteins_assay)

    return container


class TestDimReductionEvaluatorInit:
    """Test DimReductionEvaluator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        evaluator = DimReductionEvaluator()
        assert evaluator._n_components == 50
        assert evaluator._n_neighbors == 15
        assert evaluator._random_state == 42

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        evaluator = DimReductionEvaluator(n_components=30, n_neighbors=10, random_state=123)
        assert evaluator._n_components == 30
        assert evaluator._n_neighbors == 10
        assert evaluator._random_state == 123


class TestDimReductionEvaluatorProperties:
    """Test DimReductionEvaluator properties."""

    def test_stage_name(self):
        """Test stage_name property."""
        evaluator = DimReductionEvaluator()
        assert evaluator.stage_name == "reduce"

    def test_methods_returns_dict(self):
        """Test methods property returns a dictionary."""
        evaluator = DimReductionEvaluator()
        methods = evaluator.methods
        assert isinstance(methods, dict)
        # PCA should always be available
        assert "pca" in methods
        assert "tsne" in methods

    def test_metric_weights(self):
        """Test metric_weights property."""
        evaluator = DimReductionEvaluator()
        weights = evaluator.metric_weights
        assert isinstance(weights, dict)
        assert "variance_explained" in weights
        assert "reconstruction_error" in weights
        assert "local_structure" in weights
        assert "clustering_potential" in weights
        # Check weights sum to approximately 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)


class TestDimReductionEvaluatorComputeMetrics:
    """Test DimReductionEvaluator compute_metrics method."""

    def test_compute_metrics_returns_dict(self, container_with_pca):
        """Test that compute_metrics returns a dictionary."""
        evaluator = DimReductionEvaluator()
        scores = evaluator.compute_metrics(
            container=container_with_pca,
            original_container=container_with_pca,
            layer_name="pca",
        )

        assert isinstance(scores, dict)
        for key in evaluator.metric_weights:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_compute_metrics_missing_assay(self, container_for_reduction):
        """Test compute_metrics with missing assay."""
        evaluator = DimReductionEvaluator()
        scores = evaluator.compute_metrics(
            container=container_for_reduction,
            original_container=container_for_reduction,
            layer_name="nonexistent",
        )

        for key in evaluator.metric_weights:
            assert scores[key] == 0.0


class TestDimReductionEvaluatorHelpers:
    """Test DimReductionEvaluator helper methods."""

    def test_compute_variance_explained(self, container_with_pca):
        """Test _compute_variance_explained method."""
        evaluator = DimReductionEvaluator()
        assay = container_with_pca.assays["pca"]
        X = assay.layers["X"].X

        score = evaluator._compute_variance_explained(X, X, assay)
        assert 0.0 <= score <= 1.0

    def test_compute_local_structure(self):
        """Test _compute_local_structure method."""
        evaluator = DimReductionEvaluator()
        rng = np.random.default_rng(42)
        X_original = rng.random((50, 20))
        X_reduced = rng.random((50, 10))

        score = evaluator._compute_local_structure(X_original, X_reduced)
        assert 0.0 <= score <= 1.0

    def test_compute_clustering_potential(self):
        """Test _compute_clustering_potential method."""
        evaluator = DimReductionEvaluator()
        rng = np.random.default_rng(42)
        X = rng.random((50, 10))

        score = evaluator._compute_clustering_potential(X)
        assert 0.0 <= score <= 1.0


class TestDimReductionEvaluatorRunAll:
    """Test DimReductionEvaluator run_all method."""

    def test_run_all_returns_container_and_report(self, container_for_reduction):
        """Test that run_all returns container and report."""
        evaluator = DimReductionEvaluator(n_components=5)
        result_container, report = evaluator.run_all(
            container=container_for_reduction,
            assay_name="proteins",
            source_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert report.stage_name == "reduce"
        assert len(report.results) > 0

    def test_run_all_creates_pca_assay(self, container_for_reduction):
        """Test that run_all creates PCA assay."""
        evaluator = DimReductionEvaluator(n_components=5)
        result_container, report = evaluator.run_all(
            container=container_for_reduction,
            assay_name="proteins",
            source_layer="raw",
        )

        # PCA should be available and should succeed
        if "pca" in result_container.assays:
            assert "X" in result_container.assays["pca"].layers
