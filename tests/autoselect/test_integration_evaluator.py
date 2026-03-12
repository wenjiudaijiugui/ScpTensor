"""Tests for IntegrationEvaluator.

This module contains tests for the IntegrationEvaluator class.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect import auto_integrate
from scptensor.autoselect.evaluators.integration import IntegrationEvaluator
from scptensor.core import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def container_with_batches() -> ScpContainer:
    """Create a container with batch information for testing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 10

    # Create data with batch effects
    X = rng.random((n_samples, n_features))

    # Add batch effect
    batches = np.array(["B1"] * 25 + ["B2"] * 25)
    X[:25] += 0.5  # Batch 1 has higher values

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
            "batch": batches,
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


class TestIntegrationEvaluatorInit:
    """Test IntegrationEvaluator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        evaluator = IntegrationEvaluator()
        assert evaluator._batch_key == "batch"
        assert evaluator._bio_key is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        evaluator = IntegrationEvaluator(batch_key="run", bio_key="cell_type")
        assert evaluator._batch_key == "run"
        assert evaluator._bio_key == "cell_type"


class TestIntegrationEvaluatorProperties:
    """Test IntegrationEvaluator properties."""

    def test_stage_name(self):
        """Test stage_name property."""
        evaluator = IntegrationEvaluator()
        assert evaluator.stage_name == "integrate"

    def test_methods_returns_dict(self):
        """Test methods property returns a dictionary."""
        evaluator = IntegrationEvaluator()
        methods = evaluator.methods
        assert isinstance(methods, dict)
        assert set(methods) == {"none", "limma"}

    def test_methods_can_include_embedding_candidates(self):
        """Embedding-only methods should require an explicit opt-in."""
        evaluator = IntegrationEvaluator(include_embedding_methods=True)
        methods = evaluator.methods
        assert "none" in methods
        assert "limma" in methods
        assert "mnn" in methods
        assert "combat" not in methods

    def test_metric_weights(self):
        """Test metric_weights property."""
        evaluator = IntegrationEvaluator()
        weights = evaluator.metric_weights
        assert isinstance(weights, dict)
        assert "batch_asw" in weights
        assert "batch_mixing" in weights
        assert "variance_preserved" in weights

    def test_metric_weights_with_bio_key(self):
        """Test metric_weights includes bio_asw when bio_key is set."""
        evaluator = IntegrationEvaluator(bio_key="cell_type")
        weights = evaluator.metric_weights
        assert "bio_asw" in weights


class TestIntegrationEvaluatorComputeMetrics:
    """Test IntegrationEvaluator compute_metrics method."""

    def test_compute_metrics_returns_dict(self, container_with_batches):
        """Test that compute_metrics returns a dictionary."""
        evaluator = IntegrationEvaluator()
        # Add a mock integrated layer
        assay = container_with_batches.assays["proteins"]
        X_raw = assay.layers["raw"].X
        integrated_matrix = ScpMatrix(X=X_raw * 0.9)  # Mock integration
        assay.add_layer("raw_integrated", integrated_matrix)

        scores = evaluator.compute_metrics(
            container=container_with_batches,
            original_container=container_with_batches,
            layer_name="raw_integrated",
        )

        assert isinstance(scores, dict)
        for key in evaluator.metric_weights:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_compute_metrics_missing_layer(self, container_with_batches):
        """Test compute_metrics with missing layer."""
        evaluator = IntegrationEvaluator()
        scores = evaluator.compute_metrics(
            container=container_with_batches,
            original_container=container_with_batches,
            layer_name="nonexistent",
        )

        for key in evaluator.metric_weights:
            assert scores[key] == 0.0

    def test_compute_metrics_missing_batch_key(self):
        """Test compute_metrics with missing batch key."""
        evaluator = IntegrationEvaluator(batch_key="nonexistent")

        rng = np.random.default_rng(42)
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = rng.random((3, 2))
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var)
        assay.add_layer("raw", matrix)
        container = ScpContainer(obs=obs)
        container.add_assay("proteins", assay)

        scores = evaluator.compute_metrics(
            container=container,
            original_container=container,
            layer_name="raw",
        )

        for key in evaluator.metric_weights:
            assert scores[key] == 0.0


class TestIntegrationEvaluatorRunAll:
    """Test IntegrationEvaluator run_all method."""

    def test_run_all_returns_container_and_report(self, container_with_batches):
        """Test that run_all returns container and report."""
        evaluator = IntegrationEvaluator()
        result_container, report = evaluator.run_all(
            container=container_with_batches,
            assay_name="proteins",
            source_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert report.stage_name == "integrate"
        assert len(report.results) > 0
        assert set(report.method_contracts) == {"none", "limma"}
        assert {result.method_name for result in report.results} == {"none", "limma"}
        assert all(result.method_contract is not None for result in report.results)
        assert all(
            contract["integration_level"] == "matrix"
            and contract["recommended_for_de"] is True
            and contract["candidate_scope"] == "stable"
            for contract in report.method_contracts.values()
        )
        assert report.recommendation_reason.startswith(
            "Candidate set restricted to matrix-level methods "
        )

    def test_run_all_identifies_best_method(self, container_with_batches):
        """Test that run_all identifies best method."""
        evaluator = IntegrationEvaluator()
        result_container, report = evaluator.run_all(
            container=container_with_batches,
            assay_name="proteins",
            source_layer="raw",
        )

        # At least one method should succeed
        if report.best_method:
            assert report.best_result is not None
            assert report.best_result.error is None

    def test_run_all_can_include_embedding_candidates(self, container_with_batches):
        """Explicit opt-in should expose exploratory embedding methods in the report."""
        evaluator = IntegrationEvaluator()
        _, report = evaluator.run_all(
            container=container_with_batches,
            assay_name="proteins",
            source_layer="raw",
            include_embedding_methods=True,
        )

        assert "mnn" in report.method_contracts
        assert report.method_contracts["mnn"]["integration_level"] == "embedding"
        assert report.method_contracts["mnn"]["recommended_for_de"] is False
        assert report.method_contracts["mnn"]["candidate_scope"] == "exploratory"
        assert report.recommendation_reason.startswith(
            "Exploratory embedding-level integration methods were included."
        )

    def test_auto_integrate_defaults_to_stable_contract(self, container_with_batches):
        """Convenience API should default to stable matrix-level candidates only."""
        _, report = auto_integrate(
            container_with_batches,
            assay_name="proteins",
            source_layer="raw",
        )

        assert {result.method_name for result in report.results} == {"none", "limma"}
        assert set(report.method_contracts) == {"none", "limma"}

    def test_auto_integrate_can_include_embedding_candidates(self, container_with_batches):
        """Convenience API should expose exploratory methods only on explicit opt-in."""
        _, report = auto_integrate(
            container_with_batches,
            assay_name="proteins",
            source_layer="raw",
            include_embedding_methods=True,
        )

        assert "mnn" in report.method_contracts
        assert report.method_contracts["mnn"]["candidate_scope"] == "exploratory"

    def test_run_all_accepts_protein_alias(self, container_with_batches):
        """Evaluator should resolve proteins/protein assay aliases."""
        assay = container_with_batches.assays.pop("proteins")
        container_with_batches.assays["protein"] = assay

        evaluator = IntegrationEvaluator()
        result_container, report = evaluator.run_all(
            container=container_with_batches,
            assay_name="proteins",
            source_layer="raw",
        )

        assert report.best_result is not None
        assert report.best_result.error is None
        assert "protein" in result_container.assays


class TestIntegrationEvaluatorHelpers:
    """Test IntegrationEvaluator helper methods."""

    def test_compute_batch_asw(self, container_with_batches):
        """Test _compute_batch_asw method."""
        evaluator = IntegrationEvaluator()
        assay = container_with_batches.assays["proteins"]
        X = assay.layers["raw"].X
        batches = container_with_batches.obs["batch"].to_numpy()

        score = evaluator._compute_batch_asw(X, batches)
        assert 0.0 <= score <= 1.0

    def test_compute_batch_mixing(self, container_with_batches):
        """Test _compute_batch_mixing method."""
        evaluator = IntegrationEvaluator()
        assay = container_with_batches.assays["proteins"]
        X = assay.layers["raw"].X
        batches = container_with_batches.obs["batch"].to_numpy()

        score = evaluator._compute_batch_mixing(X, batches)
        assert 0.0 <= score <= 1.0

    def test_compute_variance_preserved(self, container_with_batches):
        """Test _compute_variance_preserved method."""
        evaluator = IntegrationEvaluator()
        assay = container_with_batches.assays["proteins"]
        X = assay.layers["raw"].X

        # Add integrated layer
        integrated_matrix = ScpMatrix(X=X * 0.9)
        assay.add_layer("raw_integrated", integrated_matrix)

        score = evaluator._compute_variance_preserved(
            container_with_batches, container_with_batches, "raw_integrated"
        )
        assert 0.0 <= score <= 1.0
