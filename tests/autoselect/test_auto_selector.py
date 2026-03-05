"""Tests for AutoSelector class.

This module contains tests for the AutoSelector main class that orchestrates
automatic method selection across multiple analysis stages.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect import AutoSelectReport, StageReport
from scptensor.autoselect.core import AutoSelector
from scptensor.core import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def simple_container() -> ScpContainer:
    """Create a simple container for testing."""
    rng = np.random.default_rng(42)
    X = rng.random((5, 3))

    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3", "S4", "S5"],
            "batch": ["B1", "B1", "B2", "B2", "B1"],
        }
    )

    var = pl.DataFrame(
        {
            "_index": ["P1", "P2", "P3"],
            "protein": ["A", "B", "C"],
        }
    )

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("raw", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)

    return container


@pytest.fixture
def larger_container() -> ScpContainer:
    """Create a larger container suitable for reduce->cluster routing tests."""
    rng = np.random.default_rng(123)
    n_samples = 80
    n_features = 120

    X = rng.normal(size=(n_samples, n_features))

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(n_samples)],
            "batch": np.array(["B1"] * 40 + ["B2"] * 40),
        }
    )
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("raw", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


class TestAutoSelectorInit:
    """Test AutoSelector initialization."""

    def test_init_default_stages(self):
        """Test initialization with default stages (all)."""
        selector = AutoSelector()

        assert selector.stages == AutoSelector.SUPPORTED_STAGES
        assert selector.keep_all is False
        assert selector.parallel is True
        assert selector.n_jobs == -1

    def test_init_custom_stages(self):
        """Test initialization with custom stages."""
        stages = ["normalize", "impute"]
        selector = AutoSelector(stages=stages)

        assert selector.stages == stages

    def test_init_custom_options(self):
        """Test initialization with custom options."""
        selector = AutoSelector(
            stages=["normalize"],
            keep_all=True,
            weights={"normalize": {"variance": 0.7, "batch_effect": 0.3}},
            parallel=False,
            n_jobs=4,
            selection_strategy="quality",
            n_repeats=2,
            confidence_level=0.9,
        )

        assert selector.keep_all is True
        assert selector.parallel is False
        assert selector.n_jobs == 4
        assert selector.weights["normalize"]["variance"] == 0.7
        assert selector.selection_strategy == "quality"
        assert selector.n_repeats == 2
        assert selector.confidence_level == pytest.approx(0.9)

    def test_init_invalid_strategy_raises(self):
        """Test invalid selection strategy raises ValueError."""
        with pytest.raises(ValueError, match="selection_strategy must be one of"):
            AutoSelector(selection_strategy="invalid")

    def test_init_invalid_repeats_raises(self):
        """Test invalid n_repeats raises ValueError."""
        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            AutoSelector(n_repeats=0)

    def test_init_invalid_confidence_raises(self):
        """Test invalid confidence level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            AutoSelector(confidence_level=1.0)

    def test_init_invalid_stage_raises(self):
        """Test that invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):
            AutoSelector(stages=["invalid_stage"])

    def test_init_empty_stages(self):
        """Test initialization with empty stages list."""
        selector = AutoSelector(stages=[])

        assert selector.stages == []


class TestAutoSelectorGetEvaluator:
    """Test _get_evaluator method."""

    def test_get_evaluator_normalize(self):
        """Test getting normalization evaluator."""
        selector = AutoSelector(stages=["normalize"])

        # Should not raise NotImplementedError
        evaluator = selector._get_evaluator("normalize")
        assert evaluator is not None
        assert hasattr(evaluator, "stage_name")
        assert hasattr(evaluator, "methods")
        assert hasattr(evaluator, "metric_weights")

    def test_get_evaluator_impute(self):
        """Test getting imputation evaluator."""
        selector = AutoSelector(stages=["impute"])

        # Should not raise NotImplementedError
        evaluator = selector._get_evaluator("impute")
        assert evaluator is not None

    def test_get_evaluator_integrate(self):
        """Test getting integration evaluator."""
        selector = AutoSelector(stages=["integrate"])

        # Should not raise NotImplementedError
        evaluator = selector._get_evaluator("integrate")
        assert evaluator is not None
        assert hasattr(evaluator, "stage_name")

    def test_get_evaluator_reduce(self):
        """Test getting dimensionality reduction evaluator."""
        selector = AutoSelector(stages=["reduce"])

        # Should not raise NotImplementedError
        evaluator = selector._get_evaluator("reduce")
        assert evaluator is not None
        assert hasattr(evaluator, "stage_name")

    def test_get_evaluator_cluster(self):
        """Test getting clustering evaluator."""
        selector = AutoSelector(stages=["cluster"])

        # Should not raise NotImplementedError
        evaluator = selector._get_evaluator("cluster")
        assert evaluator is not None
        assert hasattr(evaluator, "stage_name")


class TestAutoSelectorRunStage:
    """Test run_stage method."""

    def test_run_stage_returns_container_and_report(self, simple_container):
        """Test that run_stage returns correct types."""
        selector = AutoSelector(stages=["normalize"])

        result_container, report = selector.run_stage(
            container=simple_container,
            stage="normalize",
            assay_name="proteins",
            source_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, StageReport)
        assert report.stage_name == "normalize"
        assert report.stage_key == "normalize"
        assert report.input_assay == "proteins"
        assert report.input_layer == "raw"
        assert report.output_assay == "proteins"
        assert report.output_layer is not None

    def test_run_stage_invalid_stage_raises(self, simple_container):
        """Test that invalid stage raises ValueError."""
        selector = AutoSelector()

        with pytest.raises(ValueError, match="Invalid stage"):
            selector.run_stage(
                container=simple_container,
                stage="invalid",
            )

    def test_run_stage_integrate_works(self, simple_container):
        """Test that integration stage works."""
        selector = AutoSelector(stages=["integrate"])

        # Should not raise NotImplementedError anymore
        result_container, report = selector.run_stage(
            container=simple_container,
            stage="integrate",
            assay_name="proteins",
            source_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, StageReport)
        assert report.stage_name == "integrate"

    def test_run_stage_missing_assay_raises(self, simple_container):
        """Test actionable error when assay is missing."""
        selector = AutoSelector(stages=["normalize"])
        with pytest.raises(ValueError, match="requires assay 'missing_assay'"):
            selector.run_stage(
                container=simple_container,
                stage="normalize",
                assay_name="missing_assay",
                source_layer="raw",
            )

    def test_run_stage_missing_layer_raises(self, simple_container):
        """Test actionable error when source layer is missing."""
        selector = AutoSelector(stages=["normalize"])
        with pytest.raises(ValueError, match="requires layer 'missing_layer'"):
            selector.run_stage(
                container=simple_container,
                stage="normalize",
                assay_name="proteins",
                source_layer="missing_layer",
            )

    def test_run_stage_integrate_missing_batch_key_raises(self, simple_container):
        """Test actionable error when integrate stage has no batch key."""
        selector = AutoSelector(stages=["integrate"])
        with pytest.raises(ValueError, match="requires batch_key 'missing_batch'"):
            selector.run_stage(
                container=simple_container,
                stage="integrate",
                assay_name="proteins",
                source_layer="raw",
                batch_key="missing_batch",
            )


class TestAutoSelectorRun:
    """Test run method (full pipeline)."""

    def test_run_returns_container_and_report(self, simple_container):
        """Test that run returns correct types."""
        selector = AutoSelector(stages=["normalize"])

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, AutoSelectReport)
        assert isinstance(report.total_time, float)
        assert report.total_time >= 0

    def test_run_empty_stages(self, simple_container):
        """Test run with no stages."""
        selector = AutoSelector(stages=[])

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, AutoSelectReport)
        assert len(report.stages) == 0

    def test_run_single_stage(self, simple_container):
        """Test run with single stage."""
        selector = AutoSelector(stages=["normalize"])

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert len(report.stages) == 1
        assert "normalize" in report.stages
        assert report.stages["normalize"].stage_name == "normalize"

    def test_run_multiple_stages(self, simple_container):
        """Test run with multiple stages."""
        selector = AutoSelector(stages=["normalize", "impute"])

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert len(report.stages) == 2
        assert "normalize" in report.stages
        assert "impute" in report.stages

    def test_run_integrate_stage_works(self, simple_container):
        """Test that integration stage works in full pipeline."""
        selector = AutoSelector(stages=["integrate"])

        # Should not raise NotImplementedError anymore
        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, AutoSelectReport)
        assert len(report.stages) == 1
        assert "integrate" in report.stages

    def test_run_keep_all_option(self, simple_container):
        """Test run with keep_all option."""
        selector = AutoSelector(stages=["normalize"], keep_all=True)

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        # Check that keep_all was passed to evaluator
        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, AutoSelectReport)

    def test_run_reduce_then_cluster_routes_assay_context(self, larger_container):
        """Test that reduce->cluster uses reduced assay ('pca/X') automatically."""
        selector = AutoSelector(stages=["reduce", "cluster"])
        result_container, report = selector.run(
            container=larger_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert "reduce" in report.stages
        assert "cluster" in report.stages

        reduce_stage = report.stages["reduce"]
        cluster_stage = report.stages["cluster"]

        assert reduce_stage.output_assay is not None
        assert reduce_stage.output_layer == "X"
        assert cluster_stage.input_assay == reduce_stage.output_assay
        assert cluster_stage.input_layer == reduce_stage.output_layer


class TestAutoSelectorProperties:
    """Test AutoSelector properties and attributes."""

    def test_supported_stages_class_attribute(self):
        """Test SUPPORTED_STAGES class attribute."""
        assert hasattr(AutoSelector, "SUPPORTED_STAGES")
        assert isinstance(AutoSelector.SUPPORTED_STAGES, list)
        assert "normalize" in AutoSelector.SUPPORTED_STAGES
        assert "impute" in AutoSelector.SUPPORTED_STAGES
        assert "integrate" in AutoSelector.SUPPORTED_STAGES
        assert "reduce" in AutoSelector.SUPPORTED_STAGES
        assert "cluster" in AutoSelector.SUPPORTED_STAGES

    def test_selector_instance_attributes(self):
        """Test selector instance attributes."""
        selector = AutoSelector()

        assert hasattr(selector, "stages")
        assert hasattr(selector, "keep_all")
        assert hasattr(selector, "weights")
        assert hasattr(selector, "parallel")
        assert hasattr(selector, "n_jobs")
        assert hasattr(selector, "selection_strategy")
        assert hasattr(selector, "n_repeats")
        assert hasattr(selector, "confidence_level")
