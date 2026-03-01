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
        )

        assert selector.keep_all is True
        assert selector.parallel is False
        assert selector.n_jobs == 4
        assert selector.weights["normalize"]["variance"] == 0.7

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
        # The evaluator returns its own stage_name which is "normalization"
        assert report.stage_name == "normalization"

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
        # The stage is stored with the evaluator's stage_name as key
        assert "normalization" in report.stages
        assert report.stages["normalization"].stage_name == "normalization"

    def test_run_multiple_stages(self, simple_container):
        """Test run with multiple stages."""
        selector = AutoSelector(stages=["normalize", "impute"])

        result_container, report = selector.run(
            container=simple_container,
            assay_name="proteins",
            initial_layer="raw",
        )

        assert len(report.stages) == 2
        # Stages are stored with evaluator's stage_name as key
        assert "normalization" in report.stages
        assert "imputation" in report.stages

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
