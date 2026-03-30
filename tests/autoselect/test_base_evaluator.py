"""Tests for BaseEvaluator abstract base class.

This module contains tests for the BaseEvaluator abstract class and its
concrete implementations.
"""

import time
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect import EvaluationResult, StageReport
from scptensor.autoselect.evaluators.base import BaseEvaluator, create_wrapper
from scptensor.core import Assay, ScpContainer, ScpMatrix


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing purposes."""

    @property
    def stage_name(self) -> str:
        """Return stage name."""
        return "test_stage"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return available methods."""
        return {
            "method_a": self._method_a,
            "method_b": self._method_b,
            "method_failing": self._method_failing,
        }

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return metric weights."""
        return {
            "metric1": 0.5,
            "metric2": 0.3,
            "metric3": 0.2,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute mock metrics."""
        # Simple mock: return scores based on layer existence
        assay = container.assays.get("proteins")
        if assay is None or layer_name not in assay.layers:
            return {"metric1": 0.0, "metric2": 0.0, "metric3": 0.0}

        # Use layer name to generate different scores for testing
        if "method_a" in layer_name:
            return {"metric1": 0.9, "metric2": 0.8, "metric3": 0.85}
        if "method_b" in layer_name:
            return {"metric1": 0.85, "metric2": 0.9, "metric3": 0.8}
        return {"metric1": 0.7, "metric2": 0.7, "metric3": 0.7}

    def _method_a(
        self,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        **kwargs,
    ) -> ScpContainer:
        """Mock method A that succeeds."""
        assay = container.assays[assay_name]
        X = assay.layers[source_layer].X
        new_X = X * 1.1  # Simple transformation
        new_matrix = ScpMatrix(X=new_X, M=assay.layers[source_layer].M)
        new_layer_name = f"{source_layer}_method_a"
        assay.add_layer(new_layer_name, new_matrix)
        return container

    def _method_b(
        self,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        **kwargs,
    ) -> ScpContainer:
        """Mock method B that succeeds."""
        assay = container.assays[assay_name]
        X = assay.layers[source_layer].X
        new_X = X * 0.9  # Simple transformation
        new_matrix = ScpMatrix(X=new_X, M=assay.layers[source_layer].M)
        new_layer_name = f"{source_layer}_method_b"
        assay.add_layer(new_layer_name, new_matrix)
        return container

    def _method_failing(
        self,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        **kwargs,
    ) -> ScpContainer:
        """Mock method that always fails."""
        raise ValueError("Intentional failure for testing")


@pytest.fixture
def simple_container() -> ScpContainer:
    """Create a simple container for testing."""
    rng = np.random.default_rng(42)
    X = rng.random((5, 3))

    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3", "S4", "S5"],
            "batch": ["B1", "B1", "B2", "B2", "B1"],
        },
    )

    var = pl.DataFrame(
        {
            "_index": ["P1", "P2", "P3"],
            "protein": ["A", "B", "C"],
        },
    )

    matrix = ScpMatrix(X=X)
    assay = Assay(var=var)
    assay.add_layer("raw", matrix)

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)

    return container


class TestBaseEvaluatorAbstract:
    """Test that BaseEvaluator is properly abstract."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore

    def test_must_implement_stage_name(self):
        """Test that stage_name must be implemented."""

        class IncompleteEvaluator(BaseEvaluator):
            pass  # Missing all abstract methods

        with pytest.raises(TypeError):
            IncompleteEvaluator()  # type: ignore

    def test_must_implement_methods(self):
        """Test that methods property must be implemented."""

        class IncompleteEvaluator(BaseEvaluator):
            @property
            def stage_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteEvaluator()  # type: ignore


class TestComputeOverallScore:
    """Test compute_overall_score method."""

    def test_compute_overall_score_basic(self):
        """Test basic weighted score computation."""
        evaluator = MockEvaluator()

        scores = {"metric1": 0.9, "metric2": 0.8, "metric3": 0.7}
        # Expected: (0.9 * 0.5 + 0.8 * 0.3 + 0.7 * 0.2) / 1.0 = 0.83
        result = evaluator.compute_overall_score(scores)

        assert result == pytest.approx(0.83, rel=1e-3)

    def test_compute_overall_score_partial_scores(self):
        """Test with partial scores (some metrics missing)."""
        evaluator = MockEvaluator()

        scores = {"metric1": 0.9, "metric2": 0.8}  # metric3 missing
        # Expected: (0.9 * 0.5 + 0.8 * 0.3 + 0.0 * 0.2) / 1.0 = 0.69
        result = evaluator.compute_overall_score(scores)

        assert result == pytest.approx(0.69, rel=1e-3)

    def test_compute_overall_score_empty_scores(self):
        """Test with empty scores."""
        evaluator = MockEvaluator()

        scores = {}
        result = evaluator.compute_overall_score(scores)

        assert result == 0.0

    def test_compute_overall_score_zero_weights(self):
        """Test with zero total weight."""

        class ZeroWeightEvaluator(BaseEvaluator):
            @property
            def stage_name(self) -> str:
                return "zero_weight"

            @property
            def methods(self) -> dict[str, Callable]:
                return {}

            @property
            def metric_weights(self) -> dict[str, float]:
                return {"metric1": 0.0, "metric2": 0.0}

            def compute_metrics(
                self,
                container,
                original_container,
                layer_name,
            ) -> dict[str, float]:
                return {}

        evaluator = ZeroWeightEvaluator()
        scores = {"metric1": 0.9, "metric2": 0.8}
        result = evaluator.compute_overall_score(scores)

        assert result == 0.0

    def test_compute_overall_score_override_weights(self):
        """Test override metric weights."""
        evaluator = MockEvaluator()
        evaluator.set_metric_weights({"metric1": 0.0, "metric2": 1.0, "metric3": 0.0})

        scores = {"metric1": 0.2, "metric2": 0.4, "metric3": 0.9}
        result = evaluator.compute_overall_score(scores)

        assert result == pytest.approx(0.4, rel=1e-3)

    def test_set_metric_weights_validation(self):
        """Test metric weight validation."""
        evaluator = MockEvaluator()

        with pytest.raises(ValueError):
            evaluator.set_metric_weights({"unknown": 1.0})

        with pytest.raises(ValueError):
            evaluator.set_metric_weights({"metric1": -0.1})

        with pytest.raises(ValueError):
            evaluator.set_metric_weights({"metric2": np.nan})


class TestEvaluateMethod:
    """Test evaluate_method method."""

    def test_evaluate_method_success(self, simple_container):
        """Test evaluating a successful method."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, eval_result = evaluator.evaluate_method(
            container=container_copy,
            method_name="method_a",
            method_func=evaluator._method_a,
            assay_name="proteins",
            source_layer="raw",
        )

        assert result_container is not None
        assert eval_result.method_name == "method_a"
        assert eval_result.error is None
        assert eval_result.overall_score > 0
        assert eval_result.execution_time >= 0
        assert "raw_method_a" in result_container.assays["proteins"].layers

    def test_evaluate_method_failure(self, simple_container):
        """Test evaluating a failing method."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, eval_result = evaluator.evaluate_method(
            container=container_copy,
            method_name="method_failing",
            method_func=evaluator._method_failing,
            assay_name="proteins",
            source_layer="raw",
        )

        assert result_container is None
        assert eval_result.method_name == "method_failing"
        assert eval_result.error is not None
        assert "Intentional failure" in eval_result.error
        assert eval_result.overall_score == 0.0

    def test_evaluate_method_creates_new_layer(self, simple_container):
        """Test that method creates a new layer with correct name."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, eval_result = evaluator.evaluate_method(
            container=container_copy,
            method_name="method_a",
            method_func=evaluator._method_a,
            assay_name="proteins",
            source_layer="raw",
        )

        assert result_container is not None
        assert eval_result.layer_name == "raw_method_a"

    def test_get_metric_assay_has_no_hidden_default(self, simple_container):
        """Metric assay lookup should not silently fall back to proteins."""
        evaluator = MockEvaluator()

        assert evaluator._get_metric_assay(simple_container) is None


class TestRunAll:
    """Test run_all method."""

    def test_run_all_returns_container_and_report(self, simple_container):
        """Test that run_all returns container and StageReport."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        assert isinstance(result_container, ScpContainer)
        assert isinstance(report, StageReport)
        assert report.stage_name == "test_stage"

    def test_run_all_tests_all_methods(self, simple_container):
        """Test that run_all tests all registered methods."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        assert len(report.results) == 3  # method_a, method_b, method_failing

    def test_run_all_identifies_best_method(self, simple_container):
        """Test that run_all identifies the best performing method."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        assert report.best_method != ""
        assert report.best_result is not None
        assert report.best_result.error is None

    def test_run_all_keeps_best_layer_by_default(self, simple_container):
        """Test that run_all keeps only the best layer by default."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        # Should have raw layer + best method layer
        assay = result_container.assays["proteins"]
        assert "raw" in assay.layers
        assert report.best_result is not None
        assert report.best_result.layer_name in assay.layers

    def test_run_all_keep_all_true(self, simple_container):
        """Test that run_all with keep_all=True keeps all result layers."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
            keep_all=True,
        )

        assay = result_container.assays["proteins"]
        assert "raw" in assay.layers
        # Only successful methods should have layers
        assert "raw_method_a" in assay.layers
        assert "raw_method_b" in assay.layers
        # Failing method should not create a layer
        assert "raw_method_failing" not in assay.layers

    def test_run_all_handles_all_failures(self, simple_container):
        """Test run_all when all methods fail."""

        class AllFailingEvaluator(BaseEvaluator):
            @property
            def stage_name(self) -> str:
                return "all_failing"

            @property
            def methods(self) -> dict[str, Callable]:
                return {"fail1": self._fail, "fail2": self._fail}

            @property
            def metric_weights(self) -> dict[str, float]:
                return {"metric": 1.0}

            def compute_metrics(self, container, original, layer_name):
                return {"metric": 0.0}

            def _fail(self, container, assay_name, source_layer, **kwargs):
                raise RuntimeError("Always fails")

        evaluator = AllFailingEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        assert report.success_rate == 0.0
        assert report.best_method == ""
        assert report.best_result is None
        assert report.stage_valid is False
        assert "All methods failed" in report.invalid_reason
        assert "unchanged input-container copy" in report.recommendation_reason

    def test_run_all_marks_stage_invalid_when_all_quality_scores_are_zero(self, simple_container):
        """All-zero quality stages should not return a runtime-only best method."""

        class AllZeroQualityEvaluator(BaseEvaluator):
            @property
            def stage_name(self) -> str:
                return "all_zero_quality"

            @property
            def methods(self) -> dict[str, Callable]:
                return {"slow": self._slow, "fast": self._fast}

            @property
            def metric_weights(self) -> dict[str, float]:
                return {"metric": 1.0}

            def compute_metrics(self, container, original, layer_name):
                del container, original, layer_name
                return {"metric": 0.0}

            def _slow(self, container, assay_name, source_layer, **kwargs):
                time.sleep(0.01)
                assay = container.assays[assay_name]
                assay.add_layer(f"{source_layer}_slow", assay.layers[source_layer])
                return container

            def _fast(self, container, assay_name, source_layer, **kwargs):
                assay = container.assays[assay_name]
                assay.add_layer(f"{source_layer}_fast", assay.layers[source_layer])
                return container

        evaluator = AllZeroQualityEvaluator()
        result_container, report = evaluator.run_all(
            container=simple_container.copy(),
            assay_name="proteins",
            source_layer="raw",
            selection_strategy="balanced",
        )

        assert report.success_rate == 1.0
        assert report.best_method == ""
        assert report.best_result is None
        assert report.stage_valid is False
        assert "zero quality scores" in report.invalid_reason
        assert "raw" in result_container.assays["proteins"].layers
        assert "raw_fast" not in result_container.assays["proteins"].layers
        assert "raw_slow" not in result_container.assays["proteins"].layers

    def test_run_all_success_rate(self, simple_container):
        """Test that success_rate is correctly calculated."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        # 2 success (method_a, method_b) out of 3 total
        assert report.success_rate == pytest.approx(2 / 3, rel=1e-3)

    def test_run_all_recommendation_reason(self, simple_container):
        """Test that recommendation_reason is generated."""
        evaluator = MockEvaluator()

        container_copy = simple_container.copy()
        result_container, report = evaluator.run_all(
            container=container_copy,
            assay_name="proteins",
            source_layer="raw",
        )

        assert len(report.recommendation_reason) > 0

    def test_run_all_with_repeats_adds_confidence_metadata(self, simple_container):
        """Test repeat evaluation metadata is attached to results."""
        evaluator = MockEvaluator()

        result_container, report = evaluator.run_all(
            container=simple_container.copy(),
            assay_name="proteins",
            source_layer="raw",
            n_repeats=3,
            confidence_level=0.9,
        )

        assert isinstance(result_container, ScpContainer)
        assert report.n_repeats == 3
        assert report.confidence_level == pytest.approx(0.9)
        assert report.selection_strategy == "balanced"
        assert report.best_result is not None
        assert report.best_result.n_repeats == 3
        assert len(report.best_result.repeat_overall_scores) == 3
        assert report.best_result.overall_score_std is not None
        assert report.best_result.overall_score_ci_lower is not None
        assert report.best_result.overall_score_ci_upper is not None

    def test_run_all_speed_strategy_prefers_faster_method(self, simple_container):
        """Test speed strategy can prioritize runtime when quality ties."""

        class SpeedAwareEvaluator(BaseEvaluator):
            @property
            def stage_name(self) -> str:
                return "speed_test"

            @property
            def methods(self) -> dict[str, Callable]:
                return {"slow": self._slow, "fast": self._fast}

            @property
            def metric_weights(self) -> dict[str, float]:
                return {"metric": 1.0}

            def compute_metrics(self, container, original_container, layer_name):
                # Tie quality on purpose to isolate strategy behavior.
                return {"metric": 0.8}

            def _slow(self, container, assay_name, source_layer, **kwargs):
                time.sleep(0.01)
                assay = container.assays[assay_name]
                assay.add_layer(f"{source_layer}_slow", assay.layers[source_layer])
                return container

            def _fast(self, container, assay_name, source_layer, **kwargs):
                assay = container.assays[assay_name]
                assay.add_layer(f"{source_layer}_fast", assay.layers[source_layer])
                return container

        evaluator = SpeedAwareEvaluator()
        result_container, report = evaluator.run_all(
            container=simple_container.copy(),
            assay_name="proteins",
            source_layer="raw",
            selection_strategy="speed",
        )

        assert isinstance(result_container, ScpContainer)
        assert report.selection_strategy == "speed"
        assert report.best_method == "fast"

    def test_run_all_invalid_controls_raise(self, simple_container):
        """Test invalid M2 control parameters raise actionable errors."""
        evaluator = MockEvaluator()

        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            evaluator.run_all(
                container=simple_container.copy(),
                assay_name="proteins",
                source_layer="raw",
                n_repeats=0,
            )

        with pytest.raises(ValueError, match="confidence_level must be in"):
            evaluator.run_all(
                container=simple_container.copy(),
                assay_name="proteins",
                source_layer="raw",
                confidence_level=1.0,
            )

        with pytest.raises(ValueError, match="selection_strategy must be one of"):
            evaluator.run_all(
                container=simple_container.copy(),
                assay_name="proteins",
                source_layer="raw",
                selection_strategy="unknown",
            )

    def test_apply_selection_scores_uses_strategy_presets(self):
        """Test strategy presets are applied for selection-score weighting."""
        evaluator = MockEvaluator()
        quality_heavy = EvaluationResult(
            method_name="quality_heavy",
            scores={"metric1": 0.9},
            overall_score=0.9,
            execution_time=2.0,
            layer_name="raw_quality_heavy",
        )
        speed_heavy = EvaluationResult(
            method_name="speed_heavy",
            scores={"metric1": 0.8},
            overall_score=0.8,
            execution_time=1.0,
            layer_name="raw_speed_heavy",
        )

        quality_results = [quality_heavy, speed_heavy]
        evaluator._apply_selection_scores(quality_results, "quality")
        assert quality_heavy.selection_score == pytest.approx(0.9)
        assert speed_heavy.selection_score == pytest.approx(0.8)

        quality_heavy.selection_score = None
        speed_heavy.selection_score = None
        speed_results = [quality_heavy, speed_heavy]
        evaluator._apply_selection_scores(speed_results, "speed")
        assert quality_heavy.selection_score == pytest.approx(0.585)
        assert speed_heavy.selection_score == pytest.approx(0.87)


class TestCreateWrapperContract:
    """Test strict runtime/fixed-parameter contract in create_wrapper."""

    def test_create_wrapper_rejects_unsupported_runtime_kwargs(self, simple_container):
        """Unknown runtime kwargs must fail explicitly instead of being ignored."""

        def method(
            container: ScpContainer,
            assay_name: str,
            source_layer: str,
            new_layer_name: str,
            alpha: float = 1.0,
        ) -> ScpContainer:
            del assay_name, source_layer, new_layer_name, alpha
            return container

        wrapper = create_wrapper(method)

        with pytest.raises(TypeError, match="unsupported runtime kwargs.*beta"):
            wrapper(simple_container.copy(), "proteins", "raw", beta=2.0)

    def test_create_wrapper_rejects_unsupported_extra_params(self):
        """Wrapper creation should fail when fixed params are not declared."""

        def method(
            container: ScpContainer,
            assay_name: str,
            source_layer: str,
            new_layer_name: str,
        ) -> ScpContainer:
            del assay_name, source_layer, new_layer_name
            return container

        with pytest.raises(TypeError, match="unsupported fixed params.*gamma"):
            create_wrapper(method, gamma=1.0)

    def test_create_wrapper_requires_explicit_core_params(self):
        """Wrapped methods must explicitly declare the core adapter params."""

        def missing_source_layer(
            container: ScpContainer,
            assay_name: str,
            new_layer_name: str,
        ) -> ScpContainer:
            del assay_name, new_layer_name
            return container

        with pytest.raises(
            TypeError,
            match="requires `missing_source_layer` to declare parameters",
        ):
            create_wrapper(missing_source_layer)

    def test_create_wrapper_stays_strict_even_if_method_has_var_kwargs(self, simple_container):
        """`**kwargs` in target method does not relax wrapper runtime contract."""

        def method(
            container: ScpContainer,
            assay_name: str,
            source_layer: str,
            new_layer_name: str,
            alpha: float = 1.0,
            **extra,
        ) -> ScpContainer:
            del assay_name, source_layer, new_layer_name, alpha, extra
            return container

        wrapper = create_wrapper(method)
        wrapper(simple_container.copy(), "proteins", "raw", alpha=3.0)

        with pytest.raises(TypeError, match="unsupported runtime kwargs.*beta"):
            wrapper(simple_container.copy(), "proteins", "raw", beta=2.0)


class TestEvaluatorProperties:
    """Test evaluator property access."""

    def test_stage_name_property(self):
        """Test stage_name property access."""
        evaluator = MockEvaluator()
        assert evaluator.stage_name == "test_stage"

    def test_methods_property(self):
        """Test methods property returns dict."""
        evaluator = MockEvaluator()
        methods = evaluator.methods

        assert isinstance(methods, dict)
        assert "method_a" in methods
        assert "method_b" in methods
        assert callable(methods["method_a"])

    def test_metric_weights_property(self):
        """Test metric_weights property returns dict."""
        evaluator = MockEvaluator()
        weights = evaluator.metric_weights

        assert isinstance(weights, dict)
        assert weights == {"metric1": 0.5, "metric2": 0.3, "metric3": 0.2}
