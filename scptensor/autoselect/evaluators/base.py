"""Base evaluator abstract class for automatic method selection.

This module provides the abstract base class for all evaluators in the
automatic method selection system. Each analysis stage (normalization,
imputation, batch correction, etc.) should have its own evaluator
implementation.

Classes
-------
BaseEvaluator
    Abstract base class defining the evaluator interface.
"""

from __future__ import annotations

import inspect
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from scptensor.autoselect.core import EvaluationResult, StageReport
from scptensor.autoselect.strategy import get_strategy_preset
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.state_metrics import compute_state_transition_metrics

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def create_wrapper(
    func: Callable,
    source_layer_param: str = "source_layer",
    layer_namer: str | Callable[[str, str], str] = "auto",
    **extra_params,
) -> Callable:
    """Create a wrapper function for method functions.

    This factory function creates a wrapper that adapts method functions
    to the standard evaluator signature: (container, assay_name, source_layer, **kwargs).

    Parameters
    ----------
    func : Callable
        The method function to wrap
    source_layer_param : str, optional
        Parameter name in func for the source layer, by default "source_layer"
    layer_namer : str | Callable[[str, str], str], optional
        Strategy for naming the new layer:
        - "auto": Use f"{source_layer}_{func.__name__}" (default)
        - "clean": Use f"{source_layer}_{cleaned_name}" (removes common prefixes)
        - callable: Custom function (source_layer, func_name) -> layer_name
    **extra_params
        Additional parameters to pass to func (e.g., batch_key, n_components)

    Returns
    -------
    Callable
        Wrapped function with signature:
        (container, assay_name, source_layer, **kwargs) -> ScpContainer

    Examples
    --------
    >>> from scptensor.transformation import log_transform
    >>> wrapper = create_wrapper(log_transform)
    >>> result = wrapper(container, "proteins", "raw")

    >>> from scptensor.integration import integrate_combat
    >>> wrapper = create_wrapper(
    ...     integrate_combat,
    ...     source_layer_param="base_layer",
    ...     layer_namer="clean",
    ...     batch_key="batch"
    ... )

    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    accepted_params = set(parameters.keys())
    required_params = ("container", "assay_name", source_layer_param)
    missing_required = [name for name in required_params if name not in accepted_params]
    if missing_required:
        raise TypeError(
            f"create_wrapper() requires `{func.__name__}` to declare parameters "
            f"{list(required_params)}; missing {missing_required}.",
        )

    unsupported_extra = sorted(set(extra_params) - accepted_params)
    if unsupported_extra:
        raise TypeError(
            f"create_wrapper() for `{func.__name__}` received unsupported fixed params "
            f"{unsupported_extra}. Declare these parameters explicitly in the method "
            "signature before wiring through create_wrapper.",
        )

    fixed_param_names = {"container", "assay_name", source_layer_param, "new_layer_name"}
    allowed_runtime_kwargs = sorted(
        name
        for name, param in parameters.items()
        if name not in fixed_param_names
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    allowed_runtime_set = set(allowed_runtime_kwargs)

    # Determine layer naming strategy
    get_layer_name: Callable[[str, str], str]
    if layer_namer == "auto":

        def get_layer_name(src: str, fname: str) -> str:
            return f"{src}_{fname}"
    elif layer_namer == "clean":
        # Remove common prefixes like "integrate_", "reduce_", "cluster_", "impute_"

        def get_layer_name(src: str, fname: str) -> str:
            clean_name = re.sub(r"^(integrate_|reduce_|cluster_|impute_|norm_)", "", fname)
            return f"{src}_{clean_name}"
    elif callable(layer_namer):
        get_layer_name = layer_namer
    else:
        raise ValueError(f"layer_namer must be 'auto', 'clean', or callable, got {layer_namer}")

    def wrapper(
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        **kwargs,
    ) -> ScpContainer:
        """Wrapper for method functions."""
        new_layer_name = get_layer_name(source_layer, func.__name__)
        unknown_runtime_kwargs = sorted(set(kwargs) - allowed_runtime_set)
        if unknown_runtime_kwargs:
            allowed_display = allowed_runtime_kwargs if allowed_runtime_kwargs else ["<none>"]
            raise TypeError(
                f"{func.__name__} wrapper received unsupported runtime kwargs "
                f"{unknown_runtime_kwargs}. Allowed runtime kwargs: {allowed_display}.",
            )

        call_params = {
            "container": container,
            "assay_name": assay_name,
            source_layer_param: source_layer,
        }
        if "new_layer_name" in accepted_params:
            call_params["new_layer_name"] = new_layer_name
        call_params.update(extra_params)
        call_params.update(kwargs)
        return func(**call_params)

    return wrapper


class BaseEvaluator(ABC):
    """Abstract base class for method evaluators.

    This class defines the interface for evaluators that assess and compare
    different analysis methods within a specific stage (e.g., normalization,
    imputation, batch correction).

    Subclasses must implement the following abstract properties and methods:
    - stage_name: Property returning the name of the analysis stage
    - methods: Property returning a dictionary of available methods
    - metric_weights: Property returning weights for each metric
    - compute_metrics: Method to compute evaluation metrics

    The base class provides default implementations for:
    - compute_overall_score: Weighted average of metric scores
    - evaluate_method: Evaluate a single method and return results
    - run_all: Run all methods and select the best one

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage (e.g., "normalization", "imputation")
    methods : dict[str, Callable]
        Dictionary mapping method names to their implementation functions
    metric_weights : dict[str, float]
        Dictionary mapping metric names to their weights in overall score

    Examples
    --------
    >>> class NormalizationEvaluator(BaseEvaluator):
    ...     @property
    ...     def stage_name(self) -> str:
    ...         return "normalization"
    ...
    ...     @property
    ...     def methods(self) -> dict[str, Callable]:
    ...         return {"log": log_normalize, "median": median_normalize}
    ...
    ...     @property
    ...     def metric_weights(self) -> dict[str, float]:
    ...         return {"variance": 0.4, "batch_effect": 0.3, "completeness": 0.3}
    ...
    ...     def compute_metrics(self, container, original, layer_name):
    ...         # Compute and return metrics
    ...         return {"variance": 0.9, "batch_effect": 0.85, "completeness": 0.95}

    """

    _metric_assay_name: str | None = None
    _metric_source_layer: str | None = None

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name (e.g., "normalization", "imputation", "batch_correction")

        """

    @property
    @abstractmethod
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Each function should have signature:
            func(
                container: ScpContainer,
                assay_name: str,
                source_layer: str,
                **kwargs,
            ) -> ScpContainer

        """

    @property
    @abstractmethod
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Weights should sum to 1.0 for proper weighted averaging, but any
        positive values are accepted and normalized internally.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights

        """

    @abstractmethod
    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for a processed layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the processed data layer
        original_container : ScpContainer
            Original container before processing (for comparison)
        layer_name : str
            Name of the layer to evaluate

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)

        """

    def compute_report_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Compute additional reporting metrics not used for ranking.

        Subclasses can override this to surface diagnostics separately from
        ``scores``. The default implementation returns an empty mapping.
        """
        del container, original_container, layer_name, scores
        return {}

    def _compute_state_report_metrics(
        self,
        container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Return shared state-aware diagnostics for a derived layer."""
        resolved_assay_name = self._resolve_metric_assay_name(container, None)
        if resolved_assay_name is None:
            return {}

        assay = container.assays.get(resolved_assay_name)
        if assay is None or layer_name not in assay.layers:
            return {}

        metrics = compute_state_transition_metrics(
            container,
            assay_name=resolved_assay_name,
            layer_name=layer_name,
            reference="source",
        )
        return {f"state_{key}": float(value) for key, value in metrics.items()}

    def get_metric_weights(self) -> dict[str, float]:
        """Return metric weights, applying overrides if present."""
        override = getattr(self, "_metric_weights_override", None)
        return override if override is not None else self.metric_weights

    def set_metric_weights(self, weights: dict[str, float] | None) -> None:
        """Override metric weights for scoring."""
        if not weights:
            self._metric_weights_override = None
            return

        default_weights = self.metric_weights
        unknown = [key for key in weights if key not in default_weights]
        if unknown:
            raise ValueError(f"Unknown metric keys: {unknown}")

        merged = default_weights.copy()
        merged.update(weights)
        invalid = [key for key, value in merged.items() if value < 0 or not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Invalid metric weights for keys: {invalid}")

        self._metric_weights_override = merged

    def compute_overall_score(self, scores: dict[str, float]) -> float:
        """Compute weighted overall score from individual metric scores.

        Parameters
        ----------
        scores : dict[str, float]
            Dictionary mapping metric names to their scores

        Returns
        -------
        float
            Weighted average score, or 0.0 if total weight is zero

        Examples
        --------
        >>> evaluator = MyEvaluator()
        >>> scores = {"metric1": 0.9, "metric2": 0.8}
        >>> overall = evaluator.compute_overall_score(scores)

        """
        weights = self.get_metric_weights()
        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(scores.get(k, 0.0) * v for k, v in weights.items())
        return weighted_sum / total_weight

    def _extract_eval_controls(
        self,
        kwargs: dict[str, Any],
    ) -> tuple[int, float, str, dict[str, Any]]:
        """Extract and validate evaluator control arguments."""
        method_kwargs = dict(kwargs)
        n_repeats = int(method_kwargs.pop("n_repeats", 1))
        confidence_level = float(method_kwargs.pop("confidence_level", 0.95))
        strategy = get_strategy_preset(
            str(method_kwargs.pop("selection_strategy", "balanced")),
        ).name

        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        return n_repeats, confidence_level, strategy, method_kwargs

    def _with_repeat_random_state(
        self,
        method_kwargs: dict[str, Any],
        repeat_idx: int,
    ) -> dict[str, Any]:
        """Derive repeat-specific kwargs, including random_state jitter."""
        repeat_kwargs = dict(method_kwargs)
        base_rs = repeat_kwargs.get("random_state")
        if isinstance(base_rs, int):
            repeat_kwargs["random_state"] = base_rs + repeat_idx
        return repeat_kwargs

    def _default_metric_assay_name(self) -> str | None:
        """Return an evaluator-specific default assay name, if any."""
        return None

    def _resolve_metric_assay_name(
        self,
        container: ScpContainer,
        assay_name: str | None,
    ) -> str | None:
        """Resolve assay aliases for evaluator bookkeeping."""
        preferred = assay_name
        if preferred is None:
            cached = getattr(self, "_metric_assay_name", None)
            preferred = cached if isinstance(cached, str) and cached else None
        if preferred is None:
            preferred = self._default_metric_assay_name()
        if preferred is None:
            return None
        return resolve_assay_name(container, preferred)

    def _get_metric_assay(self, container: ScpContainer, assay_name: str | None = None):
        """Return resolved assay or None when unavailable."""
        resolved = self._resolve_metric_assay_name(container, assay_name)
        if resolved is None:
            return None
        return container.assays.get(resolved)

    def _get_result_name(self, method_name: str, source_layer: str) -> str:
        """Return the result identifier stored in EvaluationResult.layer_name."""
        return f"{source_layer}_{method_name}"

    def _result_output_kind(self) -> Literal["layer", "assay", "obs"]:
        """Return the output storage kind for this evaluator."""
        return "layer"

    def _prepare_evaluation_context(
        self,
        assay_name: str,
        source_layer: str,
    ) -> None:
        """Set any evaluator-local context needed by ``compute_metrics``."""
        self._metric_assay_name = assay_name
        del source_layer

    def _clear_evaluation_context(self) -> None:
        """Clear any evaluator-local context after one method evaluation."""
        self._metric_assay_name = None

    def _collect_success_artifact(
        self,
        result_container: ScpContainer,
        assay_name: str,
        result_name: str,
    ):
        """Extract the successful artifact from a result container."""
        resolved_assay_name = self._resolve_metric_assay_name(result_container, assay_name)
        if resolved_assay_name is None:
            raise ValueError(
                "Metric assay name is undefined when collecting evaluator output. "
                "This indicates evaluator context was not prepared correctly.",
            )
        return result_container.assays[resolved_assay_name].layers[result_name]

    def _attach_success_artifact(
        self,
        result_container: ScpContainer,
        assay_name: str,
        result_name: str,
        artifact,
    ) -> None:
        """Attach a stored artifact to the final result container."""
        resolved_assay_name = self._resolve_metric_assay_name(result_container, assay_name)
        if resolved_assay_name is None:
            raise ValueError(
                "Metric assay name is undefined when attaching evaluator output. "
                "This indicates evaluator context was not prepared correctly.",
            )
        result_container.assays[resolved_assay_name].add_layer(result_name, artifact)

    def _create_stage_report(
        self,
        *,
        strategy: str,
        n_repeats: int,
        confidence_level: float,
    ) -> StageReport:
        """Create a stage report with the current scoring configuration."""
        return StageReport(
            stage_name=self.stage_name,
            stage_key=self.stage_name,
            metric_weights=self.get_metric_weights(),
            selection_strategy=strategy,
            n_repeats=n_repeats,
            confidence_level=confidence_level,
        )

    def _evaluate_methods(
        self,
        *,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        n_repeats: int,
        confidence_level: float,
        method_kwargs: dict[str, Any],
    ) -> tuple[list[EvaluationResult], dict[str, tuple[str, Any]]]:
        """Evaluate all candidate methods and collect successful artifacts."""
        results: list[EvaluationResult] = []
        successful_artifacts: dict[str, tuple[str, Any]] = {}

        for method_name, method_func in self.methods.items():
            result_container, eval_result = self.evaluate_method_repeated(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                n_repeats=n_repeats,
                confidence_level=confidence_level,
                **method_kwargs,
            )
            results.append(eval_result)

            if result_container is not None and eval_result.error is None:
                result_name = eval_result.layer_name
                successful_artifacts[method_name] = (
                    result_name,
                    self._collect_success_artifact(result_container, assay_name, result_name),
                )

        return results, successful_artifacts

    def _build_result_container(
        self,
        *,
        container: ScpContainer,
        assay_name: str,
        keep_all: bool,
        successful_artifacts: dict[str, tuple[str, Any]],
        best_result: EvaluationResult,
    ) -> ScpContainer:
        """Build the final container from collected successful artifacts."""
        result_container = container.copy()

        if keep_all:
            for method_name, (result_name, artifact) in successful_artifacts.items():
                if method_name != best_result.method_name:
                    self._attach_success_artifact(
                        result_container,
                        assay_name,
                        result_name,
                        artifact,
                    )

        self._attach_success_artifact(
            result_container,
            assay_name,
            best_result.layer_name,
            successful_artifacts[best_result.method_name][1],
        )
        return result_container

    def _compute_confidence_interval(
        self,
        values: list[float],
        confidence_level: float,
    ) -> tuple[float, float]:
        """Compute empirical confidence interval using quantiles."""
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return values[0], values[0]

        alpha = 1.0 - confidence_level
        lower = float(np.quantile(values, alpha / 2.0))
        upper = float(np.quantile(values, 1.0 - alpha / 2.0))
        return lower, upper

    def evaluate_method_repeated(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        n_repeats: int = 1,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """Evaluate a method multiple times and aggregate the results."""
        repeat_results: list[EvaluationResult] = []
        best_container: ScpContainer | None = None
        best_success_score = -math.inf
        layer_name = self._get_result_name(method_name, source_layer)

        for idx in range(n_repeats):
            repeat_kwargs = self._with_repeat_random_state(kwargs, idx)
            result_container, result = self.evaluate_method(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                **repeat_kwargs,
            )
            repeat_results.append(result)
            layer_name = result.layer_name
            if (
                result.error is None
                and result_container is not None
                and result.overall_score > best_success_score
            ):
                best_success_score = result.overall_score
                best_container = result_container

        successful = [r for r in repeat_results if r.error is None]
        if not successful:
            merged_error = "; ".join(sorted({r.error or "Unknown error" for r in repeat_results}))
            failed_result = EvaluationResult(
                method_name=method_name,
                scores=dict.fromkeys(self.get_metric_weights(), 0.0),
                report_metrics={},
                overall_score=0.0,
                execution_time=float(np.mean([r.execution_time for r in repeat_results])),
                layer_name=layer_name,
                output_kind=self._result_output_kind(),
                error=merged_error,
                n_repeats=n_repeats,
                repeat_overall_scores=[],
                overall_score_std=0.0,
                overall_score_ci_lower=0.0,
                overall_score_ci_upper=0.0,
            )
            return None, failed_result

        metric_keys = set().union(*(r.scores.keys() for r in successful))
        mean_scores = {
            key: float(np.mean([r.scores.get(key, 0.0) for r in successful]))
            for key in sorted(metric_keys)
        }
        report_metric_keys = set().union(*(r.report_metrics.keys() for r in successful))
        mean_report_metrics = {
            key: float(np.mean([r.report_metrics.get(key, 0.0) for r in successful]))
            for key in sorted(report_metric_keys)
        }
        repeat_overall_scores = [float(r.overall_score) for r in successful]
        mean_overall = float(np.mean(repeat_overall_scores))
        overall_std = float(np.std(repeat_overall_scores, ddof=0))
        ci_low, ci_high = self._compute_confidence_interval(repeat_overall_scores, confidence_level)
        mean_exec = float(np.mean([r.execution_time for r in successful]))

        merged = EvaluationResult(
            method_name=method_name,
            scores=mean_scores,
            report_metrics=mean_report_metrics,
            overall_score=mean_overall,
            execution_time=mean_exec,
            layer_name=layer_name,
            output_kind=self._result_output_kind(),
            error=None,
            n_repeats=n_repeats,
            overall_score_std=overall_std,
            overall_score_ci_lower=ci_low,
            overall_score_ci_upper=ci_high,
            repeat_overall_scores=repeat_overall_scores,
        )
        return best_container, merged

    def _apply_selection_scores(
        self,
        results: list[EvaluationResult],
        strategy: str,
    ) -> None:
        """Apply strategy-aware selection score (quality + speed)."""
        successful = [r for r in results if r.error is None]
        if not successful:
            return

        preset = get_strategy_preset(strategy)
        quality_weight = preset.quality_weight
        runtime_weight = preset.runtime_weight

        times = np.array([r.execution_time for r in successful], dtype=float)
        min_time = float(np.min(times))
        max_time = float(np.max(times))
        if max_time - min_time < 1e-12:
            runtime_scores = np.ones_like(times)
        else:
            runtime_scores = 1.0 - (times - min_time) / (max_time - min_time)

        for result, runtime_score in zip(successful, runtime_scores, strict=False):
            selection_score = quality_weight * result.overall_score + runtime_weight * float(
                runtime_score,
            )
            result.selection_score = float(np.clip(selection_score, 0.0, 1.0))

        for result in results:
            if result.error is not None:
                result.selection_score = 0.0

    def _select_best_result(self, successful_results: list[EvaluationResult]) -> EvaluationResult:
        """Select best result with deterministic tie-breakers."""
        return max(
            successful_results,
            key=lambda r: (
                r.selection_score if r.selection_score is not None else r.overall_score,
                r.overall_score,
                -(r.execution_time),
                r.method_name,
            ),
        )

    def evaluate_method(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """Evaluate a single method and return results.

        This method:
        1. Executes the method on a copy of the container
        2. Computes evaluation metrics
        3. Returns the result container and evaluation result

        Parameters
        ----------
        container : ScpContainer
            Input container to process
        method_name : str
            Name of the method to evaluate
        method_func : Callable
            Method implementation function
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        source_layer : str, optional
            Name of source layer, by default "raw"
        **kwargs
            Additional parameters passed to the method

        Returns
        -------
        tuple[ScpContainer | None, EvaluationResult]
            Tuple of (result_container, evaluation_result).
            result_container is None if method failed.
            evaluation_result always contains method info and scores.

        Examples
        --------
        >>> evaluator = MyEvaluator()
        >>> container, result = evaluator.evaluate_method(
        ...     container=data,
        ...     method_name="log_normalize",
        ...     method_func=log_normalize,
        ...     assay_name="proteins",
        ...     source_layer="raw"
        ... )
        >>> if container is not None:
        ...     print(f"Score: {result.overall_score}")

        """
        new_layer_name = self._get_result_name(method_name, source_layer)
        start_time = time.perf_counter()
        result_container: ScpContainer | None = None
        error_msg: str | None = None
        scores: dict[str, float] = {}
        report_metrics: dict[str, float] = {}

        try:
            resolved_assay_name = self._resolve_metric_assay_name(container, assay_name)
            if resolved_assay_name is None:
                raise ValueError(
                    "Metric assay name is undefined for this evaluator. "
                    "Pass assay_name explicitly or set evaluator context before evaluation.",
                )
            self._metric_assay_name = resolved_assay_name
            self._prepare_evaluation_context(resolved_assay_name, source_layer)
            result_container = method_func(
                container=container.copy(),
                assay_name=resolved_assay_name,
                source_layer=source_layer,
                **kwargs,
            )
            scores = (
                self.compute_metrics(result_container, container, new_layer_name)
                if result_container
                else {}
            )
            report_metrics = (
                self.compute_report_metrics(
                    result_container,
                    container,
                    new_layer_name,
                    scores,
                )
                if result_container
                else {}
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            result_container = None
            scores = dict.fromkeys(self.get_metric_weights(), 0.0)
            report_metrics = {}
        finally:
            self._clear_evaluation_context()

        execution_time = time.perf_counter() - start_time
        overall_score = 0.0 if error_msg else self.compute_overall_score(scores)

        eval_result = EvaluationResult(
            method_name=method_name,
            scores=scores,
            report_metrics=report_metrics,
            overall_score=overall_score,
            execution_time=execution_time,
            layer_name=new_layer_name,
            output_kind=self._result_output_kind(),
            error=error_msg,
        )

        return result_container, eval_result

    def run_all(
        self,
        container: ScpContainer,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        keep_all: bool = False,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """Run all methods and select the best performing one.

        This method:
        1. Evaluates all registered methods
        2. Compares their performance
        3. Selects the best method
        4. Returns container with best result and detailed report

        Parameters
        ----------
        container : ScpContainer
            Input container to process
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        source_layer : str, optional
            Name of source layer, by default "raw"
        keep_all : bool, optional
            If True, keep all result layers; if False, keep only best, by default False
        **kwargs
            Additional parameters passed to all methods

        Returns
        -------
        tuple[ScpContainer, StageReport]
            Tuple of (result_container, stage_report).
            result_container contains the best method's output.
            stage_report contains detailed evaluation results for all methods.

        Examples
        --------
        >>> evaluator = MyEvaluator()
        >>> result_container, report = evaluator.run_all(
        ...     container=data,
        ...     assay_name="proteins",
        ...     source_layer="raw"
        ... )
        >>> print(f"Best method: {report.best_method}")
        >>> print(f"Best score: {report.best_result.overall_score}")

        """
        n_repeats, confidence_level, strategy, method_kwargs = self._extract_eval_controls(kwargs)
        report = self._create_stage_report(
            strategy=strategy,
            n_repeats=n_repeats,
            confidence_level=confidence_level,
        )
        report.results, successful_artifacts = self._evaluate_methods(
            container=container,
            assay_name=assay_name,
            source_layer=source_layer,
            n_repeats=n_repeats,
            confidence_level=confidence_level,
            method_kwargs=method_kwargs,
        )
        self._apply_selection_scores(report.results, strategy)

        # Find best method
        successful_results = [r for r in report.results if not r.error]

        if not successful_results:
            report.best_method = ""
            report.best_result = None
            report.recommendation_reason = (
                "All methods failed; returned an unchanged input-container copy."
            )
            return container.copy(), report

        # Select best and build result container
        best_result = self._select_best_result(successful_results)
        report.best_method = best_result.method_name
        report.best_result = best_result
        best_selection_score = (
            best_result.selection_score
            if best_result.selection_score is not None
            else best_result.overall_score
        )
        report.recommendation_reason = (
            f"Best '{strategy}' selection score "
            f"({best_selection_score:.4f}) "
            f"from {len(successful_results)} successful methods (n_repeats={n_repeats})."
        )
        return (
            self._build_result_container(
                container=container,
                assay_name=assay_name,
                keep_all=keep_all,
                successful_artifacts=successful_artifacts,
                best_result=best_result,
            ),
            report,
        )


__all__ = ["BaseEvaluator", "create_wrapper"]
