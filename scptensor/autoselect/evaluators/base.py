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

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from scptensor.autoselect.core import EvaluationResult, StageReport

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer, ScpMatrix


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

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name (e.g., "normalization", "imputation", "batch_correction")
        """
        pass

    @property
    @abstractmethod
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Each function should have signature:
            func(container: ScpContainer, assay_name: str, source_layer: str, **kwargs) -> ScpContainer
        """
        pass

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
        pass

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
        pass

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
        weights = self.metric_weights
        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(scores.get(k, 0.0) * v for k, v in weights.items())
        return weighted_sum / total_weight

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
        # Determine output layer name
        new_layer_name = f"{source_layer}_{method_name}"

        # Track execution time
        start_time = time.perf_counter()
        result_container: ScpContainer | None = None
        error_msg: str | None = None

        try:
            # Execute method on a copy of container
            result_container = method_func(
                container=container.copy(),
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result_container = None

        execution_time = time.perf_counter() - start_time

        # Compute metrics if method succeeded
        if result_container is not None and error_msg is None:
            try:
                scores = self.compute_metrics(
                    container=result_container,
                    original_container=container,
                    layer_name=new_layer_name,
                )
            except Exception as e:
                error_msg = f"Metric computation failed: {type(e).__name__}: {str(e)}"
                scores = dict.fromkeys(self.metric_weights, 0.0)
        else:
            scores = dict.fromkeys(self.metric_weights, 0.0)

        # Compute overall score
        overall_score = 0.0 if error_msg is not None else self.compute_overall_score(scores)

        # Create evaluation result
        eval_result = EvaluationResult(
            method_name=method_name,
            scores=scores,
            overall_score=overall_score,
            execution_time=execution_time,
            layer_name=new_layer_name,
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
        # Initialize report
        report = StageReport(stage_name=self.stage_name)
        results: list[EvaluationResult] = []

        # Store successful results for later: method_name -> (layer_name, layer_matrix)
        successful_layers: dict[str, tuple[str, ScpMatrix]] = {}

        # Evaluate each method
        for method_name, method_func in self.methods.items():
            result_container, eval_result = self.evaluate_method(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )

            results.append(eval_result)

            # Store successful result layer
            if result_container is not None and eval_result.error is None:
                layer_name = eval_result.layer_name
                layer_matrix = result_container.assays[assay_name].layers[layer_name]
                successful_layers[method_name] = (layer_name, layer_matrix)

        # Update report with all results
        report.results = results

        # Find best method (highest score among successful methods)
        successful_results = [r for r in results if r.error is None]

        if successful_results:
            best_result = max(successful_results, key=lambda r: r.overall_score)
            report.best_method = best_result.method_name
            report.best_result = best_result
            report.recommendation_reason = (
                f"Highest overall score ({best_result.overall_score:.4f}) "
                f"among {len(successful_results)} successful methods"
            )

            # Create result container with appropriate layers
            result_container = container.copy()
            assay = result_container.assays[assay_name]

            if keep_all:
                # Add all successful result layers to the container
                for method_name, (layer_name, layer_matrix) in successful_layers.items():
                    if method_name != best_result.method_name:
                        assay.add_layer(layer_name, layer_matrix)
            # else: only the best layer is added (which we do below)

            # Add the best layer (this may overwrite if keep_all=True and best already added)
            best_layer_name = best_result.layer_name
            best_layer_matrix = successful_layers[best_result.method_name][1]
            assay.add_layer(best_layer_name, best_layer_matrix)

            return result_container, report
        else:
            # All methods failed - return original container
            report.best_method = ""
            report.best_result = None
            report.recommendation_reason = "All methods failed"

            return container.copy(), report
