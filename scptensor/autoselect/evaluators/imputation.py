"""Imputation evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
imputation methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


class ImputationEvaluator(BaseEvaluator):
    """Evaluator for imputation methods.

    This evaluator tests various imputation methods and evaluates their
    performance using metrics such as RMSE, correlation preservation, and
    imputation quality.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("imputation")
    methods : dict[str, Callable]
        Dictionary of imputation methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = ImputationEvaluator()
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="log"
    ... )
    """

    def __init__(self) -> None:
        """Initialize the imputation evaluator."""
        self._available_methods: dict[str, Callable] | None = None

    def _get_available_methods(self) -> dict[str, Callable]:
        """Get available imputation methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary of available methods
        """
        from scptensor.autoselect.evaluators.base import create_wrapper

        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # Core imputation methods (always available)
        # Use "clean" layer_namer to remove "impute_" prefix from function names
        # This ensures layer names match what evaluate_method expects:
        # {source_layer}_{method_name} instead of {source_layer}_{func.__name__}
        try:
            from scptensor.impute import impute_knn

            methods["knn"] = create_wrapper(impute_knn, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_bpca

            methods["bpca"] = create_wrapper(impute_bpca, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_mf

            methods["mf"] = create_wrapper(impute_mf, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_minprob

            methods["minprob"] = create_wrapper(impute_minprob, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_qrilc

            methods["qrilc"] = create_wrapper(impute_qrilc, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_lls

            methods["lls"] = create_wrapper(impute_lls, layer_namer="clean")
        except ImportError:
            pass

        self._available_methods = methods
        return methods

    @property
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name ("imputation")
        """
        return "imputation"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available imputation methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Only methods with installed dependencies are included.
        """
        return self._get_available_methods()

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights
        """
        return {
            "rmse": 0.4,
            "correlation": 0.3,
            "completeness": 0.3,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for an imputed layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the imputed data layer
        original_container : ScpContainer
            Original container before imputation (for comparison)
        layer_name : str
            Name of the layer to evaluate

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)
        """
        # For now, use simple placeholder metrics
        # In a real implementation, these would compute actual quality metrics

        # Check if layer exists
        if "proteins" not in container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        assay = container.assays["proteins"]
        if layer_name not in assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Placeholder: return moderate scores for all metrics
        # TODO: Implement actual metric computation
        return {
            "rmse": 0.85,
            "correlation": 0.82,
            "completeness": 0.95,
        }
