"""Normalization evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
normalization methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


class NormalizationEvaluator(BaseEvaluator):
    """Evaluator for normalization methods.

    This evaluator tests various normalization methods and evaluates their
    performance using metrics such as variance stabilization, batch effect
    reduction, and data completeness.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("normalization")
    methods : dict[str, Callable]
        Dictionary of normalization methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = NormalizationEvaluator()
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="raw"
    ... )
    """

    @property
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name ("normalization")
        """
        return "normalization"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available normalization methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Only includes true normalization methods (not log_transform, which is
            a preprocessing step).
        """
        from scptensor.autoselect.evaluators.base import create_wrapper
        from scptensor.normalization import norm_mean, norm_median, norm_quantile

        return {
            "norm_mean": create_wrapper(norm_mean, layer_namer="auto"),
            "norm_median": create_wrapper(norm_median, layer_namer="auto"),
            "norm_quantile": create_wrapper(norm_quantile, layer_namer="auto"),
        }

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights
        """
        return {
            "variance_stabilization": 0.4,
            "batch_effect": 0.3,
            "completeness": 0.3,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for a normalized layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the normalized data layer
        original_container : ScpContainer
            Original container before normalization (for comparison)
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
            "variance_stabilization": 0.85,
            "batch_effect": 0.80,
            "completeness": 0.90,
        }
