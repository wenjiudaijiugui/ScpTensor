"""Evaluators for automatic method selection.

This module provides evaluator classes for each analysis stage in the
automatic method selection system. Each evaluator implements the BaseEvaluator
interface and provides methods and metrics specific to its stage.

Classes
-------
BaseEvaluator
    Abstract base class defining the evaluator interface
NormalizationEvaluator
    Evaluator for normalization methods
ImputationEvaluator
    Evaluator for imputation methods
IntegrationEvaluator
    Evaluator for batch correction/integration methods
DimReductionEvaluator
    Evaluator for dimensionality reduction methods
ClusteringEvaluator
    Evaluator for clustering methods

Examples
--------
>>> from scptensor.autoselect.evaluators import BaseEvaluator
>>> class MyEvaluator(BaseEvaluator):
...     # Implement abstract methods
...     pass
"""

from __future__ import annotations

from scptensor.autoselect.evaluators.base import BaseEvaluator
from scptensor.autoselect.evaluators.clustering import ClusteringEvaluator
from scptensor.autoselect.evaluators.dim_reduction import DimReductionEvaluator
from scptensor.autoselect.evaluators.imputation import ImputationEvaluator
from scptensor.autoselect.evaluators.integration import IntegrationEvaluator
from scptensor.autoselect.evaluators.normalization import NormalizationEvaluator

__all__ = [
    "BaseEvaluator",
    "ClusteringEvaluator",
    "DimReductionEvaluator",
    "ImputationEvaluator",
    "IntegrationEvaluator",
    "NormalizationEvaluator",
]
