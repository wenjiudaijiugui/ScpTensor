"""Evaluators for automatic method selection.

This module provides evaluator classes for each analysis stage in the
automatic method selection system. Each evaluator implements the BaseEvaluator
interface and provides methods and metrics specific to its stage.

Classes
-------
BaseEvaluator
    Abstract base class defining the evaluator interface

Examples
--------
>>> from scptensor.autoselect.evaluators import BaseEvaluator
>>> class MyEvaluator(BaseEvaluator):
...     # Implement abstract methods
...     pass
"""

from __future__ import annotations

from scptensor.autoselect.evaluators.base import BaseEvaluator

__all__ = [
    "BaseEvaluator",
]
