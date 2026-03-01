"""Automatic method selection system for ScpTensor.

This module provides intelligent automatic selection of optimal analysis methods
for single-cell proteomics data. It evaluates multiple methods across different
analysis stages and recommends the best performing ones.

Main Classes
------------
EvaluationResult
    Single method evaluation result with scores and metrics
StageReport
    Report for a single analysis stage with multiple method results
AutoSelectReport
    Complete automatic selection report across all stages

Example
-------
>>> from scptensor.autoselect import AutoSelectReport, StageReport, EvaluationResult
>>> result = EvaluationResult(
...     method_name="log_normalize",
...     scores={"variance": 0.9, "batch_effect": 0.85},
...     overall_score=0.875,
...     execution_time=1.2,
...     layer_name="log"
... )
>>> stage = StageReport(
...     stage_name="normalization",
...     results=[result],
...     best_method="log_normalize",
...     best_result=result
... )
>>> report = AutoSelectReport(stages={"normalization": stage})
>>> print(report.summary())
"""

from __future__ import annotations

from scptensor.autoselect.core import AutoSelectReport, EvaluationResult, StageReport

__all__ = [
    "EvaluationResult",
    "StageReport",
    "AutoSelectReport",
]
