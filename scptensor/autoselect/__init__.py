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
AutoSelector
    Unified automatic method selector for multi-stage pipelines

Shortcut Functions
------------------
auto_normalize
    Auto-select optimal normalization method
auto_impute
    Auto-select optimal imputation method
auto_integrate
    Auto-select optimal batch correction method
auto_reduce
    Auto-select optimal dimensionality reduction method
auto_cluster
    Auto-select optimal clustering method

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

Using shortcut functions:

>>> from scptensor.autoselect import auto_normalize
>>> container, report = auto_normalize(container)
>>> print(f"Best method: {report.best_method}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scptensor.autoselect.core import (
    AutoSelector,
    AutoSelectReport,
    EvaluationResult,
    StageReport,
)

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def auto_normalize(
    container: ScpContainer,
    assay_name: str = "proteins",
    source_layer: str | None = None,
    keep_all: bool = False,
    **kwargs,
) -> tuple[ScpContainer, StageReport]:
    """Auto-select optimal normalization method.

    Evaluates multiple normalization methods and returns the best performing one.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, optional
        Name of assay to process, by default "proteins"
    source_layer : str | None, optional
        Source layer name. If None, uses "raw".
    keep_all : bool, optional
        If True, keep all method results; if False, keep only best, by default False
    **kwargs
        Additional parameters passed to the evaluator

    Returns
    -------
    tuple[ScpContainer, StageReport]
        Tuple of (result_container, stage_report)

    Examples
    --------
    >>> container, report = auto_normalize(container)
    >>> print(f"Best method: {report.best_method}")

    Notes
    -----
    This is a convenience function that creates a single-stage AutoSelector.
    For multi-stage pipelines, use AutoSelector directly.
    """
    selector = AutoSelector(stages=["normalize"], keep_all=keep_all)
    return selector.run_stage(
        container, stage="normalize", assay_name=assay_name, source_layer=source_layer, **kwargs
    )


def auto_impute(
    container: ScpContainer,
    assay_name: str = "proteins",
    source_layer: str | None = None,
    keep_all: bool = False,
    **kwargs,
) -> tuple[ScpContainer, StageReport]:
    """Auto-select optimal imputation method.

    Evaluates multiple imputation methods and returns the best performing one.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, optional
        Name of assay to process, by default "proteins"
    source_layer : str | None, optional
        Source layer name. If None, uses "raw".
    keep_all : bool, optional
        If True, keep all method results; if False, keep only best, by default False
    **kwargs
        Additional parameters passed to the evaluator

    Returns
    -------
    tuple[ScpContainer, StageReport]
        Tuple of (result_container, stage_report)

    Examples
    --------
    >>> container, report = auto_impute(container)
    >>> print(f"Best method: {report.best_method}")

    Notes
    -----
    This is a convenience function that creates a single-stage AutoSelector.
    For multi-stage pipelines, use AutoSelector directly.
    """
    selector = AutoSelector(stages=["impute"], keep_all=keep_all)
    return selector.run_stage(
        container, stage="impute", assay_name=assay_name, source_layer=source_layer, **kwargs
    )


def auto_integrate(
    container: ScpContainer,
    assay_name: str = "proteins",
    source_layer: str | None = None,
    keep_all: bool = False,
    **kwargs,
) -> tuple[ScpContainer, StageReport]:
    """Auto-select optimal batch correction method.

    Evaluates multiple batch correction methods and returns the best performing one.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, optional
        Name of assay to process, by default "proteins"
    source_layer : str | None, optional
        Source layer name. If None, uses "raw".
    keep_all : bool, optional
        If True, keep all method results; if False, keep only best, by default False
    **kwargs
        Additional parameters passed to the evaluator

    Returns
    -------
    tuple[ScpContainer, StageReport]
        Tuple of (result_container, stage_report)

    Examples
    --------
    >>> container, report = auto_integrate(container)
    >>> print(f"Best method: {report.best_method}")

    Notes
    -----
    This is a convenience function that creates a single-stage AutoSelector.
    For multi-stage pipelines, use AutoSelector directly.
    """
    selector = AutoSelector(stages=["integrate"], keep_all=keep_all)
    return selector.run_stage(
        container, stage="integrate", assay_name=assay_name, source_layer=source_layer, **kwargs
    )


def auto_reduce(
    container: ScpContainer,
    assay_name: str = "proteins",
    source_layer: str | None = None,
    keep_all: bool = False,
    **kwargs,
) -> tuple[ScpContainer, StageReport]:
    """Auto-select optimal dimensionality reduction method.

    Evaluates multiple dimensionality reduction methods and returns the best performing one.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, optional
        Name of assay to process, by default "proteins"
    source_layer : str | None, optional
        Source layer name. If None, uses "raw".
    keep_all : bool, optional
        If True, keep all method results; if False, keep only best, by default False
    **kwargs
        Additional parameters passed to the evaluator

    Returns
    -------
    tuple[ScpContainer, StageReport]
        Tuple of (result_container, stage_report)

    Examples
    --------
    >>> container, report = auto_reduce(container)
    >>> print(f"Best method: {report.best_method}")

    Notes
    -----
    This is a convenience function that creates a single-stage AutoSelector.
    For multi-stage pipelines, use AutoSelector directly.
    """
    selector = AutoSelector(stages=["reduce"], keep_all=keep_all)
    return selector.run_stage(
        container, stage="reduce", assay_name=assay_name, source_layer=source_layer, **kwargs
    )


def auto_cluster(
    container: ScpContainer,
    assay_name: str = "proteins",
    source_layer: str | None = None,
    keep_all: bool = False,
    **kwargs,
) -> tuple[ScpContainer, StageReport]:
    """Auto-select optimal clustering method.

    Evaluates multiple clustering methods and returns the best performing one.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, optional
        Name of assay to process, by default "proteins"
    source_layer : str | None, optional
        Source layer name. If None, uses "raw".
    keep_all : bool, optional
        If True, keep all method results; if False, keep only best, by default False
    **kwargs
        Additional parameters passed to the evaluator

    Returns
    -------
    tuple[ScpContainer, StageReport]
        Tuple of (result_container, stage_report)

    Examples
    --------
    >>> container, report = auto_cluster(container)
    >>> print(f"Best method: {report.best_method}")

    Notes
    -----
    This is a convenience function that creates a single-stage AutoSelector.
    For multi-stage pipelines, use AutoSelector directly.
    """
    selector = AutoSelector(stages=["cluster"], keep_all=keep_all)
    return selector.run_stage(
        container, stage="cluster", assay_name=assay_name, source_layer=source_layer, **kwargs
    )


__all__ = [
    "EvaluationResult",
    "StageReport",
    "AutoSelectReport",
    "AutoSelector",
    "auto_normalize",
    "auto_impute",
    "auto_integrate",
    "auto_reduce",
    "auto_cluster",
]
