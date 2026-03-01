"""Core data classes for autoselect module.

This module defines the core data structures for the automatic method selection system:
- EvaluationResult: Single method evaluation result
- StageReport: Report for a single analysis stage
- AutoSelectReport: Complete automatic selection report
- AutoSelector: Unified automatic method selector
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scptensor.autoselect.evaluators.base import BaseEvaluator
    from scptensor.core.structures import ScpContainer


@dataclass
class EvaluationResult:
    """Single method evaluation result.

    Attributes
    ----------
    method_name : str
        Name of the evaluated method
    scores : dict[str, float]
        Individual metric scores
    overall_score : float
        Weighted composite score
    execution_time : float
        Execution time in seconds
    layer_name : str
        Name of the result layer
    error : str | None
        Error message if method failed, None if successful
    """

    method_name: str
    scores: dict[str, float]
    overall_score: float
    execution_time: float
    layer_name: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all evaluation result data
        """
        return {
            "method_name": self.method_name,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "layer_name": self.layer_name,
            "error": self.error,
        }


@dataclass
class StageReport:
    """Report for a single analysis stage.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage (e.g., "normalization", "imputation")
    results : list[EvaluationResult]
        List of evaluation results for all tested methods
    best_method : str
        Name of the best performing method
    best_result : EvaluationResult | None
        Evaluation result of the best method
    recommendation_reason : str
        Explanation for why this method was selected
    """

    stage_name: str
    results: list[EvaluationResult] = field(default_factory=list)
    best_method: str = ""
    best_result: EvaluationResult | None = None
    recommendation_reason: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate of tested methods.

        Returns
        -------
        float
            Proportion of methods that completed successfully (0.0 to 1.0)
        """
        if not self.results:
            return 0.0

        successful = sum(1 for r in self.results if r.error is None)
        return successful / len(self.results)


@dataclass
class AutoSelectReport:
    """Complete automatic method selection report.

    Attributes
    ----------
    stages : dict[str, StageReport]
        Dictionary mapping stage names to their reports
    total_time : float
        Total execution time in seconds
    warnings : list[str]
        List of warning messages generated during selection
    """

    stages: dict[str, StageReport] = field(default_factory=dict)
    total_time: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string of the report.

        Returns
        -------
        str
            Human-readable summary of the automatic selection process
        """
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("AutoSelect Report Summary")
        lines.append("=" * 60)

        # Check if empty
        if not self.stages:
            lines.append("\nNo stages completed.")
            return "\n".join(lines)

        # Stage summaries
        lines.append(f"\nTotal stages: {len(self.stages)}")
        lines.append(f"Total time: {self.total_time:.2f} seconds")
        lines.append("")

        for stage_name, stage_report in self.stages.items():
            lines.append(f"\n{stage_name.upper()}")
            lines.append("-" * 40)
            lines.append(f"  Methods tested: {len(stage_report.results)}")
            lines.append(f"  Success rate: {stage_report.success_rate:.1%}")

            if stage_report.best_method:
                lines.append(f"  Best method: {stage_report.best_method}")
                if stage_report.best_result:
                    lines.append(f"  Overall score: {stage_report.best_result.overall_score:.4f}")
                if stage_report.recommendation_reason:
                    lines.append(f"  Reason: {stage_report.recommendation_reason}")

        # Warnings
        if self.warnings:
            lines.append("\n" + "=" * 60)
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def save(
        self,
        filepath: str | Path,
        format: str = "markdown",
    ) -> None:
        """Save report to file.

        Parameters
        ----------
        filepath : str | Path
            Output file path
        format : str, optional
            Output format: "markdown", "json", or "csv", by default "markdown"

        Raises
        ------
        ValueError
            If format is not supported

        Examples
        --------
        >>> report.save("report.md")  # Markdown
        >>> report.save("report.json", format="json")
        >>> report.save("report.csv", format="csv")
        """
        from scptensor.autoselect.report import save_csv, save_json, save_markdown

        format_lower = format.lower()

        if format_lower == "markdown":
            save_markdown(self, filepath)
        elif format_lower == "json":
            save_json(self, filepath)
        elif format_lower == "csv":
            save_csv(self, filepath)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Supported formats: markdown, json, csv"
            )


class AutoSelector:
    """Unified automatic method selector for ScpTensor analysis pipeline.

    This class orchestrates automatic method selection across multiple analysis
    stages (normalization, imputation, integration, dimensionality reduction,
    clustering). It uses stage-specific evaluators to test methods and select
    the best performing ones.

    Attributes
    ----------
    SUPPORTED_STAGES : list[str]
        List of supported analysis stage names
    stages : list[str]
        Stages to execute (subset of SUPPORTED_STAGES)
    keep_all : bool
        Whether to keep all method results or only best
    weights : dict[str, dict[str, float]]
        Custom metric weights per stage (optional)
    parallel : bool
        Whether to execute stages in parallel (future feature)
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Examples
    --------
    >>> selector = AutoSelector(stages=["normalize", "impute"])
    >>> result_container, report = selector.run(container)
    >>> print(report.summary())
    """

    SUPPORTED_STAGES = ["normalize", "impute", "integrate", "reduce", "cluster"]

    def __init__(
        self,
        stages: list[str] | None = None,
        keep_all: bool = False,
        weights: dict[str, dict[str, float]] | None = None,
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize automatic method selector.

        Parameters
        ----------
        stages : list[str] | None, optional
            List of stages to execute. If None, executes all stages.
            Supported: "normalize", "impute", "integrate", "reduce", "cluster"
        keep_all : bool, bool, optional
            If True, keep all method results; if False, keep only best, by default False
        weights : dict[str, dict[str, float]] | None, optional
            Custom metric weights per stage, by default None
            Example: {"normalize": {"variance": 0.5, "batch_effect": 0.5}}
        parallel : bool, optional
            Enable parallel execution (future feature), by default True
        n_jobs : int, optional
            Number of parallel jobs, -1 for all cores, by default -1

        Raises
        ------
        ValueError
            If stages contains invalid stage names
        """
        # Validate stages
        if stages is None:
            self.stages = self.SUPPORTED_STAGES.copy()
        else:
            # Validate each stage
            invalid_stages = [s for s in stages if s not in self.SUPPORTED_STAGES]
            if invalid_stages:
                raise ValueError(
                    f"Invalid stage(s): {invalid_stages}. Supported stages: {self.SUPPORTED_STAGES}"
                )
            self.stages = stages.copy()

        self.keep_all = keep_all
        self.weights = weights if weights is not None else {}
        self.parallel = parallel
        self.n_jobs = n_jobs

    def _get_evaluator(self, stage: str) -> BaseEvaluator:
        """Get evaluator for a specific stage (lazy import).

        Parameters
        ----------
        stage : str
            Stage name

        Returns
        -------
        BaseEvaluator
            Evaluator instance for the stage

        Raises
        ------
        NotImplementedError
            If evaluator for stage is not yet implemented
        """
        # Lazy import to avoid circular dependencies
        if stage == "normalize":
            from scptensor.autoselect.evaluators.normalization import (
                NormalizationEvaluator,
            )

            return NormalizationEvaluator()
        elif stage == "impute":
            from scptensor.autoselect.evaluators.imputation import ImputationEvaluator

            return ImputationEvaluator()
        elif stage in ["integrate", "reduce", "cluster"]:
            raise NotImplementedError(
                f"Evaluator for stage '{stage}' is not yet implemented. "
                f"Currently supported: normalize, impute"
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def run_stage(
        self,
        container: ScpContainer,
        stage: str,
        assay_name: str = "proteins",
        source_layer: str | None = None,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """Execute automatic method selection for a single stage.

        Parameters
        ----------
        container : ScpContainer
            Input data container
        stage : str
            Stage name to execute
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        source_layer : str | None, optional
            Source layer name. If None, uses default for stage.
        **kwargs
            Additional parameters passed to evaluator

        Returns
        -------
        tuple[ScpContainer, StageReport]
            Tuple of (result_container, stage_report)

        Raises
        ------
        ValueError
            If stage is not supported
        """
        # Validate stage
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Supported stages: {self.SUPPORTED_STAGES}")

        # Get evaluator
        evaluator = self._get_evaluator(stage)

        # Determine source layer
        if source_layer is None:
            source_layer = "raw"

        # Apply custom weights if provided
        if stage in self.weights:
            # Note: This would require evaluator to support weight updates
            # For now, evaluators use their default weights
            pass

        # Run evaluator
        result_container, report = evaluator.run_all(
            container=container,
            assay_name=assay_name,
            source_layer=source_layer,
            keep_all=self.keep_all,
            **kwargs,
        )

        return result_container, report

    def run(
        self,
        container: ScpContainer,
        assay_name: str = "proteins",
        initial_layer: str = "raw",
    ) -> tuple[ScpContainer, AutoSelectReport]:
        """Execute full automatic method selection pipeline.

        This method runs all configured stages in sequence, passing the output
        of each stage as input to the next stage.

        Parameters
        ----------
        container : ScpContainer
            Input data container
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        initial_layer : str, optional
            Initial source layer name, by default "raw"

        Returns
        -------
        tuple[ScpContainer, AutoSelectReport]
            Tuple of (result_container, autoselect_report)

        Examples
        --------
        >>> selector = AutoSelector(stages=["normalize", "impute"])
        >>> result, report = selector.run(container, assay_name="proteins")
        >>> print(f"Best normalization: {report.stages['normalize'].best_method}")
        """
        # Initialize report
        report = AutoSelectReport()
        warnings: list[str] = []

        # Track total time
        start_time = time.perf_counter()

        # Current container and layer
        current_container = container
        current_layer = initial_layer

        # Run each stage
        for stage in self.stages:
            try:
                # Run stage
                result_container, stage_report = self.run_stage(
                    container=current_container,
                    stage=stage,
                    assay_name=assay_name,
                    source_layer=current_layer,
                )

                # Update container and layer for next stage
                current_container = result_container

                # Update layer to best result layer for next stage
                if stage_report.best_result is not None:
                    current_layer = stage_report.best_result.layer_name

                # Add stage report to overall report using evaluator's stage_name
                report.stages[stage_report.stage_name] = stage_report

            except NotImplementedError as e:
                # Stage not implemented - add warning and continue
                warning_msg = f"Stage '{stage}' skipped: {str(e)}"
                warnings.append(warning_msg)
                # Re-raise for now (tests expect this)
                raise
            except Exception as e:
                # Stage failed - add warning and continue
                warning_msg = f"Stage '{stage}' failed: {type(e).__name__}: {str(e)}"
                warnings.append(warning_msg)
                report.warnings.append(warning_msg)

        # Set total time and warnings
        report.total_time = time.perf_counter() - start_time
        report.warnings = warnings

        return current_container, report
