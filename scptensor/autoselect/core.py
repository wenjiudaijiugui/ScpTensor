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

from scptensor.autoselect.strategy import get_strategy_preset

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
    selection_score: float | None = None
    n_repeats: int = 1
    overall_score_std: float | None = None
    overall_score_ci_lower: float | None = None
    overall_score_ci_upper: float | None = None
    repeat_overall_scores: list[float] = field(default_factory=list)

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
            "selection_score": self.selection_score,
            "n_repeats": self.n_repeats,
            "overall_score_std": self.overall_score_std,
            "overall_score_ci_lower": self.overall_score_ci_lower,
            "overall_score_ci_upper": self.overall_score_ci_upper,
            "repeat_overall_scores": self.repeat_overall_scores,
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
    stage_key: str | None = None
    results: list[EvaluationResult] = field(default_factory=list)
    best_method: str = ""
    best_result: EvaluationResult | None = None
    recommendation_reason: str = ""
    metric_weights: dict[str, float] = field(default_factory=dict)
    input_assay: str | None = None
    input_layer: str | None = None
    output_assay: str | None = None
    output_layer: str | None = None
    output_obs_key: str | None = None
    selection_strategy: str = "balanced"
    n_repeats: int = 1
    confidence_level: float = 0.95

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

    def to_dict(self) -> dict[str, Any]:
        """Convert stage report to dictionary representation."""
        return {
            "stage_name": self.stage_name,
            "stage_key": self.stage_key or self.stage_name,
            "best_method": self.best_method,
            "recommendation_reason": self.recommendation_reason,
            "success_rate": self.success_rate,
            "metric_weights": self.metric_weights,
            "input_assay": self.input_assay,
            "input_layer": self.input_layer,
            "output_assay": self.output_assay,
            "output_layer": self.output_layer,
            "output_obs_key": self.output_obs_key,
            "selection_strategy": self.selection_strategy,
            "n_repeats": self.n_repeats,
            "confidence_level": self.confidence_level,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class _StageContext:
    """Execution context shared between stages."""

    assay_name: str
    source_layer: str


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

    # Mapping of stage names to evaluator import paths
    _EVALUATOR_MAP = {
        "normalize": ("scptensor.autoselect.evaluators.normalization", "NormalizationEvaluator"),
        "impute": ("scptensor.autoselect.evaluators.imputation", "ImputationEvaluator"),
        "integrate": ("scptensor.autoselect.evaluators.integration", "IntegrationEvaluator"),
        "reduce": ("scptensor.autoselect.evaluators.dim_reduction", "DimReductionEvaluator"),
        "cluster": ("scptensor.autoselect.evaluators.clustering", "ClusteringEvaluator"),
    }
    _STAGE_OUTPUT_KIND = {
        "normalize": "layer",
        "impute": "layer",
        "integrate": "layer",
        "reduce": "assay",
        "cluster": "obs",
    }

    def __init__(
        self,
        stages: list[str] | None = None,
        keep_all: bool = False,
        weights: dict[str, dict[str, float]] | None = None,
        parallel: bool = True,
        n_jobs: int = -1,
        selection_strategy: str = "balanced",
        n_repeats: int = 1,
        confidence_level: float = 0.95,
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
        selection_strategy : {"speed", "balanced", "quality"}, optional
            Strategy for final method ranking. Default is "balanced".
        n_repeats : int, optional
            Number of repeated evaluations per method for stochastic robustness.
            Default is 1.
        confidence_level : float, optional
            Confidence level for repeat-score interval estimation. Default is 0.95.

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
        strategy_name = get_strategy_preset(selection_strategy).name
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        self.selection_strategy = strategy_name
        self.n_repeats = n_repeats
        self.confidence_level = confidence_level

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
        ValueError
            If stage is not in the evaluator map
        """
        if stage not in self._EVALUATOR_MAP:
            raise ValueError(f"Unknown stage: {stage}")

        # Lazy import to avoid circular dependencies
        module_path, class_name = self._EVALUATOR_MAP[stage]
        import importlib

        module = importlib.import_module(module_path)
        evaluator_class = getattr(module, class_name)
        return evaluator_class()

    def _validate_stage_inputs(
        self,
        container: ScpContainer,
        stage: str,
        assay_name: str,
        source_layer: str,
        **kwargs,
    ) -> None:
        """Validate stage-level inputs with actionable errors."""
        if assay_name not in container.assays:
            available = sorted(container.assays.keys())
            raise ValueError(
                f"Stage '{stage}' requires assay '{assay_name}', but it was not found. "
                f"Available assays: {available}"
            )

        assay = container.assays[assay_name]
        if source_layer not in assay.layers:
            available_layers = sorted(assay.layers.keys())
            raise ValueError(
                f"Stage '{stage}' requires layer '{source_layer}' in assay '{assay_name}', "
                f"but it was not found. Available layers: {available_layers}"
            )

        if stage == "integrate":
            batch_key = kwargs.get("batch_key", "batch")
            if batch_key not in container.obs.columns:
                raise ValueError(
                    f"Stage 'integrate' requires batch_key '{batch_key}' in obs, "
                    f"but it was not found. Available obs columns: {container.obs.columns}"
                )

    def _infer_next_context(
        self,
        stage: str,
        context: _StageContext,
        stage_report: StageReport,
    ) -> _StageContext:
        """Infer next stage context from current stage result."""
        best_result = stage_report.best_result
        if best_result is None:
            return context

        output_kind = self._STAGE_OUTPUT_KIND[stage]
        if output_kind == "layer":
            return _StageContext(assay_name=context.assay_name, source_layer=best_result.layer_name)
        if output_kind == "assay":
            # Dim-reduction outputs are represented as new assays with layer "X".
            return _StageContext(assay_name=best_result.layer_name, source_layer="X")
        return context

    def _attach_stage_io(
        self,
        stage: str,
        context: _StageContext,
        stage_report: StageReport,
    ) -> StageReport:
        """Attach unified stage metadata and input/output descriptors."""
        stage_report.stage_name = stage
        stage_report.stage_key = stage
        stage_report.input_assay = context.assay_name
        stage_report.input_layer = context.source_layer

        next_context = self._infer_next_context(stage, context, stage_report)
        stage_report.output_assay = next_context.assay_name
        stage_report.output_layer = next_context.source_layer
        if stage == "cluster" and stage_report.best_result is not None:
            stage_report.output_obs_key = stage_report.best_result.layer_name

        return stage_report

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

        # Determine source layer
        if source_layer is None:
            source_layer = "raw"

        context = _StageContext(assay_name=assay_name, source_layer=source_layer)
        self._validate_stage_inputs(
            container=container,
            stage=stage,
            assay_name=context.assay_name,
            source_layer=context.source_layer,
            **kwargs,
        )

        # Get evaluator
        evaluator = self._get_evaluator(stage)

        if stage in self.weights:
            evaluator.set_metric_weights(self.weights[stage])

        # Run evaluator
        eval_kwargs = {
            "selection_strategy": self.selection_strategy,
            "n_repeats": self.n_repeats,
            "confidence_level": self.confidence_level,
            **kwargs,
        }
        result_container, report = evaluator.run_all(
            container=container,
            assay_name=context.assay_name,
            source_layer=context.source_layer,
            keep_all=self.keep_all,
            **eval_kwargs,
        )

        return result_container, self._attach_stage_io(stage, context, report)

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

        # Current container and stage context
        current_container = container
        current_context = _StageContext(assay_name=assay_name, source_layer=initial_layer)

        # Run each stage
        for stage in self.stages:
            try:
                # Run stage
                result_container, stage_report = self.run_stage(
                    container=current_container,
                    stage=stage,
                    assay_name=current_context.assay_name,
                    source_layer=current_context.source_layer,
                )

                # Update container and layer for next stage
                current_container = result_container
                current_context = self._infer_next_context(stage, current_context, stage_report)

                # Add stage report using canonical stage key
                report.stages[stage] = stage_report

            except Exception as e:
                # Stage failed - add warning and continue
                warning_msg = f"Stage '{stage}' failed: {type(e).__name__}: {str(e)}"
                warnings.append(warning_msg)
                report.warnings.append(warning_msg)
                # Re-raise the exception to maintain expected behavior
                raise

        # Set total time and warnings
        report.total_time = time.perf_counter() - start_time
        report.warnings = warnings

        return current_container, report
