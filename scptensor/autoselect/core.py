"""Core data classes for autoselect module.

This module defines the core data structures for the automatic method selection system:
- EvaluationResult: Single method evaluation result
- StageReport: Report for a single analysis stage
- AutoSelectReport: Complete automatic selection report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
