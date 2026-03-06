"""Report generation and export for automatic method selection.

This module provides functions to export AutoSelectReport to various formats:
- Markdown: Human-readable summary with tables
- JSON: Machine-readable structured data
- CSV: Tabular format for spreadsheet analysis

Functions
---------
save_markdown(report, filepath)
    Save report as Markdown file
save_json(report, filepath)
    Save report as JSON file
save_csv(report, filepath)
    Save report as CSV file
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scptensor.autoselect.strategy import get_strategy_preset

if TYPE_CHECKING:
    from scptensor.autoselect.core import AutoSelectReport


def save_markdown(report: AutoSelectReport, filepath: str | Path) -> None:
    """Save report as Markdown file.

    Parameters
    ----------
    report : AutoSelectReport
        Report to export
    filepath : str | Path
        Output file path

    Examples
    --------
    >>> from scptensor.autoselect import AutoSelectReport
    >>> report = AutoSelectReport()
    >>> save_markdown(report, "autoselect_report.md")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Header
    lines.append("# AutoSelect Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Stages:** {len(report.stages)}")
    lines.append(f"- **Total Time:** {report.total_time:.2f} seconds")
    lines.append("")

    if not report.stages:
        lines.append("No stages completed.")
        lines.append("")
    else:
        # Stage details
        for stage_name, stage_report in report.stages.items():
            lines.append(f"## {stage_name.title()}")
            lines.append("")
            lines.append(f"- **Methods Tested:** {len(stage_report.results)}")
            lines.append(f"- **Success Rate:** {stage_report.success_rate:.1%}")
            lines.append(f"- **Selection Strategy:** `{stage_report.selection_strategy}`")
            try:
                preset = get_strategy_preset(stage_report.selection_strategy)
                lines.append(
                    "- **Strategy Weights:** "
                    f"quality={preset.quality_weight:.2f}, "
                    f"runtime={preset.runtime_weight:.2f}"
                )
            except ValueError:
                # Keep markdown export resilient even if stage report was externally crafted.
                pass
            lines.append(f"- **Repeats per Method:** {stage_report.n_repeats}")
            lines.append(f"- **Confidence Level:** {stage_report.confidence_level:.2f}")
            if stage_report.input_assay and stage_report.input_layer:
                lines.append(
                    f"- **Input:** `{stage_report.input_assay}/{stage_report.input_layer}`"
                )
            if stage_report.output_obs_key:
                lines.append(f"- **Output:** `obs[{stage_report.output_obs_key}]`")
            elif stage_report.output_assay and stage_report.output_layer:
                lines.append(
                    f"- **Output:** `{stage_report.output_assay}/{stage_report.output_layer}`"
                )
            lines.append("")

            if stage_report.best_method:
                lines.append(f"**Best Method:** `{stage_report.best_method}`")
                if stage_report.best_result:
                    lines.append(
                        f"- **Overall Score:** {stage_report.best_result.overall_score:.4f}"
                    )
                    lines.append(
                        f"- **Execution Time:** {stage_report.best_result.execution_time:.2f}s"
                    )
                if stage_report.recommendation_reason:
                    lines.append(f"- **Reason:** {stage_report.recommendation_reason}")
                lines.append("")

            # Results table
            if stage_report.results:
                lines.append("### All Methods")
                lines.append("")
                lines.append(
                    "| Method | Selection Score | Overall Score | Std | CI | "
                    "Execution Time | Status |"
                )
                lines.append(
                    "|--------|------------------|---------------|-----|----|----------------|--------|"
                )

                for result in stage_report.results:
                    status = "✓ Success" if result.error is None else f"✗ Failed: {result.error}"
                    best_marker = " **(Best)**" if result == stage_report.best_result else ""
                    selection_score = (
                        "NA" if result.selection_score is None else f"{result.selection_score:.4f}"
                    )
                    score_std = (
                        "NA"
                        if result.overall_score_std is None
                        else f"{result.overall_score_std:.4f}"
                    )
                    if (
                        result.overall_score_ci_lower is None
                        or result.overall_score_ci_upper is None
                    ):
                        score_ci = "NA"
                    else:
                        score_ci = (
                            f"[{result.overall_score_ci_lower:.4f}, "
                            f"{result.overall_score_ci_upper:.4f}]"
                        )
                    lines.append(
                        f"| {result.method_name}{best_marker} | "
                        f"{selection_score} | "
                        f"{result.overall_score:.4f} | "
                        f"{score_std} | "
                        f"{score_ci} | "
                        f"{result.execution_time:.2f}s | "
                        f"{status} |"
                    )
                lines.append("")

    # Warnings
    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by ScpTensor AutoSelect*")
    lines.append("")

    # Write to file
    filepath.write_text("\n".join(lines))


def save_json(report: AutoSelectReport, filepath: str | Path) -> None:
    """Save report as JSON file.

    Parameters
    ----------
    report : AutoSelectReport
        Report to export
    filepath : str | Path
        Output file path

    Examples
    --------
    >>> from scptensor.autoselect import AutoSelectReport
    >>> report = AutoSelectReport()
    >>> save_json(report, "autoselect_report.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary
    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_time": report.total_time,
        "warnings": report.warnings,
        "stages": {},
    }

    for stage_name, stage_report in report.stages.items():
        stage_data = stage_report.to_dict()
        stage_data["stage_name"] = stage_name
        data["stages"][stage_name] = stage_data

    # Write to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(report: AutoSelectReport, filepath: str | Path) -> None:
    """Save report as CSV file (one row per method per stage).

    Parameters
    ----------
    report : AutoSelectReport
        Report to export
    filepath : str | Path
        Output file path

    Examples
    --------
    >>> from scptensor.autoselect import AutoSelectReport
    >>> report = AutoSelectReport()
    >>> save_csv(report, "autoselect_report.csv")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for stage_name, stage_report in report.stages.items():
        for result in stage_report.results:
            is_best = result == stage_report.best_result
            rows.append(
                {
                    "stage_name": stage_name,
                    "stage_key": stage_report.stage_key or stage_name,
                    "method_name": result.method_name,
                    "overall_score": result.overall_score,
                    "execution_time": result.execution_time,
                    "layer_name": result.layer_name,
                    "input_assay": stage_report.input_assay or "",
                    "input_layer": stage_report.input_layer or "",
                    "output_assay": stage_report.output_assay or "",
                    "output_layer": stage_report.output_layer or "",
                    "output_obs_key": stage_report.output_obs_key or "",
                    "selection_strategy": stage_report.selection_strategy,
                    "n_repeats": stage_report.n_repeats,
                    "confidence_level": stage_report.confidence_level,
                    "recommendation_reason": stage_report.recommendation_reason,
                    "error": result.error or "",
                    "selection_score": result.selection_score,
                    "overall_score_std": result.overall_score_std,
                    "overall_score_ci_lower": result.overall_score_ci_lower,
                    "overall_score_ci_upper": result.overall_score_ci_upper,
                    "repeat_overall_scores": json.dumps(result.repeat_overall_scores),
                    "scores": json.dumps(result.scores, sort_keys=True),
                    "metric_weights": json.dumps(stage_report.metric_weights, sort_keys=True),
                    "is_best": is_best,
                }
            )

    # Write to CSV
    fieldnames = [
        "stage_name",
        "stage_key",
        "method_name",
        "overall_score",
        "execution_time",
        "layer_name",
        "input_assay",
        "input_layer",
        "output_assay",
        "output_layer",
        "output_obs_key",
        "selection_strategy",
        "n_repeats",
        "confidence_level",
        "selection_score",
        "overall_score_std",
        "overall_score_ci_lower",
        "overall_score_ci_upper",
        "repeat_overall_scores",
        "scores",
        "metric_weights",
        "recommendation_reason",
        "error",
        "is_best",
    ]
    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        # Write empty CSV with header
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
