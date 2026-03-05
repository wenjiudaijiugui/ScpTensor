#!/usr/bin/env python3
"""Benchmark AutoSelect integration strategies side-by-side.

This script generates a DIA-like protein matrix with explicit batch confounding,
applies a fixed preprocessing baseline, and compares AutoSelect strategy presets:
`quality`, `balanced`, and `speed`.

Baseline preprocessing is fixed to:
log transform -> normalization -> missing-value imputation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scptensor.autoselect import AutoSelector, StageReport
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.impute import impute
from scptensor.normalization import normalize
from scptensor.transformation import log_transform


@dataclass
class StrategySummaryRow:
    """Compact summary row for one strategy."""

    strategy: str
    best_method: str
    selection_score: float
    overall_score: float
    execution_time: float
    n_repeats: int
    ci_lower: float
    ci_upper: float
    methods_tested: int
    success_rate: float
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "best_method": self.best_method,
            "selection_score": self.selection_score,
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "n_repeats": self.n_repeats,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "methods_tested": self.methods_tested,
            "success_rate": self.success_rate,
            "recommendation_reason": self.recommendation_reason,
        }


def _build_confounded_dia_container(
    random_state: int,
    n_samples: int,
    n_proteins: int,
    n_cell_types: int = 4,
) -> tuple[ScpContainer, dict[str, Any]]:
    """Create a DIA-like protein matrix with explicit batch confounding."""
    if n_samples < n_cell_types:
        raise ValueError(
            f"n_samples must be >= n_cell_types ({n_cell_types}), got {n_samples}"
        )

    rng = np.random.default_rng(random_state)

    # Cell types
    cell_type_idx = np.arange(n_samples) % n_cell_types
    rng.shuffle(cell_type_idx)

    # Batch confounding: different cell types have different batch priors.
    batch0_prob_by_type = np.linspace(0.85, 0.15, n_cell_types)
    batch_idx = np.where(rng.random(n_samples) < batch0_prob_by_type[cell_type_idx], 0, 1)

    # Latent log2-intensity signal: biological signal + batch effect + noise.
    base_log2 = rng.normal(24.0, 1.1, size=n_proteins)
    bio_effects = rng.normal(0.0, 0.3, size=(n_cell_types, n_proteins))
    marker_count = max(12, n_proteins // 8)
    for ct in range(n_cell_types):
        markers = rng.choice(n_proteins, size=marker_count, replace=False)
        bio_effects[ct, markers] += rng.normal(1.0, 0.2, size=marker_count)

    batch_effects = rng.normal(0.0, 0.2, size=(2, n_proteins))
    strong_batch_features = rng.choice(n_proteins, size=max(20, n_proteins // 6), replace=False)
    batch_effects[1, strong_batch_features] += rng.normal(
        0.6, 0.1, size=strong_batch_features.size
    )

    noise = rng.normal(0.0, 0.35, size=(n_samples, n_proteins))
    log2_x = base_log2 + bio_effects[cell_type_idx] + batch_effects[batch_idx] + noise

    # Convert to linear-scale intensities (raw DIA-like matrix).
    raw_x = np.exp2(np.clip(log2_x, a_min=0.0, a_max=None))

    # Missingness: intensity-dependent + batch-dependent + MCAR component.
    p_intensity = 0.02 + 0.22 * (1.0 / (1.0 + np.exp(log2_x - 23.0)))
    p_batch = np.where(batch_idx[:, None] == 1, 0.05, 0.0)
    p_total = np.clip(p_intensity + p_batch + 0.03, 0.0, 0.8)
    missing_mask = rng.random(size=raw_x.shape) < p_total
    raw_x[missing_mask] = np.nan

    obs = pl.DataFrame(
        {
            "_index": [f"Cell_{i:04d}" for i in range(n_samples)],
            "batch": np.where(batch_idx == 0, "Batch_0", "Batch_1"),
            "cell_type": [f"Type_{ct}" for ct in cell_type_idx],
        }
    )
    var = pl.DataFrame(
        {
            "_index": [f"P_{j:05d}" for j in range(n_proteins)],
            "protein": [f"Protein_{j:05d}" for j in range(n_proteins)],
        }
    )

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=raw_x))
    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)

    confounding_counts = (
        obs.group_by("cell_type", "batch").len().sort(["cell_type", "batch"]).to_dicts()
    )
    metadata = {
        "n_samples": n_samples,
        "n_proteins": n_proteins,
        "n_cell_types": n_cell_types,
        "missing_rate": float(np.mean(np.isnan(raw_x))),
        "batch_celltype_counts": confounding_counts,
    }
    return container, metadata


def _apply_baseline_preprocessing(container: ScpContainer) -> tuple[ScpContainer, str]:
    """Apply fixed preprocessing baseline before integration autoselection."""
    processed = container.copy()

    processed = log_transform(
        processed,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log2",
        base=2.0,
        offset=1.0,
        detect_logged=True,
        skip_if_logged=True,
    )
    processed = normalize(
        processed,
        method="median",
        assay_name="proteins",
        source_layer="log2",
        new_layer_name="median_norm",
    )
    processed = impute(
        processed,
        method="row_median",
        assay_name="proteins",
        source_layer="median_norm",
        new_layer_name="baseline_imputed",
    )

    return processed, "baseline_imputed"


def _run_single_strategy(
    container: ScpContainer,
    source_layer: str,
    strategy: str,
    n_repeats: int,
    confidence_level: float,
) -> StageReport:
    """Run integration autoselection under one strategy."""
    selector = AutoSelector(
        stages=["integrate"],
        keep_all=False,
        selection_strategy=strategy,
        n_repeats=n_repeats,
        confidence_level=confidence_level,
    )
    _, report = selector.run_stage(
        container=container.copy(),
        stage="integrate",
        assay_name="proteins",
        source_layer=source_layer,
        batch_key="batch",
    )
    return report


def _to_summary_row(strategy: str, report: StageReport) -> StrategySummaryRow:
    """Convert a stage report to compact summary row."""
    best = report.best_result
    if best is None:
        return StrategySummaryRow(
            strategy=strategy,
            best_method="",
            selection_score=0.0,
            overall_score=0.0,
            execution_time=0.0,
            n_repeats=report.n_repeats,
            ci_lower=0.0,
            ci_upper=0.0,
            methods_tested=len(report.results),
            success_rate=report.success_rate,
            recommendation_reason=report.recommendation_reason,
        )

    return StrategySummaryRow(
        strategy=strategy,
        best_method=report.best_method,
        selection_score=float(best.selection_score or 0.0),
        overall_score=float(best.overall_score),
        execution_time=float(best.execution_time),
        n_repeats=int(best.n_repeats),
        ci_lower=float(best.overall_score_ci_lower or 0.0),
        ci_upper=float(best.overall_score_ci_upper or 0.0),
        methods_tested=len(report.results),
        success_rate=report.success_rate,
        recommendation_reason=report.recommendation_reason,
    )


def _write_json(
    output_path: Path,
    dataset_meta: dict[str, Any],
    summaries: list[StrategySummaryRow],
    reports: dict[str, StageReport],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "module": "autoselect.integration.strategy_comparison",
        "dataset": dataset_meta,
        "baseline_preprocessing": ["log_transform(base=2, offset=1)", "median_norm", "row_median"],
        "summary": [row.to_dict() for row in summaries],
        "reports": {name: report.to_dict() for name, report in reports.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _write_csv(output_path: Path, summaries: list[StrategySummaryRow]) -> None:
    fieldnames = [
        "strategy",
        "best_method",
        "selection_score",
        "overall_score",
        "execution_time",
        "n_repeats",
        "ci_lower",
        "ci_upper",
        "methods_tested",
        "success_rate",
        "recommendation_reason",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.to_dict() for row in summaries])


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _write_markdown(
    output_path: Path,
    dataset_meta: dict[str, Any],
    summaries: list[StrategySummaryRow],
    reports: dict[str, StageReport],
) -> None:
    lines: list[str] = []
    lines.append("# AutoSelect Integration Strategy Comparison")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Samples: `{dataset_meta['n_samples']}`")
    lines.append(f"- Proteins: `{dataset_meta['n_proteins']}`")
    lines.append(f"- Missing rate: `{dataset_meta['missing_rate']:.2%}`")
    lines.append("")
    lines.append("## Fixed Baseline")
    lines.append("")
    lines.append("`raw -> log_transform(base=2, offset=1) -> median_norm -> row_median impute`")
    lines.append("")
    lines.append("## Batch Confounding Snapshot")
    lines.append("")
    lines.append("| Cell Type | Batch | Count |")
    lines.append("|---|---|---:|")
    for row in dataset_meta["batch_celltype_counts"]:
        lines.append(f"| {row['cell_type']} | {row['batch']} | {row['len']} |")
    lines.append("")
    lines.append("## Strategy Summary")
    lines.append("")
    lines.append(
        "| Strategy | Best Method | Selection Score | Overall Score | Time (s) | Repeats | CI (overall) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in summaries:
        lines.append(
            f"| {row.strategy} | {row.best_method} | {row.selection_score:.4f} | "
            f"{row.overall_score:.4f} | {row.execution_time:.3f} | {row.n_repeats} | "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
        )
    lines.append("")

    for strategy_name, report in reports.items():
        lines.append(f"## {strategy_name} Details")
        lines.append("")
        lines.append(f"- Best method: `{report.best_method}`")
        lines.append(f"- Success rate: `{report.success_rate:.1%}`")
        lines.append("")
        lines.append("| Method | Selection | Overall | Time (s) | Status |")
        lines.append("|---|---:|---:|---:|---|")
        ranked = sorted(
            report.results,
            key=lambda r: (
                r.error is None,
                r.selection_score if r.selection_score is not None else -1.0,
                r.overall_score,
            ),
            reverse=True,
        )
        for result in ranked:
            status = "ok" if result.error is None else "failed"
            method = (
                f"{result.method_name} (best)"
                if report.best_result is not None and result.method_name == report.best_result.method_name
                else result.method_name
            )
            lines.append(
                f"| {method} | {_fmt(result.selection_score)} | {_fmt(result.overall_score)} | "
                f"{result.execution_time:.3f} | {status} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def run_benchmark(
    output_dir: Path,
    random_state: int,
    n_samples: int,
    n_proteins: int,
    n_repeats: int,
    confidence_level: float,
) -> dict[str, Path]:
    """Run strategy comparison benchmark and write all output artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    container, dataset_meta = _build_confounded_dia_container(
        random_state=random_state,
        n_samples=n_samples,
        n_proteins=n_proteins,
    )
    container, baseline_layer = _apply_baseline_preprocessing(container)

    strategies = ["quality", "balanced", "speed"]
    reports: dict[str, StageReport] = {}
    summaries: list[StrategySummaryRow] = []

    for strategy in strategies:
        report = _run_single_strategy(
            container=container,
            source_layer=baseline_layer,
            strategy=strategy,
            n_repeats=n_repeats,
            confidence_level=confidence_level,
        )
        reports[strategy] = report
        summaries.append(_to_summary_row(strategy, report))

    json_path = output_dir / "strategy_comparison.json"
    csv_path = output_dir / "strategy_comparison.csv"
    md_path = output_dir / "strategy_comparison.md"
    _write_json(json_path, dataset_meta, summaries, reports)
    _write_csv(csv_path, summaries)
    _write_markdown(md_path, dataset_meta, summaries, reports)

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AutoSelect integration strategy comparison "
            "for quality / balanced / speed presets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "strategy_compare",
        help="Directory for output files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--samples", type=int, default=240, help="Number of samples (cells).")
    parser.add_argument("--proteins", type=int, default=600, help="Number of proteins (features).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated evaluations per method.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence interval level for repeat aggregation.",
    )
    args = parser.parse_args()

    outputs = run_benchmark(
        output_dir=args.output_dir,
        random_state=args.seed,
        n_samples=args.samples,
        n_proteins=args.proteins,
        n_repeats=args.repeats,
        confidence_level=args.confidence,
    )
    print("Strategy comparison benchmark completed:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
