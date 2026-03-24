#!/usr/bin/env python3
"""Benchmark AutoSelect integration strategies across design scenarios.

This script generates a DIA-like protein matrix, applies a fixed preprocessing
baseline, and compares AutoSelect strategy presets across three scenario
boundaries:

- balanced
- partially confounded with a bridge batch
- fully confounded

The script also reports a scenario-level state burden derived from the baseline
mask state. This produces an auxiliary script-local penalty column without
changing the core AutoSelect scoring contract.
"""

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
from scptensor.core import (
    Assay,
    FilterCriteria,
    ScpContainer,
    ScpMatrix,
    compute_layer_state_metrics,
)
from scptensor.impute import impute
from scptensor.normalization import normalize
from scptensor.transformation import log_transform

SAMPLE_GROUPS = ("S1", "S2", "S3")
AMOUNT_GROUPS = ("300ng", "600ng")

SCENARIO_SPECS: dict[str, dict[str, Any]] = {
    "balanced_amount_by_sample": {
        "batch_key": "batch_balanced",
        "bio_key": "condition_balanced",
        "design_identifiability": "fully_identifiable_balanced",
        "subset_strategy": "full_dataset",
        "description": "Balanced sample-group batches crossed with amount-group biology.",
    },
    "partially_confounded_bridge_sample": {
        "batch_key": "batch_partially_confounded",
        "bio_key": "condition_partially_confounded",
        "design_identifiability": "bridge_reference_partial_confounding",
        "subset_strategy": "bridge_sample_partial_confounding",
        "description": (
            "Bridge-style partial confounding: S1 low only, S2 both amounts, S3 high only."
        ),
    },
    "confounded_amount_as_batch": {
        "batch_key": "batch_confounded",
        "bio_key": None,
        "design_identifiability": "non_identifiable_fully_confounded",
        "subset_strategy": "full_dataset",
        "description": "Batch equals amount group; biological contrast is not separately identifiable.",
    },
}


@dataclass
class StrategySummaryRow:
    """Compact summary row for one strategy under one scenario."""

    scenario: str
    design_identifiability: str
    strategy: str
    best_method: str
    selection_score: float
    state_penalized_selection_score: float
    overall_score: float
    execution_time: float
    n_repeats: int
    ci_lower: float
    ci_upper: float
    methods_tested: int
    success_rate: float
    n_samples_scenario: int
    state_direct_observation_rate: float
    state_supported_observation_rate: float
    state_uncertainty_burden: float
    state_non_valid_fraction: float
    state_imputed_fraction: float
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "design_identifiability": self.design_identifiability,
            "strategy": self.strategy,
            "best_method": self.best_method,
            "selection_score": self.selection_score,
            "state_penalized_selection_score": self.state_penalized_selection_score,
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "n_repeats": self.n_repeats,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "methods_tested": self.methods_tested,
            "success_rate": self.success_rate,
            "n_samples_scenario": self.n_samples_scenario,
            "state_direct_observation_rate": self.state_direct_observation_rate,
            "state_supported_observation_rate": self.state_supported_observation_rate,
            "state_uncertainty_burden": self.state_uncertainty_burden,
            "state_non_valid_fraction": self.state_non_valid_fraction,
            "state_imputed_fraction": self.state_imputed_fraction,
            "recommendation_reason": self.recommendation_reason,
        }


def _assign_design_labels(n_samples: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    pairs = [
        (sample_group, amount_group)
        for sample_group in SAMPLE_GROUPS
        for amount_group in AMOUNT_GROUPS
    ]
    repeated = [pairs[i % len(pairs)] for i in range(n_samples)]
    rng.shuffle(repeated)
    sample_groups = [pair[0] for pair in repeated]
    amount_groups = [pair[1] for pair in repeated]
    return sample_groups, amount_groups


def _build_design_synthetic_container(
    random_state: int,
    n_samples: int,
    n_proteins: int,
) -> tuple[ScpContainer, dict[str, Any]]:
    """Create a DIA-like protein matrix with balanced design labels."""
    if n_samples < len(SAMPLE_GROUPS) * len(AMOUNT_GROUPS):
        raise ValueError(
            "n_samples must be >= 6 to cover all sample_group/amount_group combinations."
        )

    rng = np.random.default_rng(random_state)
    sample_groups, amount_groups = _assign_design_labels(n_samples, rng)
    sample_index = {name: idx for idx, name in enumerate(SAMPLE_GROUPS)}
    amount_index = {name: idx for idx, name in enumerate(AMOUNT_GROUPS)}

    batch_idx = np.asarray([sample_index[name] for name in sample_groups], dtype=int)
    condition_idx = np.asarray([amount_index[name] for name in amount_groups], dtype=int)

    base_log2 = rng.normal(24.0, 1.0, size=n_proteins)
    bio_effects = np.zeros((len(AMOUNT_GROUPS), n_proteins), dtype=np.float64)
    marker_count = max(16, n_proteins // 8)
    markers_high = rng.choice(n_proteins, size=marker_count, replace=False)
    bio_effects[1, markers_high] += rng.normal(0.9, 0.15, size=marker_count)

    batch_effects = rng.normal(0.0, 0.20, size=(len(SAMPLE_GROUPS), n_proteins))
    batch_specific = rng.choice(n_proteins, size=max(24, n_proteins // 6), replace=False)
    for batch_id in range(1, len(SAMPLE_GROUPS)):
        batch_effects[batch_id, batch_specific] += rng.normal(0.35, 0.08, size=batch_specific.size)

    noise = rng.normal(0.0, 0.30, size=(n_samples, n_proteins))
    log2_x = base_log2[None, :] + bio_effects[condition_idx] + batch_effects[batch_idx] + noise
    raw_x = np.exp2(np.clip(log2_x, a_min=0.0, a_max=None))

    p_intensity = 0.03 + 0.20 * (1.0 / (1.0 + np.exp(log2_x - 23.0)))
    p_batch = np.where(np.asarray(sample_groups, dtype=object)[:, None] == "S3", 0.04, 0.0)
    p_total = np.clip(p_intensity + p_batch + 0.02, 0.0, 0.75)
    missing_mask = rng.random(size=raw_x.shape) < p_total
    raw_x[missing_mask] = np.nan

    obs = pl.DataFrame(
        {
            "_index": [f"Cell_{i:04d}" for i in range(n_samples)],
            "sample_group": sample_groups,
            "amount_group": amount_groups,
            "batch_balanced": sample_groups,
            "condition_balanced": amount_groups,
            "batch_partially_confounded": sample_groups,
            "condition_partially_confounded": amount_groups,
            "batch_confounded": amount_groups,
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

    batch_condition_counts = (
        obs.group_by("sample_group", "amount_group")
        .len()
        .sort(["sample_group", "amount_group"])
        .to_dicts()
    )
    metadata = {
        "n_samples": n_samples,
        "n_proteins": n_proteins,
        "missing_rate": float(np.mean(np.isnan(raw_x))),
        "batch_condition_counts": batch_condition_counts,
    }
    return container, metadata


def _partially_confounded_mask(container: ScpContainer) -> np.ndarray:
    sample_values = container.obs["sample_group"].cast(pl.Utf8).to_numpy()
    amount_values = container.obs["amount_group"].cast(pl.Utf8).to_numpy()

    low_batch, bridge_batch, high_batch = SAMPLE_GROUPS
    low_amount = AMOUNT_GROUPS[0]
    high_amount = AMOUNT_GROUPS[-1]

    keep_mask = (
        ((sample_values == low_batch) & (amount_values == low_amount))
        | (sample_values == bridge_batch)
        | ((sample_values == high_batch) & (amount_values == high_amount))
    )
    return keep_mask


def _prepare_scenario_container(container: ScpContainer, scenario_name: str) -> ScpContainer:
    scenario = SCENARIO_SPECS[scenario_name]
    subset_strategy = str(scenario["subset_strategy"])
    if subset_strategy == "full_dataset":
        return container
    if subset_strategy == "bridge_sample_partial_confounding":
        return container.filter_samples(
            FilterCriteria.by_mask(_partially_confounded_mask(container))
        )
    raise ValueError(f"Unknown subset strategy: {subset_strategy}")


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


def _compute_state_burden(
    container: ScpContainer,
    *,
    assay_name: str,
    layer_name: str,
) -> dict[str, float]:
    state_metrics = compute_layer_state_metrics(
        container,
        assay_name=assay_name,
        layer_name=layer_name,
    )
    return {
        "state_direct_observation_rate": state_metrics["direct_observation_rate"],
        "state_supported_observation_rate": state_metrics["supported_observation_rate"],
        "state_uncertainty_burden": state_metrics["uncertainty_burden"],
        "state_non_valid_fraction": float(1.0 - state_metrics["valid_rate"]),
        "state_imputed_fraction": state_metrics["imputed_rate"],
    }


def _run_single_strategy(
    container: ScpContainer,
    source_layer: str,
    strategy: str,
    n_repeats: int,
    confidence_level: float,
    *,
    batch_key: str,
    bio_key: str | None,
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
        batch_key=batch_key,
        bio_key=bio_key,
    )
    return report


def _to_summary_row(
    *,
    scenario: str,
    report: StageReport,
    strategy: str,
    n_samples_scenario: int,
    design_identifiability: str,
    state_burden: dict[str, float],
) -> StrategySummaryRow:
    """Convert a stage report to compact summary row."""
    best = report.best_result
    if best is None:
        return StrategySummaryRow(
            scenario=scenario,
            design_identifiability=design_identifiability,
            strategy=strategy,
            best_method="",
            selection_score=0.0,
            state_penalized_selection_score=0.0,
            overall_score=0.0,
            execution_time=0.0,
            n_repeats=report.n_repeats,
            ci_lower=0.0,
            ci_upper=0.0,
            methods_tested=len(report.results),
            success_rate=report.success_rate,
            n_samples_scenario=n_samples_scenario,
            state_direct_observation_rate=state_burden["state_direct_observation_rate"],
            state_supported_observation_rate=state_burden["state_supported_observation_rate"],
            state_uncertainty_burden=state_burden["state_uncertainty_burden"],
            state_non_valid_fraction=state_burden["state_non_valid_fraction"],
            state_imputed_fraction=state_burden["state_imputed_fraction"],
            recommendation_reason=report.recommendation_reason,
        )

    selection_score = float(best.selection_score or 0.0)
    state_penalty = max(0.0, 1.0 - state_burden["state_uncertainty_burden"])
    return StrategySummaryRow(
        scenario=scenario,
        design_identifiability=design_identifiability,
        strategy=strategy,
        best_method=report.best_method,
        selection_score=selection_score,
        state_penalized_selection_score=float(selection_score * state_penalty),
        overall_score=float(best.overall_score),
        execution_time=float(best.execution_time),
        n_repeats=int(best.n_repeats),
        ci_lower=float(best.overall_score_ci_lower or 0.0),
        ci_upper=float(best.overall_score_ci_upper or 0.0),
        methods_tested=len(report.results),
        success_rate=report.success_rate,
        n_samples_scenario=n_samples_scenario,
        state_direct_observation_rate=state_burden["state_direct_observation_rate"],
        state_supported_observation_rate=state_burden["state_supported_observation_rate"],
        state_uncertainty_burden=state_burden["state_uncertainty_burden"],
        state_non_valid_fraction=state_burden["state_non_valid_fraction"],
        state_imputed_fraction=state_burden["state_imputed_fraction"],
        recommendation_reason=report.recommendation_reason,
    )


def _write_json(
    output_path: Path,
    dataset_meta: dict[str, Any],
    summaries: list[StrategySummaryRow],
    reports: dict[str, dict[str, StageReport]],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "module": "autoselect.integration.strategy_comparison",
        "dataset": dataset_meta,
        "baseline_preprocessing": ["log_transform(base=2, offset=1)", "median_norm", "row_median"],
        "state_aware_enabled": True,
        "state_reference_policy": "baseline_layer_current_state",
        "scenario_specs": SCENARIO_SPECS,
        "summary": [row.to_dict() for row in summaries],
        "reports": {
            scenario: {name: report.to_dict() for name, report in scenario_reports.items()}
            for scenario, scenario_reports in reports.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _write_csv(output_path: Path, summaries: list[StrategySummaryRow]) -> None:
    fieldnames = [
        "scenario",
        "design_identifiability",
        "strategy",
        "best_method",
        "selection_score",
        "state_penalized_selection_score",
        "overall_score",
        "execution_time",
        "n_repeats",
        "ci_lower",
        "ci_upper",
        "methods_tested",
        "success_rate",
        "n_samples_scenario",
        "state_direct_observation_rate",
        "state_supported_observation_rate",
        "state_uncertainty_burden",
        "state_non_valid_fraction",
        "state_imputed_fraction",
        "recommendation_reason",
    ]
    with output_path.open("w", newline="") as handle:
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
    reports: dict[str, dict[str, StageReport]],
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
    lines.append("## Design Snapshot")
    lines.append("")
    lines.append("| Sample Group | Amount Group | Count |")
    lines.append("|---|---|---:|")
    for row in dataset_meta["batch_condition_counts"]:
        lines.append(f"| {row['sample_group']} | {row['amount_group']} | {row['len']} |")
    lines.append("")

    for scenario_name in SCENARIO_SPECS:
        scenario_rows = [row for row in summaries if row.scenario == scenario_name]
        if not scenario_rows:
            continue

        lines.append(f"## {scenario_name}")
        lines.append("")
        lines.append(f"- Design identifiability: `{scenario_rows[0].design_identifiability}`")
        lines.append(f"- Samples in scenario: `{scenario_rows[0].n_samples_scenario}`")
        lines.append(
            f"- State direct observation rate: "
            f"`{scenario_rows[0].state_direct_observation_rate:.2%}`"
        )
        lines.append(
            f"- State supported observation rate: "
            f"`{scenario_rows[0].state_supported_observation_rate:.2%}`"
        )
        lines.append(
            f"- State uncertainty burden: `{scenario_rows[0].state_uncertainty_burden:.2%}`"
        )
        lines.append(
            f"- State non-valid fraction: `{scenario_rows[0].state_non_valid_fraction:.2%}`"
        )
        lines.append(f"- State imputed fraction: `{scenario_rows[0].state_imputed_fraction:.2%}`")
        lines.append("")
        lines.append(
            "| Strategy | Best Method | Selection Score | State-Penalized Selection | "
            "Overall Score | Time (s) | CI (overall) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in scenario_rows:
            lines.append(
                f"| {row.strategy} | {row.best_method} | {row.selection_score:.4f} | "
                f"{row.state_penalized_selection_score:.4f} | {row.overall_score:.4f} | "
                f"{row.execution_time:.3f} | [{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
            )
        lines.append("")

        for strategy_name, report in reports[scenario_name].items():
            lines.append(f"### {strategy_name} Details")
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
                    if report.best_result is not None
                    and result.method_name == report.best_result.method_name
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

    container, dataset_meta = _build_design_synthetic_container(
        random_state=random_state,
        n_samples=n_samples,
        n_proteins=n_proteins,
    )
    container, baseline_layer = _apply_baseline_preprocessing(container)

    strategies = ["quality", "balanced", "speed"]
    reports: dict[str, dict[str, StageReport]] = {}
    summaries: list[StrategySummaryRow] = []

    for scenario_name, spec in SCENARIO_SPECS.items():
        scenario_container = _prepare_scenario_container(container, scenario_name)
        state_burden = _compute_state_burden(
            scenario_container,
            assay_name="proteins",
            layer_name=baseline_layer,
        )

        scenario_reports: dict[str, StageReport] = {}
        for strategy in strategies:
            report = _run_single_strategy(
                container=scenario_container,
                source_layer=baseline_layer,
                strategy=strategy,
                n_repeats=n_repeats,
                confidence_level=confidence_level,
                batch_key=str(spec["batch_key"]),
                bio_key=spec["bio_key"],
            )
            scenario_reports[strategy] = report
            summaries.append(
                _to_summary_row(
                    scenario=scenario_name,
                    report=report,
                    strategy=strategy,
                    n_samples_scenario=scenario_container.n_samples,
                    design_identifiability=str(spec["design_identifiability"]),
                    state_burden=state_burden,
                )
            )
        reports[scenario_name] = scenario_reports

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
            "for quality / balanced / speed presets across balanced/partial/confounded scenarios."
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
