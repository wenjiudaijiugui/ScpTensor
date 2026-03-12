#!/usr/bin/env python3
"""Run large-scale synthetic normalization benchmark for AutoSelect.

This script generates many synthetic DIA-like protein matrices that cover
broad ranges of dataset size, missingness, distribution shape, and batch
effect strength. It then evaluates AutoSelect normalization methods:

- norm_none
- norm_mean
- norm_median
- norm_quantile
- norm_trqn

Outputs:
- synthetic_normalization_stress_results.json
- synthetic_normalization_stress_method_rows.csv
- synthetic_normalization_stress_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from benchmark.benchmark_data_generator import BenchmarkDataGenerator, get_actual_missing_rate
from scptensor.autoselect import auto_normalize
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.transformation import log_transform

METHODS = ["norm_none", "norm_mean", "norm_median", "norm_quantile", "norm_trqn"]


@dataclass
class SyntheticScenario:
    """Synthetic benchmark scenario definition."""

    scenario_id: str
    seed: int
    n_samples: int
    n_features: int
    missing_rate: float
    missing_pattern: str
    distribution: str
    with_batch_effect: bool
    n_batches: int
    batch_effect_strength: float
    batch_feature_fraction: float
    confounded_batches: bool
    batch_confounding_strength: float
    cluster_separation: float
    noise_scale: float
    mar_strength: float
    mnar_slope: float
    condition_tag: str = "stress"

    def to_config(self) -> dict[str, Any]:
        """Convert scenario to benchmark_data_generator config."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_rate": self.missing_rate,
            "missing_pattern": self.missing_pattern,
            "distribution": self.distribution,
            "with_batch_effect": self.with_batch_effect,
            "n_batches": self.n_batches,
            "batch_effect_strength": self.batch_effect_strength,
            "batch_feature_fraction": self.batch_feature_fraction,
            "confounded_batches": self.confounded_batches,
            "batch_confounding_strength": self.batch_confounding_strength,
            "cluster_separation": self.cluster_separation,
            "noise_scale": self.noise_scale,
            "mar_strength": self.mar_strength,
            "mnar_slope": self.mnar_slope,
        }


@dataclass
class ScenarioResult:
    """Result for one synthetic scenario."""

    scenario: SyntheticScenario
    actual_missing_rate: float
    best_method: str
    best_overall_score: float
    total_runtime: float
    quantile_trqn_mae: float | None
    quantile_trqn_max_abs: float | None
    quantile_trqn_identical: bool | None
    method_rows: list[dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.__dict__,
            "actual_missing_rate": self.actual_missing_rate,
            "best_method": self.best_method,
            "best_overall_score": self.best_overall_score,
            "total_runtime": self.total_runtime,
            "quantile_trqn_mae": self.quantile_trqn_mae,
            "quantile_trqn_max_abs": self.quantile_trqn_max_abs,
            "quantile_trqn_identical": self.quantile_trqn_identical,
            "method_rows": self.method_rows,
            "error": self.error,
        }


def _scenario_signature(s: SyntheticScenario) -> tuple[Any, ...]:
    """Dedup key for scenarios."""
    return (
        s.n_samples,
        s.n_features,
        round(float(s.missing_rate), 3),
        s.missing_pattern,
        s.distribution,
        s.with_batch_effect,
        s.n_batches,
        round(float(s.batch_effect_strength), 2),
        round(float(s.batch_feature_fraction), 2),
        s.confounded_batches,
        round(float(s.batch_confounding_strength), 2),
        round(float(s.cluster_separation), 2),
        round(float(s.noise_scale), 2),
        round(float(s.mar_strength), 2),
        round(float(s.mnar_slope), 2),
    )


def _boundary_scenarios(seed: int) -> list[SyntheticScenario]:
    """Deterministic boundary scenarios to guarantee range coverage."""
    return [
        SyntheticScenario(
            scenario_id="boundary_min",
            seed=seed,
            n_samples=50,
            n_features=500,
            missing_rate=0.05,
            missing_pattern="mcar",
            distribution="log_normal",
            with_batch_effect=False,
            n_batches=1,
            batch_effect_strength=0.0,
            batch_feature_fraction=0.25,
            confounded_batches=False,
            batch_confounding_strength=0.0,
            cluster_separation=1.0,
            noise_scale=0.9,
            mar_strength=1.0,
            mnar_slope=2.0,
            condition_tag="boundary",
        ),
        SyntheticScenario(
            scenario_id="boundary_wide",
            seed=seed + 1,
            n_samples=50,
            n_features=10000,
            missing_rate=0.30,
            missing_pattern="mar",
            distribution="normal",
            with_batch_effect=True,
            n_batches=3,
            batch_effect_strength=0.8,
            batch_feature_fraction=0.35,
            confounded_batches=True,
            batch_confounding_strength=0.6,
            cluster_separation=1.4,
            noise_scale=1.0,
            mar_strength=1.2,
            mnar_slope=2.0,
            condition_tag="boundary",
        ),
        SyntheticScenario(
            scenario_id="boundary_tall",
            seed=seed + 2,
            n_samples=500,
            n_features=500,
            missing_rate=0.35,
            missing_pattern="mnar",
            distribution="multimodal",
            with_batch_effect=False,
            n_batches=1,
            batch_effect_strength=0.0,
            batch_feature_fraction=0.3,
            confounded_batches=False,
            batch_confounding_strength=0.0,
            cluster_separation=1.7,
            noise_scale=1.1,
            mar_strength=1.0,
            mnar_slope=2.6,
            condition_tag="boundary",
        ),
        SyntheticScenario(
            scenario_id="boundary_max",
            seed=seed + 3,
            n_samples=500,
            n_features=10000,
            missing_rate=0.60,
            missing_pattern="mnar",
            distribution="heavy_tailed",
            with_batch_effect=True,
            n_batches=4,
            batch_effect_strength=1.4,
            batch_feature_fraction=0.45,
            confounded_batches=True,
            batch_confounding_strength=0.8,
            cluster_separation=2.1,
            noise_scale=1.3,
            mar_strength=1.1,
            mnar_slope=3.0,
            condition_tag="boundary",
        ),
    ]


def build_scenarios(max_scenarios: int, seed: int) -> list[SyntheticScenario]:
    """Build many diverse scenarios with guaranteed range coverage."""
    if max_scenarios < 4:
        raise ValueError("max_scenarios must be >= 4 to keep boundary coverage")

    sample_choices = [50, 100, 180, 260, 350, 500]
    feature_choices = [500, 1000, 2500, 5000, 7500, 10000]
    missing_rates = [0.05, 0.12, 0.20, 0.35, 0.50, 0.65]
    missing_patterns = ["mcar", "mar", "mnar"]
    distributions = ["normal", "log_normal", "multimodal", "heavy_tailed"]
    batch_strengths = [0.0, 0.3, 0.6, 1.0, 1.4]
    batch_feature_fractions = [0.2, 0.35, 0.5]
    confounding_strengths = [0.0, 0.4, 0.8]
    cluster_separations = [0.2, 0.5, 0.8, 1.4, 2.0]
    noise_scales = [0.7, 1.0, 1.3]
    mar_strengths = [0.7, 1.0, 1.4]
    mnar_slopes = [1.2, 2.0, 3.0]

    rng = np.random.default_rng(seed + 2026)
    scenarios = _boundary_scenarios(seed)
    seen = {_scenario_signature(s) for s in scenarios}

    while len(scenarios) < max_scenarios:
        batch_effect_strength = float(rng.choice(batch_strengths))
        with_batch_effect = batch_effect_strength > 0.0
        n_batches = int(rng.choice([2, 3, 4])) if with_batch_effect else 1
        confounded_batches = bool(rng.integers(0, 2)) if with_batch_effect else False
        batch_confounding_strength = (
            float(rng.choice(confounding_strengths)) if confounded_batches else 0.0
        )

        scenario = SyntheticScenario(
            scenario_id=f"scenario_{len(scenarios):03d}",
            seed=int(seed + 100 + len(scenarios)),
            n_samples=int(rng.choice(sample_choices)),
            n_features=int(rng.choice(feature_choices)),
            missing_rate=float(rng.choice(missing_rates)),
            missing_pattern=str(rng.choice(missing_patterns)),
            distribution=str(rng.choice(distributions)),
            with_batch_effect=with_batch_effect,
            n_batches=n_batches,
            batch_effect_strength=batch_effect_strength,
            batch_feature_fraction=float(rng.choice(batch_feature_fractions)),
            confounded_batches=confounded_batches,
            batch_confounding_strength=batch_confounding_strength,
            cluster_separation=float(rng.choice(cluster_separations)),
            noise_scale=float(rng.choice(noise_scales)),
            mar_strength=float(rng.choice(mar_strengths)),
            mnar_slope=float(rng.choice(mnar_slopes)),
            condition_tag="stress",
        )

        signature = _scenario_signature(scenario)
        if signature in seen:
            continue

        seen.add(signature)
        scenarios.append(scenario)

    return scenarios[:max_scenarios]


def build_literature_matched_scenarios(max_scenarios: int, seed: int) -> list[SyntheticScenario]:
    """Build condition-matched scenarios for literature-style comparisons."""
    if max_scenarios < 6:
        raise ValueError("max_scenarios must be >= 6 for literature-matched mode")

    rng = np.random.default_rng(seed + 8801)
    scenarios: list[SyntheticScenario] = []
    scenario_idx = 0

    def add_case(
        *,
        condition_tag: str,
        n_samples: int,
        n_features: int,
        missing_rate: float,
        missing_pattern: str,
        distribution: str,
        with_batch_effect: bool,
        n_batches: int,
        batch_effect_strength: float,
        confounded_batches: bool,
        cluster_separation: float,
        noise_scale: float,
    ) -> None:
        nonlocal scenario_idx
        scenarios.append(
            SyntheticScenario(
                scenario_id=f"{condition_tag}_{scenario_idx:03d}",
                seed=seed + 1000 + scenario_idx,
                n_samples=n_samples,
                n_features=n_features,
                missing_rate=missing_rate,
                missing_pattern=missing_pattern,
                distribution=distribution,
                with_batch_effect=with_batch_effect,
                n_batches=n_batches,
                batch_effect_strength=batch_effect_strength,
                batch_feature_fraction=0.35,
                confounded_batches=confounded_batches,
                batch_confounding_strength=0.65 if confounded_batches else 0.0,
                cluster_separation=cluster_separation,
                noise_scale=noise_scale,
                mar_strength=1.0 if missing_pattern != "mar" else 1.2,
                mnar_slope=2.2 if missing_pattern == "mnar" else 2.0,
                condition_tag=condition_tag,
            )
        )
        scenario_idx += 1

    # Same batch + low biological difference:
    # literature often suggests mean/median scaling suffices.
    for _ in range(max_scenarios // 3):
        add_case(
            condition_tag="same_batch_low_diff",
            n_samples=int(rng.choice([80, 120, 180, 260])),
            n_features=int(rng.choice([500, 1000, 2500, 5000])),
            missing_rate=float(rng.choice([0.05, 0.10, 0.15, 0.20])),
            missing_pattern=str(rng.choice(["mcar", "mar"])),
            distribution="log_normal",
            with_batch_effect=False,
            n_batches=1,
            batch_effect_strength=0.0,
            confounded_batches=False,
            cluster_separation=float(rng.choice([0.05, 0.10, 0.20, 0.30])),
            noise_scale=float(rng.choice([0.7, 0.9, 1.1])),
        )

    # Same batch + stronger biological contrast.
    for _ in range(max_scenarios // 3):
        add_case(
            condition_tag="same_batch_higher_diff",
            n_samples=int(rng.choice([80, 120, 180, 260, 350])),
            n_features=int(rng.choice([500, 1000, 2500, 5000, 7500])),
            missing_rate=float(rng.choice([0.05, 0.12, 0.20, 0.35])),
            missing_pattern=str(rng.choice(["mcar", "mar", "mnar"])),
            distribution=str(rng.choice(["normal", "log_normal", "multimodal"])),
            with_batch_effect=False,
            n_batches=1,
            batch_effect_strength=0.0,
            confounded_batches=False,
            cluster_separation=float(rng.choice([0.8, 1.2, 1.6])),
            noise_scale=float(rng.choice([0.8, 1.0, 1.2])),
        )

    # Multi-batch / confounded settings where robust methods may help.
    while len(scenarios) < max_scenarios:
        add_case(
            condition_tag="multi_batch_confounded",
            n_samples=int(rng.choice([100, 180, 260, 350, 500])),
            n_features=int(rng.choice([1000, 2500, 5000, 7500, 10000])),
            missing_rate=float(rng.choice([0.12, 0.20, 0.35, 0.50, 0.65])),
            missing_pattern=str(rng.choice(["mar", "mnar"])),
            distribution=str(rng.choice(["normal", "log_normal", "heavy_tailed"])),
            with_batch_effect=True,
            n_batches=int(rng.choice([2, 3, 4])),
            batch_effect_strength=float(rng.choice([0.6, 1.0, 1.4])),
            confounded_batches=bool(rng.integers(0, 2)),
            cluster_separation=float(rng.choice([0.5, 1.0, 1.6])),
            noise_scale=float(rng.choice([0.9, 1.1, 1.3])),
        )

    return scenarios[:max_scenarios]


def _to_dense_array(container: ScpContainer, layer_name: str) -> np.ndarray | None:
    assay = container.assays.get("proteins")
    if assay is None or layer_name not in assay.layers:
        return None

    x = assay.layers[layer_name].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=float)


def _build_same_batch_low_diff_container(scenario: SyntheticScenario) -> ScpContainer:
    """Build a literature-matched same-batch low-difference dataset.

    Design:
    - single batch (no inter-batch offsets),
    - dominant sample-level global shift,
    - very weak biological contrast.
    This setting is where mean/median scaling is commonly preferred.
    """
    rng = np.random.default_rng(scenario.seed)
    n_samples = scenario.n_samples
    n_features = scenario.n_features

    groups = np.arange(n_samples) % 2
    rng.shuffle(groups)

    base_log2 = rng.normal(24.0, 1.0 * scenario.noise_scale, size=(n_samples, n_features))
    sample_shift = rng.normal(0.0, 0.6, size=(n_samples, 1))
    x_log2 = (
        base_log2
        + sample_shift
        + rng.normal(0.0, 0.15 * scenario.noise_scale, size=(n_samples, n_features))
    )

    marker_count = min(n_features, max(20, n_features // 120))
    marker_idx = rng.choice(n_features, size=marker_count, replace=False)
    weak_effect = 0.03 + 0.12 * float(scenario.cluster_separation)
    x_log2[np.ix_(groups == 1, marker_idx)] += weak_effect

    x = np.exp2(np.clip(x_log2, a_min=0.0, a_max=None))

    if scenario.missing_pattern == "mcar":
        missing_mask = rng.random(size=x.shape) < scenario.missing_rate
    elif scenario.missing_pattern == "mar":
        sample_mean = np.mean(x_log2, axis=1, keepdims=True)
        sample_score = (sample_mean - np.mean(sample_mean)) / (np.std(sample_mean) + 1e-8)
        probs = np.clip(
            scenario.missing_rate * (1.0 + 0.5 * np.tanh(-sample_score)),
            0.0,
            0.95,
        )
        missing_mask = rng.random(size=x.shape) < probs
    else:
        logits = -(x_log2 - np.mean(x_log2)) / (np.std(x_log2) + 1e-8)
        probs = np.clip(scenario.missing_rate * (1.0 + 0.3 * np.tanh(logits)), 0.0, 0.95)
        missing_mask = rng.random(size=x.shape) < probs
    x[missing_mask] = np.nan

    obs = pl.DataFrame(
        {
            "_index": [f"Cell_{i:05d}" for i in range(n_samples)],
            "batch": ["Batch_0"] * n_samples,
            "cell_type": [f"Type_{g}" for g in groups],
        }
    )
    var = pl.DataFrame(
        {
            "_index": [f"Prot_{j:05d}" for j in range(n_features)],
            "protein": [f"Prot_{j:05d}" for j in range(n_features)],
        }
    )

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x))
    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def _quantile_trqn_diagnostics(
    container: ScpContainer,
    method_rows: list[dict[str, Any]],
) -> tuple[float | None, float | None, bool | None]:
    """Compute matrix-level difference between quantile and TRQN outputs."""
    quantile_row = next((row for row in method_rows if row["method"] == "norm_quantile"), None)
    trqn_row = next((row for row in method_rows if row["method"] == "norm_trqn"), None)

    if quantile_row is None or trqn_row is None:
        return None, None, None
    if quantile_row.get("error") or trqn_row.get("error"):
        return None, None, None

    q_layer = str(quantile_row["layer_name"])
    t_layer = str(trqn_row["layer_name"])
    qx = _to_dense_array(container, q_layer)
    tx = _to_dense_array(container, t_layer)
    if qx is None or tx is None:
        return None, None, None

    valid = np.isfinite(qx) & np.isfinite(tx)
    if not np.any(valid):
        return None, None, None

    diff = np.abs(qx[valid] - tx[valid])
    mae = float(np.mean(diff))
    max_abs = float(np.max(diff))
    identical = bool(max_abs < 1e-10)
    return mae, max_abs, identical


def run_single_scenario(
    scenario: SyntheticScenario,
    selection_strategy: str,
    n_repeats: int,
) -> ScenarioResult:
    """Generate synthetic dataset and benchmark normalization methods."""
    if scenario.condition_tag == "same_batch_low_diff":
        container = _build_same_batch_low_diff_container(scenario)
    else:
        generator = BenchmarkDataGenerator(seed=scenario.seed)
        container = generator.generate_from_config(scenario.to_config())
    actual_missing_rate = get_actual_missing_rate(container)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Source layer 'raw' appears already log-transformed.*",
            category=UserWarning,
        )
        container = log_transform(
            container,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log2",
            base=2.0,
            offset=1.0,
            detect_logged=True,
            skip_if_logged=True,
        )

    start_time = time.perf_counter()
    result_container, report = auto_normalize(
        container,
        assay_name="proteins",
        source_layer="log2",
        keep_all=True,
        selection_strategy=selection_strategy,
        n_repeats=n_repeats,
    )
    runtime = time.perf_counter() - start_time

    method_rows: list[dict[str, Any]] = []
    for result in report.results:
        row = {
            "method": result.method_name,
            "layer_name": result.layer_name,
            "error": result.error,
            "overall_score": float(result.overall_score),
            "selection_score": (
                None if result.selection_score is None else float(result.selection_score)
            ),
            "execution_time": float(result.execution_time),
            "batch_removal": float(result.scores.get("batch_removal", 0.0)),
            "bio_conservation": float(result.scores.get("bio_conservation", 0.0)),
            "technical_quality": float(result.scores.get("technical_quality", 0.0)),
            "balance_score": float(result.scores.get("balance_score", 0.0)),
        }
        method_rows.append(row)

    mae, max_abs, identical = _quantile_trqn_diagnostics(result_container, method_rows)

    best_method = report.best_method
    best_overall_score = float(report.best_result.overall_score) if report.best_result else 0.0

    return ScenarioResult(
        scenario=scenario,
        actual_missing_rate=float(actual_missing_rate),
        best_method=best_method,
        best_overall_score=best_overall_score,
        total_runtime=float(runtime),
        quantile_trqn_mae=mae,
        quantile_trqn_max_abs=max_abs,
        quantile_trqn_identical=identical,
        method_rows=method_rows,
        error=None,
    )


def _method_aggregate(results: list[ScenarioResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-method average scores."""
    buckets: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    for result in results:
        if result.error is not None:
            continue
        for row in result.method_rows:
            method = str(row["method"])
            if method in buckets and not row.get("error"):
                buckets[method].append(row)

    aggregated: dict[str, dict[str, float]] = {}
    for method, rows in buckets.items():
        if not rows:
            aggregated[method] = {
                "overall_score": 0.0,
                "batch_removal": 0.0,
                "bio_conservation": 0.0,
                "technical_quality": 0.0,
                "balance_score": 0.0,
                "execution_time": 0.0,
            }
            continue

        aggregated[method] = {
            "overall_score": float(np.mean([float(r["overall_score"]) for r in rows])),
            "batch_removal": float(np.mean([float(r["batch_removal"]) for r in rows])),
            "bio_conservation": float(np.mean([float(r["bio_conservation"]) for r in rows])),
            "technical_quality": float(np.mean([float(r["technical_quality"]) for r in rows])),
            "balance_score": float(np.mean([float(r["balance_score"]) for r in rows])),
            "execution_time": float(np.mean([float(r["execution_time"]) for r in rows])),
        }

    return aggregated


def _write_outputs(
    output_dir: Path, results: list[ScenarioResult], args: argparse.Namespace
) -> None:
    """Write benchmark artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "module": "autoselect.synthetic_normalization_stress",
        "config": {
            "max_scenarios": args.max_scenarios,
            "seed": args.seed,
            "selection_strategy": args.selection_strategy,
            "n_repeats": args.n_repeats,
        },
        "results": [r.to_dict() for r in results],
        "method_aggregates": _method_aggregate(results),
    }

    json_path = output_dir / "synthetic_normalization_stress_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "synthetic_normalization_stress_method_rows.csv"
    fieldnames = [
        "scenario_id",
        "condition_tag",
        "seed",
        "n_samples",
        "n_features",
        "missing_rate_target",
        "missing_rate_actual",
        "missing_pattern",
        "distribution",
        "with_batch_effect",
        "n_batches",
        "batch_effect_strength",
        "batch_feature_fraction",
        "confounded_batches",
        "batch_confounding_strength",
        "cluster_separation",
        "noise_scale",
        "mar_strength",
        "mnar_slope",
        "best_method",
        "method",
        "error",
        "overall_score",
        "selection_score",
        "batch_removal",
        "bio_conservation",
        "technical_quality",
        "balance_score",
        "execution_time",
        "quantile_trqn_mae",
        "quantile_trqn_max_abs",
        "quantile_trqn_identical",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result.method_rows:
                writer.writerow(
                    {
                        "scenario_id": result.scenario.scenario_id,
                        "condition_tag": result.scenario.condition_tag,
                        "seed": result.scenario.seed,
                        "n_samples": result.scenario.n_samples,
                        "n_features": result.scenario.n_features,
                        "missing_rate_target": result.scenario.missing_rate,
                        "missing_rate_actual": result.actual_missing_rate,
                        "missing_pattern": result.scenario.missing_pattern,
                        "distribution": result.scenario.distribution,
                        "with_batch_effect": result.scenario.with_batch_effect,
                        "n_batches": result.scenario.n_batches,
                        "batch_effect_strength": result.scenario.batch_effect_strength,
                        "batch_feature_fraction": result.scenario.batch_feature_fraction,
                        "confounded_batches": result.scenario.confounded_batches,
                        "batch_confounding_strength": result.scenario.batch_confounding_strength,
                        "cluster_separation": result.scenario.cluster_separation,
                        "noise_scale": result.scenario.noise_scale,
                        "mar_strength": result.scenario.mar_strength,
                        "mnar_slope": result.scenario.mnar_slope,
                        "best_method": result.best_method,
                        "method": row["method"],
                        "error": row.get("error"),
                        "overall_score": row.get("overall_score"),
                        "selection_score": row.get("selection_score"),
                        "batch_removal": row.get("batch_removal"),
                        "bio_conservation": row.get("bio_conservation"),
                        "technical_quality": row.get("technical_quality"),
                        "balance_score": row.get("balance_score"),
                        "execution_time": row.get("execution_time"),
                        "quantile_trqn_mae": result.quantile_trqn_mae,
                        "quantile_trqn_max_abs": result.quantile_trqn_max_abs,
                        "quantile_trqn_identical": result.quantile_trqn_identical,
                    }
                )

    md_path = output_dir / "synthetic_normalization_stress_summary.md"
    method_agg = payload["method_aggregates"]
    if not isinstance(method_agg, dict):
        raise TypeError("payload['method_aggregates'] must be a dict")
    best_counts: dict[str, int] = {}
    for result in results:
        best_counts[result.best_method] = best_counts.get(result.best_method, 0) + 1

    condition_wins: dict[str, dict[str, int]] = {}
    condition_method_scores: dict[str, dict[str, list[float]]] = {}
    for result in results:
        tag = result.scenario.condition_tag
        condition_wins.setdefault(tag, {})
        condition_wins[tag][result.best_method] = condition_wins[tag].get(result.best_method, 0) + 1
        condition_method_scores.setdefault(tag, {m: [] for m in METHODS})
        for row in result.method_rows:
            if row.get("error"):
                continue
            method = str(row["method"])
            if method in condition_method_scores[tag]:
                condition_method_scores[tag][method].append(float(row["overall_score"]))

    identical_cases = [r for r in results if r.quantile_trqn_identical is True]

    lines: list[str] = []
    lines.append("# Synthetic Normalization Stress Benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append(f"- Scenario count: `{len(results)}`")
    lines.append(
        "- Coverage: rows `50-500`, columns `500-10000`, varying missingness/distributions/batch effects"
    )
    lines.append(f"- Selection strategy: `{args.selection_strategy}`")
    lines.append(f"- Repeats per method: `{args.n_repeats}`")
    lines.append("")

    lines.append("## Best Method Frequency")
    lines.append("")
    lines.append("| method | count |")
    lines.append("|---|---:|")
    for method, count in sorted(best_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {method} | {count} |")
    lines.append("")

    lines.append("## Condition-Matched Winner Frequency")
    lines.append("")
    lines.append("| condition | method | count |")
    lines.append("|---|---|---:|")
    for tag in sorted(condition_wins):
        block = condition_wins[tag]
        for method, count in sorted(block.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| {tag} | {method} | {count} |")
    lines.append("")

    lines.append("## Mean Method Scores")
    lines.append("")
    lines.append(
        "| method | overall | batch_removal | bio_conservation | technical_quality | balance | time(s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        agg = method_agg.get(method, {})
        lines.append(
            "| "
            f"{method} | {float(agg.get('overall_score', 0.0)):.4f}"
            f" | {float(agg.get('batch_removal', 0.0)):.4f}"
            f" | {float(agg.get('bio_conservation', 0.0)):.4f}"
            f" | {float(agg.get('technical_quality', 0.0)):.4f}"
            f" | {float(agg.get('balance_score', 0.0)):.4f}"
            f" | {float(agg.get('execution_time', 0.0)):.3f} |"
        )
    lines.append("")

    lines.append("## Condition-Matched Mean Overall")
    lines.append("")
    lines.append("| condition | method | mean_overall |")
    lines.append("|---|---|---:|")
    for tag in sorted(condition_method_scores):
        for method in METHODS:
            vals = condition_method_scores[tag].get(method, [])
            mean_score = float(np.mean(vals)) if vals else 0.0
            lines.append(f"| {tag} | {method} | {mean_score:.4f} |")
    lines.append("")

    lines.append("## Quantile vs TRQN")
    lines.append("")
    lines.append(f"- Matrix-identical scenarios: `{len(identical_cases)}/{len(results)}`")
    lines.append(
        "- Identical means quantile/trqn outputs are numerically the same in that scenario; "
        "if scores are equal but matrices differ, it indicates evaluator sensitivity limits."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute synthetic stress benchmark end-to-end."""
    if args.scenario_mode == "literature_matched":
        scenarios = build_literature_matched_scenarios(
            max_scenarios=args.max_scenarios,
            seed=args.seed,
        )
    else:
        scenarios = build_scenarios(max_scenarios=args.max_scenarios, seed=args.seed)
    results: list[ScenarioResult] = []

    print(f"Running {len(scenarios)} scenarios...")
    for idx, scenario in enumerate(scenarios, start=1):
        print(
            f"[{idx:03d}/{len(scenarios)}] {scenario.scenario_id}: "
            f"{scenario.n_samples}x{scenario.n_features}, "
            f"missing={scenario.missing_rate:.2f}/{scenario.missing_pattern}, "
            f"dist={scenario.distribution}, batch={scenario.batch_effect_strength:.2f}"
        )

        try:
            result = run_single_scenario(
                scenario,
                selection_strategy=args.selection_strategy,
                n_repeats=args.n_repeats,
            )
            print(
                f"  best={result.best_method}, score={result.best_overall_score:.4f}, "
                f"time={result.total_runtime:.2f}s"
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    actual_missing_rate=0.0,
                    best_method="ERROR",
                    best_overall_score=0.0,
                    total_runtime=0.0,
                    quantile_trqn_mae=None,
                    quantile_trqn_max_abs=None,
                    quantile_trqn_identical=None,
                    method_rows=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    _write_outputs(args.output_dir, results, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "autoselect" / "synthetic_normalization_stress",
        help="Output directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=48,
        help="Number of synthetic scenarios to run (>=4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="literature_matched",
        choices=["literature_matched", "stress"],
        help="Scenario generation mode.",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="quality",
        choices=["quality", "balanced", "speed"],
        help="AutoSelect strategy.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Repeat count per method for AutoSelect evaluator.",
    )
    args = parser.parse_args()

    min_required = 6 if args.scenario_mode == "literature_matched" else 4
    if args.max_scenarios < min_required:
        raise ValueError(f"--max-scenarios must be >= {min_required}")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")

    run_benchmark(args)


if __name__ == "__main__":
    main()
