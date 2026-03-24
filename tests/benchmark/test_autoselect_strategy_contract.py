"""Contract tests for AutoSelect strategy comparison benchmark scenarios."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_strategy_module():
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_dir = repo_root / "benchmark" / "autoselect"
    benchmark_path = benchmark_dir / "run_strategy_comparison.py"
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))

    spec = importlib.util.spec_from_file_location(
        "autoselect_strategy_contract_module",
        benchmark_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strategy_benchmark_registers_three_design_scenarios() -> None:
    module = _load_strategy_module()

    assert set(module.SCENARIO_SPECS.keys()) == {
        "balanced_amount_by_sample",
        "partially_confounded_bridge_sample",
        "confounded_amount_as_batch",
    }
    assert module.SCENARIO_SPECS["confounded_amount_as_batch"]["bio_key"] is None


def test_partial_strategy_scenario_keeps_bridge_batch() -> None:
    module = _load_strategy_module()
    container, _ = module._build_design_synthetic_container(
        random_state=42,
        n_samples=60,
        n_proteins=80,
    )

    partial = module._prepare_scenario_container(container, "partially_confounded_bridge_sample")
    sample_groups = partial.obs["sample_group"].to_numpy().astype(str)
    amount_groups = partial.obs["amount_group"].to_numpy().astype(str)

    observed_pairs = {
        batch: sorted(np.unique(amount_groups[sample_groups == batch]).tolist())
        for batch in sorted(np.unique(sample_groups).tolist())
    }
    assert observed_pairs["S1"] == ["300ng"]
    assert observed_pairs["S2"] == ["300ng", "600ng"]
    assert observed_pairs["S3"] == ["600ng"]


def test_state_burden_summary_fields_are_serializable() -> None:
    module = _load_strategy_module()
    row = module.StrategySummaryRow(
        scenario="balanced_amount_by_sample",
        design_identifiability="fully_identifiable_balanced",
        strategy="balanced",
        best_method="limma",
        selection_score=0.8,
        state_penalized_selection_score=0.6,
        overall_score=0.75,
        execution_time=0.1,
        n_repeats=3,
        ci_lower=0.7,
        ci_upper=0.8,
        methods_tested=2,
        success_rate=1.0,
        n_samples_scenario=40,
        state_direct_observation_rate=0.8,
        state_supported_observation_rate=0.9,
        state_uncertainty_burden=0.15,
        state_non_valid_fraction=0.2,
        state_imputed_fraction=0.15,
        recommendation_reason="contract test",
    )
    payload = row.to_dict()

    assert payload["scenario"] == "balanced_amount_by_sample"
    assert payload["state_penalized_selection_score"] == 0.6
    assert payload["state_direct_observation_rate"] == 0.8
    assert payload["state_supported_observation_rate"] == 0.9
    assert payload["state_uncertainty_burden"] == 0.15
    assert payload["state_non_valid_fraction"] == 0.2
    assert payload["state_imputed_fraction"] == 0.15
