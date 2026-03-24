"""Contract tests for the integration benchmark scenario layer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

from scptensor.core import Assay, ScpContainer, ScpMatrix


def _load_integration_benchmark_module():
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_dir = repo_root / "benchmark" / "integration"
    benchmark_path = benchmark_dir / "run_real_dia_batch_confounding.py"
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))

    spec = importlib.util.spec_from_file_location(
        "integration_benchmark_contract_module",
        benchmark_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_container() -> ScpContainer:
    sample_ids: list[str] = []
    for sample_group in ("S1", "S2", "S3"):
        for amount_group in ("300ng", "600ng"):
            for replicate in ("r1", "r2"):
                sample_ids.append(f"RUN_SCamount{amount_group}_{sample_group}_{replicate}")

    obs = pl.DataFrame({"_index": sample_ids})
    n_features = 12
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(1, n_features + 1)]})
    x = np.arange(len(sample_ids) * n_features, dtype=float).reshape(len(sample_ids), n_features)

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x))
    assay.add_layer("imputed_baseline", ScpMatrix(X=x + 1.0))

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def test_partial_scenario_is_registered_and_enabled_by_default() -> None:
    module = _load_integration_benchmark_module()

    assert "partially_confounded_bridge_sample" in module.SCENARIO_SPECS
    assert "partially_confounded_bridge_sample" in module.DEFAULT_SCENARIOS
    scenario = module.SCENARIO_SPECS["partially_confounded_bridge_sample"]
    assert scenario["score_profile"] == "partially_confounded"
    assert scenario["guardrail_expectation"] == "success"
    assert scenario["subset_strategy"] == "bridge_sample_partial_confounding"


def test_partial_scenario_builds_bridge_style_subset() -> None:
    module = _load_integration_benchmark_module()
    container = _make_container()

    module._attach_scenario_labels(container)
    partial = module._prepare_scenario_container(container, "partially_confounded_bridge_sample")

    assert partial.n_samples < container.n_samples

    sample_groups = partial.obs["sample_group"].cast(pl.Utf8).to_numpy()
    amount_groups = partial.obs["amount_group"].cast(pl.Utf8).to_numpy()

    observed_pairs = {
        batch: sorted(np.unique(amount_groups[sample_groups == batch]).tolist())
        for batch in sorted(np.unique(sample_groups).tolist())
    }

    assert observed_pairs["S1"] == ["300ng"]
    assert observed_pairs["S2"] == ["300ng", "600ng"]
    assert observed_pairs["S3"] == ["600ng"]


def test_marker_consistency_axis_is_present_for_identifiable_scenario() -> None:
    module = _load_integration_benchmark_module()
    container = _make_container()

    module._attach_scenario_labels(container)
    rows, failures, guardrail = module._run_scenario(
        container,
        dataset_key="synthetic_contract",
        assay_name="proteins",
        baseline_layer="imputed_baseline",
        scenario_name="balanced_amount_by_sample",
        methods=[],
    )

    assert failures == []
    assert len(guardrail) == 2

    baseline_row = rows[0]
    assert baseline_row["method"] == "none"
    assert "uncertainty_burden" in baseline_row
    assert "reference_uncertainty_burden" in baseline_row
    assert "delta_uncertainty_burden" in baseline_row
    assert "marker_log2fc_pearson" in baseline_row
    assert "marker_topk_jaccard" in baseline_row
    assert "marker_topk_sign_agreement" in baseline_row
    assert baseline_row["marker_log2fc_pearson"] == 1.0
    assert baseline_row["marker_topk_jaccard"] == 1.0
    assert baseline_row["marker_topk_sign_agreement"] == 1.0
