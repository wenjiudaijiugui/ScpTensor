"""Unit tests for benchmark metric extension helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module(relative_dir: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_dir = repo_root / "benchmark" / relative_dir
    module_path = benchmark_dir / "metrics.py"
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imputation_de_extensions_return_expected_keys() -> None:
    module = _load_module("imputation", "benchmark_imputation_metrics")

    matrix = np.array(
        [
            [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            [10.5, 11.5, 12.5, 20.5, 21.5, 22.5],
            [30.0, 31.0, 32.0, 40.0, 41.0, 42.0],
            [30.5, 31.5, 32.5, 40.5, 41.5, 42.5],
        ],
        dtype=np.float64,
    )
    groups = ["A", "A", "B", "B"]

    metrics = module.compute_de_consistency_metrics(matrix, matrix, groups, top_k=2)

    assert "de_topk_f1" in metrics
    assert "de_pauc_01" in metrics
    assert "de_pauc_05" in metrics
    assert "de_pauc_10" in metrics
    assert metrics["de_topk_f1"] == 1.0
    assert metrics["de_topk_jaccard"] == 1.0
    assert metrics["de_topk_sign_agreement"] == 1.0


def test_aggregation_mapping_and_state_extensions() -> None:
    module = _load_module("aggregation", "benchmark_aggregation_metrics")

    mapping = module.summarize_mapping_burden(["P1", "P2;P3", "P4|P5|P6", None])
    assert mapping["ambiguous_mapping_fraction"] == 2 / 3
    assert mapping["mapping_targets_per_peptide_mean"] == 2.0

    mask = np.array(
        [
            [0, 0, 2],
            [0, 6, 2],
        ],
        dtype=np.int8,
    )
    state = module.summarize_state_burden(mask, shape=mask.shape)
    assert state["state_valid_fraction"] == 0.5
    assert state["state_non_valid_fraction"] == 0.5
    assert state["state_lod_fraction"] == 2 / 6
    assert state["state_uncertain_fraction"] == 1 / 6

    quantified = __import__("pandas").DataFrame(
        {
            "species": ["HUMAN", "YEAST", "ECOLI"],
            "expected_log2_fc_ab": [0.0, 1.0, -2.0],
            "log2_fc_ab": [0.1, 0.8, -1.9],
        }
    )
    de_proxy = module.summarize_de_consistency_proxy(quantified)
    assert de_proxy["de_changed_direction_accuracy"] == 1.0
    assert de_proxy["de_background_stability_rate"] == 1.0
    assert de_proxy["de_consistency_score"] == 1.0
