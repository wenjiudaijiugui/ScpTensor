"""Regression tests for benchmark-local sidecar import isolation."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_entrypoints_keep_local_metrics_and_plots_isolated() -> None:
    imputation = _load_module(
        "benchmark/imputation/run_benchmark.py",
        "benchmark_imputation_run_contract",
    )
    integration = _load_module(
        "benchmark/integration/run_real_dia_batch_confounding.py",
        "benchmark_integration_run_contract",
    )
    normalization = _load_module(
        "benchmark/normalization/run_benchmark.py",
        "benchmark_normalization_run_contract",
    )
    aggregation = _load_module(
        "benchmark/aggregation/run_benchmark.py",
        "benchmark_aggregation_run_contract",
    )

    assert callable(imputation.compute_reconstruction_metrics)
    assert hasattr(integration, "BALANCED_METRIC_DIRECTIONS")
    assert callable(normalization.compute_distribution_metrics)
    assert callable(aggregation.summarize_method)


def test_imputation_benchmark_module_does_not_depend_on_installed_benchmark_package() -> None:
    original = sys.modules.get("benchmark")
    sys.modules["benchmark"] = types.ModuleType("benchmark")
    try:
        imputation = _load_module(
            "benchmark/imputation/run_benchmark.py",
            "benchmark_imputation_run_contract_poisoned",
        )
    finally:
        if original is None:
            sys.modules.pop("benchmark", None)
        else:
            sys.modules["benchmark"] = original

    assert callable(imputation.compute_ratio_metrics)
