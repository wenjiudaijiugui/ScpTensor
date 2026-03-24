"""Contract tests for the state-aware imputation benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scptensor.core import MaskCode, ScpMatrix


def _load_imputation_benchmark_module():
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_dir = repo_root / "benchmark" / "imputation"
    benchmark_path = benchmark_dir / "run_benchmark.py"
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))

    spec = importlib.util.spec_from_file_location(
        "imputation_benchmark_contract_module",
        benchmark_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_dataset():
    x = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [10.5, 11.5, 12.5, 13.5],
            [20.0, 21.0, 22.0, 23.0],
            [20.5, 21.5, 22.5, 23.5],
        ],
        dtype=np.float64,
    )
    source_mask = np.array(
        [
            [MaskCode.VALID, MaskCode.LOD, MaskCode.UNCERTAIN, MaskCode.MBR],
            [MaskCode.VALID, MaskCode.LOD, MaskCode.UNCERTAIN, MaskCode.MBR],
            [MaskCode.VALID, MaskCode.VALID, MaskCode.VALID, MaskCode.VALID],
            [MaskCode.LOD, MaskCode.VALID, MaskCode.UNCERTAIN, MaskCode.VALID],
        ],
        dtype=np.int8,
    )
    sample_ids = ["S1", "S2", "S3", "S4"]
    protein_ids = ["P1", "P2", "P3", "P4"]
    groups = ["A", "A", "B", "B"]
    meta = {
        "dataset_key": "synthetic_contract",
        "shape": [4, 4],
        "missing_rate": 0.0,
        "recoverable_state_candidates": {
            "all_observed": 16,
            "valid": 9,
            "lod": 3,
            "uncertain": 3,
        },
    }
    return x, source_mask, sample_ids, protein_ids, groups, meta


def _synthetic_auxiliary_dataset():
    x = np.array(
        [
            [10.0, 12.0, 20.0, 22.0],
            [11.0, 13.0, 21.0, 23.0],
            [30.0, 32.0, 40.0, 42.0],
            [31.0, 33.0, 41.0, 43.0],
        ],
        dtype=np.float64,
    )
    source_mask = np.array(
        [
            [MaskCode.VALID, MaskCode.LOD, MaskCode.VALID, MaskCode.UNCERTAIN],
            [MaskCode.VALID, MaskCode.LOD, MaskCode.VALID, MaskCode.UNCERTAIN],
            [MaskCode.VALID, MaskCode.VALID, MaskCode.VALID, MaskCode.VALID],
            [MaskCode.VALID, MaskCode.VALID, MaskCode.LOD, MaskCode.VALID],
        ],
        dtype=np.int8,
    )
    sample_ids = ["S1", "S2", "S3", "S4"]
    precursor_ids = ["pep1", "pep2", "pep3", "pep4"]
    selected_var = __import__("polars").DataFrame(
        {
            "_index": precursor_ids,
            "EG.ProteinId": ["P1", "P1", "P2", "P2"],
        }
    )
    protein_truth = np.array(
        [
            [22.0, 42.0],
            [24.0, 44.0],
            [62.0, 82.0],
            [64.0, 84.0],
        ],
        dtype=np.float64,
    )
    protein_ids = ["P1", "P2"]
    linkage = __import__("polars").DataFrame(
        {
            "source_id": ["pep1", "pep2", "pep3", "pep4"],
            "target_id": ["P1", "P1", "P2", "P2"],
        }
    )
    groups = ["A", "A", "B", "B"]
    meta = {
        "dataset_key": "lfqbench_hye124_spectronaut",
        "shape": [4, 4],
        "protein_eval_shape": [4, 2],
        "missing_rate": 0.0,
        "aux_aggregation_method": "sum",
        "recoverable_state_candidates": {
            "all_observed": 16,
            "lod": 3,
            "uncertain": 2,
        },
    }
    return (
        x,
        source_mask,
        sample_ids,
        precursor_ids,
        groups,
        protein_truth,
        protein_ids,
        linkage,
        selected_var,
        meta,
    )


def test_generate_holdout_mask_respects_source_state_stratum() -> None:
    module = _load_imputation_benchmark_module()
    x, source_mask, *_ = _synthetic_dataset()

    holdout_mask, meta = module.generate_holdout_mask(
        x,
        source_mask,
        holdout_state="lod",
        holdout_rate=0.5,
        mechanism="mcar",
        mnar_fraction=0.75,
        mnar_low_quantile=0.3,
        random_seed=42,
    )

    assert int(np.sum(holdout_mask)) > 0
    assert np.all(source_mask[holdout_mask] == MaskCode.LOD.value)
    assert meta["holdout_state"] == "lod"
    assert meta["state_lod_fraction"] == 1.0
    assert meta["state_valid_fraction"] == 0.0
    assert float(meta["holdout_state_fraction"]) == pytest.approx(3 / 16)


def test_literature_tier_profile_enables_dense_grid_and_dual_board() -> None:
    module = _load_imputation_benchmark_module()

    profile = module._resolve_benchmark_profile(
        benchmark_tier="literature",
        datasets=None,
        methods=None,
        holdout_rates=None,
        holdout_states=None,
        mechanisms=None,
        repeats=None,
        max_features=None,
        board=None,
    )

    assert profile["board"] == "both"
    assert profile["repeats"] == 5
    assert profile["holdout_rates"] == [0.1, 0.2, 0.3, 0.5]
    assert profile["max_features"] == 2000
    assert profile["methods"] == [
        "none",
        "half_row_min",
        "row_mean",
        "knn",
        "lls",
        "iterative_svd",
    ]


def test_run_benchmark_fails_closed_on_missing_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_imputation_benchmark_module()

    monkeypatch.setattr(module, "list_impute_methods", lambda: ["softimpute"])
    monkeypatch.setattr(
        module,
        "_dependency_available",
        lambda dependency: dependency != "fancyimpute",
    )

    called: dict[str, bool] = {"main": False}

    def fake_run_main_board(**_kwargs):
        called["main"] = True
        return {}

    monkeypatch.setattr(module, "_run_main_board", fake_run_main_board)

    with pytest.raises(RuntimeError, match="softimpute -> fancyimpute"):
        module.run_benchmark(
            datasets=["pxd054343_diann_2x"],
            methods=["softimpute"],
            holdout_rates=[0.5],
            holdout_states=["all_observed"],
            mechanisms=["mcar"],
            repeats=1,
            mnar_fraction=0.75,
            mnar_low_quantile=0.3,
            normalization_method="mean",
            min_observed_per_feature=1,
            max_features=None,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "outputs",
            force_download=False,
            board="main",
            benchmark_tier="literature",
        )

    assert called["main"] is False


def test_run_benchmark_writes_state_aware_seed_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_imputation_benchmark_module()
    x, source_mask, sample_ids, protein_ids, groups, meta = _synthetic_dataset()

    def fake_load_dataset(_dataset_key: str, **_kwargs):
        return (
            x.copy(),
            source_mask.copy(),
            list(sample_ids),
            list(protein_ids),
            list(groups),
            dict(meta),
        )

    def fake_impute(
        container,
        *,
        method: str,
        assay_name: str,
        source_layer: str,
        new_layer_name: str,
        **_kwargs,
    ):
        assay = container.assays[assay_name]
        masked = assay.layers[source_layer]
        x_masked = np.asarray(masked.X, dtype=np.float64).copy()
        col_means = np.nanmean(x_masked, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing = np.isnan(x_masked)
        if np.any(missing):
            x_masked[missing] = col_means[np.where(missing)[1]]
        assay.add_layer(new_layer_name, ScpMatrix(X=x_masked, M=masked.M.copy()))
        return container

    monkeypatch.setattr(module, "_load_dataset", fake_load_dataset)
    monkeypatch.setattr(module, "list_impute_methods", lambda: ["simple_mean"])
    monkeypatch.setattr(module, "impute", fake_impute)
    monkeypatch.setattr(module, "plot_overall_scores", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_metric_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nrmse_curves", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_runtime_vs_accuracy", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "outputs"
    module.run_benchmark(
        datasets=["pxd054343_diann_2x"],
        methods=["simple_mean"],
        holdout_rates=[0.5],
        holdout_states=["all_observed", "lod", "uncertain"],
        mechanisms=["mcar"],
        repeats=2,
        mnar_fraction=0.75,
        mnar_low_quantile=0.3,
        normalization_method="mean",
        min_observed_per_feature=1,
        max_features=None,
        data_dir=tmp_path / "data",
        output_dir=output_dir,
        force_download=False,
    )

    raw_df = pd.read_csv(output_dir / "metrics_raw.csv")
    summary_df = pd.read_csv(output_dir / "metrics_summary.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert "holdout_state" in raw_df.columns
    assert "holdout_state" in summary_df.columns
    assert "holdout_rate_within_state" in raw_df.columns
    assert "nrmse_sd" in summary_df.columns
    assert raw_df["seed"].nunique() == 2
    assert set(raw_df["holdout_state"]) == {"all_observed", "lod", "uncertain"}

    runs_by_state = {
        str(row.holdout_state): int(row.runs)
        for row in summary_df.loc[:, ["holdout_state", "runs"]].itertuples(index=False)
    }
    assert runs_by_state == {
        "all_observed": 2,
        "lod": 2,
        "uncertain": 2,
    }

    assert metadata["state_aware_enabled"] is True
    assert metadata["benchmark_tier"] == "default"
    assert metadata["board_type"] == "protein_direct_state_aware_masked_recovery"
    assert metadata["config"]["holdout_states"] == ["all_observed", "lod", "uncertain"]
    assert metadata["n_skipped_holdout_scenarios"] == 0


def test_main_board_flushes_partial_outputs_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_imputation_benchmark_module()
    x, source_mask, sample_ids, protein_ids, groups, meta = _synthetic_dataset()

    def fake_load_dataset(_dataset_key: str, **_kwargs):
        return (
            x.copy(),
            source_mask.copy(),
            list(sample_ids),
            list(protein_ids),
            list(groups),
            dict(meta),
        )

    def fake_impute(
        container,
        *,
        method: str,
        assay_name: str,
        source_layer: str,
        new_layer_name: str,
        **_kwargs,
    ):
        if method == "stop_now":
            raise KeyboardInterrupt("stop benchmark")

        assay = container.assays[assay_name]
        masked = assay.layers[source_layer]
        x_masked = np.asarray(masked.X, dtype=np.float64).copy()
        col_means = np.nanmean(x_masked, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing = np.isnan(x_masked)
        if np.any(missing):
            x_masked[missing] = col_means[np.where(missing)[1]]
        assay.add_layer(new_layer_name, ScpMatrix(X=x_masked, M=masked.M.copy()))
        return container

    monkeypatch.setattr(module, "_load_dataset", fake_load_dataset)
    monkeypatch.setattr(module, "list_impute_methods", lambda: ["simple_mean", "stop_now"])
    monkeypatch.setattr(module, "impute", fake_impute)
    monkeypatch.setattr(module, "plot_overall_scores", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_metric_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nrmse_curves", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_runtime_vs_accuracy", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "partial-main"
    with pytest.raises(KeyboardInterrupt, match="stop benchmark"):
        module.run_benchmark(
            datasets=["pxd054343_diann_2x"],
            methods=["simple_mean", "stop_now"],
            holdout_rates=[0.5],
            holdout_states=["all_observed"],
            mechanisms=["mcar"],
            repeats=1,
            mnar_fraction=0.75,
            mnar_low_quantile=0.3,
            normalization_method="mean",
            min_observed_per_feature=1,
            max_features=None,
            data_dir=tmp_path / "data",
            output_dir=output_dir,
            force_download=False,
            board="main",
            benchmark_tier="literature",
        )

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    raw_df = pd.read_csv(output_dir / "metrics_raw.csv")

    assert metadata["run_status"] == "interrupted"
    assert metadata["final_error"]["type"] == "KeyboardInterrupt"
    assert len(raw_df) == 1
    assert raw_df.loc[0, "method"] == "simple_mean"
    assert bool(raw_df.loc[0, "success"]) is True


def test_auxiliary_board_writes_precursor_to_protein_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_imputation_benchmark_module()
    (
        x,
        source_mask,
        sample_ids,
        precursor_ids,
        groups,
        protein_truth,
        protein_ids,
        linkage,
        selected_var,
        meta,
    ) = _synthetic_auxiliary_dataset()

    def fake_load_auxiliary_dataset(_dataset_key: str, **_kwargs):
        return (
            x.copy(),
            source_mask.copy(),
            list(sample_ids),
            list(precursor_ids),
            list(groups),
            protein_truth.copy(),
            list(protein_ids),
            linkage.clone(),
            selected_var.clone(),
            dict(meta),
        )

    def fake_impute(
        container,
        *,
        method: str,
        assay_name: str,
        source_layer: str,
        new_layer_name: str,
        **_kwargs,
    ):
        assay = container.assays[assay_name]
        masked = assay.layers[source_layer]
        x_masked = np.asarray(masked.X, dtype=np.float64).copy()
        col_means = np.nanmean(x_masked, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing = np.isnan(x_masked)
        if np.any(missing):
            x_masked[missing] = col_means[np.where(missing)[1]]
        assay.add_layer(new_layer_name, ScpMatrix(X=x_masked, M=masked.M.copy()))
        return container

    def fake_aux_aggregate(container, *, aux_aggregation_method: str, protein_column: str):
        assert aux_aggregation_method == "sum"
        assert protein_column == "EG.ProteinId"
        assert protein_column in container.assays["peptides"].var.columns
        x_imputed = np.asarray(container.assays["peptides"].layers["imputed"].X, dtype=np.float64)
        return np.column_stack(
            [x_imputed[:, 0] + x_imputed[:, 1], x_imputed[:, 2] + x_imputed[:, 3]]
        )

    monkeypatch.setattr(module, "_load_auxiliary_dataset", fake_load_auxiliary_dataset)
    monkeypatch.setattr(module, "list_impute_methods", lambda: ["simple_mean"])
    monkeypatch.setattr(module, "impute", fake_impute)
    monkeypatch.setattr(module, "_aggregate_auxiliary_imputed_protein_matrix", fake_aux_aggregate)
    monkeypatch.setattr(module, "plot_overall_scores", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_metric_heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nrmse_curves", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_runtime_vs_accuracy", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "auxiliary"
    module.run_benchmark(
        datasets=["lfqbench_hye124_spectronaut"],
        methods=["simple_mean"],
        holdout_rates=[0.5],
        holdout_states=["lod"],
        mechanisms=["mcar"],
        repeats=2,
        mnar_fraction=0.75,
        mnar_low_quantile=0.3,
        normalization_method="mean",
        min_observed_per_feature=1,
        max_features=None,
        data_dir=tmp_path / "data",
        output_dir=output_dir,
        force_download=False,
        board="auxiliary",
        benchmark_tier="literature",
        aux_aggregation_method="sum",
    )

    raw_df = pd.read_csv(output_dir / "metrics_raw.csv")
    summary_df = pd.read_csv(output_dir / "metrics_summary.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert set(raw_df["board_scope"]) == {"auxiliary"}
    assert set(raw_df["input_level"]) == {"precursor"}
    assert set(raw_df["eval_level"]) == {"protein"}
    assert "n_holdout_source" in raw_df.columns
    assert "n_holdout_eval" in raw_df.columns
    assert int(raw_df["n_holdout_eval"].min()) > 0
    assert set(summary_df["holdout_state"]) == {"lod"}

    assert metadata["benchmark_tier"] == "literature"
    assert metadata["board_type"] == "precursor_to_protein_auxiliary_masked_recovery"
    assert metadata["config"]["aux_aggregation_method"] == "sum"
    assert metadata["config"]["datasets"] == ["lfqbench_hye124_spectronaut"]


def test_dual_board_root_metadata_survives_partial_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_imputation_benchmark_module()

    monkeypatch.setattr(module, "list_impute_methods", lambda: ["simple_mean"])

    def fake_run_main_board(*, output_dir: Path, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps({"board_type": "protein_direct_state_aware_masked_recovery"}),
            encoding="utf-8",
        )
        return {"board_type": "protein_direct_state_aware_masked_recovery"}

    def fake_run_auxiliary_board(**_kwargs):
        raise KeyboardInterrupt("stop after main")

    monkeypatch.setattr(module, "_run_main_board", fake_run_main_board)
    monkeypatch.setattr(module, "_run_auxiliary_board", fake_run_auxiliary_board)

    output_dir = tmp_path / "dual-board"
    with pytest.raises(KeyboardInterrupt, match="stop after main"):
        module.run_benchmark(
            datasets=["lfqbench_hye124_spectronaut"],
            methods=["simple_mean"],
            holdout_rates=[0.5],
            holdout_states=["all_observed"],
            mechanisms=["mcar"],
            repeats=1,
            mnar_fraction=0.75,
            mnar_low_quantile=0.3,
            normalization_method="mean",
            min_observed_per_feature=1,
            max_features=None,
            data_dir=tmp_path / "data",
            output_dir=output_dir,
            force_download=False,
            board="both",
            benchmark_tier="literature",
            aux_aggregation_method="sum",
        )

    root_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert root_metadata["run_status"] == "interrupted"
    assert root_metadata["boards"]["main"]["status"] == "completed"
    assert root_metadata["boards"]["auxiliary"]["status"] == "interrupted"
    assert root_metadata["error"]["board"] == "auxiliary"
