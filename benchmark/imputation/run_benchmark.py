"""Run benchmark for protein-level imputation methods."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests
import scipy.sparse as sp

from scptensor.aggregation import aggregate_to_protein
from scptensor.core import Assay, MaskCode, MatrixMetadata, ScpContainer, ScpMatrix
from scptensor.impute import impute
from scptensor.impute.base import list_impute_methods
from scptensor.io import load_diann, load_spectronaut
from scptensor.normalization import normalize
from scptensor.transformation import log_transform

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _SCRIPT_DIR.parent
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

load_sidecar_module = import_module("benchmark_local_import").load_sidecar_module

_metrics_module = load_sidecar_module(__file__, "metrics")
_plots_module = load_sidecar_module(__file__, "plots")
_normalization_metrics_module = load_sidecar_module(
    __file__,
    "metrics",
    relative_dir=Path("..") / "normalization",
)

compute_cluster_metrics = _metrics_module.compute_cluster_metrics
compute_de_consistency_metrics = _metrics_module.compute_de_consistency_metrics
compute_reconstruction_metrics = _metrics_module.compute_reconstruction_metrics
compute_within_group_cv_median = _metrics_module.compute_within_group_cv_median
score_methods = _metrics_module.score_methods
summarize_holdout_state_profile = _metrics_module.summarize_holdout_state_profile

EXPECTED_LOG2FC_HYE124 = _normalization_metrics_module.EXPECTED_LOG2FC_HYE124
compute_ratio_metrics = _normalization_metrics_module.compute_ratio_metrics

plot_metric_heatmap = _plots_module.plot_metric_heatmap
plot_nrmse_curves = _plots_module.plot_nrmse_curves
plot_overall_scores = _plots_module.plot_overall_scores
plot_runtime_vs_accuracy = _plots_module.plot_runtime_vs_accuracy

DEFAULT_METHODS = [
    "none",
    "half_row_min",
    "row_mean",
    "knn",
    "lls",
    "iterative_svd",
    "softimpute",
    "missforest",
]
LITERATURE_DEFAULT_METHODS = [
    "none",
    "half_row_min",
    "row_mean",
    "knn",
    "lls",
    "iterative_svd",
]
OPTIONAL_METHOD_DEPENDENCIES = {
    "softimpute": "fancyimpute",
}
BOARD_CHECKPOINT_SCENARIO_INTERVAL = 5

DEFAULT_DATASETS = ["pxd054343_diann_2x", "lfqbench_hye124_spectronaut"]
DEFAULT_HOLDOUT_RATES = [0.1, 0.3, 0.5]
DEFAULT_MECHANISMS = ["mcar", "mixed_mnar"]
DEFAULT_HOLDOUT_STATES = ["all_observed", "valid", "mbr", "lod", "uncertain"]
DEFAULT_MAX_FEATURES = 1200
DEFAULT_BOARD = "main"
DEFAULT_BENCHMARK_TIER = "default"
DEFAULT_AUX_AGGREGATION_METHOD = "sum"

BOARD_CHOICES = ["main", "auxiliary", "both"]
BENCHMARK_TIER_CHOICES = ["smoke", "default", "literature"]

HOLDOUT_STATE_CODES: dict[str, int] = {
    "valid": int(MaskCode.VALID.value),
    "mbr": int(MaskCode.MBR.value),
    "lod": int(MaskCode.LOD.value),
    "filtered": int(MaskCode.FILTERED.value),
    "outlier": int(MaskCode.OUTLIER.value),
    "uncertain": int(MaskCode.UNCERTAIN.value),
}
HOLDOUT_STATE_CHOICES = ["all_observed", *HOLDOUT_STATE_CODES.keys()]
SUMMARY_GROUP_COLUMNS = ["dataset", "method", "mechanism", "holdout_rate", "holdout_state"]

TIER_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": ["lfqbench_hye124_spectronaut"],
        "methods": ["none", "knn"],
        "holdout_rates": [0.1],
        "holdout_states": ["all_observed", "lod"],
        "mechanisms": ["mcar"],
        "repeats": 1,
        "max_features": 250,
        "board": "main",
    },
    "default": {
        "datasets": DEFAULT_DATASETS,
        "methods": DEFAULT_METHODS,
        "holdout_rates": DEFAULT_HOLDOUT_RATES,
        "holdout_states": DEFAULT_HOLDOUT_STATES,
        "mechanisms": DEFAULT_MECHANISMS,
        "repeats": 1,
        "max_features": DEFAULT_MAX_FEATURES,
        "board": DEFAULT_BOARD,
    },
    "literature": {
        "datasets": DEFAULT_DATASETS,
        "methods": LITERATURE_DEFAULT_METHODS,
        "holdout_rates": [0.1, 0.2, 0.3, 0.5],
        "holdout_states": DEFAULT_HOLDOUT_STATES,
        "mechanisms": DEFAULT_MECHANISMS,
        "repeats": 5,
        "max_features": 2000,
        "board": "both",
    },
}

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "pxd054343_diann_2x": {
        "kind": "diann_protein_long_local",
        "path": "data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv",
        "quantity_column": "PG.Quantity",
        "group_regex": r"_(S\d+)_",
        "supported_boards": ["main"],
        "references": {
            "paper": "https://www.nature.com/articles/s41467-025-65174-4",
            "proteomexchange": "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD054343",
        },
    },
    "lfqbench_hye124_spectronaut": {
        "kind": "spectronaut_peptide_long",
        "url": (
            "https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/"
            "vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv"
        ),
        "filename": "Spectronaut_TTOF6600_64w_example.tsv",
        "quantity_column": "FG.NormalizedTotalPeakArea",
        "protein_column": "EG.ProteinId",
        "sample_column": "R.FileName",
        "group_column": "R.Condition",
        "group_a": "A",
        "group_b": "B",
        "expected_log2_fc": EXPECTED_LOG2FC_HYE124,
        "supported_boards": ["main", "auxiliary"],
        "references": {
            "paper": "https://doi.org/10.1021/acs.jproteome.5b00791",
            "dataset_repo": "https://github.com/IFIproteomics/LFQbench",
            "raw_table": (
                "https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/"
                "vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv"
            ),
        },
    },
}

LITERATURE_REFERENCES = {
    "dia_sc_workflow_benchmark": "https://www.nature.com/articles/s41467-025-65174-4",
    "multpro_dia_missing_benchmark": "https://pubmed.ncbi.nlm.nih.gov/40947414/",
    "naguidr_proteomics_impute_benchmark": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/",
    "pimms_single_cell_imputation": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10949645/",
}

METHOD_KWARGS: dict[str, dict[str, Any]] = {
    "half_row_min": {"fraction": 0.5},
    "knn": {"k": 5, "weights": "distance"},
    "lls": {"k": 10, "max_iter": 50, "tol": 1e-5},
    "iterative_svd": {"n_components": 5, "max_iter": 80, "tol": 1e-5},
    "softimpute": {"rank": 5, "max_iter": 80, "convergence_threshold": 1e-5},
    "missforest": {"max_iter": 8, "n_estimators": 60, "n_jobs": -1},
    "qrilc": {"q": 0.01},
    "minprob": {"sigma": 2.0, "q": 0.01},
}


def _download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    return destination


def _normalize_line_endings_if_needed(path: Path) -> Path:
    with path.open("rb") as fh:
        sample = fh.read(2 * 1024 * 1024)

    if b"\n" in sample or b"\r" not in sample:
        return path

    normalized = path.with_name(f"{path.stem}.normalized{path.suffix}")
    if normalized.exists() and normalized.stat().st_mtime >= path.stat().st_mtime:
        return normalized

    with path.open("rb") as src, normalized.open("wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk.replace(b"\r", b"\n"))

    return normalized


def _extract_sample_group_mapping(
    path: Path,
    *,
    sample_column: str,
    group_column: str,
) -> dict[str, str]:
    table = pl.read_csv(
        path,
        separator="\t",
        columns=[sample_column, group_column],
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        infer_schema_length=0,
    ).with_columns(
        pl.col(sample_column)
        .cast(pl.Utf8, strict=False)
        .str.replace(r"(?i)\.raw$", "")
        .str.strip_chars()
        .alias("_sample_id"),
        pl.col(group_column).cast(pl.Utf8, strict=False).str.strip_chars().alias("_group"),
    )

    pairs = table.select(["_sample_id", "_group"]).drop_nulls().unique()
    duplicates = (
        pairs.group_by("_sample_id")
        .agg(pl.col("_group").n_unique().alias("n_groups"))
        .filter(pl.col("n_groups") > 1)
    )
    if duplicates.height > 0:
        bad = duplicates["_sample_id"].to_list()
        raise ValueError(f"Sample IDs map to multiple groups: {bad[:5]}")

    return {
        row["_sample_id"]: row["_group"]
        for row in pairs.select(["_sample_id", "_group"]).to_dicts()
    }


def _infer_groups_by_regex(sample_ids: list[str], pattern: str) -> list[str] | None:
    regex = re.compile(pattern)
    groups: list[str] = []
    for sid in sample_ids:
        match = regex.search(str(sid))
        groups.append(match.group(1) if match else "UNKNOWN")

    uniq = sorted(set(groups))
    if len([g for g in uniq if g != "UNKNOWN"]) < 2:
        return None

    counts = {grp: groups.count(grp) for grp in uniq if grp != "UNKNOWN"}
    if not counts or min(counts.values()) < 2:
        return None

    return groups


def _safe_nanvar_matrix(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape[1], dtype=np.float64)
    finite = np.isfinite(x)
    for j in range(x.shape[1]):
        vals = x[finite[:, j], j]
        if vals.size >= 2:
            out[j] = float(np.var(vals, ddof=1))
    return out


def _dense_mask_array(
    mask: np.ndarray | sp.spmatrix | None,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.int8)
    if sp.issparse(mask):
        return mask.toarray().astype(np.int8, copy=False)
    return np.asarray(mask, dtype=np.int8)


def _select_feature_indices(
    x: np.ndarray,
    *,
    min_observed_per_feature: int,
    max_features: int | None,
) -> np.ndarray:
    finite = np.isfinite(x)
    observed_counts = np.sum(finite, axis=0)
    keep = observed_counts >= min_observed_per_feature

    if not np.any(keep):
        raise RuntimeError(
            "No features left after filtering. "
            f"Try lowering min_observed_per_feature (current={min_observed_per_feature})."
        )

    idx = np.where(keep)[0]
    if max_features is not None and idx.size > max_features:
        var_scores = _safe_nanvar_matrix(x[:, idx])
        missing_rate = 1.0 - (observed_counts[idx] / float(x.shape[0]))
        # Prefer informative features while preserving natural missingness structure.
        score = var_scores * (0.5 + missing_rate)
        order = np.argsort(-score)
        idx = idx[order[:max_features]]

    return idx


def _select_features(
    x: np.ndarray,
    source_mask: np.ndarray,
    protein_ids: list[str],
    *,
    min_observed_per_feature: int,
    max_features: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    idx = _select_feature_indices(
        x,
        min_observed_per_feature=min_observed_per_feature,
        max_features=max_features,
    )
    return x[:, idx], source_mask[:, idx], [protein_ids[i] for i in idx.tolist()]


def _build_single_layer_container(
    *,
    assay_name: str,
    layer_name: str,
    x: np.ndarray,
    mask: np.ndarray | None,
    sample_ids: list[str],
    var: pl.DataFrame,
    metadata: MatrixMetadata | None = None,
) -> ScpContainer:
    obs = pl.DataFrame({"_index": sample_ids})
    assay = Assay(var=var.clone())
    assay.add_layer(
        layer_name,
        ScpMatrix(
            X=np.asarray(x, dtype=np.float64),
            M=(None if mask is None else np.asarray(mask, dtype=np.int8).copy()),
            metadata=metadata,
        ),
    )
    return ScpContainer(obs=obs, assays={assay_name: assay})


def _build_impute_container(
    x_masked: np.ndarray,
    source_mask: np.ndarray,
    sample_ids: list[str],
    feature_ids: list[str],
    *,
    assay_name: str,
    source_layer_name: str,
    var: pl.DataFrame | None = None,
) -> ScpContainer:
    var_df = var.clone() if var is not None else pl.DataFrame({"_index": feature_ids})
    return _build_single_layer_container(
        assay_name=assay_name,
        layer_name="masked",
        x=x_masked,
        mask=source_mask,
        sample_ids=sample_ids,
        var=var_df,
        metadata=MatrixMetadata(
            creation_info={
                "source_assay": assay_name,
                "source_layer": source_layer_name,
                "action": "benchmark_holdout_masking",
                "output_layer": "masked",
            }
        ),
    )


def _resolve_method_kwargs(method: str, n_samples: int, n_features: int) -> dict[str, Any]:
    kwargs = dict(METHOD_KWARGS.get(method, {}))

    if method in {"bpca", "iterative_svd", "softimpute"}:
        max_rank = max(1, min(n_samples, n_features) - 1)
        if "n_components" in kwargs:
            kwargs["n_components"] = int(min(max_rank, int(kwargs["n_components"])))
        if "rank" in kwargs:
            kwargs["rank"] = int(min(max_rank, int(kwargs["rank"])))

    return kwargs


def _dependency_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _resolve_missing_optional_dependencies(methods: list[str]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for method in methods:
        dependency = OPTIONAL_METHOD_DEPENDENCIES.get(method)
        if dependency is None:
            continue
        if not _dependency_available(dependency):
            missing[method] = dependency
    return missing


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _build_board_metadata(
    *,
    output_dir: Path,
    raw_df: pd.DataFrame,
    failures: list[dict[str, Any]],
    skipped_holdouts: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    config: dict[str, Any],
    board_type: str,
    benchmark_tier: str,
    state_reference_policy: str,
    endpoint_semantics: dict[str, str],
    run_status: str,
    final_error: dict[str, Any] | None,
) -> dict[str, Any]:
    outputs = {
        "run_metadata": str(output_dir / "run_metadata.json"),
    }
    for key, filename in (
        ("raw", "metrics_raw.csv"),
        ("summary", "metrics_summary.csv"),
        ("scores", "metrics_scores.csv"),
        ("overall_scores_plot", "overall_scores.png"),
        ("score_heatmap_plot", "score_heatmap.png"),
        ("nrmse_curves_plot", "nrmse_curves.png"),
        ("runtime_vs_accuracy_plot", "runtime_vs_accuracy.png"),
    ):
        path = output_dir / filename
        if path.exists():
            outputs[key] = str(path)

    n_failed_runs = 0
    if not raw_df.empty and "success" in raw_df.columns:
        n_failed_runs = int((~raw_df["success"]).sum())

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "dataset_meta": dataset_meta,
        "literature_references": LITERATURE_REFERENCES,
        "n_total_runs": int(raw_df.shape[0]),
        "n_failed_runs": n_failed_runs,
        "n_skipped_holdout_scenarios": len(skipped_holdouts),
        "failures": failures,
        "skipped_holdouts": skipped_holdouts,
        "state_aware_enabled": True,
        "benchmark_tier": benchmark_tier,
        "board_type": board_type,
        "state_reference_policy": state_reference_policy,
        "endpoint_semantics": endpoint_semantics,
        "run_status": run_status,
        "final_error": final_error,
        "outputs": outputs,
    }


def _flush_board_progress(
    *,
    output_dir: Path,
    raw_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped_holdouts: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    config: dict[str, Any],
    board_type: str,
    benchmark_tier: str,
    state_reference_policy: str,
    endpoint_semantics: dict[str, str],
    run_status: str,
    write_plots: bool = False,
    final_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_df = pd.DataFrame(raw_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame()
    scores = pd.DataFrame()
    if not raw_df.empty:
        raw_out = output_dir / "metrics_raw.csv"
        raw_df.to_csv(raw_out, index=False)

        summary = _summarize_raw(raw_df)
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)

        scores = _score_summary(summary)
        scores.to_csv(output_dir / "metrics_scores.csv", index=False)

        if write_plots and not summary.empty and bool(raw_df["success"].any()):
            plot_overall_scores(scores, output_dir / "overall_scores.png")
            plot_metric_heatmap(scores, output_dir / "score_heatmap.png")
            plot_nrmse_curves(raw_df, output_dir / "nrmse_curves.png")
            plot_runtime_vs_accuracy(summary, output_dir / "runtime_vs_accuracy.png")

    metadata = _build_board_metadata(
        output_dir=output_dir,
        raw_df=raw_df,
        failures=failures,
        skipped_holdouts=skipped_holdouts,
        dataset_meta=dataset_meta,
        config=config,
        board_type=board_type,
        benchmark_tier=benchmark_tier,
        state_reference_policy=state_reference_policy,
        endpoint_semantics=endpoint_semantics,
        run_status=run_status,
        final_error=final_error,
    )
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _write_root_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    payload = dict(metadata)
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(output_dir / "run_metadata.json", payload)


def _resolve_holdout_candidate_mask(
    x: np.ndarray,
    source_mask: np.ndarray,
    *,
    holdout_state: str,
) -> np.ndarray:
    finite = np.isfinite(x)
    if holdout_state == "all_observed":
        return finite
    if holdout_state not in HOLDOUT_STATE_CODES:
        raise ValueError(
            f"Unsupported holdout_state: {holdout_state}. Choices: {HOLDOUT_STATE_CHOICES}"
        )
    return finite & (source_mask == HOLDOUT_STATE_CODES[holdout_state])


def _recoverable_state_candidate_counts(
    x: np.ndarray,
    source_mask: np.ndarray,
) -> dict[str, int]:
    return {
        state: int(np.sum(_resolve_holdout_candidate_mask(x, source_mask, holdout_state=state)))
        for state in HOLDOUT_STATE_CHOICES
    }


def _dataset_supports_board(dataset_key: str, board: str) -> bool:
    spec = DATASET_REGISTRY[dataset_key]
    supported = {str(value) for value in spec.get("supported_boards", ["main"])}
    return board in supported


def _resolve_groups_for_dataset(
    spec: dict[str, Any],
    *,
    kind: str,
    source_info: dict[str, Any],
    sample_ids: list[str],
) -> list[str] | None:
    if "group_regex" in spec:
        return _infer_groups_by_regex(sample_ids, str(spec["group_regex"]))
    if kind == "spectronaut_peptide_long":
        mapping = _extract_sample_group_mapping(
            Path(source_info.get("parse_path", source_info["path"])),
            sample_column=str(spec["sample_column"]),
            group_column=str(spec["group_column"]),
        )
        return [mapping.get(sid, "UNKNOWN") for sid in sample_ids]
    return None


def _build_protein_holdout_mask_from_linkage(
    holdout_mask: np.ndarray,
    *,
    source_feature_ids: list[str],
    protein_ids: list[str],
    linkage: pl.DataFrame,
) -> np.ndarray:
    out = np.zeros((holdout_mask.shape[0], len(protein_ids)), dtype=bool)
    if linkage.is_empty():
        return out

    source_index = {feature_id: idx for idx, feature_id in enumerate(source_feature_ids)}
    protein_index = {protein_id: idx for idx, protein_id in enumerate(protein_ids)}
    for row in linkage.select(["source_id", "target_id"]).iter_rows(named=True):
        source_id = str(row["source_id"])
        target_id = str(row["target_id"])
        source_idx = source_index.get(source_id)
        protein_idx = protein_index.get(target_id)
        if source_idx is None or protein_idx is None:
            continue
        out[:, protein_idx] |= holdout_mask[:, source_idx]
    return out


def _choose_holdout_indices(
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    n_target: int,
    *,
    row_remaining: np.ndarray,
    col_remaining: np.ndarray,
    rng: np.random.Generator,
    min_row_remaining: int,
    min_col_remaining: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_target <= 0 or candidate_rows.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = rng.permutation(candidate_rows.size)
    selected_r: list[int] = []
    selected_c: list[int] = []

    for idx in order.tolist():
        r = int(candidate_rows[idx])
        c = int(candidate_cols[idx])
        if row_remaining[r] <= min_row_remaining or col_remaining[c] <= min_col_remaining:
            continue

        row_remaining[r] -= 1
        col_remaining[c] -= 1
        selected_r.append(r)
        selected_c.append(c)
        if len(selected_r) >= n_target:
            break

    return np.asarray(selected_r, dtype=int), np.asarray(selected_c, dtype=int)


def generate_holdout_mask(
    x: np.ndarray,
    source_mask: np.ndarray,
    *,
    holdout_state: str,
    holdout_rate: float,
    mechanism: str,
    mnar_fraction: float,
    mnar_low_quantile: float,
    random_seed: int,
    min_row_remaining: int = 1,
    min_col_remaining: int = 2,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Generate a state-aware holdout mask from source-state-stratified entries."""
    if not 0 < holdout_rate < 1:
        raise ValueError(f"holdout_rate must be in (0, 1), got {holdout_rate}")

    rng = np.random.default_rng(random_seed)
    finite = np.isfinite(x)
    n_observed = int(np.sum(finite))
    if n_observed == 0:
        raise RuntimeError("No observed entries available for holdout masking.")

    candidate_mask = _resolve_holdout_candidate_mask(x, source_mask, holdout_state=holdout_state)
    candidate_rows, candidate_cols = np.where(candidate_mask)
    n_candidates = int(candidate_rows.size)
    row_remaining = np.sum(finite, axis=1).astype(int)
    col_remaining = np.sum(finite, axis=0).astype(int)
    holdout = np.zeros_like(finite, dtype=bool)
    meta: dict[str, float | int | str] = {
        "holdout_state": holdout_state,
        "n_state_candidates": n_candidates,
        "holdout_state_fraction": (
            float(n_candidates / n_observed) if n_observed > 0 else float("nan")
        ),
        "n_holdout": 0,
        "n_holdout_source": 0,
        "n_target_holdout": 0,
        "holdout_rate_within_state": float("nan"),
        "holdout_share_of_total_observed": 0.0,
        **summarize_holdout_state_profile(np.asarray([], dtype=np.int8)),
    }
    if n_candidates == 0:
        return holdout, meta

    n_target = max(1, int(round(n_candidates * holdout_rate)))
    meta["n_target_holdout"] = int(n_target)

    if mechanism == "mcar":
        sel_r, sel_c = _choose_holdout_indices(
            candidate_rows,
            candidate_cols,
            n_target,
            row_remaining=row_remaining,
            col_remaining=col_remaining,
            rng=rng,
            min_row_remaining=min_row_remaining,
            min_col_remaining=min_col_remaining,
        )
        holdout[sel_r, sel_c] = True
    else:
        if mechanism != "mixed_mnar":
            raise ValueError(f"Unsupported mechanism: {mechanism}")

        candidate_values = x[candidate_mask]
        cutoff = float(np.quantile(candidate_values, mnar_low_quantile))
        mnar_mask = candidate_mask & (x <= cutoff)
        mnar_rows, mnar_cols = np.where(mnar_mask)

        n_mnar_target = int(round(n_target * mnar_fraction))
        sel_mnar_r, sel_mnar_c = _choose_holdout_indices(
            mnar_rows,
            mnar_cols,
            n_mnar_target,
            row_remaining=row_remaining,
            col_remaining=col_remaining,
            rng=rng,
            min_row_remaining=min_row_remaining,
            min_col_remaining=min_col_remaining,
        )
        holdout[sel_mnar_r, sel_mnar_c] = True

        n_remaining = n_target - int(np.sum(holdout))
        if n_remaining > 0:
            available = candidate_mask & (~holdout)
            avail_rows, avail_cols = np.where(available)
            sel_rest_r, sel_rest_c = _choose_holdout_indices(
                avail_rows,
                avail_cols,
                n_remaining,
                row_remaining=row_remaining,
                col_remaining=col_remaining,
                rng=rng,
                min_row_remaining=min_row_remaining,
                min_col_remaining=min_col_remaining,
            )
            holdout[sel_rest_r, sel_rest_c] = True

    selected_mask_codes = source_mask[holdout]
    n_holdout = int(np.sum(holdout))
    meta.update(
        {
            "n_holdout": n_holdout,
            "n_holdout_source": n_holdout,
            "holdout_rate_within_state": (
                float(n_holdout / n_candidates) if n_candidates > 0 else float("nan")
            ),
            "holdout_share_of_total_observed": (
                float(n_holdout / n_observed) if n_observed > 0 else float("nan")
            ),
        }
    )
    meta.update(summarize_holdout_state_profile(selected_mask_codes))
    return holdout, meta


def _load_main_dataset(
    dataset_key: str,
    *,
    data_dir: Path,
    force_download: bool,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str] | None, dict[str, Any]]:
    spec = DATASET_REGISTRY[dataset_key]
    kind = str(spec["kind"])

    if kind == "diann_protein_long_local":
        path = Path(str(spec["path"]))
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        container = load_diann(
            path,
            level="protein",
            table_format="long",
            quantity_column=str(spec["quantity_column"]),
            assay_name="proteins",
            layer_name="raw",
            fdr_threshold=0.01,
        )
        source_info = {"path": str(path), "kind": kind}

    elif kind == "spectronaut_peptide_long":
        path = _download_file(
            str(spec["url"]), data_dir / str(spec["filename"]), force=force_download
        )
        parse_path = _normalize_line_endings_if_needed(path)

        peptide = load_spectronaut(
            parse_path,
            level="peptide",
            table_format="long",
            quantity_column=str(spec["quantity_column"]),
            assay_name="peptides",
            layer_name="raw",
            fdr_threshold=0.01,
        )
        container = aggregate_to_protein(
            peptide,
            source_assay="peptides",
            source_layer="raw",
            target_assay="proteins",
            method="sum",
            protein_column=str(spec["protein_column"]),
            keep_unmapped=False,
        )
        source_info = {
            "path": str(path),
            "parse_path": str(parse_path),
            "kind": kind,
        }
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

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
    container = normalize(
        container,
        method=normalization_method,
        assay_name="proteins",
        source_layer="log2",
        new_layer_name="norm",
    )

    assay = container.assays["proteins"]
    norm_layer = assay.layers["norm"]
    x = np.asarray(norm_layer.X, dtype=np.float64)
    source_mask = _dense_mask_array(norm_layer.get_m(), shape=x.shape)
    sample_ids = [str(v) for v in container.sample_ids.to_list()]
    protein_ids = [str(v) for v in assay.var["_index"].to_list()]

    x, source_mask, protein_ids = _select_features(
        x,
        source_mask,
        protein_ids,
        min_observed_per_feature=min_observed_per_feature,
        max_features=max_features,
    )

    groups = _resolve_groups_for_dataset(
        spec,
        kind=kind,
        source_info=source_info,
        sample_ids=sample_ids,
    )

    meta = {
        "dataset_key": dataset_key,
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "missing_rate": float(np.mean(np.isnan(x))),
        "source": source_info,
        "input_level": "protein",
        "eval_level": "protein",
        "groups_available": groups is not None,
        "recoverable_state_candidates": _recoverable_state_candidate_counts(x, source_mask),
        "references": dict(spec.get("references", {})),
        "group_a": spec.get("group_a"),
        "group_b": spec.get("group_b"),
        "expected_log2_fc": spec.get("expected_log2_fc"),
    }

    return x, source_mask, sample_ids, protein_ids, groups, meta


def _load_dataset(
    dataset_key: str,
    *,
    data_dir: Path,
    force_download: bool,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str] | None, dict[str, Any]]:
    """Backward-compatible main-board loader used by existing tests."""
    return _load_main_dataset(
        dataset_key,
        data_dir=data_dir,
        force_download=force_download,
        normalization_method=normalization_method,
        min_observed_per_feature=min_observed_per_feature,
        max_features=max_features,
    )


def _load_auxiliary_dataset(
    dataset_key: str,
    *,
    data_dir: Path,
    force_download: bool,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
    aux_aggregation_method: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str] | None,
    np.ndarray,
    list[str],
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
]:
    spec = DATASET_REGISTRY[dataset_key]
    kind = str(spec["kind"])
    if kind != "spectronaut_peptide_long":
        raise ValueError(f"Dataset '{dataset_key}' does not currently support the auxiliary board.")

    path = _download_file(str(spec["url"]), data_dir / str(spec["filename"]), force=force_download)
    parse_path = _normalize_line_endings_if_needed(path)
    peptide = load_spectronaut(
        parse_path,
        level="peptide",
        table_format="long",
        quantity_column=str(spec["quantity_column"]),
        assay_name="peptides",
        layer_name="raw",
        fdr_threshold=0.01,
    )
    peptide = log_transform(
        peptide,
        assay_name="peptides",
        source_layer="raw",
        new_layer_name="log2",
        base=2.0,
        offset=1.0,
        detect_logged=True,
        skip_if_logged=True,
    )
    peptide = normalize(
        peptide,
        method=normalization_method,
        assay_name="peptides",
        source_layer="log2",
        new_layer_name="norm",
    )

    assay = peptide.assays["peptides"]
    norm_layer = assay.layers["norm"]
    x = np.asarray(norm_layer.X, dtype=np.float64)
    source_mask = _dense_mask_array(norm_layer.get_m(), shape=x.shape)
    sample_ids = [str(v) for v in peptide.sample_ids.to_list()]
    precursor_ids = [str(v) for v in assay.var["_index"].to_list()]
    selected_idx = _select_feature_indices(
        x,
        min_observed_per_feature=min_observed_per_feature,
        max_features=max_features,
    )
    x = x[:, selected_idx]
    source_mask = source_mask[:, selected_idx]
    precursor_ids = [precursor_ids[i] for i in selected_idx.tolist()]
    selected_var = assay.var[selected_idx.tolist(), :]

    filtered_container = _build_single_layer_container(
        assay_name="peptides",
        layer_name="norm",
        x=x,
        mask=source_mask,
        sample_ids=sample_ids,
        var=selected_var,
    )
    reference = aggregate_to_protein(
        filtered_container,
        source_assay="peptides",
        source_layer="norm",
        target_assay="proteins",
        method=aux_aggregation_method,
        protein_column=str(spec["protein_column"]),
        keep_unmapped=False,
    )
    protein_assay = reference.assays["proteins"]
    protein_truth = np.asarray(protein_assay.layers["norm"].X, dtype=np.float64)
    protein_ids = [str(v) for v in protein_assay.var["_index"].to_list()]
    linkage = reference.links[-1].linkage.clone()
    source_info = {
        "path": str(path),
        "parse_path": str(parse_path),
        "kind": kind,
        "aux_aggregation_method": aux_aggregation_method,
    }
    groups = _resolve_groups_for_dataset(
        spec,
        kind=kind,
        source_info=source_info,
        sample_ids=sample_ids,
    )

    meta = {
        "dataset_key": dataset_key,
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "protein_eval_shape": [int(protein_truth.shape[0]), int(protein_truth.shape[1])],
        "missing_rate": float(np.mean(np.isnan(x))),
        "source": source_info,
        "input_level": "precursor",
        "eval_level": "protein",
        "aux_aggregation_method": aux_aggregation_method,
        "groups_available": groups is not None,
        "recoverable_state_candidates": _recoverable_state_candidate_counts(x, source_mask),
        "references": dict(spec.get("references", {})),
        "group_a": spec.get("group_a"),
        "group_b": spec.get("group_b"),
        "expected_log2_fc": spec.get("expected_log2_fc"),
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


def _aggregate_auxiliary_imputed_protein_matrix(
    container: ScpContainer,
    *,
    aux_aggregation_method: str,
    protein_column: str,
) -> np.ndarray:
    aggregated = aggregate_to_protein(
        container,
        source_assay="peptides",
        source_layer="imputed",
        target_assay="proteins",
        method=aux_aggregation_method,
        protein_column=protein_column,
        keep_unmapped=False,
    )
    return np.asarray(aggregated.assays["proteins"].layers["imputed"].X, dtype=np.float64)


def _summarize_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        column
        for column in raw_df.columns
        if column not in {*SUMMARY_GROUP_COLUMNS, "seed", "error"}
        and column != "success"
        and pd.api.types.is_numeric_dtype(raw_df[column])
    ]
    grouped = raw_df.groupby(SUMMARY_GROUP_COLUMNS, dropna=False)
    summary = grouped[numeric_cols].mean(numeric_only=True).reset_index()

    sd_cols = [column for column in numeric_cols if column != "success"]
    if sd_cols:
        sd_summary = grouped[sd_cols].std(numeric_only=True, ddof=1).reset_index()
        sd_summary = sd_summary.rename(columns={column: f"{column}_sd" for column in sd_cols})
        summary = summary.merge(sd_summary, on=SUMMARY_GROUP_COLUMNS, how="left")

    summary["runs"] = grouped["success"].size().values
    summary["successful_runs"] = grouped["success"].sum().astype(int).values
    summary["success_rate"] = grouped["success"].mean().values
    summary["seed_min"] = grouped["seed"].min().values
    summary["seed_max"] = grouped["seed"].max().values

    return summary.sort_values(
        ["dataset", "holdout_state", "mechanism", "holdout_rate", "method"]
    ).reset_index(drop=True)


def _score_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    scored = score_methods(
        summary_df,
        group_by=["dataset", "holdout_state", "mechanism", "holdout_rate"],
    )
    return scored.sort_values(
        ["dataset", "holdout_state", "mechanism", "holdout_rate", "method"]
    ).reset_index(drop=True)


def _populate_evaluation_metrics(
    row: dict[str, Any],
    *,
    matrix_true: np.ndarray,
    matrix_imputed: np.ndarray,
    eval_holdout_mask: np.ndarray,
    groups: list[str] | None,
    feature_ids: list[str],
    dataset_key: str,
) -> None:
    recon = compute_reconstruction_metrics(
        y_true=matrix_true[eval_holdout_mask],
        y_pred=matrix_imputed[eval_holdout_mask],
    )
    row.update(recon)
    row["n_holdout_eval"] = int(np.sum(eval_holdout_mask))
    row["post_missing_rate"] = float(np.mean(np.isnan(matrix_imputed)))
    finite_counts = np.sum(np.isfinite(matrix_imputed), axis=0)
    row["retained_proteins"] = int(np.sum(finite_counts > 0))
    row["retained_proteins_ratio"] = float(np.mean(finite_counts > 0))
    row["fully_observed_proteins"] = int(np.sum(finite_counts == matrix_imputed.shape[0]))
    row["fully_observed_proteins_ratio"] = float(np.mean(finite_counts == matrix_imputed.shape[0]))
    row["within_group_cv_median"] = compute_within_group_cv_median(matrix_imputed, groups)
    row.update(compute_cluster_metrics(matrix_imputed, groups))
    row.update(
        compute_de_consistency_metrics(
            matrix_true=matrix_true,
            matrix_imputed=matrix_imputed,
            groups=groups,
            top_k=50,
        )
    )

    expected = DATASET_REGISTRY[dataset_key].get("expected_log2_fc")
    group_a = DATASET_REGISTRY[dataset_key].get("group_a")
    group_b = DATASET_REGISTRY[dataset_key].get("group_b")
    if expected is not None and groups is not None and group_a and group_b:
        ratio_metrics, _ = compute_ratio_metrics(
            matrix_imputed,
            feature_ids,
            groups,
            group_a=str(group_a),
            group_b=str(group_b),
            expected_log2fc=expected,
        )
        row.update(ratio_metrics)


def _write_board_outputs(
    *,
    output_dir: Path,
    raw_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped_holdouts: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    config: dict[str, Any],
    board_type: str,
    benchmark_tier: str,
    state_reference_policy: str,
    endpoint_semantics: dict[str, str],
) -> dict[str, Any]:
    if not raw_rows:
        raise RuntimeError("No benchmark results generated.")
    metadata = _flush_board_progress(
        output_dir=output_dir,
        raw_rows=raw_rows,
        failures=failures,
        skipped_holdouts=skipped_holdouts,
        dataset_meta=dataset_meta,
        config=config,
        board_type=board_type,
        benchmark_tier=benchmark_tier,
        state_reference_policy=state_reference_policy,
        endpoint_semantics=endpoint_semantics,
        run_status="completed",
        write_plots=True,
    )

    raw_df = pd.DataFrame(raw_rows)
    summary = _summarize_raw(raw_df)
    if summary.empty or not bool(raw_df["success"].any()):
        raise RuntimeError("No successful benchmark runs. Check failures in metadata output.")

    print("[INFO] Benchmark completed.")
    print(f"[INFO] Raw metrics: {output_dir / 'metrics_raw.csv'}")
    print(f"[INFO] Summary metrics: {output_dir / 'metrics_summary.csv'}")
    print(f"[INFO] Score table: {output_dir / 'metrics_scores.csv'}")
    print(f"[INFO] Failures: {len(failures)}")
    return metadata


def _run_main_board(
    *,
    datasets: list[str],
    methods: list[str],
    holdout_rates: list[float],
    holdout_states: list[str],
    mechanisms: list[str],
    repeats: int,
    mnar_fraction: float,
    mnar_low_quantile: float,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
    data_dir: Path,
    output_dir: Path,
    force_download: bool,
    benchmark_tier: str,
) -> dict[str, Any]:
    config = {
        "requested_datasets": datasets,
        "datasets": datasets,
        "methods": methods,
        "holdout_rates": holdout_rates,
        "holdout_states": holdout_states,
        "mechanisms": mechanisms,
        "repeats": repeats,
        "mnar_fraction": mnar_fraction,
        "mnar_low_quantile": mnar_low_quantile,
        "normalization_method": normalization_method,
        "min_observed_per_feature": min_observed_per_feature,
        "max_features": max_features,
    }
    board_type = "protein_direct_state_aware_masked_recovery"
    state_reference_policy = "source_layer_mask_codes_on_current_finite_entries"
    endpoint_semantics = {
        "masked_recovery": "pseudo_truth_protein_holdout_recovery",
        "de_consistency": "pseudo_truth_group_contrast_proxy",
        "ratio_metrics": "external_truth_species_mixture_when_available",
    }
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_holdouts: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}
    scenario_counter = 0

    try:
        for dataset_key in datasets:
            if dataset_key not in DATASET_REGISTRY:
                raise ValueError(f"Unknown dataset key: {dataset_key}")

            print(f"[INFO] Loading dataset: {dataset_key}")
            x_base, source_mask, sample_ids, protein_ids, groups, meta = _load_dataset(
                dataset_key,
                data_dir=data_dir,
                force_download=force_download,
                normalization_method=normalization_method,
                min_observed_per_feature=min_observed_per_feature,
                max_features=max_features,
            )
            dataset_meta[dataset_key] = meta
            print(
                f"[INFO] Dataset {dataset_key} -> shape={x_base.shape}, missing={np.isnan(x_base).mean():.2%}"
            )

            n_samples, n_features = x_base.shape
            for mechanism in mechanisms:
                for holdout_state in holdout_states:
                    for holdout_rate in holdout_rates:
                        for rep in range(repeats):
                            seed = 20260304 + rep * 1000 + int(round(holdout_rate * 100))
                            holdout_mask, holdout_meta = generate_holdout_mask(
                                x_base,
                                source_mask,
                                holdout_state=holdout_state,
                                holdout_rate=holdout_rate,
                                mechanism=mechanism,
                                mnar_fraction=mnar_fraction,
                                mnar_low_quantile=mnar_low_quantile,
                                random_seed=seed,
                            )
                            if int(np.sum(holdout_mask)) == 0:
                                skipped_holdouts.append(
                                    {
                                        "dataset": dataset_key,
                                        "mechanism": mechanism,
                                        "holdout_rate": float(holdout_rate),
                                        "seed": int(seed),
                                        **holdout_meta,
                                        "reason": "no_recoverable_entries_after_state_stratification",
                                    }
                                )
                                scenario_counter += 1
                                continue

                            x_masked = np.array(x_base, copy=True)
                            x_masked[holdout_mask] = np.nan

                            for method in methods:
                                print(
                                    "[INFO] "
                                    f"board=main dataset={dataset_key} state={holdout_state} "
                                    f"mechanism={mechanism} rate={holdout_rate:.2f} "
                                    f"rep={rep + 1}/{repeats} method={method}"
                                )
                                row: dict[str, Any] = {
                                    "dataset": dataset_key,
                                    "method": method,
                                    "mechanism": mechanism,
                                    "holdout_rate": float(holdout_rate),
                                    "holdout_state": holdout_state,
                                    "seed": int(seed),
                                    "board_scope": "main",
                                    "input_level": "protein",
                                    "eval_level": "protein",
                                    "n_source_features": int(n_features),
                                    "n_eval_features": int(x_base.shape[1]),
                                    "n_observed_before": int(np.sum(np.isfinite(x_base))),
                                    "success": False,
                                    **holdout_meta,
                                }

                                container_in = _build_impute_container(
                                    x_masked,
                                    source_mask,
                                    sample_ids,
                                    protein_ids,
                                    assay_name="protein",
                                    source_layer_name="norm",
                                    var=pl.DataFrame({"_index": protein_ids}),
                                )
                                kwargs = _resolve_method_kwargs(
                                    method, n_samples=n_samples, n_features=n_features
                                )

                                start = time.perf_counter()
                                try:
                                    out = impute(
                                        container_in,
                                        method=method,
                                        assay_name="protein",
                                        source_layer="masked",
                                        new_layer_name="imputed",
                                        **kwargs,
                                    )
                                    elapsed = float(time.perf_counter() - start)
                                    x_imputed = np.asarray(
                                        out.assays["protein"].layers["imputed"].X,
                                        dtype=np.float64,
                                    )
                                    _populate_evaluation_metrics(
                                        row,
                                        matrix_true=x_base,
                                        matrix_imputed=x_imputed,
                                        eval_holdout_mask=holdout_mask,
                                        groups=groups,
                                        feature_ids=protein_ids,
                                        dataset_key=dataset_key,
                                    )
                                    row["runtime_sec"] = elapsed
                                    row["success"] = True
                                except Exception as exc:  # noqa: BLE001
                                    row["runtime_sec"] = float(time.perf_counter() - start)
                                    row["error"] = str(exc)
                                    failures.append(
                                        {
                                            "dataset": dataset_key,
                                            "method": method,
                                            "mechanism": mechanism,
                                            "holdout_rate": float(holdout_rate),
                                            "holdout_state": holdout_state,
                                            "seed": int(seed),
                                            "error": str(exc),
                                        }
                                    )

                                raw_rows.append(row)

                            scenario_counter += 1
                            if scenario_counter % BOARD_CHECKPOINT_SCENARIO_INTERVAL == 0:
                                _flush_board_progress(
                                    output_dir=output_dir,
                                    raw_rows=raw_rows,
                                    failures=failures,
                                    skipped_holdouts=skipped_holdouts,
                                    dataset_meta=dataset_meta,
                                    config=config,
                                    board_type=board_type,
                                    benchmark_tier=benchmark_tier,
                                    state_reference_policy=state_reference_policy,
                                    endpoint_semantics=endpoint_semantics,
                                    run_status="running",
                                )
    except BaseException as exc:  # noqa: BLE001
        _flush_board_progress(
            output_dir=output_dir,
            raw_rows=raw_rows,
            failures=failures,
            skipped_holdouts=skipped_holdouts,
            dataset_meta=dataset_meta,
            config=config,
            board_type=board_type,
            benchmark_tier=benchmark_tier,
            state_reference_policy=state_reference_policy,
            endpoint_semantics=endpoint_semantics,
            run_status=("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"),
            final_error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    return _write_board_outputs(
        output_dir=output_dir,
        raw_rows=raw_rows,
        failures=failures,
        skipped_holdouts=skipped_holdouts,
        dataset_meta=dataset_meta,
        config=config,
        board_type=board_type,
        benchmark_tier=benchmark_tier,
        state_reference_policy=state_reference_policy,
        endpoint_semantics=endpoint_semantics,
    )


def _run_auxiliary_board(
    *,
    datasets: list[str],
    methods: list[str],
    holdout_rates: list[float],
    holdout_states: list[str],
    mechanisms: list[str],
    repeats: int,
    mnar_fraction: float,
    mnar_low_quantile: float,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
    data_dir: Path,
    output_dir: Path,
    force_download: bool,
    benchmark_tier: str,
    aux_aggregation_method: str,
) -> dict[str, Any]:
    supported_datasets = [key for key in datasets if _dataset_supports_board(key, "auxiliary")]
    skipped_datasets = [key for key in datasets if key not in supported_datasets]
    if not supported_datasets:
        raise RuntimeError("No datasets support the auxiliary precursor-to-protein board.")

    config = {
        "requested_datasets": datasets,
        "datasets": supported_datasets,
        "skipped_datasets": skipped_datasets,
        "methods": methods,
        "holdout_rates": holdout_rates,
        "holdout_states": holdout_states,
        "mechanisms": mechanisms,
        "repeats": repeats,
        "mnar_fraction": mnar_fraction,
        "mnar_low_quantile": mnar_low_quantile,
        "normalization_method": normalization_method,
        "min_observed_per_feature": min_observed_per_feature,
        "max_features": max_features,
        "aux_aggregation_method": aux_aggregation_method,
    }
    board_type = "precursor_to_protein_auxiliary_masked_recovery"
    state_reference_policy = "source_layer_mask_codes_on_current_finite_precursor_entries"
    endpoint_semantics = {
        "masked_recovery": "precursor_holdout_to_protein_pseudo_truth_recovery",
        "de_consistency": "aggregated_protein_group_contrast_proxy",
        "ratio_metrics": "external_truth_species_mixture_when_available",
    }
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_holdouts: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}
    scenario_counter = 0

    try:
        for dataset_key in supported_datasets:
            print(f"[INFO] Loading auxiliary-board dataset: {dataset_key}")
            (
                x_source,
                source_mask,
                sample_ids,
                precursor_ids,
                groups,
                protein_truth,
                protein_ids,
                linkage,
                selected_var,
                meta,
            ) = _load_auxiliary_dataset(
                dataset_key,
                data_dir=data_dir,
                force_download=force_download,
                normalization_method=normalization_method,
                min_observed_per_feature=min_observed_per_feature,
                max_features=max_features,
                aux_aggregation_method=aux_aggregation_method,
            )
            dataset_meta[dataset_key] = meta
            n_samples, n_features = x_source.shape
            print(
                "[INFO] "
                f"Aux dataset {dataset_key} -> precursor_shape={x_source.shape}, "
                f"protein_eval_shape={protein_truth.shape}, missing={np.isnan(x_source).mean():.2%}"
            )

            for mechanism in mechanisms:
                for holdout_state in holdout_states:
                    for holdout_rate in holdout_rates:
                        for rep in range(repeats):
                            seed = 20260304 + rep * 1000 + int(round(holdout_rate * 100))
                            holdout_mask, holdout_meta = generate_holdout_mask(
                                x_source,
                                source_mask,
                                holdout_state=holdout_state,
                                holdout_rate=holdout_rate,
                                mechanism=mechanism,
                                mnar_fraction=mnar_fraction,
                                mnar_low_quantile=mnar_low_quantile,
                                random_seed=seed,
                            )
                            if int(np.sum(holdout_mask)) == 0:
                                skipped_holdouts.append(
                                    {
                                        "dataset": dataset_key,
                                        "mechanism": mechanism,
                                        "holdout_rate": float(holdout_rate),
                                        "seed": int(seed),
                                        **holdout_meta,
                                        "reason": "no_recoverable_entries_after_state_stratification",
                                    }
                                )
                                scenario_counter += 1
                                continue

                            protein_holdout_mask = _build_protein_holdout_mask_from_linkage(
                                holdout_mask,
                                source_feature_ids=precursor_ids,
                                protein_ids=protein_ids,
                                linkage=linkage,
                            )
                            if int(np.sum(protein_holdout_mask)) == 0:
                                skipped_holdouts.append(
                                    {
                                        "dataset": dataset_key,
                                        "mechanism": mechanism,
                                        "holdout_rate": float(holdout_rate),
                                        "seed": int(seed),
                                        **holdout_meta,
                                        "reason": "no_protein_targets_affected_after_auxiliary_mapping",
                                    }
                                )
                                scenario_counter += 1
                                continue

                            x_masked = np.array(x_source, copy=True)
                            x_masked[holdout_mask] = np.nan

                            for method in methods:
                                print(
                                    "[INFO] "
                                    f"board=auxiliary dataset={dataset_key} state={holdout_state} "
                                    f"mechanism={mechanism} rate={holdout_rate:.2f} "
                                    f"rep={rep + 1}/{repeats} method={method}"
                                )
                                row: dict[str, Any] = {
                                    "dataset": dataset_key,
                                    "method": method,
                                    "mechanism": mechanism,
                                    "holdout_rate": float(holdout_rate),
                                    "holdout_state": holdout_state,
                                    "seed": int(seed),
                                    "board_scope": "auxiliary",
                                    "input_level": "precursor",
                                    "eval_level": "protein",
                                    "n_source_features": int(n_features),
                                    "n_eval_features": int(protein_truth.shape[1]),
                                    "n_observed_before": int(np.sum(np.isfinite(x_source))),
                                    "success": False,
                                    **holdout_meta,
                                }

                                container_in = _build_impute_container(
                                    x_masked,
                                    source_mask,
                                    sample_ids,
                                    precursor_ids,
                                    assay_name="peptides",
                                    source_layer_name="norm",
                                    var=selected_var,
                                )
                                kwargs = _resolve_method_kwargs(
                                    method, n_samples=n_samples, n_features=n_features
                                )

                                start = time.perf_counter()
                                try:
                                    out = impute(
                                        container_in,
                                        method=method,
                                        assay_name="peptides",
                                        source_layer="masked",
                                        new_layer_name="imputed",
                                        **kwargs,
                                    )
                                    elapsed = float(time.perf_counter() - start)
                                    x_imputed_protein = _aggregate_auxiliary_imputed_protein_matrix(
                                        out,
                                        aux_aggregation_method=aux_aggregation_method,
                                        protein_column=str(
                                            DATASET_REGISTRY[dataset_key]["protein_column"]
                                        ),
                                    )
                                    _populate_evaluation_metrics(
                                        row,
                                        matrix_true=protein_truth,
                                        matrix_imputed=x_imputed_protein,
                                        eval_holdout_mask=protein_holdout_mask,
                                        groups=groups,
                                        feature_ids=protein_ids,
                                        dataset_key=dataset_key,
                                    )
                                    row["runtime_sec"] = elapsed
                                    row["success"] = True
                                except Exception as exc:  # noqa: BLE001
                                    row["runtime_sec"] = float(time.perf_counter() - start)
                                    row["error"] = str(exc)
                                    failures.append(
                                        {
                                            "dataset": dataset_key,
                                            "method": method,
                                            "mechanism": mechanism,
                                            "holdout_rate": float(holdout_rate),
                                            "holdout_state": holdout_state,
                                            "seed": int(seed),
                                            "error": str(exc),
                                        }
                                    )

                                raw_rows.append(row)

                            scenario_counter += 1
                            if scenario_counter % BOARD_CHECKPOINT_SCENARIO_INTERVAL == 0:
                                _flush_board_progress(
                                    output_dir=output_dir,
                                    raw_rows=raw_rows,
                                    failures=failures,
                                    skipped_holdouts=skipped_holdouts,
                                    dataset_meta=dataset_meta,
                                    config=config,
                                    board_type=board_type,
                                    benchmark_tier=benchmark_tier,
                                    state_reference_policy=state_reference_policy,
                                    endpoint_semantics=endpoint_semantics,
                                    run_status="running",
                                )
    except BaseException as exc:  # noqa: BLE001
        _flush_board_progress(
            output_dir=output_dir,
            raw_rows=raw_rows,
            failures=failures,
            skipped_holdouts=skipped_holdouts,
            dataset_meta=dataset_meta,
            config=config,
            board_type=board_type,
            benchmark_tier=benchmark_tier,
            state_reference_policy=state_reference_policy,
            endpoint_semantics=endpoint_semantics,
            run_status=("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"),
            final_error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    return _write_board_outputs(
        output_dir=output_dir,
        raw_rows=raw_rows,
        failures=failures,
        skipped_holdouts=skipped_holdouts,
        dataset_meta=dataset_meta,
        config=config,
        board_type=board_type,
        benchmark_tier=benchmark_tier,
        state_reference_policy=state_reference_policy,
        endpoint_semantics=endpoint_semantics,
    )


def _resolve_benchmark_profile(
    *,
    benchmark_tier: str,
    datasets: list[str] | None,
    methods: list[str] | None,
    holdout_rates: list[float] | None,
    holdout_states: list[str] | None,
    mechanisms: list[str] | None,
    repeats: int | None,
    max_features: int | None,
    board: str | None,
) -> dict[str, Any]:
    if benchmark_tier not in TIER_PROFILES:
        raise ValueError(
            f"Unknown benchmark_tier: {benchmark_tier}. Choices: {BENCHMARK_TIER_CHOICES}"
        )
    profile = TIER_PROFILES[benchmark_tier]
    return {
        "datasets": list(datasets or profile["datasets"]),
        "methods": list(methods or profile["methods"]),
        "holdout_rates": [float(value) for value in (holdout_rates or profile["holdout_rates"])],
        "holdout_states": list(holdout_states or profile["holdout_states"]),
        "mechanisms": list(mechanisms or profile["mechanisms"]),
        "repeats": int(profile["repeats"] if repeats is None else repeats),
        "max_features": profile["max_features"] if max_features is None else max_features,
        "board": str(board or profile["board"]),
    }


def run_benchmark(
    *,
    datasets: list[str],
    methods: list[str],
    holdout_rates: list[float],
    holdout_states: list[str],
    mechanisms: list[str],
    repeats: int,
    mnar_fraction: float,
    mnar_low_quantile: float,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
    data_dir: Path,
    output_dir: Path,
    force_download: bool,
    board: str = DEFAULT_BOARD,
    benchmark_tier: str = DEFAULT_BENCHMARK_TIER,
    aux_aggregation_method: str = DEFAULT_AUX_AGGREGATION_METHOD,
) -> dict[str, Any]:
    if board not in BOARD_CHOICES:
        raise ValueError(f"Unknown board: {board}. Choices: {BOARD_CHOICES}")
    if benchmark_tier not in BENCHMARK_TIER_CHOICES:
        raise ValueError(
            f"Unknown benchmark_tier: {benchmark_tier}. Choices: {BENCHMARK_TIER_CHOICES}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in datasets:
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

    available_methods = set(list_impute_methods())
    missing_methods = [m for m in methods if m not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Unknown imputation methods: {missing_methods}. Available: {sorted(available_methods)}"
        )
    missing_dependencies = _resolve_missing_optional_dependencies(methods)
    if missing_dependencies:
        detail = ", ".join(
            f"{method} -> {dependency}"
            for method, dependency in sorted(missing_dependencies.items())
        )
        raise RuntimeError(
            "Requested benchmark methods require optional dependencies that are not installed: "
            f"{detail}. Install the dependency or remove the method from --methods."
        )

    board_results: dict[str, Any] = {}
    root_metadata: dict[str, Any] | None = None
    current_board: str | None = None
    if board == "both":
        root_metadata = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "benchmark_tier": benchmark_tier,
            "board_type": "dual_board",
            "requested_board": board,
            "run_status": "running",
            "config": {
                "datasets": datasets,
                "methods": methods,
                "holdout_rates": holdout_rates,
                "holdout_states": holdout_states,
                "mechanisms": mechanisms,
                "repeats": repeats,
                "mnar_fraction": mnar_fraction,
                "mnar_low_quantile": mnar_low_quantile,
                "normalization_method": normalization_method,
                "min_observed_per_feature": min_observed_per_feature,
                "max_features": max_features,
                "aux_aggregation_method": aux_aggregation_method,
            },
            "boards": {
                "main": {
                    "status": "pending",
                    "run_metadata": str(output_dir / "main" / "run_metadata.json"),
                },
                "auxiliary": {
                    "status": "pending",
                    "run_metadata": str(output_dir / "auxiliary" / "run_metadata.json"),
                },
            },
        }
        _write_root_metadata(output_dir, root_metadata)

    try:
        if board in {"main", "both"}:
            current_board = "main"
            if root_metadata is not None:
                root_metadata["boards"]["main"]["status"] = "running"
                _write_root_metadata(output_dir, root_metadata)
            board_output_dir = output_dir if board == "main" else output_dir / "main"
            board_results["main"] = _run_main_board(
                datasets=datasets,
                methods=methods,
                holdout_rates=holdout_rates,
                holdout_states=holdout_states,
                mechanisms=mechanisms,
                repeats=repeats,
                mnar_fraction=mnar_fraction,
                mnar_low_quantile=mnar_low_quantile,
                normalization_method=normalization_method,
                min_observed_per_feature=min_observed_per_feature,
                max_features=max_features,
                data_dir=data_dir,
                output_dir=board_output_dir,
                force_download=force_download,
                benchmark_tier=benchmark_tier,
            )
            if root_metadata is not None:
                root_metadata["boards"]["main"]["status"] = "completed"
                root_metadata["boards"]["main"]["board_type"] = board_results["main"]["board_type"]
                _write_root_metadata(output_dir, root_metadata)
        if board in {"auxiliary", "both"}:
            current_board = "auxiliary"
            if root_metadata is not None:
                root_metadata["boards"]["auxiliary"]["status"] = "running"
                _write_root_metadata(output_dir, root_metadata)
            board_output_dir = output_dir if board == "auxiliary" else output_dir / "auxiliary"
            board_results["auxiliary"] = _run_auxiliary_board(
                datasets=datasets,
                methods=methods,
                holdout_rates=holdout_rates,
                holdout_states=holdout_states,
                mechanisms=mechanisms,
                repeats=repeats,
                mnar_fraction=mnar_fraction,
                mnar_low_quantile=mnar_low_quantile,
                normalization_method=normalization_method,
                min_observed_per_feature=min_observed_per_feature,
                max_features=max_features,
                data_dir=data_dir,
                output_dir=board_output_dir,
                force_download=force_download,
                benchmark_tier=benchmark_tier,
                aux_aggregation_method=aux_aggregation_method,
            )
            if root_metadata is not None:
                root_metadata["boards"]["auxiliary"]["status"] = "completed"
                root_metadata["boards"]["auxiliary"]["board_type"] = board_results["auxiliary"][
                    "board_type"
                ]
                _write_root_metadata(output_dir, root_metadata)
    except BaseException as exc:  # noqa: BLE001
        if root_metadata is not None and current_board is not None:
            root_metadata["run_status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            root_metadata["boards"][current_board]["status"] = root_metadata["run_status"]
            root_metadata["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "board": current_board,
            }
            _write_root_metadata(output_dir, root_metadata)
        raise

    if board == "both":
        assert root_metadata is not None
        root_metadata["run_status"] = "completed"
        _write_root_metadata(output_dir, root_metadata)
        return root_metadata

    return next(iter(board_results.values()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ScpTensor imputation methods.")
    parser.add_argument(
        "--tier",
        type=str,
        default=DEFAULT_BENCHMARK_TIER,
        choices=BENCHMARK_TIER_CHOICES,
        help="Benchmark run profile: smoke, default, or literature.",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        choices=BOARD_CHOICES,
        help="Board to run. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Datasets to benchmark. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Imputation methods to benchmark. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--holdout-rates",
        nargs="+",
        type=float,
        default=None,
        help="Holdout rates for masked-value recovery. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--holdout-states",
        nargs="+",
        default=None,
        choices=HOLDOUT_STATE_CHOICES,
        help="Source-state strata used to build state-aware holdout masks.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=None,
        choices=["mcar", "mixed_mnar"],
        help="Missingness mechanisms used for additional holdout masking.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Replicates per scenario. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--mnar-fraction",
        type=float,
        default=0.75,
        help="Fraction of holdout entries assigned to low-intensity MNAR in mixed mode.",
    )
    parser.add_argument(
        "--mnar-low-quantile",
        type=float,
        default=0.3,
        help="Low-intensity quantile threshold for MNAR candidate pool.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="mean",
        choices=["none", "mean", "median", "quantile", "trqn"],
        help="Normalization method applied before imputation benchmark.",
    )
    parser.add_argument(
        "--min-observed-per-feature",
        type=int,
        default=3,
        help="Drop features with fewer observed values than this threshold.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Upper bound for retained features per dataset. Defaults to the selected tier profile.",
    )
    parser.add_argument(
        "--aux-aggregation-method",
        type=str,
        default=DEFAULT_AUX_AGGREGATION_METHOD,
        choices=["sum", "mean", "median", "max", "weighted_mean", "top_n", "maxlfq", "tmp", "ibaq"],
        help="Fixed aggregation method used by the precursor-to-protein auxiliary board.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/imputation/data"),
        help="Directory for downloaded benchmark data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/imputation/outputs"),
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of remote datasets.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = _resolve_benchmark_profile(
        benchmark_tier=args.tier,
        datasets=args.datasets,
        methods=args.methods,
        holdout_rates=args.holdout_rates,
        holdout_states=args.holdout_states,
        mechanisms=args.mechanisms,
        repeats=args.repeats,
        max_features=args.max_features,
        board=args.board,
    )
    run_benchmark(
        datasets=profile["datasets"],
        methods=profile["methods"],
        holdout_rates=profile["holdout_rates"],
        holdout_states=profile["holdout_states"],
        mechanisms=profile["mechanisms"],
        repeats=profile["repeats"],
        mnar_fraction=args.mnar_fraction,
        mnar_low_quantile=args.mnar_low_quantile,
        normalization_method=args.normalization,
        min_observed_per_feature=args.min_observed_per_feature,
        max_features=profile["max_features"],
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force_download=args.force_download,
        board=profile["board"],
        benchmark_tier=args.tier,
        aux_aggregation_method=args.aux_aggregation_method,
    )
