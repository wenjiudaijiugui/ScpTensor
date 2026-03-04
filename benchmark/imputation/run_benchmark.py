"""Run benchmark for protein-level imputation methods."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests
from metrics import compute_reconstruction_metrics, compute_within_group_cv_median, score_methods
from plots import (
    plot_metric_heatmap,
    plot_nrmse_curves,
    plot_overall_scores,
    plot_runtime_vs_accuracy,
)

from benchmark.normalization.metrics import EXPECTED_LOG2FC_HYE124, compute_ratio_metrics
from scptensor.aggregation import aggregate_to_protein
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.impute import impute, list_impute_methods
from scptensor.io import load_diann, load_spectronaut
from scptensor.normalization import normalize
from scptensor.transformation import log_transform

DEFAULT_METHODS = [
    "none",
    "zero",
    "row_mean",
    "row_median",
    "half_row_min",
    "knn",
    "lls",
    "bpca",
    "iterative_svd",
    "missforest",
    "qrilc",
    "minprob",
]

DEFAULT_DATASETS = ["pxd054343_diann_2x"]
DEFAULT_HOLDOUT_RATES = [0.1, 0.3, 0.5]
DEFAULT_MECHANISMS = ["mcar", "mixed_mnar"]

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "pxd054343_diann_2x": {
        "kind": "diann_protein_long_local",
        "path": "data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv",
        "quantity_column": "PG.Quantity",
        "group_regex": r"_(S\d+)_",
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
    "bpca": {"n_components": 5, "max_iter": 80},
    "iterative_svd": {"n_components": 5, "max_iter": 80, "tol": 1e-5},
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


def _select_features(
    x: np.ndarray,
    protein_ids: list[str],
    *,
    min_observed_per_feature: int,
    max_features: int | None,
) -> tuple[np.ndarray, list[str]]:
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

    return x[:, idx], [protein_ids[i] for i in idx.tolist()]


def _build_impute_container(
    x_masked: np.ndarray,
    sample_ids: list[str],
    protein_ids: list[str],
) -> ScpContainer:
    obs = pl.DataFrame({"_index": sample_ids})
    var = pl.DataFrame({"_index": protein_ids})

    assay = Assay(var=var)
    assay.add_layer("masked", ScpMatrix(X=np.asarray(x_masked, dtype=np.float64), M=None))
    return ScpContainer(obs=obs, assays={"protein": assay})


def _resolve_method_kwargs(method: str, n_samples: int, n_features: int) -> dict[str, Any]:
    kwargs = dict(METHOD_KWARGS.get(method, {}))

    if method in {"bpca", "iterative_svd", "softimpute"}:
        max_rank = max(1, min(n_samples, n_features) - 1)
        if "n_components" in kwargs:
            kwargs["n_components"] = int(min(max_rank, int(kwargs["n_components"])))
        if "rank" in kwargs:
            kwargs["rank"] = int(min(max_rank, int(kwargs["rank"])))

    return kwargs


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
    *,
    holdout_rate: float,
    mechanism: str,
    mnar_fraction: float,
    mnar_low_quantile: float,
    random_seed: int,
    min_row_remaining: int = 1,
    min_col_remaining: int = 2,
) -> np.ndarray:
    """Generate holdout mask from observed entries using MCAR or mixed MNAR/MCAR."""
    if not 0 < holdout_rate < 1:
        raise ValueError(f"holdout_rate must be in (0, 1), got {holdout_rate}")

    rng = np.random.default_rng(random_seed)
    finite = np.isfinite(x)
    observed_rows, observed_cols = np.where(finite)
    n_observed = observed_rows.size
    if n_observed == 0:
        raise RuntimeError("No observed entries available for holdout masking.")

    n_target = max(1, int(round(n_observed * holdout_rate)))
    row_remaining = np.sum(finite, axis=1).astype(int)
    col_remaining = np.sum(finite, axis=0).astype(int)

    holdout = np.zeros_like(finite, dtype=bool)

    if mechanism == "mcar":
        sel_r, sel_c = _choose_holdout_indices(
            observed_rows,
            observed_cols,
            n_target,
            row_remaining=row_remaining,
            col_remaining=col_remaining,
            rng=rng,
            min_row_remaining=min_row_remaining,
            min_col_remaining=min_col_remaining,
        )
        holdout[sel_r, sel_c] = True
        return holdout

    if mechanism != "mixed_mnar":
        raise ValueError(f"Unsupported mechanism: {mechanism}")

    observed_values = x[finite]
    cutoff = float(np.quantile(observed_values, mnar_low_quantile))
    mnar_mask = finite & (x <= cutoff)
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
        available = finite & (~holdout)
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

    return holdout


def _load_dataset(
    dataset_key: str,
    *,
    data_dir: Path,
    force_download: bool,
    normalization_method: str,
    min_observed_per_feature: int,
    max_features: int | None,
) -> tuple[np.ndarray, list[str], list[str], list[str] | None, dict[str, Any]]:
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
    x = np.asarray(assay.layers["norm"].X, dtype=np.float64)
    sample_ids = [str(v) for v in container.sample_ids.to_list()]
    protein_ids = [str(v) for v in assay.var["_index"].to_list()]

    x, protein_ids = _select_features(
        x,
        protein_ids,
        min_observed_per_feature=min_observed_per_feature,
        max_features=max_features,
    )

    groups: list[str] | None = None
    if "group_regex" in spec:
        groups = _infer_groups_by_regex(sample_ids, str(spec["group_regex"]))
    elif kind == "spectronaut_peptide_long":
        mapping = _extract_sample_group_mapping(
            Path(source_info.get("parse_path", source_info["path"])),
            sample_column=str(spec["sample_column"]),
            group_column=str(spec["group_column"]),
        )
        groups = [mapping.get(sid, "UNKNOWN") for sid in sample_ids]

    meta = {
        "dataset_key": dataset_key,
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "missing_rate": float(np.mean(np.isnan(x))),
        "source": source_info,
        "groups_available": groups is not None,
        "references": dict(spec.get("references", {})),
        "group_a": spec.get("group_a"),
        "group_b": spec.get("group_b"),
        "expected_log2_fc": spec.get("expected_log2_fc"),
    }

    return x, sample_ids, protein_ids, groups, meta


def run_benchmark(
    *,
    datasets: list[str],
    methods: list[str],
    holdout_rates: list[float],
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    available_methods = set(list_impute_methods())
    missing_methods = [m for m in methods if m not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Unknown imputation methods: {missing_methods}. Available: {sorted(available_methods)}"
        )

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}

    for dataset_key in datasets:
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        print(f"[INFO] Loading dataset: {dataset_key}")
        x_base, sample_ids, protein_ids, groups, meta = _load_dataset(
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
            for holdout_rate in holdout_rates:
                for rep in range(repeats):
                    seed = 20260304 + rep * 1000 + int(round(holdout_rate * 100))
                    holdout_mask = generate_holdout_mask(
                        x_base,
                        holdout_rate=holdout_rate,
                        mechanism=mechanism,
                        mnar_fraction=mnar_fraction,
                        mnar_low_quantile=mnar_low_quantile,
                        random_seed=seed,
                    )
                    n_holdout = int(np.sum(holdout_mask))
                    if n_holdout == 0:
                        continue

                    x_masked = np.array(x_base, copy=True)
                    x_masked[holdout_mask] = np.nan

                    for method in methods:
                        print(
                            "[INFO] "
                            f"dataset={dataset_key} mechanism={mechanism} rate={holdout_rate:.2f} "
                            f"rep={rep + 1}/{repeats} method={method}"
                        )

                        row: dict[str, Any] = {
                            "dataset": dataset_key,
                            "method": method,
                            "mechanism": mechanism,
                            "holdout_rate": float(holdout_rate),
                            "seed": int(seed),
                            "n_holdout": int(n_holdout),
                            "n_observed_before": int(np.sum(np.isfinite(x_base))),
                            "success": False,
                        }

                        container_in = _build_impute_container(x_masked, sample_ids, protein_ids)
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
                            x_imputed = np.asarray(
                                out.assays["protein"].layers["imputed"].X,
                                dtype=np.float64,
                            )
                            elapsed = float(time.perf_counter() - start)

                            recon = compute_reconstruction_metrics(
                                y_true=x_base[holdout_mask],
                                y_pred=x_imputed[holdout_mask],
                            )

                            row.update(recon)
                            row["runtime_sec"] = elapsed
                            row["post_missing_rate"] = float(np.mean(np.isnan(x_imputed)))
                            row["within_group_cv_median"] = compute_within_group_cv_median(
                                x_imputed,
                                groups,
                            )

                            expected = DATASET_REGISTRY[dataset_key].get("expected_log2_fc")
                            group_a = DATASET_REGISTRY[dataset_key].get("group_a")
                            group_b = DATASET_REGISTRY[dataset_key].get("group_b")
                            if expected is not None and groups is not None and group_a and group_b:
                                ratio_metrics, _ = compute_ratio_metrics(
                                    x_imputed,
                                    protein_ids,
                                    groups,
                                    group_a=str(group_a),
                                    group_b=str(group_b),
                                    expected_log2fc=expected,
                                )
                                row.update(ratio_metrics)

                            row["success"] = True
                        except Exception as exc:  # noqa: BLE001
                            elapsed = float(time.perf_counter() - start)
                            row["runtime_sec"] = elapsed
                            row["error"] = str(exc)
                            failures.append(
                                {
                                    "dataset": dataset_key,
                                    "method": method,
                                    "mechanism": mechanism,
                                    "holdout_rate": holdout_rate,
                                    "seed": seed,
                                    "error": str(exc),
                                }
                            )

                        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise RuntimeError("No benchmark results generated.")

    raw_out = output_dir / "metrics_raw.csv"
    raw_df.to_csv(raw_out, index=False)

    numeric_cols = [
        c
        for c in raw_df.columns
        if c not in {"dataset", "method", "mechanism", "seed", "error"}
        and pd.api.types.is_numeric_dtype(raw_df[c])
    ]

    summary = (
        raw_df.groupby(["dataset", "method"], dropna=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["runs"] = raw_df.groupby(["dataset", "method"], dropna=False)["success"].size().values
    summary["success_rate"] = (
        raw_df.groupby(["dataset", "method"], dropna=False)["success"].mean().values
    )

    summary_out = output_dir / "metrics_summary.csv"
    summary.to_csv(summary_out, index=False)

    scores = score_methods(summary)
    scores_out = output_dir / "metrics_scores.csv"
    scores.to_csv(scores_out, index=False)

    plot_overall_scores(scores, output_dir / "overall_scores.png")
    plot_metric_heatmap(scores, output_dir / "score_heatmap.png")
    plot_nrmse_curves(raw_df, output_dir / "nrmse_curves.png")
    plot_runtime_vs_accuracy(summary, output_dir / "runtime_vs_accuracy.png")

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": {
            "datasets": datasets,
            "methods": methods,
            "holdout_rates": holdout_rates,
            "mechanisms": mechanisms,
            "repeats": repeats,
            "mnar_fraction": mnar_fraction,
            "mnar_low_quantile": mnar_low_quantile,
            "normalization_method": normalization_method,
            "min_observed_per_feature": min_observed_per_feature,
            "max_features": max_features,
        },
        "dataset_meta": dataset_meta,
        "literature_references": LITERATURE_REFERENCES,
        "n_total_runs": int(raw_df.shape[0]),
        "n_failed_runs": int((~raw_df["success"]).sum()),
        "failures": failures,
        "outputs": {
            "raw": str(raw_out),
            "summary": str(summary_out),
            "scores": str(scores_out),
            "overall_scores_plot": str(output_dir / "overall_scores.png"),
            "score_heatmap_plot": str(output_dir / "score_heatmap.png"),
            "nrmse_curves_plot": str(output_dir / "nrmse_curves.png"),
            "runtime_vs_accuracy_plot": str(output_dir / "runtime_vs_accuracy.png"),
        },
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    if summary.empty:
        raise RuntimeError("No successful benchmark runs. Check failures in metadata output.")

    print("[INFO] Benchmark completed.")
    print(f"[INFO] Raw metrics: {raw_out}")
    print(f"[INFO] Summary metrics: {summary_out}")
    print(f"[INFO] Score table: {scores_out}")
    print(f"[INFO] Failures: {len(failures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ScpTensor imputation methods.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Imputation methods to benchmark.",
    )
    parser.add_argument(
        "--holdout-rates",
        nargs="+",
        type=float,
        default=DEFAULT_HOLDOUT_RATES,
        help="Holdout rates for masked-value recovery.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=DEFAULT_MECHANISMS,
        choices=["mcar", "mixed_mnar"],
        help="Missingness mechanisms used for additional holdout masking.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Replicates per scenario.")
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
        default=1200,
        help="Upper bound for retained protein features per dataset.",
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
    run_benchmark(
        datasets=args.datasets,
        methods=args.methods,
        holdout_rates=args.holdout_rates,
        mechanisms=args.mechanisms,
        repeats=args.repeats,
        mnar_fraction=args.mnar_fraction,
        mnar_low_quantile=args.mnar_low_quantile,
        normalization_method=args.normalization,
        min_observed_per_feature=args.min_observed_per_feature,
        max_features=args.max_features,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force_download=args.force_download,
    )
