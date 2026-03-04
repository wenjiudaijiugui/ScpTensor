"""Run benchmark for protein-level normalization methods."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests
from metrics import (
    EXPECTED_LOG2FC_HYE124,
    compute_distribution_metrics,
    compute_group_metrics,
    compute_ratio_metrics,
    guess_groups_from_sample_ids,
)
from plots import (
    plot_overall_scores,
    plot_ratio_distributions,
    plot_score_heatmap,
    plot_summary_metrics,
)

from scptensor.aggregation import aggregate_to_protein
from scptensor.io import load_diann, load_spectronaut
from scptensor.normalization import normalize
from scptensor.transformation import log_transform

DEFAULT_METHODS = ["none", "mean", "median", "quantile", "trqn"]

DEFAULT_DATASETS = ["lfqbench_hye124_spectronaut", "pxd054343_diann_2x"]

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
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
        "expected_log2_fc_ab": EXPECTED_LOG2FC_HYE124,
        "references": {
            "paper": "https://doi.org/10.1021/acs.jproteome.5b00791",
            "dataset_repo": "https://github.com/IFIproteomics/LFQbench",
            "vignette": "https://github.com/IFIproteomics/LFQbench/blob/master/vignettes/LFQbench.Rmd",
        },
    },
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
}

REFERENCE_DOWNLOADS = {
    "natcomm_2025_source_data": {
        "url": (
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65174-4/"
            "MediaObjects/41467_2025_65174_MOESM12_ESM.xlsx"
        ),
        "filename": "41467_2025_65174_MOESM12_ESM.xlsx",
        "description": "Nature Communications 2025 source data (protein-level workflow benchmark)",
        "paper": "https://www.nature.com/articles/s41467-025-65174-4",
    }
}

METRIC_DIRECTIONS: dict[str, bool] = {
    "coverage_ratio": True,
    "feature_quantified_ratio": True,
    "sample_median_mad": False,
    "sample_iqr_cv": False,
    "rle_mad_median": False,
    "pairwise_wasserstein_median": False,
    "within_group_sd_median": False,
    "group_eta2_median": True,
    "ratio_mae": False,
    "ratio_rmse": False,
    "ratio_pairwise_auc_mean": True,
    "ratio_changed_vs_bg_auc": True,
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
    """Convert CR-only files to LF for stable CSV parsing."""
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


def _score_methods(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _dataset, block in summary_df.groupby("dataset", sort=False):
        work = block.copy()
        score_cols: list[str] = []

        for metric, higher_better in METRIC_DIRECTIONS.items():
            if metric not in work.columns:
                continue
            vals = work[metric].astype(float)
            finite = np.isfinite(vals.to_numpy(dtype=np.float64))
            score_col = f"score_{metric}"
            score_cols.append(score_col)

            if np.sum(finite) < 2:
                work[score_col] = np.nan
                continue

            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if np.isclose(vmin, vmax):
                work[score_col] = np.where(np.isfinite(vals), 1.0, np.nan)
                continue

            if higher_better:
                work[score_col] = (vals - vmin) / (vmax - vmin)
            else:
                work[score_col] = (vmax - vals) / (vmax - vmin)

        if score_cols:
            work["overall_score"] = np.nanmean(work[score_cols].to_numpy(dtype=np.float64), axis=1)
        else:
            work["overall_score"] = np.nan

        for row in work.to_dict("records"):
            rows.append(row)

    return pd.DataFrame(rows)


def _load_dataset(
    dataset_key: str,
    *,
    data_dir: Path,
    force_download: bool,
    fdr_threshold: float,
) -> tuple[Any, list[str] | None, dict[str, Any] | None, dict[str, Any]]:
    spec = DATASET_REGISTRY[dataset_key]
    kind = str(spec["kind"])

    if kind == "spectronaut_peptide_long":
        data_path = _download_file(
            str(spec["url"]),
            data_dir / dataset_key / str(spec["filename"]),
            force=force_download,
        )
        parse_path = _normalize_line_endings_if_needed(data_path)

        peptide_container = load_spectronaut(
            parse_path,
            level="peptide",
            table_format="long",
            quantity_column=str(spec["quantity_column"]),
            assay_name="peptides",
            layer_name="raw",
            fdr_threshold=fdr_threshold,
        )

        container = aggregate_to_protein(
            peptide_container,
            source_assay="peptides",
            source_layer="raw",
            target_assay="proteins",
            method="sum",
            protein_column=str(spec["protein_column"]),
            keep_unmapped=False,
        )

        sample_to_group = _extract_sample_group_mapping(
            parse_path,
            sample_column=str(spec["sample_column"]),
            group_column=str(spec["group_column"]),
        )
        sample_ids = [str(v) for v in container.sample_ids.to_list()]
        groups = [sample_to_group.get(sid, "UNKNOWN") for sid in sample_ids]

        ratio_config = {
            "group_a": str(spec["group_a"]),
            "group_b": str(spec["group_b"]),
            "expected_log2fc": dict(spec["expected_log2_fc_ab"]),
        }

        metadata = {
            "dataset_key": dataset_key,
            "kind": kind,
            "source_file": str(data_path),
            "parse_file": str(parse_path),
            "references": dict(spec["references"]),
        }
        return container, groups, ratio_config, metadata

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
            fdr_threshold=fdr_threshold,
        )

        sample_ids = [str(v) for v in container.sample_ids.to_list()]
        groups = None
        if "group_regex" in spec:
            groups = _infer_groups_by_regex(sample_ids, str(spec["group_regex"]))
        if groups is None:
            groups = guess_groups_from_sample_ids(sample_ids)

        metadata = {
            "dataset_key": dataset_key,
            "kind": kind,
            "source_file": str(path),
            "references": dict(spec["references"]),
        }
        return container, groups, None, metadata

    raise ValueError(f"Unsupported dataset kind for '{dataset_key}': {kind}")


def _download_reference_materials(data_dir: Path, force_download: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    reference_dir = data_dir / "references"
    for key, spec in REFERENCE_DOWNLOADS.items():
        file_path = _download_file(
            str(spec["url"]),
            reference_dir / str(spec["filename"]),
            force=force_download,
        )
        out[key] = str(file_path)
    return out


def run_benchmark(
    *,
    datasets: list[str],
    methods: list[str],
    data_dir: Path,
    output_dir: Path,
    fdr_threshold: float,
    log_base: float,
    log_offset: float,
    force_download: bool,
    download_references: bool,
) -> None:
    for key in datasets:
        if key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset key: {key}")

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Datasets: {', '.join(datasets)}")
    print(f"[INFO] Methods: {', '.join(methods)}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    reference_paths: dict[str, str] = {}
    if download_references:
        reference_paths = _download_reference_materials(data_dir, force_download=force_download)

    summary_rows: list[dict[str, Any]] = []
    ratio_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    dataset_metadata: list[dict[str, Any]] = []

    for dataset_key in datasets:
        print(f"[INFO] Loading dataset: {dataset_key}")
        try:
            container, groups, ratio_cfg, meta = _load_dataset(
                dataset_key,
                data_dir=data_dir,
                force_download=force_download,
                fdr_threshold=fdr_threshold,
            )
            dataset_metadata.append(meta)
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset_key,
                    "method": "<load>",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        container = log_transform(
            container,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log2",
            base=log_base,
            offset=log_offset,
            detect_logged=True,
            skip_if_logged=True,
        )

        protein_ids = [str(v) for v in container.assays["proteins"].var["_index"].to_list()]
        for method in methods:
            print(f"[INFO] Dataset={dataset_key} Method={method}")
            try:
                method_layer = f"norm_{method}"
                normalized = normalize(
                    container.copy(),
                    method=method,
                    assay_name="proteins",
                    source_layer="log2",
                    new_layer_name=method_layer,
                )

                matrix = np.asarray(
                    normalized.assays["proteins"].layers[method_layer].X,
                    dtype=np.float64,
                )

                row: dict[str, Any] = {
                    "dataset": dataset_key,
                    "method": method,
                }
                row.update(compute_distribution_metrics(matrix))
                row.update(compute_group_metrics(matrix, groups))

                if ratio_cfg is not None:
                    ratio_metrics, ratio_table = compute_ratio_metrics(
                        matrix,
                        protein_ids,
                        groups,
                        group_a=str(ratio_cfg["group_a"]),
                        group_b=str(ratio_cfg["group_b"]),
                        expected_log2fc=dict(ratio_cfg["expected_log2fc"]),
                    )
                    row.update(ratio_metrics)
                    if not ratio_table.empty:
                        ratio_table["dataset"] = dataset_key
                        ratio_table["method"] = method
                        ratio_frames.append(ratio_table)
                else:
                    row.update(
                        {
                            "ratio_n_quantified": float("nan"),
                            "ratio_mae": float("nan"),
                            "ratio_rmse": float("nan"),
                            "ratio_bias": float("nan"),
                            "ratio_pairwise_auc_mean": float("nan"),
                            "ratio_changed_vs_bg_auc": float("nan"),
                        }
                    )

                summary_rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset_key,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("No successful benchmark runs. Check failures in metadata output.")

    summary_df = summary_df.sort_values(["dataset", "method"]).reset_index(drop=True)
    score_df = _score_methods(summary_df)
    ratio_df = pd.concat(ratio_frames, ignore_index=True) if ratio_frames else pd.DataFrame()

    summary_path = output_dir / "metrics_summary.csv"
    score_path = output_dir / "metrics_scores.csv"
    ratio_path = output_dir / "ratio_protein_table.csv"
    summary_df.to_csv(summary_path, index=False)
    score_df.to_csv(score_path, index=False)
    if not ratio_df.empty:
        ratio_df.to_csv(ratio_path, index=False)

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_df.to_csv(output_dir / "failures.csv", index=False)

    plot_summary_metrics(summary_df, output_dir / "summary_metrics.png")
    plot_score_heatmap(score_df, output_dir / "score_heatmap.png")
    plot_overall_scores(score_df, output_dir / "overall_scores.png")
    if not ratio_df.empty:
        plot_ratio_distributions(ratio_df, output_dir / "ratio_distribution.png")

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "datasets": datasets,
        "methods": methods,
        "fdr_threshold": fdr_threshold,
        "log_base": log_base,
        "log_offset": log_offset,
        "summary_file": str(summary_path),
        "score_file": str(score_path),
        "ratio_file": str(ratio_path) if ratio_path.exists() else None,
        "reference_downloads": reference_paths,
        "dataset_metadata": dataset_metadata,
        "failures": failures,
        "metric_directions": METRIC_DIRECTIONS,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print("[INFO] Benchmark completed.")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Scores: {score_path}")
    if ratio_path.exists():
        print(f"[INFO] Ratio table: {ratio_path}")
    if failures:
        print(f"[WARN] Some runs failed. See: {output_dir / 'failures.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/normalization/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/normalization/outputs"))
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--log-base", type=float, default=2.0)
    parser.add_argument("--log-offset", type=float, default=1.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-reference-download", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        datasets=[str(v) for v in args.datasets],
        methods=[str(v) for v in args.methods],
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fdr_threshold=float(args.fdr_threshold),
        log_base=float(args.log_base),
        log_offset=float(args.log_offset),
        force_download=bool(args.force_download),
        download_references=not bool(args.no_reference_download),
    )


if __name__ == "__main__":
    main()
