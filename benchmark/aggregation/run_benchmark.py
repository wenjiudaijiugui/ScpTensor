"""Run benchmark for peptide->protein aggregation methods."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests
from metrics import (
    EXPECTED_LOG2FC_HYE124,
    compute_protein_level_table,
    summarize_de_consistency_proxy,
    summarize_mapping_burden,
    summarize_method,
    summarize_state_burden,
)
from plots import (
    plot_cv_distribution,
    plot_log2fc_distribution,
    plot_metric_heatmap,
    plot_observed_vs_expected,
    plot_species_coverage,
    plot_summary_metrics,
)

from scptensor.aggregation import aggregate_to_protein
from scptensor.io import load_spectronaut

DEFAULT_METHODS = [
    "sum",
    "mean",
    "median",
    "max",
    "weighted_mean",
    "top_n",
    "maxlfq",
    "tmp",
    "ibaq",
]

DATASET_REGISTRY = {
    "lfqbench_hye124_spectronaut": {
        "url": (
            "https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/"
            "vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv"
        ),
        "filename": "Spectronaut_TTOF6600_64w_example.tsv",
        "software": "spectronaut",
        "quantity_column": "FG.NormalizedTotalPeakArea",
        "protein_column": "EG.ProteinId",
        "sample_column": "R.FileName",
        "group_column": "R.Condition",
        "group_a": "A",
        "group_b": "B",
        "expected_log2_fc_ab": EXPECTED_LOG2FC_HYE124,
        "references": {
            "dataset": "https://github.com/IFIproteomics/LFQbench",
            "vignette": "https://github.com/IFIproteomics/LFQbench/blob/master/vignettes/LFQbench.Rmd",
        },
    }
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
    """Convert CR-only files to LF so Polars can parse them reliably."""
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
    if pairs.is_empty():
        raise ValueError(
            "Failed to derive sample-group mapping from source table. "
            f"Columns used: sample='{sample_column}', group='{group_column}'."
        )

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


def _resolve_group_indices(
    sample_ids: list[str],
    sample_to_group: dict[str, str],
    *,
    group_a: str,
    group_b: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: list[str] = []
    missing: list[str] = []

    for sid in sample_ids:
        grp = sample_to_group.get(sid)
        if grp is None:
            missing.append(sid)
            grp = "<MISSING_GROUP>"
        groups.append(grp)

    if missing:
        raise ValueError(
            "Some sample IDs from ScpContainer were not found in source sample-group mapping: "
            f"{missing}"
        )

    groups_array = np.asarray(groups)
    idx_a = np.where(groups_array == group_a)[0]
    idx_b = np.where(groups_array == group_b)[0]

    if idx_a.size < 2 or idx_b.size < 2:
        raise ValueError(
            "Insufficient replicates for A/B comparison. "
            f"group '{group_a}' count={idx_a.size}, group '{group_b}' count={idx_b.size}."
        )

    return idx_a, idx_b, groups


def run_benchmark(
    *,
    dataset_key: str,
    methods: list[str],
    data_dir: Path,
    output_dir: Path,
    fdr_threshold: float,
    quantity_column_override: str | None,
    force_download: bool,
) -> None:
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    spec = DATASET_REGISTRY[dataset_key]
    data_path = _download_file(
        spec["url"],
        data_dir / str(spec["filename"]),
        force=force_download,
    )
    parse_path = _normalize_line_endings_if_needed(data_path)

    quantity_col = quantity_column_override or str(spec["quantity_column"])

    print(f"[INFO] Dataset: {dataset_key}")
    print(f"[INFO] Data file: {data_path}")
    if parse_path != data_path:
        print(f"[INFO] Normalized parse file: {parse_path}")
    print(f"[INFO] Quantity column: {quantity_col}")
    print(f"[INFO] Methods: {', '.join(methods)}")

    peptide_container = load_spectronaut(
        parse_path,
        level="peptide",
        table_format="long",
        quantity_column=quantity_col,
        fdr_threshold=fdr_threshold,
        assay_name="peptides",
        layer_name="raw",
    )

    sample_to_group = _extract_sample_group_mapping(
        parse_path,
        sample_column=str(spec["sample_column"]),
        group_column=str(spec["group_column"]),
    )

    sample_ids = [str(x) for x in peptide_container.sample_ids.to_list()]
    idx_a, idx_b, sample_groups = _resolve_group_indices(
        sample_ids,
        sample_to_group,
        group_a=str(spec["group_a"]),
        group_b=str(spec["group_b"]),
    )
    mapping_burden = summarize_mapping_burden(
        peptide_container.assays["peptides"]
        .var[str(spec["protein_column"])]
        .cast(pl.Utf8, strict=False)
        .to_list()
    )

    summary_rows: list[dict[str, object]] = []
    protein_frames: list[pd.DataFrame] = []
    species_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    expected_map = spec["expected_log2_fc_ab"]

    for method in methods:
        print(f"[INFO] Running method: {method}")
        try:
            aggregated = aggregate_to_protein(
                peptide_container,
                source_assay="peptides",
                source_layer="raw",
                target_assay="proteins",
                method=method,
                protein_column=str(spec["protein_column"]),
                keep_unmapped=False,
            )

            protein_assay = aggregated.assays["proteins"]
            matrix = np.asarray(protein_assay.layers["raw"].X, dtype=np.float64)
            protein_ids = [str(x) for x in protein_assay.var["_index"].to_list()]

            protein_table = compute_protein_level_table(
                method=method,
                protein_ids=protein_ids,
                matrix=matrix,
                group_a_idx=idx_a,
                group_b_idx=idx_b,
            )

            summary, quantified, per_species, pairwise_auc = summarize_method(
                protein_table,
                expected_log2fc=expected_map,
                background_species="HUMAN",
            )
            summary.update(
                summarize_state_burden(
                    protein_assay.layers["raw"].M,
                    shape=matrix.shape,
                )
            )
            summary.update(summarize_de_consistency_proxy(quantified, background_species="HUMAN"))
            summary.update(mapping_burden)

            summary_rows.append(summary)
            protein_frames.append(quantified)
            species_frames.append(per_species)
            pairwise_frames.append(pairwise_auc)
        except Exception as exc:  # pragma: no cover
            failures.append({"method": method, "error": str(exc)})
            print(f"[WARN] Method '{method}' failed: {exc}")

    if not summary_rows:
        raise RuntimeError("All aggregation methods failed; no benchmark results generated.")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    protein_df = pd.concat(protein_frames, ignore_index=True)
    species_df = pd.concat(species_frames, ignore_index=True)
    pairwise_df = pd.concat(pairwise_frames, ignore_index=True)

    summary_df = summary_df.sort_values(
        ["mae", "species_overlap_auc_mean"], ascending=[True, False]
    )

    summary_path = output_dir / "metrics_summary.csv"
    proteins_path = output_dir / "protein_level_results.csv"
    species_path = output_dir / "species_coverage_summary.csv"
    pairwise_path = output_dir / "pairwise_auc.csv"

    summary_df.to_csv(summary_path, index=False)
    protein_df.to_csv(proteins_path, index=False)
    species_df.to_csv(species_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_methods.csv", index=False)

    plot_summary_metrics(summary_df, output_dir / "summary_metrics.png")
    plot_log2fc_distribution(protein_df, output_dir / "log2fc_distribution.png")
    plot_observed_vs_expected(protein_df, output_dir / "observed_vs_expected.png")
    plot_cv_distribution(protein_df, output_dir / "cv_distribution.png")
    plot_species_coverage(species_df, output_dir / "species_coverage.png")
    plot_metric_heatmap(summary_df, output_dir / "metric_heatmap.png")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_key": dataset_key,
        "dataset_path": str(data_path),
        "parse_path": str(parse_path),
        "references": spec["references"],
        "group_a": spec["group_a"],
        "group_b": spec["group_b"],
        "sample_ids": sample_ids,
        "sample_groups": sample_groups,
        "methods_requested": methods,
        "methods_succeeded": summary_df["method"].tolist(),
        "methods_failed": failures,
        "fdr_threshold": fdr_threshold,
        "quantity_column": quantity_col,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print("\n[INFO] Benchmark finished.")
    print(f"[INFO] Summary table: {summary_path}")
    print(f"[INFO] Protein-level table: {proteins_path}")
    print(f"[INFO] Plots folder: {output_dir}")

    top_cols = [
        "method",
        "mae",
        "rmse",
        "species_overlap_auc_mean",
        "changed_vs_background_auc",
        "coverage_ratio",
    ]
    print("\n[INFO] Top methods (sorted by MAE):")
    print(summary_df[top_cols].head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark peptide->protein aggregation methods.")
    parser.add_argument(
        "--dataset",
        default="lfqbench_hye124_spectronaut",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Benchmark dataset key.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Aggregation methods to benchmark.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/aggregation/data"),
        help="Directory for downloaded benchmark data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/aggregation/outputs"),
        help="Directory for benchmark outputs (tables + figures).",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.01,
        help="FDR threshold passed to load_spectronaut (set <0 to disable).",
    )
    parser.add_argument(
        "--quantity-column",
        type=str,
        default=None,
        help="Override quantity column (default uses dataset preset).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download dataset even if local file exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fdr_threshold = None if args.fdr_threshold < 0 else args.fdr_threshold

    run_benchmark(
        dataset_key=args.dataset,
        methods=args.methods,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fdr_threshold=fdr_threshold,
        quantity_column_override=args.quantity_column,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
