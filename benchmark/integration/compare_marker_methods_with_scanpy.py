#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ScpTensor marker ranking against a scanpy-based max_fold_change ranking."
        )
    )
    parser.add_argument(
        "--container",
        type=Path,
        required=True,
        help="Path to ScpTensor cluster container pickle (.pkl.gz).",
    )
    parser.add_argument(
        "--reference-h5ad",
        type=Path,
        required=True,
        help="Path to reference AnnData .h5ad file.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        required=False,
        default=None,
        help=(
            "Optional CSV with columns gene_symbol_combined, matched_accession "
            "for article marker mapping."
        ),
    )
    parser.add_argument(
        "--cluster-col",
        type=str,
        default="scptensor_cluster",
        help="Cluster column in container obs used for group-based marker ranking.",
    )
    parser.add_argument(
        "--maxfc-layer",
        type=str,
        default="log2_norm_median_row_mean_none",
        help="Protein layer used to compute scanpy max_fold_change ranking.",
    )
    parser.add_argument(
        "--heuristic-layer",
        type=str,
        default="zscore",
        help="Protein layer used by existing heuristic marker score.",
    )
    parser.add_argument(
        "--scanpy-method",
        type=str,
        default="wilcoxon",
        choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
        help="scanpy rank_genes_groups method.",
    )
    parser.add_argument(
        "--maxfc-layer-is-log2",
        action="store_true",
        default=True,
        help=(
            "Interpret --maxfc-layer as log2 scale and convert to scanpy-friendly "
            "log1p(linear) before rank_genes_groups."
        ),
    )
    parser.add_argument(
        "--no-maxfc-layer-is-log2",
        dest="maxfc_layer_is_log2",
        action="store_false",
        help="Disable log2->log1p conversion for --maxfc-layer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write comparison outputs.",
    )
    return parser.parse_args()


def _require_column(df: pd.DataFrame, column: str, context: str) -> None:
    if column not in df.columns:
        raise ValueError(f"missing required column '{column}' in {context}")


def load_container(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def get_obs_frame(container: Any) -> pd.DataFrame:
    obs = getattr(container, "obs", None)
    if obs is None:
        raise ValueError("container has no obs table")
    if hasattr(obs, "to_pandas"):
        obs_pd = obs.to_pandas()
    elif isinstance(obs, pd.DataFrame):
        obs_pd = obs.copy()
    else:
        raise TypeError(f"unsupported obs type: {type(obs)!r}")
    return obs_pd


def get_protein_layer(container: Any, layer_name: str) -> tuple[np.ndarray, list[str]]:
    assays = getattr(container, "assays", None)
    if assays is None or "proteins" not in assays:
        raise ValueError("container has no 'proteins' assay")
    proteins = assays["proteins"]
    if layer_name not in proteins.layers:
        available = ", ".join(sorted(proteins.layers.keys()))
        raise ValueError(
            f"layer '{layer_name}' not found in proteins assay; available: {available}"
        )
    matrix = proteins.layers[layer_name].X
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"layer '{layer_name}' is not 2D")
    feature_ids = proteins.feature_ids
    if hasattr(feature_ids, "to_list"):
        accessions = [str(x) for x in feature_ids.to_list()]
    else:
        accessions = [str(x) for x in feature_ids]
    if matrix.shape[1] != len(accessions):
        raise ValueError("protein matrix column count does not match number of protein accessions")
    return matrix, accessions


def heuristic_marker_scores(
    matrix: np.ndarray,
    clusters: pd.Series,
    accessions: list[str],
) -> pd.DataFrame:
    cluster_values = clusters.astype(str).to_numpy()
    unique_clusters = sorted(pd.unique(cluster_values))
    means = []
    for cluster in unique_clusters:
        idx = np.where(cluster_values == cluster)[0]
        if idx.size == 0:
            continue
        means.append(matrix[idx, :].mean(axis=0))
    if not means:
        raise ValueError("no clusters found for heuristic marker scoring")
    mean_mat = np.vstack(means)
    score = mean_mat.max(axis=0) - np.median(mean_mat, axis=0)
    out = pd.DataFrame(
        {"accession": accessions, "heuristic_marker_score": score.astype(float)}
    ).sort_values("heuristic_marker_score", ascending=False, kind="stable")
    out["heuristic_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out.reset_index(drop=True)


def scanpy_max_fold_change(
    matrix: np.ndarray,
    clusters: pd.Series,
    accessions: list[str],
    method: str,
    layer_is_log2: bool,
) -> pd.DataFrame:
    matrix_scanpy = matrix.astype(np.float64, copy=False)
    if layer_is_log2:
        # scanpy rank_genes_groups computes fold changes assuming log1p input.
        # Convert log2 intensities to positive linear scale first, then log1p.
        matrix_scanpy = np.log1p(np.exp2(matrix_scanpy))
    adata = ad.AnnData(
        X=matrix_scanpy,
        obs=pd.DataFrame({"cluster": clusters.astype(str).to_numpy()}),
        var=pd.DataFrame(index=pd.Index(accessions, name="accession")),
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby="cluster",
        groups="all",
        reference="rest",
        method=method,
        n_genes=adata.n_vars,
        use_raw=False,
    )
    groups = [str(x) for x in pd.unique(adata.obs["cluster"])]
    all_rows: list[pd.DataFrame] = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata, group=group)
        if "names" not in df.columns:
            continue
        if "logfoldchanges" not in df.columns:
            df["logfoldchanges"] = np.nan
        all_rows.append(
            pd.DataFrame(
                {
                    "accession": df["names"].astype(str).to_numpy(),
                    "cluster": group,
                    "logfoldchanges": pd.to_numeric(
                        df["logfoldchanges"], errors="coerce"
                    ).to_numpy(),
                }
            )
        )

    if not all_rows:
        raise ValueError("scanpy did not return rank_genes_groups rows")
    long_df = pd.concat(all_rows, ignore_index=True)
    max_fc = (
        long_df.groupby("accession", as_index=False)["logfoldchanges"]
        .max()
        .rename(columns={"logfoldchanges": "max_fold_change_scanpy"})
    )
    max_fc["max_fold_change_scanpy"] = pd.to_numeric(
        max_fc["max_fold_change_scanpy"], errors="coerce"
    ).fillna(float("-inf"))
    max_fc = max_fc.sort_values(
        "max_fold_change_scanpy", ascending=False, kind="stable"
    ).reset_index(drop=True)
    max_fc["maxfc_rank"] = np.arange(1, len(max_fc) + 1, dtype=int)
    return max_fc


def load_reference_maxfc(reference_h5ad: Path) -> pd.DataFrame:
    ref = ad.read_h5ad(reference_h5ad)
    var = ref.var.copy()
    _require_column(var, "Accession", "reference var")
    _require_column(var, "max_fold_change", "reference var")
    out = pd.DataFrame(
        {
            "accession": var["Accession"].astype(str).to_numpy(),
            "reference_max_fold_change": pd.to_numeric(
                var["max_fold_change"], errors="coerce"
            ).to_numpy(),
        }
    )
    out = out.dropna(subset=["reference_max_fold_change"]).drop_duplicates(
        subset=["accession"], keep="first"
    )
    out = out.sort_values("reference_max_fold_change", ascending=False, kind="stable").reset_index(
        drop=True
    )
    out["reference_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def topk_overlap(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return float("nan")
    denom = max(len(set_a), len(set_b))
    return float(len(set_a & set_b) / denom)


def maybe_article_marker_table(
    mapping_csv: Path | None,
    merged: pd.DataFrame,
    output_dir: Path,
) -> tuple[int, int]:
    if mapping_csv is None:
        return (0, 0)
    mapping = pd.read_csv(mapping_csv)
    _require_column(mapping, "gene_symbol_combined", "mapping csv")
    _require_column(mapping, "matched_accession", "mapping csv")
    table = mapping.rename(
        columns={
            "gene_symbol_combined": "gene_symbol",
            "matched_accession": "accession",
        }
    )
    table["accession"] = table["accession"].astype(str).replace({"nan": ""})
    merged_table = table.merge(merged, on="accession", how="left")
    merged_table.to_csv(output_dir / "article_marker_method_comparison.csv", index=False)
    present = int((merged_table["accession"].str.len() > 0).sum())
    matched = int(merged_table["maxfc_rank"].notna().sum())
    return (present, matched)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    container = load_container(args.container)
    obs = get_obs_frame(container)
    _require_column(obs, args.cluster_col, "container obs")

    matrix_fc, accessions_fc = get_protein_layer(container, args.maxfc_layer)
    matrix_heur, accessions_heur = get_protein_layer(container, args.heuristic_layer)
    if accessions_fc != accessions_heur:
        raise ValueError("feature IDs differ between maxfc-layer and heuristic-layer")

    clusters = obs[args.cluster_col]
    heuristic_df = heuristic_marker_scores(matrix_heur, clusters, accessions_fc)
    maxfc_df = scanpy_max_fold_change(
        matrix=matrix_fc,
        clusters=clusters,
        accessions=accessions_fc,
        method=args.scanpy_method,
        layer_is_log2=args.maxfc_layer_is_log2,
    )
    reference_df = load_reference_maxfc(args.reference_h5ad)

    merged = (
        maxfc_df.merge(heuristic_df, on="accession", how="outer")
        .merge(reference_df, on="accession", how="left")
        .sort_values("maxfc_rank", ascending=True, kind="stable")
        .reset_index(drop=True)
    )

    top20_ref = set(reference_df.nsmallest(20, "reference_rank")["accession"].tolist())
    top20_maxfc = set(maxfc_df.nsmallest(20, "maxfc_rank")["accession"].tolist())
    top20_heur = set(heuristic_df.nsmallest(20, "heuristic_rank")["accession"].tolist())

    corr_df = merged.dropna(
        subset=["reference_max_fold_change", "max_fold_change_scanpy", "heuristic_marker_score"]
    )
    spearman_maxfc = (
        float(
            corr_df["reference_max_fold_change"].corr(
                corr_df["max_fold_change_scanpy"], method="spearman"
            )
        )
        if not corr_df.empty
        else float("nan")
    )
    spearman_heur = (
        float(
            corr_df["reference_max_fold_change"].corr(
                corr_df["heuristic_marker_score"], method="spearman"
            )
        )
        if not corr_df.empty
        else float("nan")
    )

    present_article, matched_article = maybe_article_marker_table(
        mapping_csv=args.mapping_csv,
        merged=merged,
        output_dir=args.output_dir,
    )

    maxfc_df.to_csv(args.output_dir / "scptensor_markers_max_fold_change.csv", index=False)
    heuristic_df.to_csv(args.output_dir / "scptensor_markers_heuristic_full.csv", index=False)
    merged.to_csv(args.output_dir / "marker_method_comparison_full.csv", index=False)

    summary = {
        "container": str(args.container),
        "reference_h5ad": str(args.reference_h5ad),
        "cluster_column": args.cluster_col,
        "maxfc_layer": args.maxfc_layer,
        "heuristic_layer": args.heuristic_layer,
        "scanpy_method": args.scanpy_method,
        "maxfc_layer_is_log2": bool(args.maxfc_layer_is_log2),
        "n_features": int(len(accessions_fc)),
        "n_samples": int(matrix_fc.shape[0]),
        "reference_features_with_maxfc": int(len(reference_df)),
        "shared_features_with_reference": int(merged["reference_max_fold_change"].notna().sum()),
        "spearman_vs_reference_maxfc": {
            "scanpy_max_fold_change": spearman_maxfc,
            "heuristic_marker_score": spearman_heur,
        },
        "top20_overlap_with_reference": {
            "scanpy_max_fold_change": topk_overlap(top20_maxfc, top20_ref),
            "heuristic_marker_score": topk_overlap(top20_heur, top20_ref),
        },
        "article_marker_mapping": {
            "mapped_accessions_in_table": present_article,
            "matched_to_scptensor_features": matched_article,
        },
    }
    (args.output_dir / "marker_method_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
