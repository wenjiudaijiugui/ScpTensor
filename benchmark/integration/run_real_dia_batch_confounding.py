"""Run DIA single-cell integration benchmark with balanced/confounded scenarios.

Output contract is aligned with other benchmark modules:
- metrics_raw.csv
- metrics_summary.csv
- metrics_scores.csv
- run_metadata.json
- summary_metrics.png
- score_heatmap.png
- overall_scores.png

Legacy report JSON is still generated for backward compatibility.
"""

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
from metrics import (
    BALANCED_METRIC_DIRECTIONS,
    CONFOUNDED_METRIC_DIRECTIONS,
    score_methods,
)
from plots import plot_overall_scores, plot_score_heatmap, plot_summary_metrics
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from scptensor.impute import impute
from scptensor.integration import integrate_combat, integrate_limma, integrate_mnn
from scptensor.integration.diagnostics import (
    compute_batch_asw,
    compute_batch_mixing_metric,
    compute_lisi_approx,
)
from scptensor.io import load_diann
from scptensor.normalization import normalize
from scptensor.transformation import log_transform

DEFAULT_DATASET_KEY = "pxd054343_diann_1_sc_lf"
DEFAULT_DATA_PATH = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
DEFAULT_OUTPUT_DIR = Path("benchmark/integration/outputs")
DEFAULT_SCENARIOS = [
    "balanced_amount_by_sample",
    "confounded_amount_as_batch",
]
DEFAULT_METHODS = [
    "combat_parametric",
    "combat_nonparametric",
    "limma",
    "mnn",
]

METHOD_SPECS: dict[str, tuple[Any, dict[str, Any]]] = {
    "combat_parametric": (integrate_combat, {"eb_mode": "parametric"}),
    "combat_nonparametric": (integrate_combat, {"eb_mode": "nonparametric"}),
    "limma": (integrate_limma, {}),
    "mnn": (integrate_mnn, {"k": 10, "sigma": 1.0}),
}

SCENARIO_SPECS: dict[str, dict[str, Any]] = {
    "balanced_amount_by_sample": {
        "batch_key": "batch_balanced",
        "condition_key": "condition_balanced",
        "score_profile": "balanced",
        "guardrail_expectation": "success",
        "description": "Batch=S1/S2/S3, Condition=SCamount(300ng/600ng); balanced design",
    },
    "confounded_amount_as_batch": {
        "batch_key": "batch_confounded",
        "condition_key": "condition_confounded",
        "score_profile": "confounded",
        "guardrail_expectation": "rank_deficient_error",
        "description": "Batch=Condition=SCamount; fully confounded design",
    },
}

SCORE_PROFILE_DIRECTIONS: dict[str, dict[str, bool]] = {
    "balanced": BALANCED_METRIC_DIRECTIONS,
    "confounded": CONFOUNDED_METRIC_DIRECTIONS,
}

LITERATURE_REFERENCES = {
    "dia_sc_workflow_benchmark": "https://www.nature.com/articles/s41467-025-65174-4",
    "scplainer_benchmark": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4",
    "protein_level_batch_benchmark": "https://www.nature.com/articles/s41467-025-64718-y",
    "scib_metrics_framework": "https://doi.org/10.1038/s41592-021-01336-8",
    "combat_original": "https://pubmed.ncbi.nlm.nih.gov/16632515/",
    "limma_linear_model": "https://pubmed.ncbi.nlm.nih.gov/25605792/",
    "mnn_original": "https://pubmed.ncbi.nlm.nih.gov/29608177/",
}


def _layer_to_dense(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _extract_by_regex(values: list[str], pattern: str, fallback: str) -> list[str]:
    rx = re.compile(pattern)
    out: list[str] = []
    for value in values:
        matched = rx.search(str(value))
        out.append(matched.group(1) if matched else fallback)
    return out


def _attach_scenario_labels(container) -> None:
    runs = [str(x) for x in container.obs["_index"].to_list()]
    amount_group = _extract_by_regex(runs, r"SCamount(\d+ng)", fallback="unknown")
    sample_group = _extract_by_regex(runs, r"_(S\d+)_", fallback="unknown")

    if len(set(amount_group)) < 2:
        raise ValueError(
            "Cannot build amount groups from run names: less than 2 groups detected."
        )
    if len(set(sample_group)) < 2:
        raise ValueError(
            "Cannot build sample groups from run names: less than 2 groups detected."
        )

    container.obs = container.obs.with_columns(
        pl.Series("amount_group", amount_group),
        pl.Series("sample_group", sample_group),
        pl.Series("batch_balanced", sample_group),
        pl.Series("condition_balanced", amount_group),
        pl.Series("batch_confounded", amount_group),
        pl.Series("condition_confounded", amount_group),
    )


def _between_group_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return 0.0

    group_means = [np.mean(x[labels == g], axis=0) for g in unique_labels]
    grand_mean = np.mean(x, axis=0)

    between_ss = sum(
        len(x[labels == g]) * np.sum((g_mean - grand_mean) ** 2)
        for g, g_mean in zip(unique_labels, group_means, strict=False)
    )
    total_ss = np.sum((x - grand_mean) ** 2)
    if total_ss <= 0:
        return 0.0
    return float(between_ss / total_ss)


def _safe_condition_asw(x: np.ndarray, condition_labels: np.ndarray) -> float:
    unique, counts = np.unique(condition_labels, return_counts=True)
    if unique.size < 2 or np.min(counts) < 2:
        return float("nan")
    try:
        return float(silhouette_score(x, condition_labels))
    except Exception:  # noqa: BLE001
        return float("nan")


def _safe_condition_cluster_scores(x: np.ndarray, condition_labels: np.ndarray) -> tuple[float, float]:
    unique = np.unique(condition_labels)
    n_clusters = int(unique.size)
    if n_clusters < 2 or x.shape[0] <= n_clusters:
        return float("nan"), float("nan")

    try:
        pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")

    try:
        ari = float(adjusted_rand_score(condition_labels, pred))
    except Exception:  # noqa: BLE001
        ari = float("nan")
    try:
        nmi = float(normalized_mutual_info_score(condition_labels, pred))
    except Exception:  # noqa: BLE001
        nmi = float("nan")
    return ari, nmi


def _condition_knn_purity(x: np.ndarray, condition_labels: np.ndarray, n_neighbors: int) -> float:
    unique = np.unique(condition_labels)
    if unique.size < 2 or x.shape[0] <= 2:
        return float("nan")

    n_neighbors = max(2, min(n_neighbors, x.shape[0] - 1))
    try:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x)
        _, indices = nbrs.kneighbors(x)
    except Exception:  # noqa: BLE001
        return float("nan")

    scores: list[float] = []
    for idx in indices:
        neigh = condition_labels[idx[1:]]
        if neigh.size == 0:
            continue
        majority = np.sum(neigh == condition_labels[idx[0]]) / float(neigh.size)
        scores.append(float(majority))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _compute_metrics(
    container,
    *,
    assay_name: str,
    layer_name: str,
    batch_key: str,
    condition_key: str,
) -> dict[str, float]:
    x = _layer_to_dense(container.assays[assay_name].layers[layer_name].X)
    batch_labels = container.obs[batch_key].cast(pl.Utf8).to_numpy()
    condition_labels = container.obs[condition_key].cast(pl.Utf8).to_numpy()
    n_neighbors = max(3, min(10, x.shape[0] - 1))

    condition_ari, condition_nmi = _safe_condition_cluster_scores(x, condition_labels)

    return {
        "between_batch_ratio": _between_group_ratio(x, batch_labels),
        "batch_asw": float(compute_batch_asw(container, assay_name, layer_name, batch_key)),
        "batch_mixing": float(
            compute_batch_mixing_metric(
                container,
                assay_name,
                layer_name,
                batch_key,
                n_neighbors=n_neighbors,
            )
        ),
        "lisi_approx": float(
            compute_lisi_approx(
                container,
                assay_name,
                layer_name,
                batch_key,
                n_neighbors=n_neighbors,
            )
        ),
        "between_condition_ratio": _between_group_ratio(x, condition_labels),
        "condition_asw": _safe_condition_asw(x, condition_labels),
        "condition_ari": condition_ari,
        "condition_nmi": condition_nmi,
        "condition_knn_purity": _condition_knn_purity(x, condition_labels, n_neighbors=n_neighbors),
    }


def _run_guardrail_checks(
    container,
    *,
    assay_name: str,
    base_layer: str,
    batch_key: str,
    condition_key: str,
    scenario_name: str,
    dataset_key: str,
    expectation: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    check_specs = [
        ("combat_parametric_with_covariate", integrate_combat, {"eb_mode": "parametric"}),
        ("limma_with_covariate", integrate_limma, {}),
    ]

    for method_name, fn, kwargs in check_specs:
        start = time.perf_counter()
        try:
            fn(
                container,
                batch_key=batch_key,
                assay_name=assay_name,
                base_layer=base_layer,
                new_layer_name=f"{scenario_name}_{method_name}_guardrail",
                covariates=[condition_key],
                **kwargs,
            )
            elapsed = time.perf_counter() - start
            got_error = False
            status = "success"
            err_type = None
            message = ""
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            got_error = True
            status = "error"
            err_type = type(exc).__name__
            message = str(exc)

        expect_error = expectation == "rank_deficient_error"
        guardrail_pass = (expect_error and got_error) or ((not expect_error) and (not got_error))

        checks.append(
            {
                "dataset": dataset_key,
                "scenario": scenario_name,
                "method": method_name,
                "batch_key": batch_key,
                "covariate": condition_key,
                "expectation": expectation,
                "status": status,
                "error_type": err_type,
                "message": message,
                "runtime_sec": round(float(elapsed), 6),
                "guardrail_pass": bool(guardrail_pass),
            }
        )

    return checks


def _run_scenario(
    container,
    *,
    dataset_key: str,
    assay_name: str,
    baseline_layer: str,
    scenario_name: str,
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    scenario = SCENARIO_SPECS[scenario_name]
    batch_key = str(scenario["batch_key"])
    condition_key = str(scenario["condition_key"])
    score_profile = str(scenario["score_profile"])

    baseline_metrics = _compute_metrics(
        container,
        assay_name=assay_name,
        layer_name=baseline_layer,
        batch_key=batch_key,
        condition_key=condition_key,
    )

    rows: list[dict[str, Any]] = [
        {
            "dataset": dataset_key,
            "scenario": scenario_name,
            "score_profile": score_profile,
            "batch_key": batch_key,
            "condition_key": condition_key,
            "method": "none",
            "layer": baseline_layer,
            "runtime_sec": 0.0,
            "delta_between_batch_ratio": 0.0,
            "success": True,
            "error": "",
            **baseline_metrics,
        }
    ]
    failures: list[dict[str, str]] = []

    for method_name in methods:
        if method_name not in METHOD_SPECS:
            failures.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "method": method_name,
                    "error": "Unknown integration method",
                }
            )
            rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "score_profile": score_profile,
                    "batch_key": batch_key,
                    "condition_key": condition_key,
                    "method": method_name,
                    "layer": "",
                    "runtime_sec": float("nan"),
                    "delta_between_batch_ratio": float("nan"),
                    "success": False,
                    "error": "Unknown integration method",
                }
            )
            continue

        fn, kwargs = METHOD_SPECS[method_name]
        layer_name = f"{scenario_name}_{method_name}"
        start = time.perf_counter()
        try:
            fn(
                container,
                batch_key=batch_key,
                assay_name=assay_name,
                base_layer=baseline_layer,
                new_layer_name=layer_name,
                **kwargs,
            )
            elapsed = time.perf_counter() - start
            metrics = _compute_metrics(
                container,
                assay_name=assay_name,
                layer_name=layer_name,
                batch_key=batch_key,
                condition_key=condition_key,
            )
            rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "score_profile": score_profile,
                    "batch_key": batch_key,
                    "condition_key": condition_key,
                    "method": method_name,
                    "layer": layer_name,
                    "runtime_sec": round(float(elapsed), 6),
                    "delta_between_batch_ratio": metrics["between_batch_ratio"]
                    - baseline_metrics["between_batch_ratio"],
                    "success": True,
                    "error": "",
                    **metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            msg = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "method": method_name,
                    "error": msg,
                }
            )
            rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "score_profile": score_profile,
                    "batch_key": batch_key,
                    "condition_key": condition_key,
                    "method": method_name,
                    "layer": layer_name,
                    "runtime_sec": round(float(elapsed), 6),
                    "delta_between_batch_ratio": float("nan"),
                    "success": False,
                    "error": msg,
                }
            )

    guardrail = _run_guardrail_checks(
        container,
        assay_name=assay_name,
        base_layer=baseline_layer,
        batch_key=batch_key,
        condition_key=condition_key,
        scenario_name=scenario_name,
        dataset_key=dataset_key,
        expectation=str(scenario["guardrail_expectation"]),
    )
    return rows, failures, guardrail


def _summarize_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in raw_df.columns
        if c
        not in {
            "dataset",
            "scenario",
            "score_profile",
            "batch_key",
            "condition_key",
            "method",
            "layer",
            "success",
            "error",
        }
        and pd.api.types.is_numeric_dtype(raw_df[c])
    ]

    summary = (
        raw_df.groupby(
            ["dataset", "scenario", "score_profile", "batch_key", "condition_key", "method"],
            dropna=False,
        )[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = raw_df.groupby(
        ["dataset", "scenario", "score_profile", "batch_key", "condition_key", "method"],
        dropna=False,
    )
    summary["runs"] = grouped["success"].size().values
    summary["success_rate"] = grouped["success"].mean().values
    return summary.sort_values(["dataset", "scenario", "method"]).reset_index(drop=True)


def _score_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    score_blocks: list[pd.DataFrame] = []
    for _scenario_name, block in summary_df.groupby("scenario", sort=False, dropna=False):
        if block.empty:
            continue
        score_profile = str(block["score_profile"].iloc[0])
        directions = SCORE_PROFILE_DIRECTIONS.get(score_profile, BALANCED_METRIC_DIRECTIONS)
        scored = score_methods(
            block,
            metric_directions=directions,
            group_by=["dataset", "scenario"],
        )
        score_blocks.append(scored)

    if not score_blocks:
        return pd.DataFrame()
    return pd.concat(score_blocks, ignore_index=True)


def run_benchmark(
    *,
    data_path: Path,
    output_dir: Path,
    output_json: Path | None,
    dataset_key: str,
    scenarios: list[str],
    methods: list[str],
    fdr_threshold: float,
) -> dict[str, Any]:
    start_all = time.perf_counter()

    container = load_diann(
        data_path,
        level="protein",
        table_format="long",
        assay_name="proteins",
        quantity_column="PG.Quantity",
        fdr_threshold=fdr_threshold,
        layer_name="raw",
    )
    _attach_scenario_labels(container)

    assay_name = "proteins"
    source_layer = "raw"

    # Fixed preprocessing chain for all scenarios.
    container = log_transform(
        container,
        assay_name=assay_name,
        source_layer=source_layer,
        new_layer_name="log2",
        base=2.0,
        offset=1.0,
        detect_logged=False,
    )
    container = normalize(
        container,
        method="median",
        assay_name=assay_name,
        source_layer="log2",
        new_layer_name="median_norm",
    )
    container = impute(
        container,
        method="row_median",
        assay_name=assay_name,
        source_layer="median_norm",
        new_layer_name="imputed_baseline",
    )

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    guardrail_rows: list[dict[str, Any]] = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIO_SPECS:
            failures.append(
                {
                    "dataset": dataset_key,
                    "scenario": scenario_name,
                    "method": "<scenario>",
                    "error": "Unknown scenario",
                }
            )
            continue
        rows, method_failures, guardrail = _run_scenario(
            container,
            dataset_key=dataset_key,
            assay_name=assay_name,
            baseline_layer="imputed_baseline",
            scenario_name=scenario_name,
            methods=methods,
        )
        raw_rows.extend(rows)
        failures.extend(method_failures)
        guardrail_rows.extend(guardrail)

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise RuntimeError("No benchmark results generated. Check scenarios and method settings.")

    summary_df = _summarize_raw(raw_df)
    scores_df = _score_summary(summary_df)
    guardrail_df = pd.DataFrame(guardrail_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_out = output_dir / "metrics_raw.csv"
    summary_out = output_dir / "metrics_summary.csv"
    score_out = output_dir / "metrics_scores.csv"
    metadata_out = output_dir / "run_metadata.json"
    guardrail_out = output_dir / "guardrail_checks.csv"

    raw_df.to_csv(raw_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    scores_df.to_csv(score_out, index=False)
    if not guardrail_df.empty:
        guardrail_df.to_csv(guardrail_out, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    plot_summary_metrics(summary_df, output_dir / "summary_metrics.png")
    plot_score_heatmap(scores_df, output_dir / "score_heatmap.png")
    plot_overall_scores(scores_df, output_dir / "overall_scores.png")

    total_runtime = float(time.perf_counter() - start_all)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": str(data_path),
        "dataset_key": dataset_key,
        "n_samples": int(container.n_samples),
        "n_features": int(container.assays["proteins"].n_features),
        "fdr_threshold": fdr_threshold,
        "pipeline": [
            "log_transform(base=2,offset=1)",
            "normalize(median)",
            "impute(row_median)",
        ],
        "scenarios": {k: SCENARIO_SPECS[k] for k in scenarios if k in SCENARIO_SPECS},
        "methods_requested": methods,
        "methods_failed": failures,
        "guardrail_pass_rate": (
            float(guardrail_df["guardrail_pass"].mean()) if not guardrail_df.empty else float("nan")
        ),
        "score_profiles": SCORE_PROFILE_DIRECTIONS,
        "literature_references": LITERATURE_REFERENCES,
        "total_runtime_sec": round(total_runtime, 6),
        "outputs": {
            "raw": str(raw_out),
            "summary": str(summary_out),
            "scores": str(score_out),
            "guardrail": str(guardrail_out) if guardrail_out.exists() else None,
            "failures": str(output_dir / "failures.csv") if (output_dir / "failures.csv").exists() else None,
            "summary_plot": str(output_dir / "summary_metrics.png"),
            "heatmap_plot": str(output_dir / "score_heatmap.png"),
            "overall_plot": str(output_dir / "overall_scores.png"),
        },
    }
    with metadata_out.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    # Legacy JSON compatibility payload.
    legacy_report = {
        "dataset": str(data_path),
        "dataset_key": dataset_key,
        "scenarios": [
            {
                "name": scenario,
                "score_profile": SCENARIO_SPECS[scenario]["score_profile"],
                "batch_key": SCENARIO_SPECS[scenario]["batch_key"],
                "condition_key": SCENARIO_SPECS[scenario]["condition_key"],
                "baseline_metrics": (
                    raw_df[
                        (raw_df["scenario"] == scenario)
                        & (raw_df["method"] == "none")
                    ]
                    .drop(columns=["error"])
                    .head(1)
                    .to_dict("records")
                ),
                "integration_results": (
                    raw_df[
                        (raw_df["scenario"] == scenario)
                        & (raw_df["method"] != "none")
                    ]
                    .drop(columns=["error"])
                    .to_dict("records")
                ),
            }
            for scenario in scenarios
            if scenario in SCENARIO_SPECS
        ],
        "guardrail_checks": guardrail_rows,
        "total_runtime_sec": round(total_runtime, 6),
    }
    legacy_json = output_json or (output_dir / "real_dia_batch_confounding_report.json")
    legacy_json.write_text(json.dumps(legacy_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Integration benchmark completed.")
    print(f"[INFO] Raw metrics: {raw_out}")
    print(f"[INFO] Summary metrics: {summary_out}")
    print(f"[INFO] Score table: {score_out}")
    print(f"[INFO] Metadata: {metadata_out}")
    print(f"[INFO] Legacy report: {legacy_json}")
    if failures:
        print(f"[WARN] Some runs failed. See: {output_dir / 'failures.csv'}")

    if not scores_df.empty and "overall_score" in scores_df.columns:
        rank_df = scores_df.sort_values(["scenario", "overall_score"], ascending=[True, False])
        keep_cols = [
            "scenario",
            "method",
            "overall_score",
            "between_batch_ratio",
            "batch_asw",
            "batch_mixing",
            "lisi_approx",
            "condition_asw",
            "condition_ari",
            "condition_nmi",
        ]
        show_cols = [c for c in keep_cols if c in rank_df.columns]
        print("\n[INFO] Method ranking by scenario:")
        print(rank_df[show_cols].to_string(index=False))

    return legacy_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="DIA-NN report TSV path.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=DEFAULT_DATASET_KEY,
        help="Dataset key used in output tables.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=sorted(SCENARIO_SPECS.keys()),
        help="Benchmark scenarios to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(METHOD_SPECS.keys()),
        help="Integration methods to benchmark (none baseline is always included).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional legacy JSON report path.",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.01,
        help="FDR threshold for loading DIA-NN long report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        output_json=args.output_json,
        dataset_key=str(args.dataset_key),
        scenarios=[str(v) for v in args.scenarios],
        methods=[str(v) for v in args.methods],
        fdr_threshold=float(args.fdr_threshold),
    )


if __name__ == "__main__":
    main()
