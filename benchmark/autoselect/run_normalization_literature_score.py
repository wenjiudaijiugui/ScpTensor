#!/usr/bin/env python3
"""Benchmark literature-driven normalization scoring on multiple datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark_data_generator import BenchmarkDataGenerator
from scptensor.autoselect import auto_normalize
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.io import load_diann
from scptensor.transformation import log_transform


@dataclass
class DatasetRunResult:
    """Compact per-dataset run summary."""

    dataset_name: str
    dataset_type: str
    n_samples: int
    n_features: int
    missing_rate: float
    best_method: str
    best_overall_score: float
    method_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_rate": self.missing_rate,
            "best_method": self.best_method,
            "best_overall_score": self.best_overall_score,
            "method_rows": self.method_rows,
        }


def _missing_rate(
    container: ScpContainer, assay_name: str = "proteins", layer: str = "raw"
) -> float:
    x = container.assays[assay_name].layers[layer].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return float(np.mean(np.isnan(np.asarray(x))))


def _subset_features(container: ScpContainer, max_features: int) -> ScpContainer:
    """Subset to top observed proteins for faster/robust benchmarking."""
    assay = container.assays["proteins"]
    x = assay.layers["raw"].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)

    if x.shape[1] <= max_features:
        return container

    obs_counts = np.sum(np.isfinite(x), axis=0)
    keep_idx = np.argsort(-obs_counts)[:max_features]
    keep_idx = np.sort(keep_idx)

    x_sub = x[:, keep_idx]
    var_sub = assay.var[keep_idx]
    new_assay = Assay(var=var_sub)
    new_assay.add_layer("raw", ScpMatrix(X=x_sub))

    out = ScpContainer(obs=container.obs.clone())
    out.add_assay("proteins", new_assay)
    return out


def _attach_pxd054343_metadata(container: ScpContainer) -> ScpContainer:
    """Add batch and biological group labels parsed from sample names."""
    obs = container.obs.clone()
    sample_ids = obs["_index"].to_list()

    batch = []
    amount = []
    for sample in sample_ids:
        s_match = re.search(r"_(S\\d+)_", sample)
        a_match = re.search(r"SCamount(\\d+ng)", sample)
        batch.append(s_match.group(1) if s_match else "unknown")
        amount.append(a_match.group(1) if a_match else "unknown")

    obs = obs.with_columns(
        [
            pl.Series(name="batch", values=batch),
            pl.Series(name="cell_type", values=amount),
        ]
    )

    out = container.copy()
    out.obs = obs
    return out


def _build_dataset_real(path: Path, max_features: int) -> ScpContainer:
    container = load_diann(path, assay_name="proteins", level="protein")
    container = _attach_pxd054343_metadata(container)
    return _subset_features(container, max_features=max_features)


def _build_dataset_synth_batch(seed: int) -> ScpContainer:
    gen = BenchmarkDataGenerator(seed=seed)
    return gen.generate_from_config(
        {
            "scale": "medium",
            "missing_rate": "medium",
            "missing_pattern": "mar",
            "distribution": "log_normal",
            "with_batch_effect": True,
            "n_batches": 3,
        }
    )


def _build_dataset_synth_no_batch(seed: int) -> ScpContainer:
    gen = BenchmarkDataGenerator(seed=seed)
    return gen.generate_from_config(
        {
            "scale": "small",
            "missing_rate": "medium",
            "missing_pattern": "mnar",
            "distribution": "multimodal",
            "with_batch_effect": False,
            "n_batches": 1,
        }
    )


def _run_single(dataset_name: str, dataset_type: str, container: ScpContainer) -> DatasetRunResult:
    """Run auto-normalization and return scored method table."""
    processed = container.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Source layer 'raw' appears already log-transformed.*",
            category=UserWarning,
        )
        processed = log_transform(
            processed,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log2",
            base=2.0,
            offset=1.0,
            detect_logged=True,
            skip_if_logged=True,
        )

    _, report = auto_normalize(
        processed,
        assay_name="proteins",
        source_layer="log2",
        keep_all=True,
        selection_strategy="quality",
    )

    rows: list[dict[str, Any]] = []
    for result in report.results:
        if result.error is not None:
            rows.append(
                {
                    "method": result.method_name,
                    "error": result.error,
                    "overall_score": 0.0,
                }
            )
            continue

        scores = result.scores
        rows.append(
            {
                "method": result.method_name,
                "overall_score": float(result.overall_score),
                "batch_removal": float(scores.get("batch_removal", 0.0)),
                "bio_conservation": float(scores.get("bio_conservation", 0.0)),
                "technical_quality": float(scores.get("technical_quality", 0.0)),
                "balance_score": float(scores.get("balance_score", 0.0)),
                "batch_asw": float(scores.get("batch_asw", 0.0)),
                "batch_mixing": float(scores.get("batch_mixing", 0.0)),
                "bio_asw": float(scores.get("bio_asw", 0.0)),
                "signal_preservation": float(scores.get("signal_preservation", 0.0)),
                "execution_time": float(result.execution_time),
                "selection_score": (
                    None if result.selection_score is None else float(result.selection_score)
                ),
            }
        )

    rows_sorted = sorted(rows, key=lambda x: x.get("overall_score", 0.0), reverse=True)
    best_method = report.best_method
    best_score = report.best_result.overall_score if report.best_result else 0.0

    assay = container.assays["proteins"]
    return DatasetRunResult(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        n_samples=container.n_samples,
        n_features=assay.n_features,
        missing_rate=_missing_rate(container, assay_name="proteins", layer="raw"),
        best_method=best_method,
        best_overall_score=float(best_score),
        method_rows=rows_sorted,
    )


def _write_outputs(output_dir: Path, results: list[DatasetRunResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "module": "autoselect.normalize.literature_scoring",
        "results": [r.to_dict() for r in results],
    }
    json_path = output_dir / "normalization_literature_scoring_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_dir / "normalization_literature_scoring_results.md"
    lines: list[str] = []
    lines.append("# Normalization Literature Scoring Benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append("- Selection strategy: `quality`")
    lines.append(
        "- Composite score axes: `batch_removal`, `bio_conservation`, `technical_quality`, `balance_score`"
    )
    lines.append("")

    for result in results:
        lines.append(f"## {result.dataset_name}")
        lines.append("")
        lines.append(f"- Type: `{result.dataset_type}`")
        lines.append(f"- Shape: `{result.n_samples} x {result.n_features}`")
        lines.append(f"- Missing rate: `{result.missing_rate:.3f}`")
        lines.append(f"- Best method: `{result.best_method}` (`{result.best_overall_score:.4f}`)")
        lines.append("")
        lines.append(
            "| method | overall | batch_removal | bio_conservation | technical_quality | balance |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in result.method_rows:
            if "error" in row:
                lines.append(f"| {row['method']} | 0.0000 | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {row['method']} | {row['overall_score']:.4f} | "
                f"{row['batch_removal']:.4f} | {row['bio_conservation']:.4f} | "
                f"{row['technical_quality']:.4f} | {row['balance_score']:.4f} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "autoselect" / "normalization_literature",
        help="Output directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--real-diann-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "dia" / "diann" / "PXD054343" / "1_SC_LF_report.tsv",
        help="Path to DIA-NN real dataset.",
    )
    parser.add_argument(
        "--real-max-features",
        type=int,
        default=1200,
        help="Maximum number of proteins retained for the real dataset run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic datasets.")
    args = parser.parse_args()

    datasets: list[tuple[str, str, ScpContainer]] = []
    datasets.append(
        ("synthetic_batch_confounded", "synthetic", _build_dataset_synth_batch(seed=args.seed))
    )
    datasets.append(
        (
            "synthetic_multimodal_no_batch",
            "synthetic",
            _build_dataset_synth_no_batch(seed=args.seed + 17),
        )
    )
    datasets.append(
        (
            "real_pxd054343_diann",
            "real",
            _build_dataset_real(args.real_diann_path, max_features=args.real_max_features),
        )
    )

    results: list[DatasetRunResult] = []
    for dataset_name, dataset_type, container in datasets:
        print(f"Running: {dataset_name} ({dataset_type})")
        result = _run_single(dataset_name, dataset_type, container)
        print(f"  best={result.best_method}, overall={result.best_overall_score:.4f}")
        results.append(result)

    _write_outputs(args.output_dir, results)


if __name__ == "__main__":
    main()
