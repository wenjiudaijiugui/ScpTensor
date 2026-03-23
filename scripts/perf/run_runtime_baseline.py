#!/usr/bin/env python3
"""Run runtime baselines for stable ScpTensor preprocessing paths.

This script is intentionally separate from ``benchmark/``:

- ``benchmark/`` tracks method/quality evaluation contracts
- ``scripts/perf`` tracks runtime regression baselines for engineering work
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import polars as pl
import psutil
import scipy.sparse as sp

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scptensor
from scptensor.aggregation import aggregate_to_protein
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.impute import impute
from scptensor.integration import integrate_limma
from scptensor.io import load_diann
from scptensor.normalization import normalize
from scptensor.qc import calculate_feature_qc_metrics, calculate_sample_qc_metrics
from scptensor.transformation import log_transform

PROFILE_CONFIG: dict[str, dict[str, int | float]] = {
    "quick": {
        "io_samples": 12,
        "io_features": 32,
        "io_missing_rate": 0.20,
        "agg_samples": 12,
        "agg_proteins": 24,
        "agg_peptides_per_protein": 3,
        "agg_missing_rate": 0.15,
        "dense_samples": 24,
        "dense_features": 64,
        "dense_missing_rate": 0.22,
        "quantile_samples": 18,
        "quantile_features": 48,
        "quantile_missing_rate": 0.18,
        "sparse_samples": 28,
        "sparse_features": 96,
        "sparse_zero_rate": 0.82,
    },
    "default": {
        "io_samples": 24,
        "io_features": 160,
        "io_missing_rate": 0.25,
        "agg_samples": 24,
        "agg_proteins": 120,
        "agg_peptides_per_protein": 4,
        "agg_missing_rate": 0.20,
        "dense_samples": 96,
        "dense_features": 320,
        "dense_missing_rate": 0.25,
        "quantile_samples": 64,
        "quantile_features": 256,
        "quantile_missing_rate": 0.20,
        "sparse_samples": 96,
        "sparse_features": 512,
        "sparse_zero_rate": 0.90,
    },
}

SCENARIOS = (
    "import_diann_protein_long",
    "aggregate_peptide_to_protein",
    "stable_chain_dense",
    "stable_chain_quantile",
    "stable_chain_trqn",
    "normalize_quantile_only",
    "normalize_trqn_only",
    "sparse_log_only",
    "sparse_transform_normalize",
    "autoselect_integrate_only",
    "viz_qc_overview",
)


@dataclass(frozen=True)
class ScenarioSummary:
    """Compact summary for one runtime-baseline scenario."""

    scenario: str
    status: str
    n_stage_rows: int
    total_elapsed_s: float
    max_peak_delta_mb: float
    densify_stages: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "status": self.status,
            "n_stage_rows": self.n_stage_rows,
            "total_elapsed_s": self.total_elapsed_s,
            "max_peak_delta_mb": self.max_peak_delta_mb,
            "densify_stages": self.densify_stages,
            "note": self.note,
        }


class MemorySampler:
    """Track process RSS peak during one operation."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss = int(self.process.memory_info().rss)

    def start(self) -> None:
        def _run() -> None:
            while not self._stop.is_set():
                rss = int(self.process.memory_info().rss)
                if rss > self.peak_rss:
                    self.peak_rss = rss
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 10.0, 0.1))
        rss = int(self.process.memory_info().rss)
        if rss > self.peak_rss:
            self.peak_rss = rss


def _matrix_storage_kind(matrix: np.ndarray | sp.spmatrix | None) -> str:
    if matrix is None:
        return "none"
    return "sparse" if sp.issparse(matrix) else "dense"


def _matrix_nbytes(matrix: np.ndarray | sp.spmatrix | None) -> int:
    if matrix is None:
        return 0
    if sp.issparse(matrix):
        sparse_matrix = cast(sp.spmatrix, matrix).tocsr()
        return int(
            sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes
        )
    return int(np.asarray(matrix).nbytes)


def _hash_bytes(hasher: hashlib._Hash, values: bytes) -> None:
    hasher.update(len(values).to_bytes(8, byteorder="little", signed=False))
    hasher.update(values)


def _matrix_signature(matrix: np.ndarray | sp.spmatrix | None) -> str:
    if matrix is None:
        return "none"

    hasher = hashlib.sha1()
    if sp.issparse(matrix):
        sparse_matrix = cast(sp.spmatrix, matrix).tocsr()
        hasher.update(b"sparse")
        hasher.update(str(sparse_matrix.shape).encode("utf-8"))
        _hash_bytes(hasher, np.asarray(sparse_matrix.data).tobytes())
        _hash_bytes(hasher, np.asarray(sparse_matrix.indices).tobytes())
        _hash_bytes(hasher, np.asarray(sparse_matrix.indptr).tobytes())
        return hasher.hexdigest()

    dense = np.ascontiguousarray(np.asarray(matrix))
    hasher.update(b"dense")
    hasher.update(str(dense.shape).encode("utf-8"))
    _hash_bytes(hasher, dense.tobytes())
    return hasher.hexdigest()


def _bool_flag(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _format_mb(value: int) -> float:
    return round(value / (1024.0 * 1024.0), 3)


def _stringify_list(values: list[str]) -> str:
    return ";".join(values)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_obs(n_samples: int, n_batches: int, n_groups: int) -> pl.DataFrame:
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    batch = [f"Batch_{i % n_batches}" for i in range(n_samples)]
    group = [f"Group_{i % n_groups}" for i in range(n_samples)]
    return pl.DataFrame(
        {
            "_index": sample_ids,
            "batch": batch,
            "group": group,
        }
    )


def _make_var(n_features: int) -> pl.DataFrame:
    feature_ids = [f"P{i:05d}" for i in range(n_features)]
    return pl.DataFrame(
        {
            "_index": feature_ids,
            "gene_name": [f"G{i:05d}" for i in range(n_features)],
        }
    )


def _build_dense_intensity_matrix(
    *,
    n_samples: int,
    n_features: int,
    missing_rate: float,
    seed: int,
    n_batches: int = 3,
    n_groups: int = 4,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    obs = _make_obs(n_samples, n_batches=n_batches, n_groups=n_groups)
    var = _make_var(n_features)

    batch_index = np.asarray([i % n_batches for i in range(n_samples)])
    group_index = np.asarray([i % n_groups for i in range(n_samples)])

    base_log2 = rng.normal(loc=18.0, scale=1.0, size=n_features)
    group_effect = rng.normal(loc=0.0, scale=0.35, size=(n_groups, n_features))
    batch_effect = rng.normal(loc=0.0, scale=0.18, size=(n_batches, n_features))
    noise = rng.normal(loc=0.0, scale=0.22, size=(n_samples, n_features))

    log2_x = base_log2[None, :] + group_effect[group_index] + batch_effect[batch_index] + noise
    x = np.exp2(np.clip(log2_x, a_min=0.0, a_max=None)).astype(np.float64)

    raw_missing = 0.03 + 0.28 * (1.0 / (1.0 + np.exp(log2_x - 18.0)))
    scale = 0.0 if raw_missing.mean() == 0 else missing_rate / float(raw_missing.mean())
    missing_prob = np.clip(raw_missing * scale, 0.0, 0.80)
    missing_mask = rng.random(size=x.shape) < missing_prob

    # Keep all samples/features represented for stable importer and downstream checks.
    missing_mask[0, :] = False
    missing_mask[:, 0] = False

    x[missing_mask] = np.nan
    m = np.zeros(x.shape, dtype=np.int8)
    m[missing_mask] = 2
    return x, m, obs, var


def _build_dense_protein_container(
    *,
    n_samples: int,
    n_features: int,
    missing_rate: float,
    seed: int,
) -> ScpContainer:
    x, m, obs, var = _build_dense_intensity_matrix(
        n_samples=n_samples,
        n_features=n_features,
        missing_rate=missing_rate,
        seed=seed,
    )
    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x, M=m))
    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def _build_sparse_protein_container(
    *,
    n_samples: int,
    n_features: int,
    zero_rate: float,
    seed: int,
) -> ScpContainer:
    rng = np.random.default_rng(seed)
    obs = _make_obs(n_samples, n_batches=3, n_groups=4)
    var = _make_var(n_features)

    log2_x = rng.normal(loc=17.5, scale=0.8, size=(n_samples, n_features))
    x = np.exp2(np.clip(log2_x, a_min=0.0, a_max=None)).astype(np.float64)
    zero_mask = rng.random(size=x.shape) < zero_rate
    zero_mask[0, :] = False
    zero_mask[:, 0] = False
    x[zero_mask] = 0.0

    x_sparse = sp.csr_matrix(x)
    m_sparse = sp.csr_matrix(np.zeros(x.shape, dtype=np.int8))

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x_sparse, M=m_sparse))
    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def _build_peptide_container(
    *,
    n_samples: int,
    n_proteins: int,
    peptides_per_protein: int,
    missing_rate: float,
    seed: int,
) -> ScpContainer:
    rng = np.random.default_rng(seed)
    n_peptides = n_proteins * peptides_per_protein
    obs = _make_obs(n_samples, n_batches=3, n_groups=4)

    protein_ids = [f"P{i:05d}" for i in range(n_proteins)]
    peptide_ids = [f"pep_{i:05d}" for i in range(n_peptides)]
    protein_map = np.repeat(protein_ids, peptides_per_protein)
    var = pl.DataFrame(
        {
            "_index": peptide_ids,
            "Protein.Group": protein_map.tolist(),
        }
    )

    base_log2 = rng.normal(loc=15.5, scale=1.0, size=n_peptides)
    noise = rng.normal(loc=0.0, scale=0.30, size=(n_samples, n_peptides))
    x = np.exp2(np.clip(base_log2[None, :] + noise, a_min=0.0, a_max=None)).astype(np.float64)
    missing_mask = rng.random(size=x.shape) < missing_rate
    missing_mask[0, :] = False
    missing_mask[:, 0] = False
    x[missing_mask] = np.nan
    m = np.zeros(x.shape, dtype=np.int8)
    m[missing_mask] = 2

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x, M=m))
    container = ScpContainer(obs=obs)
    container.add_assay("peptides", assay)
    return container


def _write_diann_protein_long_table(
    *,
    destination: Path,
    n_samples: int,
    n_features: int,
    missing_rate: float,
    seed: int,
) -> dict[str, Any]:
    x, missing_mask, obs, _ = _build_dense_intensity_matrix(
        n_samples=n_samples,
        n_features=n_features,
        missing_rate=missing_rate,
        seed=seed,
    )

    sample_ids = obs["_index"].to_list()
    protein_ids = [f"P{i:05d}" for i in range(n_features)]
    gene_names = [f"G{i:05d}" for i in range(n_features)]

    rows = np.argwhere(~np.isnan(x))
    frame = pl.DataFrame(
        {
            "Run": [f"{sample_ids[i]}.raw" for i, _ in rows],
            "Protein.Group": [protein_ids[j] for _, j in rows],
            "Genes": [gene_names[j] for _, j in rows],
            "PG.Quantity": x[~np.isnan(x)].tolist(),
            "PG.Q.Value": np.full(rows.shape[0], 0.001, dtype=np.float64).tolist(),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(destination, separator="\t")
    return {
        "n_rows": int(frame.height),
        "n_samples": n_samples,
        "n_features": n_features,
        "target_missing_rate": missing_rate,
    }


def _profile_import_stage(
    *,
    scenario: str,
    stage: str,
    note: str,
    runner: Callable[[], ScpContainer],
) -> tuple[ScpContainer, dict[str, Any]]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result = runner()
    finally:
        elapsed = time.perf_counter() - started
        sampler.stop()
    rss_after = int(process.memory_info().rss)

    assay = result.assays["proteins"]
    layer = assay.layers["raw"]
    row = {
        "scenario": scenario,
        "stage": stage,
        "status": "ok",
        "elapsed_s": round(elapsed, 6),
        "rss_before_mb": _format_mb(rss_before),
        "rss_after_mb": _format_mb(rss_after),
        "rss_delta_mb": _format_mb(rss_after - rss_before),
        "rss_peak_mb": _format_mb(sampler.peak_rss),
        "rss_peak_delta_mb": _format_mb(sampler.peak_rss - rss_before),
        "returned_same_object": "",
        "source_assay_same_object": "",
        "source_layer_same_object": "",
        "source_x_same_object": "",
        "source_m_same_object": "",
        "source_x_unchanged": "",
        "source_m_unchanged": "",
        "input_x_storage": "",
        "input_m_storage": "",
        "input_x_bytes": "",
        "input_m_bytes": "",
        "output_kind": "layer",
        "output_assay": "proteins",
        "output_layer": "raw",
        "output_x_storage": _matrix_storage_kind(layer.X),
        "output_m_storage": _matrix_storage_kind(layer.M),
        "output_x_bytes": _matrix_nbytes(layer.X),
        "output_m_bytes": _matrix_nbytes(layer.M),
        "output_shares_x_with_source": "",
        "output_shares_m_with_source": "",
        "densified_output": "",
        "obs_cols_added": "",
        "var_cols_added": "",
        "n_samples": result.n_samples,
        "n_features": assay.n_features,
        "note": note,
    }
    return result, row


def _profile_container_stage(
    *,
    scenario: str,
    stage: str,
    container: ScpContainer,
    source_assay: str,
    source_layer: str,
    runner: Callable[[], ScpContainer],
    output_kind: Literal["layer", "obs", "var"],
    target_assay: str | None = None,
    target_layer: str | None = None,
    note: str = "",
) -> tuple[ScpContainer, dict[str, Any]]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)

    before_assay = container.assays[source_assay]
    before_layer = before_assay.layers[source_layer]
    before_x = before_layer.X
    before_m = before_layer.M
    before_obs_cols = list(container.obs.columns)
    before_var_cols = list(before_assay.var.columns)
    before_x_sig = _matrix_signature(before_x)
    before_m_sig = _matrix_signature(before_m)

    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result = runner()
    finally:
        elapsed = time.perf_counter() - started
        sampler.stop()
    rss_after = int(process.memory_info().rss)

    after_source_assay = result.assays[source_assay]
    after_source_layer = after_source_assay.layers[source_layer]

    row: dict[str, Any] = {
        "scenario": scenario,
        "stage": stage,
        "status": "ok",
        "elapsed_s": round(elapsed, 6),
        "rss_before_mb": _format_mb(rss_before),
        "rss_after_mb": _format_mb(rss_after),
        "rss_delta_mb": _format_mb(rss_after - rss_before),
        "rss_peak_mb": _format_mb(sampler.peak_rss),
        "rss_peak_delta_mb": _format_mb(sampler.peak_rss - rss_before),
        "returned_same_object": _bool_flag(result is container),
        "source_assay_same_object": _bool_flag(after_source_assay is before_assay),
        "source_layer_same_object": _bool_flag(after_source_layer is before_layer),
        "source_x_same_object": _bool_flag(after_source_layer.X is before_x),
        "source_m_same_object": _bool_flag(
            None if before_m is None else after_source_layer.M is before_m
        ),
        "source_x_unchanged": _bool_flag(_matrix_signature(after_source_layer.X) == before_x_sig),
        "source_m_unchanged": _bool_flag(_matrix_signature(after_source_layer.M) == before_m_sig),
        "input_x_storage": _matrix_storage_kind(before_x),
        "input_m_storage": _matrix_storage_kind(before_m),
        "input_x_bytes": _matrix_nbytes(before_x),
        "input_m_bytes": _matrix_nbytes(before_m),
        "output_kind": output_kind,
        "output_assay": target_assay or "",
        "output_layer": target_layer or "",
        "output_x_storage": "",
        "output_m_storage": "",
        "output_x_bytes": "",
        "output_m_bytes": "",
        "output_shares_x_with_source": "",
        "output_shares_m_with_source": "",
        "densified_output": "",
        "obs_cols_added": "",
        "var_cols_added": "",
        "n_samples": result.n_samples,
        "n_features": after_source_assay.n_features,
        "note": note,
    }

    if output_kind == "layer":
        if target_assay is None or target_layer is None:
            raise ValueError("layer output requires target_assay and target_layer")
        output_matrix = result.assays[target_assay].layers[target_layer]
        row.update(
            {
                "output_x_storage": _matrix_storage_kind(output_matrix.X),
                "output_m_storage": _matrix_storage_kind(output_matrix.M),
                "output_x_bytes": _matrix_nbytes(output_matrix.X),
                "output_m_bytes": _matrix_nbytes(output_matrix.M),
                "output_shares_x_with_source": _bool_flag(output_matrix.X is after_source_layer.X),
                "output_shares_m_with_source": _bool_flag(
                    None
                    if output_matrix.M is None or after_source_layer.M is None
                    else output_matrix.M is after_source_layer.M
                ),
                "densified_output": _bool_flag(
                    sp.issparse(before_x) and not sp.issparse(output_matrix.X)
                ),
            }
        )
    elif output_kind == "obs":
        added = [col for col in result.obs.columns if col not in before_obs_cols]
        row["obs_cols_added"] = _stringify_list(added)
    elif output_kind == "var":
        added = [
            col for col in result.assays[source_assay].var.columns if col not in before_var_cols
        ]
        row["var_cols_added"] = _stringify_list(added)

    return result, row


def _profile_artifact_stage(
    *,
    scenario: str,
    stage: str,
    note: str,
    runner: Callable[[], Any],
    n_samples: int,
    n_features: int,
) -> dict[str, Any]:
    """Profile a read-only or artifact-producing stage."""
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        runner()
    finally:
        elapsed = time.perf_counter() - started
        sampler.stop()
    rss_after = int(process.memory_info().rss)

    return {
        "scenario": scenario,
        "stage": stage,
        "status": "ok",
        "elapsed_s": round(elapsed, 6),
        "rss_before_mb": _format_mb(rss_before),
        "rss_after_mb": _format_mb(rss_after),
        "rss_delta_mb": _format_mb(rss_after - rss_before),
        "rss_peak_mb": _format_mb(sampler.peak_rss),
        "rss_peak_delta_mb": _format_mb(sampler.peak_rss - rss_before),
        "returned_same_object": "",
        "source_assay_same_object": "",
        "source_layer_same_object": "",
        "source_x_same_object": "",
        "source_m_same_object": "",
        "source_x_unchanged": "",
        "source_m_unchanged": "",
        "input_x_storage": "",
        "input_m_storage": "",
        "input_x_bytes": "",
        "input_m_bytes": "",
        "output_kind": "artifact",
        "output_assay": "",
        "output_layer": "",
        "output_x_storage": "",
        "output_m_storage": "",
        "output_x_bytes": "",
        "output_m_bytes": "",
        "output_shares_x_with_source": "",
        "output_shares_m_with_source": "",
        "densified_output": "",
        "obs_cols_added": "",
        "var_cols_added": "",
        "n_samples": n_samples,
        "n_features": n_features,
        "note": note,
    }


def _run_import_diann_protein_long(
    *,
    output_dir: Path,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    input_path = output_dir / "_generated_inputs" / f"diann_protein_long_{profile}.tsv"
    input_meta = _write_diann_protein_long_table(
        destination=input_path,
        n_samples=int(cfg["io_samples"]),
        n_features=int(cfg["io_features"]),
        missing_rate=float(cfg["io_missing_rate"]),
        seed=seed,
    )
    _, row = _profile_import_stage(
        scenario="import_diann_protein_long",
        stage="load_diann",
        note="Synthetic DIA-NN protein long table import baseline.",
        runner=lambda: load_diann(
            input_path,
            assay_name="proteins",
            level="protein",
            table_format="long",
            quantity_column="PG.Quantity",
        ),
    )
    row.update(input_meta)
    rows = [row]
    summary = ScenarioSummary(
        scenario="import_diann_protein_long",
        status="ok",
        n_stage_rows=1,
        total_elapsed_s=float(row["elapsed_s"]),
        max_peak_delta_mb=float(row["rss_peak_delta_mb"]),
        densify_stages=[],
        note="Measures vendor-style long-table import only.",
    )
    return rows, summary


def _run_aggregate_peptide_to_protein(
    *,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    container = _build_peptide_container(
        n_samples=int(cfg["agg_samples"]),
        n_proteins=int(cfg["agg_proteins"]),
        peptides_per_protein=int(cfg["agg_peptides_per_protein"]),
        missing_rate=float(cfg["agg_missing_rate"]),
        seed=seed,
    )
    result, row = _profile_container_stage(
        scenario="aggregate_peptide_to_protein",
        stage="aggregate_to_protein_sum",
        container=container,
        source_assay="peptides",
        source_layer="raw",
        runner=lambda: aggregate_to_protein(
            container,
            source_assay="peptides",
            source_layer="raw",
            target_assay="proteins",
            method="sum",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="raw",
        note="Protein aggregation baseline using stable sum method.",
    )
    row["n_features"] = result.assays["proteins"].n_features
    rows = [row]
    summary = ScenarioSummary(
        scenario="aggregate_peptide_to_protein",
        status="ok",
        n_stage_rows=1,
        total_elapsed_s=float(row["elapsed_s"]),
        max_peak_delta_mb=float(row["rss_peak_delta_mb"]),
        densify_stages=[],
        note="Measures stable peptide/precursor -> protein conversion only.",
    )
    return rows, summary


def _run_stable_chain_dense(
    *,
    profile: str,
    seed: int,
    normalization_method: str,
    scenario_name: str,
    size_prefix: str,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    container = _build_dense_protein_container(
        n_samples=int(cfg[f"{size_prefix}_samples"]),
        n_features=int(cfg[f"{size_prefix}_features"]),
        missing_rate=float(cfg[f"{size_prefix}_missing_rate"]),
        seed=seed,
    )
    rows: list[dict[str, Any]] = []

    container, row = _profile_container_stage(
        scenario=scenario_name,
        stage="log_transform",
        container=container,
        source_assay="proteins",
        source_layer="raw",
        runner=lambda: log_transform(
            container,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log",
            base=2.0,
            offset=1.0,
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="log",
        note="Stable log-transform baseline on dense protein matrix.",
    )
    rows.append(row)

    container, row = _profile_container_stage(
        scenario=scenario_name,
        stage=f"normalize_{normalization_method}",
        container=container,
        source_assay="proteins",
        source_layer="log",
        runner=lambda: normalize(
            container,
            method=normalization_method,
            assay_name="proteins",
            source_layer="log",
            new_layer_name="norm",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="norm",
        note=f"Stable normalization baseline using method={normalization_method}.",
    )
    rows.append(row)

    container, row = _profile_container_stage(
        scenario=scenario_name,
        stage="impute_row_median",
        container=container,
        source_assay="proteins",
        source_layer="norm",
        runner=lambda: impute(
            container,
            method="row_median",
            assay_name="proteins",
            source_layer="norm",
            new_layer_name="imputed",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="imputed",
        note="Stable dense imputation baseline.",
    )
    rows.append(row)

    container, row = _profile_container_stage(
        scenario=scenario_name,
        stage="integrate_limma",
        container=container,
        source_assay="proteins",
        source_layer="imputed",
        runner=lambda: integrate_limma(
            container,
            batch_key="batch",
            assay_name="proteins",
            base_layer="imputed",
            new_layer_name="limma",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="limma",
        note="Stable matrix-level integration baseline.",
    )
    rows.append(row)

    container, row = _profile_container_stage(
        scenario=scenario_name,
        stage="calculate_sample_qc_metrics",
        container=container,
        source_assay="proteins",
        source_layer="limma",
        runner=lambda: calculate_sample_qc_metrics(
            container,
            assay_name="proteins",
            layer_name="limma",
        ),
        output_kind="obs",
        note="Adds sample-level QC summary columns to obs.",
    )
    rows.append(row)

    _, row = _profile_container_stage(
        scenario=scenario_name,
        stage="calculate_feature_qc_metrics",
        container=container,
        source_assay="proteins",
        source_layer="limma",
        runner=lambda: calculate_feature_qc_metrics(
            container,
            assay_name="proteins",
            layer_name="limma",
        ),
        output_kind="var",
        note="Adds feature-level QC summary columns to var.",
    )
    rows.append(row)

    summary = ScenarioSummary(
        scenario=scenario_name,
        status="ok",
        n_stage_rows=len(rows),
        total_elapsed_s=round(sum(float(r["elapsed_s"]) for r in rows), 6),
        max_peak_delta_mb=max(float(r["rss_peak_delta_mb"]) for r in rows),
        densify_stages=[r["stage"] for r in rows if r["densified_output"] == "true"],
        note=("Stable dense preprocessing chain: log -> normalize -> impute -> integrate -> QC."),
    )
    return rows, summary


def _run_logged_normalization_only(
    *,
    profile: str,
    seed: int,
    normalization_method: str,
    scenario_name: str,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    container = _build_dense_protein_container(
        n_samples=int(cfg["quantile_samples"]),
        n_features=int(cfg["quantile_features"]),
        missing_rate=float(cfg["quantile_missing_rate"]),
        seed=seed,
    )
    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
        base=2.0,
        offset=1.0,
    )

    _, row = _profile_container_stage(
        scenario=scenario_name,
        stage=f"normalize_{normalization_method}",
        container=container,
        source_assay="proteins",
        source_layer="log",
        runner=lambda: normalize(
            container,
            method=normalization_method,
            assay_name="proteins",
            source_layer="log",
            new_layer_name="norm",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="norm",
        note=(
            "Isolated logged normalization baseline on precomputed log layer "
            f"using method={normalization_method}."
        ),
    )
    rows = [row]
    summary = ScenarioSummary(
        scenario=scenario_name,
        status="ok",
        n_stage_rows=1,
        total_elapsed_s=float(row["elapsed_s"]),
        max_peak_delta_mb=float(row["rss_peak_delta_mb"]),
        densify_stages=[row["stage"]] if row["densified_output"] == "true" else [],
        note=(
            "Measures logged normalization stage only; "
            "log-transform setup is excluded from timed stages."
        ),
    )
    return rows, summary


def _run_sparse_transform_normalize(
    *,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    container = _build_sparse_protein_container(
        n_samples=int(cfg["sparse_samples"]),
        n_features=int(cfg["sparse_features"]),
        zero_rate=float(cfg["sparse_zero_rate"]),
        seed=seed,
    )
    rows: list[dict[str, Any]] = []

    container, row = _profile_container_stage(
        scenario="sparse_transform_normalize",
        stage="log_transform_sparse",
        container=container,
        source_assay="proteins",
        source_layer="raw",
        runner=lambda: log_transform(
            container,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log",
            base=2.0,
            offset=1.0,
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="log",
        note="Sparse input transform baseline; should preserve sparse storage on output path.",
    )
    rows.append(row)

    _, row = _profile_container_stage(
        scenario="sparse_transform_normalize",
        stage="normalize_median_after_sparse_log",
        container=container,
        source_assay="proteins",
        source_layer="log",
        runner=lambda: normalize(
            container,
            method="median",
            assay_name="proteins",
            source_layer="log",
            new_layer_name="norm",
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="norm",
        note="Captures current sparse->dense normalization boundary.",
    )
    rows.append(row)

    summary = ScenarioSummary(
        scenario="sparse_transform_normalize",
        status="ok",
        n_stage_rows=len(rows),
        total_elapsed_s=round(sum(float(r["elapsed_s"]) for r in rows), 6),
        max_peak_delta_mb=max(float(r["rss_peak_delta_mb"]) for r in rows),
        densify_stages=[r["stage"] for r in rows if r["densified_output"] == "true"],
        note="Sparse path baseline stops at normalization because stable imputation uses NaN-based semantics.",
    )
    return rows, summary


def _run_sparse_log_only(
    *,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    cfg = PROFILE_CONFIG[profile]
    container = _build_sparse_protein_container(
        n_samples=int(cfg["sparse_samples"]),
        n_features=int(cfg["sparse_features"]),
        zero_rate=float(cfg["sparse_zero_rate"]),
        seed=seed,
    )

    _, row = _profile_container_stage(
        scenario="sparse_log_only",
        stage="log_transform_sparse",
        container=container,
        source_assay="proteins",
        source_layer="raw",
        runner=lambda: log_transform(
            container,
            assay_name="proteins",
            source_layer="raw",
            new_layer_name="log",
            base=2.0,
            offset=1.0,
        ),
        output_kind="layer",
        target_assay="proteins",
        target_layer="log",
        note=(
            "Isolated sparse log-transform baseline for JIT/NumPy branch alignment; "
            "normalization is intentionally excluded."
        ),
    )
    rows = [row]
    summary = ScenarioSummary(
        scenario="sparse_log_only",
        status="ok",
        n_stage_rows=1,
        total_elapsed_s=float(row["elapsed_s"]),
        max_peak_delta_mb=float(row["rss_peak_delta_mb"]),
        densify_stages=[row["stage"]] if row["densified_output"] == "true" else [],
        note="Isolated sparse log-transform baseline.",
    )
    return rows, summary


def _run_autoselect_integrate_only(
    *,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    from scptensor.autoselect import AutoSelector

    cfg = PROFILE_CONFIG[profile]
    source_container = _build_dense_protein_container(
        n_samples=int(cfg["dense_samples"]),
        n_features=int(cfg["dense_features"]),
        missing_rate=float(cfg["dense_missing_rate"]),
        seed=seed,
    )
    source_container = log_transform(
        source_container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
        base=2.0,
        offset=1.0,
    )
    source_container = normalize(
        source_container,
        method="median",
        assay_name="proteins",
        source_layer="log",
        new_layer_name="norm",
    )
    source_container = impute(
        source_container,
        method="row_median",
        assay_name="proteins",
        source_layer="norm",
        new_layer_name="imputed",
    )

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    before_assay = source_container.assays["proteins"]
    before_layer = before_assay.layers["imputed"]
    before_x = before_layer.X
    before_m = before_layer.M
    before_x_sig = _matrix_signature(before_x)
    before_m_sig = _matrix_signature(before_m)

    selector = AutoSelector(
        stages=["integrate"],
        keep_all=False,
        selection_strategy="balanced",
        n_repeats=1,
    )

    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result_container, report = selector.run_stage(
            container=source_container.copy(),
            stage="integrate",
            assay_name="proteins",
            source_layer="imputed",
            batch_key="batch",
            bio_key="group",
        )
    finally:
        elapsed = time.perf_counter() - started
        sampler.stop()
    rss_after = int(process.memory_info().rss)

    output_layer = "imputed"
    if report.best_result is not None:
        output_layer = report.best_result.layer_name
    elif report.output_layer:
        output_layer = report.output_layer

    output_matrix = result_container.assays["proteins"].layers[output_layer]

    row = {
        "scenario": "autoselect_integrate_only",
        "stage": "autoselect_integrate",
        "status": "ok",
        "elapsed_s": round(elapsed, 6),
        "rss_before_mb": _format_mb(rss_before),
        "rss_after_mb": _format_mb(rss_after),
        "rss_delta_mb": _format_mb(rss_after - rss_before),
        "rss_peak_mb": _format_mb(sampler.peak_rss),
        "rss_peak_delta_mb": _format_mb(sampler.peak_rss - rss_before),
        "returned_same_object": _bool_flag(result_container is source_container),
        "source_assay_same_object": _bool_flag(result_container.assays["proteins"] is before_assay),
        "source_layer_same_object": _bool_flag(
            result_container.assays["proteins"].layers["imputed"] is before_layer
        ),
        "source_x_same_object": _bool_flag(
            result_container.assays["proteins"].layers["imputed"].X is before_x
        ),
        "source_m_same_object": _bool_flag(
            None
            if before_m is None
            else result_container.assays["proteins"].layers["imputed"].M is before_m
        ),
        "source_x_unchanged": _bool_flag(
            _matrix_signature(result_container.assays["proteins"].layers["imputed"].X)
            == before_x_sig
        ),
        "source_m_unchanged": _bool_flag(
            _matrix_signature(result_container.assays["proteins"].layers["imputed"].M)
            == before_m_sig
        ),
        "input_x_storage": _matrix_storage_kind(before_x),
        "input_m_storage": _matrix_storage_kind(before_m),
        "input_x_bytes": _matrix_nbytes(before_x),
        "input_m_bytes": _matrix_nbytes(before_m),
        "output_kind": "layer",
        "output_assay": "proteins",
        "output_layer": output_layer,
        "output_x_storage": _matrix_storage_kind(output_matrix.X),
        "output_m_storage": _matrix_storage_kind(output_matrix.M),
        "output_x_bytes": _matrix_nbytes(output_matrix.X),
        "output_m_bytes": _matrix_nbytes(output_matrix.M),
        "output_shares_x_with_source": _bool_flag(output_matrix.X is before_x),
        "output_shares_m_with_source": _bool_flag(
            None if output_matrix.M is None or before_m is None else output_matrix.M is before_m
        ),
        "densified_output": _bool_flag(sp.issparse(before_x) and not sp.issparse(output_matrix.X)),
        "obs_cols_added": "",
        "var_cols_added": "",
        "n_samples": result_container.n_samples,
        "n_features": result_container.assays["proteins"].n_features,
        "note": (
            "Profiles AutoSelect integrate stage only on a preprocessed stable input. "
            f"best_method={report.best_method or 'none'}"
        ),
    }

    rows = [row]
    summary = ScenarioSummary(
        scenario="autoselect_integrate_only",
        status="ok",
        n_stage_rows=1,
        total_elapsed_s=float(row["elapsed_s"]),
        max_peak_delta_mb=float(row["rss_peak_delta_mb"]),
        densify_stages=[row["stage"]] if row["densified_output"] == "true" else [],
        note="Profiles stable AutoSelect integration selection only.",
    )
    return rows, summary


def _run_viz_qc_overview(
    *,
    profile: str,
    seed: int,
) -> tuple[list[dict[str, Any]], ScenarioSummary]:
    from matplotlib import pyplot as plt

    from scptensor.viz import plot_data_overview, plot_qc_completeness, plot_qc_matrix_spy

    cfg = PROFILE_CONFIG[profile]
    container = _build_dense_protein_container(
        n_samples=int(cfg["dense_samples"]),
        n_features=int(cfg["dense_features"]),
        missing_rate=float(cfg["dense_missing_rate"]),
        seed=seed,
    )

    rows = [
        _profile_artifact_stage(
            scenario="viz_qc_overview",
            stage="plot_data_overview",
            note="Read-only visualization baseline for workflow overview panels.",
            n_samples=container.n_samples,
            n_features=container.assays["proteins"].n_features,
            runner=lambda: (lambda axes: plt.close(axes[0].figure))(
                plot_data_overview(
                    container,
                    assay_name="proteins",
                    layer="raw",
                    groupby="batch",
                )
            ),
        ),
        _profile_artifact_stage(
            scenario="viz_qc_overview",
            stage="plot_qc_completeness",
            note="Read-only QC completeness visualization baseline.",
            n_samples=container.n_samples,
            n_features=container.assays["proteins"].n_features,
            runner=lambda: (lambda ax: plt.close(ax.figure))(
                plot_qc_completeness(
                    container,
                    assay_name="proteins",
                    layer="raw",
                    group_by="batch",
                )
            ),
        ),
        _profile_artifact_stage(
            scenario="viz_qc_overview",
            stage="plot_qc_matrix_spy",
            note="Read-only missingness matrix visualization baseline.",
            n_samples=container.n_samples,
            n_features=container.assays["proteins"].n_features,
            runner=lambda: (lambda ax: plt.close(ax.figure))(
                plot_qc_matrix_spy(
                    container,
                    assay_name="proteins",
                    layer="raw",
                )
            ),
        ),
    ]

    summary = ScenarioSummary(
        scenario="viz_qc_overview",
        status="ok",
        n_stage_rows=len(rows),
        total_elapsed_s=round(sum(float(r["elapsed_s"]) for r in rows), 6),
        max_peak_delta_mb=max(float(r["rss_peak_delta_mb"]) for r in rows),
        densify_stages=[],
        note="Read-only visualization baseline for stable QC/workflow overview plots.",
    )
    return rows, summary


def _write_stage_runs(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        raise ValueError("No stage rows available to write.")
    fieldnames = list(rows[0].keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any] | list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _run_suite(
    *,
    scenarios: list[str],
    profile: str,
    output_dir: Path,
    seed: int,
    continue_on_error: bool,
) -> int:
    rows: list[dict[str, Any]] = []
    summaries: list[ScenarioSummary] = []
    errors: list[dict[str, Any]] = []

    scenario_map: dict[str, Callable[[], tuple[list[dict[str, Any]], ScenarioSummary]]] = {
        "import_diann_protein_long": lambda: _run_import_diann_protein_long(
            output_dir=output_dir, profile=profile, seed=seed
        ),
        "aggregate_peptide_to_protein": lambda: _run_aggregate_peptide_to_protein(
            profile=profile, seed=seed + 101
        ),
        "stable_chain_dense": lambda: _run_stable_chain_dense(
            profile=profile,
            seed=seed + 202,
            normalization_method="median",
            scenario_name="stable_chain_dense",
            size_prefix="dense",
        ),
        "stable_chain_quantile": lambda: _run_stable_chain_dense(
            profile=profile,
            seed=seed + 303,
            normalization_method="quantile",
            scenario_name="stable_chain_quantile",
            size_prefix="quantile",
        ),
        "stable_chain_trqn": lambda: _run_stable_chain_dense(
            profile=profile,
            seed=seed + 353,
            normalization_method="trqn",
            scenario_name="stable_chain_trqn",
            size_prefix="quantile",
        ),
        "normalize_quantile_only": lambda: _run_logged_normalization_only(
            profile=profile,
            seed=seed + 373,
            normalization_method="quantile",
            scenario_name="normalize_quantile_only",
        ),
        "normalize_trqn_only": lambda: _run_logged_normalization_only(
            profile=profile,
            seed=seed + 383,
            normalization_method="trqn",
            scenario_name="normalize_trqn_only",
        ),
        "sparse_log_only": lambda: _run_sparse_log_only(profile=profile, seed=seed + 393),
        "sparse_transform_normalize": lambda: _run_sparse_transform_normalize(
            profile=profile, seed=seed + 404
        ),
        "autoselect_integrate_only": lambda: _run_autoselect_integrate_only(
            profile=profile, seed=seed + 454
        ),
        "viz_qc_overview": lambda: _run_viz_qc_overview(profile=profile, seed=seed + 505),
    }

    for scenario in scenarios:
        print(f"[INFO] Running scenario: {scenario}")
        started = time.perf_counter()
        try:
            scenario_rows, summary = scenario_map[scenario]()
            rows.extend(scenario_rows)
            summaries.append(summary)
            print(
                "[OK] "
                f"{scenario}: stages={summary.n_stage_rows}, "
                f"elapsed={summary.total_elapsed_s:.3f}s, "
                f"peak_delta={summary.max_peak_delta_mb:.3f} MB"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[ERROR] {scenario}: {type(exc).__name__}: {exc}", file=sys.stderr)
            errors.append(
                {
                    "scenario": scenario,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "elapsed_s": round(elapsed, 6),
                }
            )
            summaries.append(
                ScenarioSummary(
                    scenario=scenario,
                    status="failed",
                    n_stage_rows=0,
                    total_elapsed_s=round(elapsed, 6),
                    max_peak_delta_mb=0.0,
                    densify_stages=[],
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            if not continue_on_error:
                break

    stage_runs_path = output_dir / "stage_runs.csv"
    summary_path = output_dir / "scenario_summary.json"
    env_path = output_dir / "environment.json"
    errors_path = output_dir / "errors.json"

    if rows:
        _write_stage_runs(rows, stage_runs_path)
    _write_json([summary.to_dict() for summary in summaries], summary_path)
    _write_json(errors, errors_path)
    _write_json(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "profile": profile,
            "scenarios": scenarios,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "scptensor_version": getattr(scptensor, "__version__", "unknown"),
            "scptensor_jit_threshold": os.getenv("SCPTENSOR_JIT_THRESHOLD", "10000000"),
            "pid": psutil.Process().pid,
            "output_dir": str(output_dir),
            "cwd": str(PROJECT_ROOT),
        },
        env_path,
    )

    return 1 if errors else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime baselines for stable ScpTensor preprocessing paths."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIG.keys()),
        default="default",
        help="Baseline size profile. Use 'quick' for smoke checks and 'default' for local baselines.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        help="Run only the named scenario. Repeat to select multiple scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runtime_baseline"),
        help="Output directory for CSV/JSON baseline artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic synthetic inputs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining scenarios after one failure.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenarios and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(scenario)
        return 0

    selected = args.scenario or list(SCENARIOS)
    output_dir = _ensure_dir(args.output_dir)
    return _run_suite(
        scenarios=selected,
        profile=args.profile,
        output_dir=output_dir,
        seed=args.seed,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
