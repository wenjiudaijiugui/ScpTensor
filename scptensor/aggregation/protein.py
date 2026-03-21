"""Protein-level aggregation methods for peptide/precursor assays."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, ValidationError
from scptensor.core.structures import (
    AggregationLink,
    Assay,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)

BasicAggMethod = Literal["sum", "mean", "median", "max", "weighted_mean"]
AggMethod = Literal[
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

_DEFAULT_PROTEIN_COLUMNS: tuple[str, ...] = (
    "PG.ProteinGroups",
    "PG.ProteinAccessions",
    "Protein.Group",
    "Protein.Ids",
    "EG.ProteinId",
    "FG.ProteinGroups",
)


def _to_dense_float64(matrix: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert dense/sparse matrices to dense float64 arrays."""
    if sp.issparse(matrix):
        return cast(sp.spmatrix, matrix).toarray().astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64)


def _to_dense_int8(mask: np.ndarray | sp.spmatrix | None, shape: tuple[int, int]) -> np.ndarray:
    """Convert mask to dense int8 array (or zeros when missing)."""
    if mask is None:
        return np.zeros(shape, dtype=np.int8)
    if sp.issparse(mask):
        return cast(sp.spmatrix, mask).toarray().astype(np.int8, copy=False)
    return np.asarray(mask, dtype=np.int8)


def _safe_row_stat(values: np.ndarray, reducer: Literal["mean", "median", "max"]) -> np.ndarray:
    """Row-wise nan-aware reducer without all-NaN warnings."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    has_value = np.isfinite(values).any(axis=1)
    if not np.any(has_value):
        return out

    if reducer == "mean":
        out[has_value] = np.nanmean(values[has_value], axis=1)
    elif reducer == "median":
        out[has_value] = np.nanmedian(values[has_value], axis=1)
    else:
        out[has_value] = np.nanmax(values[has_value], axis=1)
    return out


def _nan_sum_preserving_missing(values: np.ndarray) -> np.ndarray:
    """Row-wise sum that keeps all-missing rows as NaN."""
    out = np.nansum(values, axis=1)
    out[~np.isfinite(values).any(axis=1)] = np.nan
    return out


def _weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted mean ignoring missing values."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    safe_weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if not np.any(safe_weights > 0):
        return out

    weighted_mask = np.isfinite(values) & (safe_weights > 0)[None, :]
    if not np.any(weighted_mask):
        return out

    weighted_sums = np.where(weighted_mask, values * safe_weights, 0.0).sum(axis=1)
    weight_sums = np.where(weighted_mask, safe_weights, 0.0).sum(axis=1)
    valid_rows = weight_sums > 0
    out[valid_rows] = weighted_sums[valid_rows] / weight_sums[valid_rows]
    return out


def _select_top_n_indices(values: np.ndarray, top_n: int) -> np.ndarray:
    """Select peptide indices by global abundance rank (median across samples)."""
    n_peptides = values.shape[1]
    if top_n <= 0 or top_n >= n_peptides:
        return np.arange(n_peptides, dtype=np.int64)

    scores = np.nanmedian(values, axis=0)
    finite_idx = np.where(np.isfinite(scores))[0]
    if finite_idx.size <= top_n:
        return finite_idx

    ranked = finite_idx[np.argsort(scores[finite_idx])[::-1]]
    return ranked[:top_n]


def _aggregate_basic(values: np.ndarray, method: BasicAggMethod) -> np.ndarray:
    """Aggregate peptide matrix (samples x peptides) with a basic method."""
    if method == "sum":
        return _nan_sum_preserving_missing(values)
    if method == "mean":
        return _safe_row_stat(values, "mean")
    if method == "median":
        return _safe_row_stat(values, "median")
    if method == "max":
        return _safe_row_stat(values, "max")

    peptide_weights = np.nanmedian(values, axis=0)
    return _weighted_row_mean(values, peptide_weights)


def _aggregate_maxlfq(values: np.ndarray, min_ratio_count: int = 1) -> np.ndarray:
    """Approximate MaxLFQ via pairwise median log-ratios and least-squares fitting."""
    n_samples = values.shape[0]
    out = np.full(n_samples, np.nan, dtype=np.float64)

    positive = np.where(values > 0, values, np.nan)
    log_vals = np.log(positive)

    edges: list[tuple[int, int, float, int]] = []
    adjacency: dict[int, set[int]] = {i: set() for i in range(n_samples)}

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diffs = log_vals[i] - log_vals[j]
            finite = np.isfinite(diffs)
            ratio_count = int(np.sum(finite))
            if ratio_count < min_ratio_count:
                continue
            ratio = float(np.nanmedian(diffs[finite]))
            edges.append((i, j, ratio, ratio_count))
            adjacency[i].add(j)
            adjacency[j].add(i)

    if not edges:
        return np.exp(_safe_row_stat(log_vals, "median"))

    visited = set()
    for start in range(n_samples):
        if start in visited:
            continue

        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)

        if len(component) == 1:
            node = component[0]
            row = log_vals[node]
            if np.isfinite(row).any():
                out[node] = float(np.exp(np.nanmedian(row)))
            continue

        index_map = {node: idx for idx, node in enumerate(component)}
        comp_edges = [e for e in edges if e[0] in index_map and e[1] in index_map]

        design_matrix = np.zeros((len(comp_edges) + 1, len(component)), dtype=np.float64)
        b = np.zeros(len(comp_edges) + 1, dtype=np.float64)

        for row_idx, (i, j, ratio, ratio_count) in enumerate(comp_edges):
            weight = np.sqrt(max(ratio_count, 1))
            design_matrix[row_idx, index_map[i]] = weight
            design_matrix[row_idx, index_map[j]] = -weight
            b[row_idx] = weight * ratio

        # Gauge fixing: first sample in component has zero latent value.
        design_matrix[-1, 0] = 1.0

        y, *_ = np.linalg.lstsq(design_matrix, b, rcond=None)

        offsets: list[float] = []
        for node in component:
            row = log_vals[node]
            finite = np.isfinite(row)
            if np.any(finite):
                offsets.extend((row[finite] - y[index_map[node]]).tolist())

        offset = float(np.median(offsets)) if offsets else 0.0
        for node in component:
            out[node] = float(np.exp(y[index_map[node]] + offset))

    return out


def _aggregate_tmp(
    values: np.ndarray, *, log_base: float = 2.0, max_iter: int = 20, tol: float = 1e-6
) -> np.ndarray:
    """Tukey median polish summarization on log-intensity scale."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)

    positive = np.where(values > 0, values, np.nan)
    z = np.log(positive) / np.log(log_base)
    if not np.isfinite(z).any():
        return out

    overall = float(np.nanmedian(z))
    residual = z - overall
    row_effect = np.zeros(z.shape[0], dtype=np.float64)

    for _ in range(max_iter):
        max_delta = 0.0

        for i in range(residual.shape[0]):
            row = residual[i]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            med = float(np.nanmedian(row[finite]))
            row_effect[i] += med
            residual[i, finite] -= med
            max_delta = max(max_delta, abs(med))

        for j in range(residual.shape[1]):
            col = residual[:, j]
            finite = np.isfinite(col)
            if not np.any(finite):
                continue
            med = float(np.nanmedian(col[finite]))
            residual[finite, j] -= med
            max_delta = max(max_delta, abs(med))

        if max_delta < tol:
            break

    has_data = np.isfinite(z).any(axis=1)
    out[has_data] = np.power(log_base, overall + row_effect[has_data])
    return out


def resolve_protein_mapping_column(var: pl.DataFrame, protein_column: str = "auto") -> str:
    """Resolve protein mapping column in peptide/precursor ``var`` metadata."""
    if protein_column != "auto":
        if protein_column not in var.columns:
            raise ValidationError(
                f"protein_column='{protein_column}' not found. Available columns: {var.columns}"
            )
        return protein_column

    for candidate in _DEFAULT_PROTEIN_COLUMNS:
        if candidate in var.columns:
            return candidate

    protein_like = [col for col in var.columns if "protein" in col.lower()]
    if len(protein_like) == 1:
        return protein_like[0]
    if len(protein_like) > 1:
        raise ValidationError(
            "Multiple protein-like columns detected. "
            f"Please pass protein_column explicitly. Candidates: {protein_like}"
        )

    raise ValidationError(
        "No protein mapping column found in peptide metadata. "
        f"Tried defaults: {_DEFAULT_PROTEIN_COLUMNS}. Available columns: {var.columns}"
    )


def _resolve_ibaq_denominator(
    protein_id: str,
    peptide_count: int,
    ibaq_denominator: dict[str, int] | None,
) -> float:
    if ibaq_denominator is None:
        return float(max(peptide_count, 1))

    if protein_id not in ibaq_denominator:
        raise ValidationError(
            "Missing iBAQ denominator for protein. "
            f"protein_id='{protein_id}'. Provide all denominators in ibaq_denominator."
        )

    denom = float(ibaq_denominator[protein_id])
    if denom <= 0:
        raise ValidationError(f"Invalid iBAQ denominator for protein '{protein_id}': {denom}")
    return denom


def _group_indices_by_protein(protein_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group feature indices by sorted protein identifier."""
    unique_proteins, inverse, counts = np.unique(
        np.asarray(protein_ids), return_inverse=True, return_counts=True
    )
    if unique_proteins.size == 0:
        return unique_proteins, []

    order = np.argsort(inverse, kind="stable")
    split_points = np.cumsum(counts[:-1], dtype=np.int64)
    grouped = np.split(order, split_points.tolist())
    return unique_proteins, grouped


def _aggregate_protein_values(
    values: np.ndarray,
    *,
    method: AggMethod,
    top_n: int,
    top_n_aggregate: BasicAggMethod,
    lfq_min_ratio_count: int,
    tmp_log_base: float,
    ibaq_denominator: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate one protein's (samples x peptides) values and return used peptide indices."""
    if method == "top_n":
        selected = _select_top_n_indices(values, top_n)
        return _aggregate_basic(values[:, selected], top_n_aggregate), selected

    if method == "maxlfq":
        selected = np.arange(values.shape[1], dtype=np.int64)
        return _aggregate_maxlfq(values, min_ratio_count=lfq_min_ratio_count), selected

    if method == "tmp":
        selected = np.arange(values.shape[1], dtype=np.int64)
        return _aggregate_tmp(values, log_base=tmp_log_base), selected

    if method == "ibaq":
        selected = np.arange(values.shape[1], dtype=np.int64)
        return _nan_sum_preserving_missing(values) / ibaq_denominator, selected

    selected = np.arange(values.shape[1], dtype=np.int64)
    return _aggregate_basic(values, method), selected


def aggregate_to_protein(
    container: ScpContainer,
    *,
    source_assay: str = "peptides",
    source_layer: str = "raw",
    target_assay: str = "proteins",
    method: AggMethod = "sum",
    protein_column: str = "auto",
    keep_unmapped: bool = True,
    unmapped_label: str = "NA",
    top_n: int = 3,
    top_n_aggregate: BasicAggMethod = "median",
    lfq_min_ratio_count: int = 1,
    tmp_log_base: float = 2.0,
    ibaq_denominator: dict[str, int] | None = None,
) -> ScpContainer:
    """Aggregate peptide/precursor quantification to protein-level assay.

    Supported methods:
    - Basic: ``sum``, ``mean``, ``median``, ``max``, ``weighted_mean``
    - Top-N: ``top_n`` with ``top_n`` and ``top_n_aggregate``
    - Ratio-based: ``maxlfq``
    - Robust log-scale summarization: ``tmp`` (Tukey Median Polish style)
    - Absolute proxy: ``ibaq`` (sum divided by peptide denominator)
    """
    valid_methods: tuple[str, ...] = (
        "sum",
        "mean",
        "median",
        "max",
        "weighted_mean",
        "top_n",
        "maxlfq",
        "tmp",
        "ibaq",
    )
    if method not in valid_methods:
        raise ValidationError(f"Unsupported aggregation method: {method}")
    if top_n < 0:
        raise ValidationError(f"top_n must be >= 0, got {top_n}")
    if lfq_min_ratio_count < 1:
        raise ValidationError(f"lfq_min_ratio_count must be >= 1, got {lfq_min_ratio_count}")
    if tmp_log_base <= 1:
        raise ValidationError(f"tmp_log_base must be > 1, got {tmp_log_base}")

    if source_assay not in container.assays:
        raise AssayNotFoundError(source_assay, available_assays=container.assays.keys())

    peptide_assay = container.assays[source_assay]
    if source_layer not in peptide_assay.layers:
        raise ValidationError(
            f"Layer '{source_layer}' not found in assay '{source_assay}'. "
            f"Available: {list(peptide_assay.layers.keys())}"
        )

    protein_col = resolve_protein_mapping_column(peptide_assay.var, protein_column)
    protein_map = peptide_assay.var[protein_col].cast(pl.Utf8, strict=False)

    source_ids = (
        peptide_assay.var[peptide_assay.feature_id_col]
        .cast(pl.Utf8, strict=False)
        .fill_null("__MISSING_SOURCE_ID__")
        .to_numpy()
    )

    if keep_unmapped:
        protein_ids = protein_map.fill_null(unmapped_label).to_numpy()
        valid_idx = np.arange(len(protein_ids), dtype=np.int64)
    else:
        map_values = protein_map.to_numpy()
        valid_idx = np.array(
            [i for i, value in enumerate(map_values) if value is not None], dtype=np.int64
        )
        if valid_idx.size == 0:
            raise ValidationError(
                "No mapped peptides available after removing null protein mapping values."
            )
        protein_ids = map_values[valid_idx]

    layer = peptide_assay.layers[source_layer]
    x_src = _to_dense_float64(layer.X)
    m_src = _to_dense_int8(layer.M, x_src.shape)

    if valid_idx.size != x_src.shape[1]:
        x_src = x_src[:, valid_idx]
        m_src = m_src[:, valid_idx]
        source_ids = source_ids[valid_idx]

    unique_proteins, grouped_indices = _group_indices_by_protein(protein_ids)
    if unique_proteins.size == 0:
        raise ValidationError("No protein groups found to aggregate.")

    x_protein = np.zeros((container.n_samples, len(unique_proteins)), dtype=np.float64)
    m_protein = np.zeros((container.n_samples, len(unique_proteins)), dtype=np.int8)
    linkage_source: list[str] = []
    linkage_target: list[str] = []

    for j, (protein_id, idx) in enumerate(zip(unique_proteins, grouped_indices, strict=True)):
        vals = x_src[:, idx]
        masks = m_src[:, idx]

        denom = 1.0
        if method == "ibaq":
            denom = _resolve_ibaq_denominator(str(protein_id), idx.size, ibaq_denominator)
        agg_vals, used = _aggregate_protein_values(
            vals,
            method=method,
            top_n=top_n,
            top_n_aggregate=top_n_aggregate,
            lfq_min_ratio_count=lfq_min_ratio_count,
            tmp_log_base=tmp_log_base,
            ibaq_denominator=denom,
        )

        x_protein[:, j] = agg_vals
        used_masks = masks if used.size == masks.shape[1] else masks[:, used]
        m_protein[:, j] = np.max(used_masks, axis=1)

        group_source_ids = source_ids[idx]
        linkage_source.extend(group_source_ids.tolist())
        linkage_target.extend([str(protein_id)] * group_source_ids.size)

    protein_var = pl.DataFrame({"_index": unique_proteins.astype(str).tolist()})
    if protein_col != "_index":
        protein_var = protein_var.with_columns(pl.col("_index").alias(protein_col))

    protein_assay = Assay(
        var=protein_var,
        layers={source_layer: ScpMatrix(X=x_protein, M=m_protein)},
        feature_id_col="_index",
    )

    link = AggregationLink(
        source_assay=source_assay,
        target_assay=target_assay,
        linkage=pl.DataFrame({"source_id": linkage_source, "target_id": linkage_target}),
    )

    history = list(container.history)
    history.append(
        ProvenanceLog(
            timestamp=datetime.now().isoformat(),
            action="aggregate_to_protein",
            params={
                "source_assay": source_assay,
                "source_layer": source_layer,
                "target_assay": target_assay,
                "method": method,
                "protein_column": protein_col,
                "keep_unmapped": keep_unmapped,
                "top_n": top_n,
                "top_n_aggregate": top_n_aggregate,
                "lfq_min_ratio_count": lfq_min_ratio_count,
                "tmp_log_base": tmp_log_base,
                "has_ibaq_denominator": ibaq_denominator is not None,
            },
            description=(
                f"Aggregated {source_assay}/{source_layer} -> {target_assay} via {method} "
                f"(protein_column={protein_col})."
            ),
        )
    )

    new_assays = dict(container.assays)
    new_assays[target_assay] = protein_assay

    return ScpContainer(
        obs=container.obs.clone(),
        assays=new_assays,
        links=[*container.links, link],
        history=history,
        sample_id_col=container.sample_id_col,
    )
