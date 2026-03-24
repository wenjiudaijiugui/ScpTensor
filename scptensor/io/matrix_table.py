"""Matrix/pivot-table importer helpers for DIA quant-table input."""

from __future__ import annotations

import numpy as np
import polars as pl

from scptensor.core._structure_assay import Assay
from scptensor.core._structure_container import ScpContainer
from scptensor.core._structure_matrix import MaskCode, ScpMatrix
from scptensor.core.exceptions import ValidationError
from scptensor.io.profiles import (
    DIANN_PEPTIDE_MATRIX_RE,
    SPECTRONAUT_MATRIX_SUFFIXES,
    ImportProfile,
    Level,
    ResolvedSoftware,
)
from scptensor.io.readers import clean_sample_name, make_unique


def numeric_like_columns(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    """Keep columns that can hold numeric quant values."""
    keep: list[str] = []
    for col in candidates:
        series = df[col]
        if series.dtype.is_numeric():
            keep.append(col)
            continue
        casted = series.cast(pl.Float64, strict=False)
        if casted.null_count() < casted.len():
            keep.append(col)
    return keep


def resolve_matrix_sample_columns(
    df: pl.DataFrame,
    software: ResolvedSoftware,
    level: Level,
    feature_col: str,
    profile: ImportProfile,
) -> tuple[list[str], list[str]]:
    """Resolve quantitative matrix columns and stable sample IDs."""
    columns = df.columns

    metadata_like = set(profile.metadata_candidates) | {feature_col}
    generic_candidates = [col for col in columns if col not in metadata_like]
    pattern_candidates: list[str] = []
    if software == "diann" and level == "peptide":
        pattern_candidates = [
            col for col in generic_candidates if DIANN_PEPTIDE_MATRIX_RE.search(col)
        ]
    elif software == "spectronaut":
        pattern_candidates = [
            col for col in generic_candidates if col.endswith(SPECTRONAUT_MATRIX_SUFFIXES)
        ]

    candidates_to_check = pattern_candidates if pattern_candidates else generic_candidates
    sample_cols = numeric_like_columns(df, candidates_to_check)
    if not sample_cols:
        raise ValidationError(
            "No quantitative sample columns detected for matrix input. "
            f"Feature column resolved as '{feature_col}'. "
            "If this is a long-format report, set table_format='long'; "
            "otherwise pass feature_column explicitly and verify quantitative columns are numeric.",
        )

    sample_ids: list[str] = []
    for col in sample_cols:
        if software == "diann" and level == "peptide":
            match = DIANN_PEPTIDE_MATRIX_RE.search(col)
            sample_ids.append(clean_sample_name(match.group(1) if match else col))
            continue

        matched_suffix = next(
            (suffix for suffix in SPECTRONAUT_MATRIX_SUFFIXES if col.endswith(suffix)),
            None,
        )
        raw_name = col[: -len(matched_suffix)] if matched_suffix else col
        sample_ids.append(clean_sample_name(raw_name))

    return sample_cols, make_unique(sample_ids)


def build_var(var_df: pl.DataFrame, feature_col: str) -> pl.DataFrame:
    """Build the assay ``var`` table with a stable ``_index`` column."""
    if feature_col not in var_df.columns:
        raise ValidationError(f"Feature column '{feature_col}' not found in var metadata table.")

    feature_vals = (
        var_df[feature_col]
        .cast(pl.Utf8, strict=False)
        .fill_null("__MISSING_FEATURE_ID__")
        .to_list()
    )
    index_vals = make_unique([str(value) for value in feature_vals])

    out = var_df
    if "_index" in out.columns and feature_col != "_index":
        out = out.drop("_index")
    out = out.with_columns(pl.Series("_index", index_vals))
    ordered = ["_index", *[col for col in out.columns if col != "_index"]]
    return out.select(ordered)


def matrix_to_assay(
    matrix_df: pl.DataFrame,
    sample_cols: list[str],
    sample_ids: list[str],
    var_df: pl.DataFrame,
    assay_name: str,
    layer_name: str,
    *,
    feature_col: str | None = None,
    filtered_pairs: pl.DataFrame | None = None,
) -> ScpContainer:
    """Convert a matrix-shaped DataFrame into a one-assay ``ScpContainer``."""
    x_t = matrix_df.select(
        [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in sample_cols],
    ).to_numpy()
    x = np.asarray(x_t, dtype=np.float64).T

    m_t = np.where(np.isfinite(x_t), MaskCode.VALID.value, MaskCode.UNCERTAIN.value).astype(np.int8)
    if filtered_pairs is not None and feature_col is not None and filtered_pairs.height > 0:
        feature_values = (
            matrix_df[feature_col]
            .cast(pl.Utf8, strict=False)
            .fill_null("__MISSING_FEATURE_ID__")
            .to_list()
        )
        feature_index = {str(feature_id): idx for idx, feature_id in enumerate(feature_values)}
        sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_cols)}
        for feature_id, sample_id in filtered_pairs.iter_rows():
            feature_idx = feature_index.get(str(feature_id))
            sample_idx = sample_index.get(str(sample_id))
            if feature_idx is None or sample_idx is None:
                continue
            if not np.isfinite(x_t[feature_idx, sample_idx]):
                m_t[feature_idx, sample_idx] = MaskCode.FILTERED.value

    m = m_t.T
    obs = pl.DataFrame({"_index": sample_ids})
    assay = Assay(var=var_df, layers={layer_name: ScpMatrix(X=x, M=m)}, feature_id_col="_index")
    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def load_matrix_table(
    df: pl.DataFrame,
    *,
    assay_name: str,
    feature_col: str,
    sample_cols: list[str],
    sample_ids: list[str],
    layer_name: str,
) -> ScpContainer:
    """Load a matrix/pivot table into a one-assay ``ScpContainer``."""
    var_cols = [col for col in df.columns if col not in sample_cols]
    if feature_col not in var_cols:
        var_cols = [feature_col, *var_cols]
    var_df = build_var(df.select(var_cols), feature_col)
    return matrix_to_assay(df, sample_cols, sample_ids, var_df, assay_name, layer_name)


__all__ = [
    "build_var",
    "load_matrix_table",
    "matrix_to_assay",
    "numeric_like_columns",
    "resolve_matrix_sample_columns",
]
