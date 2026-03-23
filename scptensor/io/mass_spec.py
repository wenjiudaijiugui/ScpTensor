"""Unified DIA-NN / Spectronaut quant-table importers.

Supported software:
- DIA-NN
- Spectronaut

Supported table styles:
- Long report tables
- Pivot/matrix tables

Supported quant levels:
- Protein-level
- Peptide/precursor-level
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from scptensor.aggregation import aggregate_to_protein as _aggregate_to_protein
from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import (
    Assay,
    MaskCode,
    ScpContainer,
    ScpMatrix,
)

Software = Literal["diann", "spectronaut", "auto"]
ResolvedSoftware = Literal["diann", "spectronaut"]
Level = Literal["protein", "peptide"]
TableFormat = Literal["auto", "matrix", "long"]
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
TopNAggregate = Literal["sum", "mean", "median", "max", "weighted_mean"]

_SUPPORTED_SUFFIXES = {".tsv", ".txt", ".csv", ".parquet"}

_DIANN_PEPTIDE_MATRIX_RE = re.compile(
    r"(?:\[[^\]]+\]\s*)?(.+?)\.(?:PEP|Precursor)\.(?:Quantity|Intensity)$",
    re.IGNORECASE,
)

# Common Spectronaut pivot suffixes seen in run-pivot exports.
_SPECTRONAUT_MATRIX_SUFFIXES = (
    "_Quantity",
    "_PeptideQuantity",
    "_PrecursorQuantity",
    "_TotalQuantity",
    "_Intensity",
    "_PeakArea",
    "_NormalizedPeakArea",
)


@dataclass(frozen=True)
class _ImportProfile:
    """Column-resolution hints for a (software, level) pair."""

    feature_candidates: tuple[str, ...]
    quantity_candidates: tuple[str, ...]
    sample_candidates_long: tuple[str, ...]
    fdr_candidates: tuple[str, ...]
    metadata_candidates: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedLongColumns:
    """Resolved long-table columns used by a single import run."""

    feature: str
    quantity: str
    sample: str
    fdr: str | None


_PROFILES: dict[tuple[ResolvedSoftware, Level], _ImportProfile] = {
    ("diann", "protein"): _ImportProfile(
        feature_candidates=("Protein.Group", "Protein.Groups", "Protein.Ids"),
        quantity_candidates=("PG.MaxLFQ", "PG.TopN", "PG.Quantity", "PG.Normalised"),
        sample_candidates_long=("Run",),
        fdr_candidates=("PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value"),
        metadata_candidates=(
            "Protein.Group",
            "Protein.Groups",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
            "PG.Q.Value",
            "Global.PG.Q.Value",
            "Lib.PG.Q.Value",
        ),
    ),
    ("diann", "peptide"): _ImportProfile(
        feature_candidates=(
            "Precursor.Id",
            "EG.PrecursorId",
            "Modified.Sequence",
            "Stripped.Sequence",
        ),
        quantity_candidates=("Precursor.Normalised", "Precursor.Quantity"),
        sample_candidates_long=("Run",),
        fdr_candidates=("Q.Value", "Global.Q.Value", "Lib.Q.Value"),
        metadata_candidates=(
            "Precursor.Id",
            "Modified.Sequence",
            "Stripped.Sequence",
            "Precursor.Charge",
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "Q.Value",
            "Global.Q.Value",
            "Lib.Q.Value",
        ),
    ),
    ("spectronaut", "protein"): _ImportProfile(
        feature_candidates=(
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "ProteinGroups",
            "ProteinGroup",
        ),
        quantity_candidates=("PG.Quantity", "PG.Normalized", "PG.Normalised"),
        sample_candidates_long=("R.FileName", "R.File.Name"),
        fdr_candidates=("PG.Qvalue", "PG.QValue", "PG.Q.Value"),
        metadata_candidates=(
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "PG.ProteinNames",
            "PG.Genes",
            "PG.Description",
            "PG.Qvalue",
            "PG.QValue",
            "PG.Q.Value",
        ),
    ),
    ("spectronaut", "peptide"): _ImportProfile(
        feature_candidates=(
            "EG.PrecursorId",
            "PEP.GroupingKey",
            "EG.ModifiedSequence",
            "FG.PrecursorId",
            "FG.Id",
        ),
        quantity_candidates=(
            "EG.TotalQuantity",
            "PEP.Quantity",
            "FG.Quantity",
            "F.NormalizedPeakArea",
            "F.PeakArea",
        ),
        sample_candidates_long=("R.FileName", "R.File.Name"),
        fdr_candidates=(
            "EG.Qvalue",
            "EG.QValue",
            "PEP.Qvalue",
            "PEP.QValue",
            "Q.Value",
        ),
        metadata_candidates=(
            "EG.PrecursorId",
            "PEP.GroupingKey",
            "PEP.GroupingKeyType",
            "EG.ModifiedSequence",
            "FG.PrecursorId",
            "FG.Id",
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "EG.ProteinId",
            "EG.Qvalue",
            "EG.QValue",
            "PEP.Qvalue",
            "PEP.QValue",
            "Q.Value",
        ),
    ),
}


def _preview_columns(columns: list[str], limit: int = 12) -> str:
    head = columns[:limit]
    suffix = "" if len(columns) <= limit else f", ... (+{len(columns) - limit} more)"
    return ", ".join(head) + suffix


def _validate_table_format(table_format: str) -> None:
    """Validate runtime table_format input."""
    valid_formats = {"auto", "matrix", "long"}
    if table_format not in valid_formats:
        raise ValidationError(
            f"Unsupported table_format='{table_format}'. "
            "Supported values: 'auto', 'matrix', 'long'."
        )


def _is_vendor_normalized_column(column: str) -> bool:
    """Return whether a resolved quantity column looks vendor-normalized."""
    normalized_tokens = ("normalized", "normalised")
    return any(token in column.lower() for token in normalized_tokens)


def _clean_sample_name(name: str) -> str:
    """Normalize sample names from column values."""
    return re.sub(r"(?i)\.raw$", "", str(name).strip())


def _make_unique(values: list[str]) -> list[str]:
    """Make values unique by suffixing repeated entries."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for val in values:
        n = seen.get(val, 0)
        if n == 0:
            out.append(val)
        else:
            out.append(f"{val}_{n}")
        seen[val] = n + 1
    return out


def _infer_separator(path: Path, delimiter: str | None = None) -> str:
    """Infer delimiter for text tables."""
    if delimiter is not None:
        return delimiter
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    return ","


def _read_table(
    path: Path, delimiter: str | None = None, n_rows: int | None = None
) -> pl.DataFrame:
    """Read table as Polars DataFrame (CSV/TSV/Parquet)."""
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValidationError(
            "Unsupported file extension for quant-table import: "
            f"'{suffix or '<none>'}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    try:
        if suffix == ".parquet":
            return pl.read_parquet(path) if n_rows is None else pl.read_parquet(path).head(n_rows)
        return pl.read_csv(
            path,
            separator=_infer_separator(path, delimiter),
            n_rows=n_rows,
            null_values=["", "NA", "NaN", "nan", "NULL", "null"],
            infer_schema_length=0,
            ignore_errors=n_rows is not None,
        )
    except Exception as exc:
        raise ValidationError(
            f"Failed to read quant table '{path}'. "
            "Check delimiter/file encoding and ensure the export is a valid DIA-NN/Spectronaut table."
        ) from exc


def detect_software(columns: list[str]) -> Literal["diann", "spectronaut", "unknown"]:
    """Detect vendor software from column names."""
    cols = set(columns)

    if {"R.FileName", "R.File.Name"} & cols:
        return "spectronaut"
    if "Run" in cols:
        return "diann"

    if "PG.ProteinGroups" in cols or "PEP.GroupingKey" in cols or "EG.PrecursorId" in cols:
        return "spectronaut"
    if "Protein.Group" in cols or "Precursor.Id" in cols or "Modified.Sequence" in cols:
        return "diann"

    if any(_DIANN_PEPTIDE_MATRIX_RE.search(col) for col in columns):
        return "diann"
    if any(col.endswith(_SPECTRONAUT_MATRIX_SUFFIXES) for col in columns):
        return "spectronaut"

    return "unknown"


def _resolve_profile(software: ResolvedSoftware, level: Level) -> _ImportProfile:
    key = (software, level)
    if key not in _PROFILES:
        raise ValidationError(f"Unsupported combination: software={software}, level={level}")
    return _PROFILES[key]


def _resolve_feature_column(
    columns: list[str], profile: _ImportProfile, feature_column: str
) -> str:
    if feature_column != "auto":
        if feature_column not in columns:
            raise ValidationError(
                f"feature_column='{feature_column}' not found. "
                f"Available columns: {_preview_columns(columns)}"
            )
        return feature_column

    for candidate in profile.feature_candidates:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect feature column. "
        f"Expected one of: {profile.feature_candidates}. "
        f"Available columns: {_preview_columns(columns)}"
    )


def _resolve_quantity_column(
    columns: list[str], profile: _ImportProfile, quantity_column: str
) -> str:
    if quantity_column != "auto":
        if quantity_column not in columns:
            raise ValidationError(
                f"quantity_column='{quantity_column}' not found. "
                f"Available columns: {_preview_columns(columns)}"
            )
        return quantity_column

    for candidate in profile.quantity_candidates:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect quantity column for long-format input. "
        f"Expected one of: {profile.quantity_candidates}. "
        f"Available columns: {_preview_columns(columns)}"
    )


def _resolve_sample_column_long(
    columns: list[str], profile: _ImportProfile, sample_column: str
) -> str:
    if sample_column != "auto":
        if sample_column not in columns:
            raise ValidationError(
                f"sample_column='{sample_column}' not found. "
                f"Available columns: {_preview_columns(columns)}"
            )
        return sample_column

    for candidate in profile.sample_candidates_long:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect sample column for long-format input. "
        f"Expected one of: {profile.sample_candidates_long}. "
        f"Available columns: {_preview_columns(columns)}"
    )


def _resolve_long_columns(
    columns: list[str],
    profile: _ImportProfile,
    *,
    feature_column: str,
    quantity_column: str,
    sample_column: str,
) -> _ResolvedLongColumns:
    """Resolve all required long-format columns once per import."""
    return _ResolvedLongColumns(
        feature=_resolve_feature_column(columns, profile, feature_column),
        quantity=_resolve_quantity_column(columns, profile, quantity_column),
        sample=_resolve_sample_column_long(columns, profile, sample_column),
        fdr=_resolve_fdr_column(columns, profile),
    )


def _numeric_like_columns(df: pl.DataFrame, candidates: list[str]) -> list[str]:
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


def _is_long_format(
    path: Path,
    columns: list[str],
    profile: _ImportProfile,
    sample_column: str,
    quantity_column: str,
    table_format: TableFormat,
) -> bool:
    if table_format == "long":
        return True
    if table_format == "matrix":
        return False

    has_sample = (
        sample_column in columns
        if sample_column != "auto"
        else any(c in columns for c in profile.sample_candidates_long)
    )
    has_quantity = (
        quantity_column in columns
        if quantity_column != "auto"
        else any(c in columns for c in profile.quantity_candidates)
    )
    return has_sample and has_quantity


def _resolve_matrix_sample_columns(
    df: pl.DataFrame,
    software: ResolvedSoftware,
    level: Level,
    feature_col: str,
    profile: _ImportProfile,
) -> tuple[list[str], list[str]]:
    columns = df.columns

    metadata_like = set(profile.metadata_candidates) | {feature_col}
    generic_candidates = [col for col in columns if col not in metadata_like]
    pattern_candidates: list[str] = []
    if software == "diann" and level == "peptide":
        pattern_candidates = [
            col for col in generic_candidates if _DIANN_PEPTIDE_MATRIX_RE.search(col)
        ]
    elif software == "spectronaut":
        pattern_candidates = [
            col for col in generic_candidates if col.endswith(_SPECTRONAUT_MATRIX_SUFFIXES)
        ]

    candidates_to_check = pattern_candidates if pattern_candidates else generic_candidates
    sample_cols = _numeric_like_columns(df, candidates_to_check)
    if not sample_cols:
        raise ValidationError(
            "No quantitative sample columns detected for matrix input. "
            f"Feature column resolved as '{feature_col}'. "
            "If this is a long-format report, set table_format='long'; "
            "otherwise pass feature_column explicitly and verify quantitative columns are numeric."
        )

    sample_ids: list[str] = []
    for col in sample_cols:
        if software == "diann" and level == "peptide":
            match = _DIANN_PEPTIDE_MATRIX_RE.search(col)
            sample_ids.append(_clean_sample_name(match.group(1) if match else col))
            continue

        matched_suffix = next(
            (sfx for sfx in _SPECTRONAUT_MATRIX_SUFFIXES if col.endswith(sfx)), None
        )
        raw_name = col[: -len(matched_suffix)] if matched_suffix else col
        sample_ids.append(_clean_sample_name(raw_name))

    return sample_cols, _make_unique(sample_ids)


def _build_var(var_df: pl.DataFrame, feature_col: str) -> pl.DataFrame:
    if feature_col not in var_df.columns:
        raise ValidationError(f"Feature column '{feature_col}' not found in var metadata table.")

    feature_vals = (
        var_df[feature_col]
        .cast(pl.Utf8, strict=False)
        .fill_null("__MISSING_FEATURE_ID__")
        .to_list()
    )
    index_vals = _make_unique([str(v) for v in feature_vals])

    out = var_df
    if "_index" in out.columns and feature_col != "_index":
        out = out.drop("_index")
    out = out.with_columns(pl.Series("_index", index_vals))
    ordered = ["_index", *[c for c in out.columns if c != "_index"]]
    return out.select(ordered)


def _matrix_to_assay(
    matrix_df: pl.DataFrame,
    sample_cols: list[str],
    sample_ids: list[str],
    var_df: pl.DataFrame,
    assay_name: str,
    layer_name: str,
) -> ScpContainer:
    x_t = matrix_df.select(
        [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in sample_cols]
    ).to_numpy()
    x = np.asarray(x_t, dtype=np.float64).T

    m = np.where(np.isfinite(x), MaskCode.VALID.value, MaskCode.UNCERTAIN.value).astype(np.int8)
    obs = pl.DataFrame({"_index": sample_ids})

    assay = Assay(var=var_df, layers={layer_name: ScpMatrix(X=x, M=m)}, feature_id_col="_index")
    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def _load_matrix_table(
    df: pl.DataFrame,
    *,
    assay_name: str,
    feature_col: str,
    sample_cols: list[str],
    sample_ids: list[str],
    layer_name: str,
) -> ScpContainer:
    var_cols = [col for col in df.columns if col not in sample_cols]
    if feature_col not in var_cols:
        var_cols = [feature_col, *var_cols]
    var_df = _build_var(df.select(var_cols), feature_col)
    return _matrix_to_assay(df, sample_cols, sample_ids, var_df, assay_name, layer_name)


def _resolve_fdr_column(columns: list[str], profile: _ImportProfile) -> str | None:
    for candidate in profile.fdr_candidates:
        if candidate in columns:
            return candidate
    return None


def _apply_fdr_filter(
    df: pl.DataFrame,
    profile: _ImportProfile,
    threshold: float | None,
) -> tuple[pl.DataFrame, str | None]:
    if threshold is None:
        return df, None

    fdr_col = _resolve_fdr_column(df.columns, profile)
    if fdr_col is None:
        return df, None

    metric = pl.col(fdr_col).cast(pl.Float64, strict=False)
    return df.filter(metric.is_null() | (metric <= threshold)), fdr_col


def _load_long_table(
    df: pl.DataFrame,
    *,
    assay_name: str,
    profile: _ImportProfile,
    resolved_cols: _ResolvedLongColumns,
    fdr_threshold: float | None,
    layer_name: str,
) -> ScpContainer:
    feature_col = resolved_cols.feature
    qty_col = resolved_cols.quantity
    sample_col = resolved_cols.sample

    work = df.with_columns(
        pl.col(feature_col).cast(pl.Utf8, strict=False),
        pl.col(sample_col).cast(pl.Utf8, strict=False),
        pl.col(qty_col).cast(pl.Float64, strict=False),
    )

    work = work.filter(pl.col(feature_col).is_not_null() & pl.col(sample_col).is_not_null())
    work = work.with_columns(
        pl.col(sample_col).str.replace(r"(?i)\.raw$", "").str.strip_chars().alias("_sample_id")
    )

    before_rows = work.height
    work, used_fdr_col = _apply_fdr_filter(work, profile, fdr_threshold)
    if work.is_empty():
        if used_fdr_col is not None:
            raise ValidationError(
                "No rows remain after FDR filtering. "
                f"Applied '{used_fdr_col} <= {fdr_threshold}' to {before_rows} rows."
            )
        raise ValidationError(
            "No rows remain after removing null feature/sample values. "
            "Check feature/sample columns and file content."
        )

    matrix_df = work.select([feature_col, "_sample_id", qty_col]).pivot(
        index=feature_col,
        on="_sample_id",
        values=qty_col,
        aggregate_function="max",
    )

    sample_cols = [col for col in matrix_df.columns if col != feature_col]
    if not sample_cols:
        raise ValidationError(
            "No sample columns produced after long-to-matrix pivot. "
            f"sample_column='{sample_col}', quantity_column='{qty_col}'."
        )

    meta_cols = [col for col in profile.metadata_candidates if col in work.columns]
    if feature_col not in meta_cols:
        meta_cols = [feature_col, *meta_cols]
    var_meta = work.select(meta_cols).unique(subset=[feature_col], keep="first")

    aligned = matrix_df.join(var_meta, on=feature_col, how="left")
    var_df = _build_var(
        aligned.select([col for col in aligned.columns if col not in sample_cols]), feature_col
    )
    sample_ids = _make_unique([_clean_sample_name(col) for col in sample_cols])
    return _matrix_to_assay(aligned, sample_cols, sample_ids, var_df, assay_name, layer_name)


def _resolve_software(software: Software, columns: list[str]) -> ResolvedSoftware:
    if software != "auto":
        if software not in ("diann", "spectronaut"):
            raise ValidationError(
                f"Unsupported software='{software}'. Supported values: 'diann', 'spectronaut', 'auto'."
            )
        return software

    guessed = detect_software(columns)
    if guessed == "unknown":
        raise ValidationError(
            "Unable to detect software type from columns. "
            "Please set software='diann' or software='spectronaut' explicitly. "
            f"Columns preview: {_preview_columns(columns)}"
        )
    return guessed


def load_quant_table(
    path: str | Path,
    *,
    software: Software = "auto",
    level: Level = "protein",
    assay_name: str | None = None,
    table_format: TableFormat = "auto",
    quantity_column: str = "auto",
    sample_column: str = "auto",
    feature_column: str = "auto",
    fdr_threshold: float | None = 0.01,
    layer_name: str = "raw",
    delimiter: str | None = None,
) -> ScpContainer:
    """Load DIA-NN/Spectronaut quant tables into a unified ScpContainer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if level not in ("protein", "peptide"):
        raise ValidationError(f"Unsupported level='{level}'. Use 'protein' or 'peptide'.")
    _validate_table_format(table_format)

    if fdr_threshold is not None and not (0 <= fdr_threshold <= 1):
        raise ValidationError(f"fdr_threshold must be within [0, 1], got {fdr_threshold}")

    preview = _read_table(path, delimiter=delimiter, n_rows=50)
    if preview.is_empty():
        raise ValidationError(f"Input file is empty: {path}")

    resolved_software = _resolve_software(software, preview.columns)
    profile = _resolve_profile(resolved_software, level)
    resolved_assay = assay_name or ("proteins" if level == "protein" else "peptides")
    resolved_feature_column: str | None = None
    resolved_quantity_column: str | None = None
    resolved_sample_column: str | None = None
    used_fdr_column: str | None = None
    vendor_normalized_input = False

    full_df = _read_table(path, delimiter=delimiter)
    is_long = _is_long_format(
        path,
        preview.columns,
        profile,
        sample_column=sample_column,
        quantity_column=quantity_column,
        table_format=table_format,
    )

    if is_long:
        resolved_long = _resolve_long_columns(
            full_df.columns,
            profile,
            feature_column=feature_column,
            quantity_column=quantity_column,
            sample_column=sample_column,
        )
        resolved_feature_column = resolved_long.feature
        resolved_quantity_column = resolved_long.quantity
        resolved_sample_column = resolved_long.sample
        used_fdr_column = resolved_long.fdr
        vendor_normalized_input = _is_vendor_normalized_column(resolved_quantity_column)
        container = _load_long_table(
            full_df,
            assay_name=resolved_assay,
            profile=profile,
            resolved_cols=resolved_long,
            fdr_threshold=fdr_threshold,
            layer_name=layer_name,
        )
    else:
        full_df, used_fdr_column = _apply_fdr_filter(full_df, profile, fdr_threshold)
        if full_df.is_empty():
            if used_fdr_column is not None:
                raise ValidationError(
                    "No rows remain after FDR filtering. "
                    f"Applied '{used_fdr_column} <= {fdr_threshold}' to matrix-format input."
                )
            raise ValidationError("No rows remain in matrix-format input after preprocessing.")

        resolved_feature_column = _resolve_feature_column(full_df.columns, profile, feature_column)
        sample_cols, sample_ids = _resolve_matrix_sample_columns(
            full_df,
            resolved_software,
            level,
            resolved_feature_column,
            profile,
        )
        vendor_normalized_input = any(
            _is_vendor_normalized_column(column) for column in sample_cols
        )
        container = _load_matrix_table(
            full_df,
            assay_name=resolved_assay,
            feature_col=resolved_feature_column,
            sample_cols=sample_cols,
            sample_ids=sample_ids,
            layer_name=layer_name,
        )

    container.log_operation(
        action="load_quant_table",
        params={
            "path": str(path),
            "software": resolved_software,
            "level": level,
            "assay_name": resolved_assay,
            "format": "long" if is_long else "matrix",
            "quantity_column": quantity_column,
            "sample_column": sample_column,
            "feature_column": feature_column,
            "resolved_feature_column": resolved_feature_column,
            "resolved_quantity_column": resolved_quantity_column,
            "resolved_sample_column": resolved_sample_column,
            "used_fdr_column": used_fdr_column,
            "input_quantity_is_vendor_normalized": vendor_normalized_input,
            "fdr_threshold": fdr_threshold,
            "layer_name": layer_name,
        },
        description=(
            f"Loaded {resolved_software} {level}-level quant table from {path.name}."
            + (" Source quantity appears vendor-normalized." if vendor_normalized_input else "")
        ),
    )
    return container


def aggregate_to_protein(
    container: ScpContainer,
    *,
    source_assay: str = "peptides",
    source_layer: str = "raw",
    target_assay: str = "proteins",
    method: AggMethod = "sum",
    protein_column: str = "auto",
    keep_unmapped: bool = True,
    top_n: int = 3,
    top_n_aggregate: TopNAggregate = "median",
    lfq_min_ratio_count: int = 1,
    tmp_log_base: float = 2.0,
    ibaq_denominator: dict[str, int] | None = None,
) -> ScpContainer:
    """Backward-compatible wrapper around :mod:`scptensor.aggregation`."""
    return _aggregate_to_protein(
        container,
        source_assay=source_assay,
        source_layer=source_layer,
        target_assay=target_assay,
        method=method,
        protein_column=protein_column,
        keep_unmapped=keep_unmapped,
        top_n=top_n,
        top_n_aggregate=top_n_aggregate,
        lfq_min_ratio_count=lfq_min_ratio_count,
        tmp_log_base=tmp_log_base,
        ibaq_denominator=ibaq_denominator,
    )


def load_diann(
    path: str | Path,
    *,
    assay_name: str | None = None,
    quantity_column: str = "auto",
    level: Level = "protein",
    table_format: TableFormat = "auto",
    fdr_threshold: float | None = 0.01,
    layer_name: str = "raw",
) -> ScpContainer:
    """Load DIA-NN protein/peptide quant tables."""
    return load_quant_table(
        path,
        software="diann",
        level=level,
        assay_name=assay_name,
        table_format=table_format,
        quantity_column=quantity_column,
        fdr_threshold=fdr_threshold,
        layer_name=layer_name,
    )


def load_spectronaut(
    path: str | Path,
    *,
    assay_name: str | None = None,
    quantity_column: str = "auto",
    level: Level = "protein",
    table_format: TableFormat = "auto",
    fdr_threshold: float | None = 0.01,
    layer_name: str = "raw",
) -> ScpContainer:
    """Load Spectronaut protein/peptide quant tables."""
    return load_quant_table(
        path,
        software="spectronaut",
        level=level,
        assay_name=assay_name,
        table_format=table_format,
        quantity_column=quantity_column,
        fdr_threshold=fdr_threshold,
        layer_name=layer_name,
    )


def load_peptide_pivot(
    path: str | Path,
    *,
    assay_name: str = "peptides",
    software: Software = "auto",
    protein_agg: bool = False,
    protein_assay_name: str = "proteins",
    agg_method: AggMethod = "sum",
    agg_top_n: int = 3,
    agg_top_n_aggregate: TopNAggregate = "median",
    agg_lfq_min_ratio_count: int = 1,
    agg_tmp_log_base: float = 2.0,
    agg_ibaq_denominator: dict[str, int] | None = None,
    quantity_column: str = "auto",
    table_format: TableFormat = "auto",
    fdr_threshold: float | None = 0.01,
    layer_name: str = "raw",
) -> ScpContainer:
    """Load peptide/precursor pivot matrix (DIA-NN / Spectronaut)."""
    container = load_quant_table(
        path,
        software=software,
        level="peptide",
        assay_name=assay_name,
        table_format=table_format,
        quantity_column=quantity_column,
        fdr_threshold=fdr_threshold,
        layer_name=layer_name,
    )
    if protein_agg:
        container = aggregate_to_protein(
            container,
            source_assay=assay_name,
            source_layer=layer_name,
            target_assay=protein_assay_name,
            method=agg_method,
            top_n=agg_top_n,
            top_n_aggregate=agg_top_n_aggregate,
            lfq_min_ratio_count=agg_lfq_min_ratio_count,
            tmp_log_base=agg_tmp_log_base,
            ibaq_denominator=agg_ibaq_denominator,
        )
    return container
