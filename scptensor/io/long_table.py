"""Long-table importer helpers for DIA quant-table input."""

from __future__ import annotations

import polars as pl

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ValidationError
from scptensor.io.matrix_table import build_var, matrix_to_assay
from scptensor.io.profiles import ImportProfile, ResolvedLongColumns, resolve_fdr_column
from scptensor.io.readers import clean_sample_name, make_unique


def is_long_format(
    columns: list[str],
    profile: ImportProfile,
    sample_column: str,
    quantity_column: str,
    table_format: str,
) -> bool:
    """Determine whether the import should use the long-table path."""
    if table_format == "long":
        return True
    if table_format == "matrix":
        return False

    has_sample = (
        sample_column in columns
        if sample_column != "auto"
        else any(candidate in columns for candidate in profile.sample_candidates_long)
    )
    has_quantity = (
        quantity_column in columns
        if quantity_column != "auto"
        else any(candidate in columns for candidate in profile.quantity_candidates)
    )
    return has_sample and has_quantity


def apply_fdr_filter(
    df: pl.DataFrame,
    profile: ImportProfile,
    threshold: float | None,
) -> tuple[pl.DataFrame, str | None]:
    """Apply optional FDR filtering and return the used FDR column."""
    if threshold is None:
        return df, None

    fdr_col = resolve_fdr_column(df.columns, profile)
    if fdr_col is None:
        return df, None

    metric = pl.col(fdr_col).cast(pl.Float64, strict=False)
    return df.filter(metric.is_null() | (metric <= threshold)), fdr_col


def collect_fdr_filtered_pairs(
    df: pl.DataFrame,
    *,
    feature_col: str,
    sample_col: str,
    fdr_col: str,
    threshold: float,
) -> pl.DataFrame:
    """Return feature-sample pairs that exist upstream but are fully removed by FDR filtering."""
    metric = pl.col(fdr_col).cast(pl.Float64, strict=False)
    return (
        df.select(
            pl.col(feature_col).cast(pl.Utf8, strict=False).alias(feature_col),
            pl.col(sample_col).cast(pl.Utf8, strict=False).alias(sample_col),
            (metric.is_null() | (metric <= threshold)).alias("_passes_fdr"),
        )
        .group_by([feature_col, sample_col])
        .agg(pl.col("_passes_fdr").any().alias("_has_pass"))
        .filter(~pl.col("_has_pass"))
        .select([feature_col, sample_col])
    )


def load_long_table(
    df: pl.DataFrame,
    *,
    assay_name: str,
    profile: ImportProfile,
    resolved_cols: ResolvedLongColumns,
    fdr_threshold: float | None,
    layer_name: str,
) -> ScpContainer:
    """Load a long-format quant table into a one-assay ``ScpContainer``."""
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
        pl.col(sample_col).str.replace(r"(?i)\.raw$", "").str.strip_chars().alias("_sample_id"),
    )

    before_rows = work.height
    filtered_pairs: pl.DataFrame | None = None
    resolved_fdr_col = resolve_fdr_column(work.columns, profile)
    if resolved_fdr_col is not None and fdr_threshold is not None:
        filtered_pairs = collect_fdr_filtered_pairs(
            work,
            feature_col=feature_col,
            sample_col="_sample_id",
            fdr_col=resolved_fdr_col,
            threshold=fdr_threshold,
        )

    work, used_fdr_col = apply_fdr_filter(work, profile, fdr_threshold)
    if work.is_empty():
        if used_fdr_col is not None:
            raise ValidationError(
                "No rows remain after FDR filtering. "
                f"Applied '{used_fdr_col} <= {fdr_threshold}' to {before_rows} rows.",
            )
        raise ValidationError(
            "No rows remain after removing null feature/sample values. "
            "Check feature/sample columns and file content.",
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
            f"sample_column='{sample_col}', quantity_column='{qty_col}'.",
        )

    meta_cols = [col for col in profile.metadata_candidates if col in work.columns]
    if feature_col not in meta_cols:
        meta_cols = [feature_col, *meta_cols]
    var_meta = work.select(meta_cols).unique(subset=[feature_col], keep="first")

    aligned = matrix_df.join(var_meta, on=feature_col, how="left")
    var_df = build_var(
        aligned.select([col for col in aligned.columns if col not in sample_cols]),
        feature_col,
    )
    sample_ids = make_unique([clean_sample_name(col) for col in sample_cols])
    return matrix_to_assay(
        aligned,
        sample_cols,
        sample_ids,
        var_df,
        assay_name,
        layer_name,
        feature_col=feature_col,
        filtered_pairs=filtered_pairs,
    )


__all__ = [
    "apply_fdr_filter",
    "collect_fdr_filtered_pairs",
    "is_long_format",
    "load_long_table",
]
