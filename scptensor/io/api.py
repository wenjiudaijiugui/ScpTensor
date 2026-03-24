"""Public DIA-NN / Spectronaut quant-table import API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ValidationError
from scptensor.io.long_table import apply_fdr_filter, is_long_format, load_long_table
from scptensor.io.matrix_table import load_matrix_table, resolve_matrix_sample_columns
from scptensor.io.profiles import (
    Level,
    ResolvedSoftware,
    Software,
    TableFormat,
    resolve_feature_column,
    resolve_long_columns,
    resolve_profile,
    resolve_software,
    validate_table_format,
)
from scptensor.io.readers import is_vendor_normalized_column, read_table

TopNAggregate = Literal["sum", "mean", "median", "max", "weighted_mean"]


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
    """Load DIA-NN/Spectronaut quant tables into a unified ``ScpContainer``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if level not in ("protein", "peptide"):
        raise ValidationError(f"Unsupported level='{level}'. Use 'protein' or 'peptide'.")
    validate_table_format(table_format)

    if fdr_threshold is not None and not (0 <= fdr_threshold <= 1):
        raise ValidationError(f"fdr_threshold must be within [0, 1], got {fdr_threshold}")

    preview = read_table(path, delimiter=delimiter, n_rows=50)
    if preview.is_empty():
        raise ValidationError(f"Input file is empty: {path}")

    resolved_software: ResolvedSoftware = resolve_software(software, preview.columns)
    profile = resolve_profile(resolved_software, level)
    resolved_assay = assay_name or ("proteins" if level == "protein" else "peptides")
    resolved_feature_column: str | None = None
    resolved_quantity_column: str | None = None
    resolved_sample_column: str | None = None
    used_fdr_column: str | None = None
    vendor_normalized_input = False

    full_df = read_table(path, delimiter=delimiter)
    use_long_format = is_long_format(
        preview.columns,
        profile,
        sample_column=sample_column,
        quantity_column=quantity_column,
        table_format=table_format,
    )

    if use_long_format:
        resolved_long = resolve_long_columns(
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
        vendor_normalized_input = is_vendor_normalized_column(resolved_quantity_column)
        container = load_long_table(
            full_df,
            assay_name=resolved_assay,
            profile=profile,
            resolved_cols=resolved_long,
            fdr_threshold=fdr_threshold,
            layer_name=layer_name,
        )
    else:
        full_df, used_fdr_column = apply_fdr_filter(full_df, profile, fdr_threshold)
        if full_df.is_empty():
            if used_fdr_column is not None:
                raise ValidationError(
                    "No rows remain after FDR filtering. "
                    f"Applied '{used_fdr_column} <= {fdr_threshold}' to matrix-format input.",
                )
            raise ValidationError("No rows remain in matrix-format input after preprocessing.")

        resolved_feature_column = resolve_feature_column(full_df.columns, profile, feature_column)
        sample_cols, sample_ids = resolve_matrix_sample_columns(
            full_df,
            resolved_software,
            level,
            resolved_feature_column,
            profile,
        )
        vendor_normalized_input = any(is_vendor_normalized_column(column) for column in sample_cols)
        container = load_matrix_table(
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
            "format": "long" if use_long_format else "matrix",
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
    quantity_column: str = "auto",
    table_format: TableFormat = "auto",
    fdr_threshold: float | None = 0.01,
    layer_name: str = "raw",
) -> ScpContainer:
    """Load a peptide/precursor pivot matrix without triggering downstream aggregation.

    Protein aggregation is a separate explicit stage via ``aggregate_to_protein``.
    """
    return load_quant_table(
        path,
        software=software,
        level="peptide",
        assay_name=assay_name,
        table_format=table_format,
        quantity_column=quantity_column,
        fdr_threshold=fdr_threshold,
        layer_name=layer_name,
    )


__all__ = [
    "Level",
    "Software",
    "TableFormat",
    "TopNAggregate",
    "load_diann",
    "load_peptide_pivot",
    "load_quant_table",
    "load_spectronaut",
]
