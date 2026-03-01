"""Spectronaut format I/O for ScpContainer.

Provides functions for loading Spectronaut output files, including:
- Protein Group Matrix (matrix format)
- Long format report (one row per protein-sample pair)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

__all__ = [
    "load_spectronaut",
]


def load_spectronaut(
    path: str | Path,
    *,
    assay_name: str = "proteins",
    quantity_column: str = "auto",
) -> ScpContainer:
    """Import Spectronaut output.

    Supports:
    1. **Matrix format**: Protein group matrix with samples as columns.
    2. **Long format**: One row per protein-sample pair.

    For long format, performs:
    - Filtering: ``Q.Value <= 0.01`` and ``PG.Q.Value <= 0.01``.
    - Aggregation: Pivots quantity column to create the protein matrix.

    Parameters
    ----------
    path : str | Path
        Path to the Spectronaut export file (.tsv, .txt, or .csv).
    assay_name : str, optional
        Name of the assay to create. Default is "proteins".
    quantity_column : str, optional
        Which quantity column to use. Default "auto" tries:
        "PG.MaxLFQ", "PG.Quantity", "PG.Normalised".

    Returns
    -------
    ScpContainer
        Container with:
        - obs: Sample metadata (filenames).
        - assays[assay_name]:
            - X: Intensities (n_samples x n_features).
            - var: Protein metadata.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValidationError
        If file structure is invalid.

    Examples
    --------
    >>> from scptensor.io import load_spectronaut
    >>>
    >>> # Load Spectronaut protein matrix
    >>> container = load_spectronaut("PG_Matrix.tsv")
    >>>
    >>> # Load long format with specific quantity column
    >>> container = load_spectronaut("report_long.tsv", quantity_column="PG.MaxLFQ")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Detect format type
    df_sample = pl.read_csv(
        path,
        separator="\t" if path.suffix in [".tsv", ".txt"] else ",",
        n_rows=10,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        ignore_errors=True,
    )

    if _is_matrix_format(df_sample.columns):
        return _load_spectronaut_matrix(path, assay_name)
    else:
        return _load_spectronaut_long(path, assay_name, quantity_column)


def _is_matrix_format(columns: list[str]) -> bool:
    """Check if columns indicate matrix format."""
    # Matrix format has protein ID column and many sample columns
    # Long format has R.FileName, EG.ProteinId, Q.Value, etc.
    long_format_indicators = {
        "R.FileName",
        "EG.ProteinId",
        "FG.ProteinGroups",
        "Q.Value",
        "PG.Q.Value",
        "PEP.Q.Value",
    }
    return not long_format_indicators.intersection(set(columns))


def _load_spectronaut_matrix(path: Path, assay_name: str) -> ScpContainer:
    """Load Spectronaut protein group matrix format."""
    df = pl.read_csv(
        path,
        separator="\t" if path.suffix in [".tsv", ".txt"] else ",",
        null_values=["", "NA", "NaN", "nan"],
        infer_schema_length=0,
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    # Identify protein ID column
    protein_id_col = None
    for potential in ["PG.ProteinGroups", "PG.ProteinNames", "PG.Genes", "ProteinGroup", "ProteinGroups"]:
        if potential in df.columns:
            protein_id_col = potential
            break

    if protein_id_col is None:
        # Use first column
        protein_id_col = df.columns[0]

    # Metadata columns
    meta_cols_set = {
        "PG.ProteinGroups",
        "PG.ProteinNames",
        "PG.Genes",
        "PG.Q.Value",
        "PG.Count",
        "F.UniquePeptides",
        "PG.Description",
        "ProteinGroup",
        "ProteinGroups",
        "Genes",
    }
    found_meta_cols = [c for c in df.columns if c in meta_cols_set]

    if not found_meta_cols:
        found_meta_cols = [protein_id_col]

    # Extract var
    var = df.select(found_meta_cols)

    # Ensure _index is unique
    if protein_id_col in var.columns and var[protein_id_col].is_duplicated().any():
        var = var.with_row_index(name="_row_num")
        var = var.rename({protein_id_col: "_original_id"})
        var = var.with_columns(
            pl.concat_str([
                pl.col("_original_id").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("_row_num").cast(pl.Utf8),
            ]).alias("_index")
        )
        var = var.drop("_row_num", "_original_id")
    elif protein_id_col in var.columns:
        var = var.rename({protein_id_col: "_index"})
    else:
        var = var.with_row_index("_index")

    # Sample columns (non-metadata)
    all_cols = set(df.columns)
    meta_cols_set_extended = set(found_meta_cols) | {protein_id_col}
    sample_cols = [c for c in df.columns if c not in meta_cols_set_extended]

    if not sample_cols:
        raise ValidationError("No sample columns found in Spectronaut matrix.")

    # Convert to numeric
    x_df = df.select(sample_cols).with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0)
        for c in sample_cols
    ])

    x = x_df.to_numpy().T.astype(np.float64)

    obs = pl.DataFrame({"_index": sample_cols})

    assay = Assay(
        var=var,
        layers={"Quantity": ScpMatrix(X=x)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


def _load_spectronaut_long(
    path: Path,
    assay_name: str,
    quantity_column: str = "auto",
) -> ScpContainer:
    """Load Spectronaut long format report."""
    df = pl.read_csv(
        path,
        separator="\t" if path.suffix in [".tsv", ".txt"] else ",",
        null_values=["", "NA", "NaN", "nan"],
        infer_schema_length=0,
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    # Auto-detect quantity column
    if quantity_column == "auto":
        for col in ["PG.MaxLFQ", "PG.Quantity", "PG.Normalised", "FG.Quantity", "FG.MaxLFQ"]:
            if col in df.columns:
                quantity_column = col
                break
        if quantity_column == "auto":
            raise ValidationError(
                f"No quantity column found in {path}. "
                f"Expected one of: PG.MaxLFQ, PG.Quantity, PG.Normalised, FG.Quantity"
            )

    # Required columns
    if "R.FileName" in df.columns:
        sample_col = "R.FileName"
    elif "R.File.Name" in df.columns:
        sample_col = "R.File.Name"
    else:
        raise ValidationError(
            f"Sample column not found in {path}. "
            f"Expected 'R.FileName' or 'R.File.Name'."
        )

    if "PG.ProteinGroups" in df.columns:
        protein_col = "PG.ProteinGroups"
    elif "FG.ProteinGroups" in df.columns:
        protein_col = "FG.ProteinGroups"
    else:
        raise ValidationError(
            f"Protein column not found in {path}. "
            f"Expected 'PG.ProteinGroups' or 'FG.ProteinGroups'."
        )

    # Convert numeric columns
    for col in [quantity_column, "Q.Value", "PG.Q.Value"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # FDR filter
    filter_expr = []
    if "Q.Value" in df.columns:
        filter_expr.append(pl.col("Q.Value") <= 0.01)
    if "PG.Q.Value" in df.columns:
        filter_expr.append(pl.col("PG.Q.Value") <= 0.01)
    if "PEP.Q.Value" in df.columns:
        filter_expr.append(pl.col("PEP.Q.Value") <= 0.01)

    if filter_expr:
        df_filtered = df.filter(pl.all_horizontal(filter_expr))
    else:
        df_filtered = df

    if df_filtered.is_empty():
        raise ValidationError("No data remains after FDR filtering.")

    # Fill null quantities
    df_filtered = df_filtered.with_columns(
        pl.col(quantity_column).fill_null(0.0)
    )

    # Pivot to matrix
    matrix_df = df_filtered.pivot(
        index=protein_col,
        on=sample_col,
        values=quantity_column,
        aggregate_function="max",
    ).fill_null(0.0)

    # Get metadata columns
    meta_cols_set = {
        "PG.ProteinGroups",
        "PG.ProteinNames",
        "PG.Genes",
        "PG.Description",
        "FG.ProteinGroups",
        "FG.ProteinNames",
        "FG.Genes",
        protein_col,
    }
    found_meta_cols = [c for c in df_filtered.columns if c in meta_cols_set]

    if found_meta_cols:
        var_df = df_filtered.select(found_meta_cols).unique(subset=[protein_col])
        aligned_df = matrix_df.join(var_df, on=protein_col, how="left")
    else:
        aligned_df = matrix_df

    sample_cols = [c for c in matrix_df.columns if c != protein_col]

    obs = pl.DataFrame({"_index": sample_cols})

    if found_meta_cols:
        var = aligned_df.select(found_meta_cols)
        # Handle duplicate IDs
        if var[protein_col].is_duplicated().any():
            var = var.with_row_index("_row_num")
            var = var.rename({protein_col: "_original_id"})
            var = var.with_columns(
                pl.concat_str([
                    pl.col("_original_id").cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col("_row_num").cast(pl.Utf8),
                ]).alias("_index")
            )
            var = var.drop("_row_num", "_original_id")
        else:
            var = var.rename({protein_col: "_index"})
    else:
        var = pl.DataFrame({"_index": aligned_df[protein_col]})

    x = aligned_df.select(sample_cols).to_numpy().T.astype(np.float64)

    layer_name = quantity_column.replace(".", "_")

    assay = Assay(
        var=var,
        layers={layer_name: ScpMatrix(X=x)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )
