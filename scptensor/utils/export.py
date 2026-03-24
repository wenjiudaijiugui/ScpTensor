"""Export helpers for stable protein-matrix workflows."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from scptensor.core._layer_processing import ensure_dense_matrix, resolve_layer_context
from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import ScpContainer


def _stringify_column_names(values: list[object], *, kind: str) -> list[str]:
    """Return unique string column names for tabular export."""
    names = ["" if value is None else str(value) for value in values]
    if len(names) != len(set(names)):
        raise ValidationError(
            f"{kind} identifiers must remain unique after string conversion for export.",
        )
    return names


def protein_matrix_to_table(
    container: ScpContainer,
    *,
    assay_name: str = "proteins",
    layer: str = "imputed",
    feature_id_column: str | None = None,
) -> pl.DataFrame:
    """Return a protein matrix table with proteins as rows and samples as columns."""
    ctx = resolve_layer_context(container, assay_name, layer)
    sample_columns = _stringify_column_names(
        container.sample_ids.to_list(),
        kind="Sample",
    )
    export_feature_id = feature_id_column or ctx.assay.feature_id_col

    if export_feature_id in sample_columns:
        raise ValidationError(
            f"feature_id_column='{export_feature_id}' conflicts with a sample column name.",
        )

    x_dense = ensure_dense_matrix(ctx.layer.X)
    matrix_table = pl.DataFrame(
        x_dense.T,
        schema=sample_columns,
    ).with_columns([pl.col(name).fill_nan(None) for name in sample_columns])

    return matrix_table.insert_column(
        0,
        pl.Series(export_feature_id, ctx.assay.feature_ids.to_list()),
    )


def write_protein_matrix_bundle(
    container: ScpContainer,
    output_dir: str | Path,
    *,
    assay_name: str = "proteins",
    layer: str = "imputed",
    feature_id_column: str | None = None,
    matrix_filename: str = "protein_matrix.tsv",
    sample_metadata_filename: str = "sample_metadata.tsv",
    feature_metadata_filename: str = "protein_metadata.tsv",
    separator: str = "\t",
) -> dict[str, Path]:
    """Write a stable protein-matrix export bundle to disk."""
    ctx = resolve_layer_context(container, assay_name, layer)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matrix_path = output_path / matrix_filename
    sample_metadata_path = output_path / sample_metadata_filename
    feature_metadata_path = output_path / feature_metadata_filename

    protein_matrix_to_table(
        container,
        assay_name=ctx.resolved_assay_name,
        layer=ctx.layer_name,
        feature_id_column=feature_id_column,
    ).write_csv(matrix_path, separator=separator)
    container.obs.write_csv(sample_metadata_path, separator=separator)
    ctx.assay.var.write_csv(feature_metadata_path, separator=separator)

    return {
        "protein_matrix": matrix_path,
        "sample_metadata": sample_metadata_path,
        "protein_metadata": feature_metadata_path,
    }


__all__ = [
    "protein_matrix_to_table",
    "write_protein_matrix_bundle",
]
