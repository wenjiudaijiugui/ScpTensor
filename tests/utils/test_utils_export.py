"""Tests for stable protein-matrix export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import LayerNotFoundError, ValidationError
from scptensor.impute import impute_row_median
from scptensor.io import aggregate_to_protein, load_quant_table
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform
from scptensor.utils.export import protein_matrix_to_table, write_protein_matrix_bundle


def _write_tsv(tmp_path: Path, name: str, frame: pl.DataFrame) -> Path:
    path = tmp_path / name
    frame.write_csv(path, separator="\t")
    return path


@pytest.fixture
def processed_protein_container(tmp_path: Path):
    path = _write_tsv(
        tmp_path,
        "diann_peptide_matrix.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2", "pep3", "pep4"],
                "Protein.Group": ["P1", "P1", "P2", "P2"],
                "[1] S1.raw.PEP.Quantity": [100.0, 50.0, 80.0, 40.0],
                "[2] S2.raw.PEP.Quantity": [120.0, None, 90.0, 45.0],
                "[3] S3.raw.PEP.Quantity": [None, 55.0, None, None],
            },
        ),
    )

    container = load_quant_table(
        path,
        software="diann",
        level="peptide",
        table_format="matrix",
        assay_name="peptides",
    )
    container = aggregate_to_protein(
        container,
        source_assay="peptides",
        source_layer="raw",
        target_assay="proteins",
        method="sum",
    )
    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
        base=2.0,
    )
    container = norm_median(
        container,
        assay_name="proteins",
        source_layer="log",
        new_layer_name="norm",
    )
    return impute_row_median(
        container,
        assay_name="proteins",
        source_layer="norm",
        new_layer_name="imputed",
    )


def test_protein_matrix_to_table_returns_feature_by_sample_export_shape(
    processed_protein_container,
) -> None:
    table = protein_matrix_to_table(
        processed_protein_container,
        assay_name="protein",
        layer="imputed",
        feature_id_column="protein_id",
    )

    assert table.columns == ["protein_id", "S1", "S2", "S3"]
    assert table["protein_id"].to_list() == ["P1", "P2"]

    numeric = table.select(["S1", "S2", "S3"]).to_numpy()
    assert numeric.shape == (2, 3)
    assert np.isfinite(numeric).all()


def test_protein_matrix_to_table_rejects_conflicting_feature_id_column(
    processed_protein_container,
) -> None:
    with pytest.raises(ValidationError, match="conflicts with a sample column name"):
        protein_matrix_to_table(
            processed_protein_container,
            assay_name="proteins",
            layer="imputed",
            feature_id_column="S1",
        )


def test_protein_matrix_to_table_raises_on_missing_layer(processed_protein_container) -> None:
    with pytest.raises(LayerNotFoundError, match="Layer 'scaled' not found"):
        protein_matrix_to_table(
            processed_protein_container,
            assay_name="proteins",
            layer="scaled",
        )


def test_write_protein_matrix_bundle_writes_matrix_and_metadata_files(
    processed_protein_container,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "export_bundle"
    paths = write_protein_matrix_bundle(
        processed_protein_container,
        output_dir,
        assay_name="proteins",
        layer="imputed",
        feature_id_column="protein_id",
    )

    assert set(paths) == {"protein_matrix", "sample_metadata", "protein_metadata"}
    assert all(path.exists() for path in paths.values())

    matrix = pl.read_csv(paths["protein_matrix"], separator="\t")
    sample_metadata = pl.read_csv(paths["sample_metadata"], separator="\t")
    protein_metadata = pl.read_csv(paths["protein_metadata"], separator="\t")

    assert matrix.columns == ["protein_id", "S1", "S2", "S3"]
    assert matrix["protein_id"].to_list() == ["P1", "P2"]
    assert sample_metadata["_index"].to_list() == ["S1", "S2", "S3"]
    assert protein_metadata["_index"].to_list() == ["P1", "P2"]
