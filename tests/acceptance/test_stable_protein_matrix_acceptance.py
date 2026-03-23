"""Acceptance regression for the stable DIA preprocessing mainline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scptensor.core.structures import MaskCode
from scptensor.impute import impute_row_median
from scptensor.io import aggregate_to_protein, load_quant_table
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform


def _write_tsv(tmp_path: Path, name: str, frame: pl.DataFrame) -> Path:
    path = tmp_path / name
    frame.write_csv(path, separator="\t")
    return path


def test_diann_peptide_matrix_to_complete_protein_matrix_acceptance(tmp_path: Path) -> None:
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
            }
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

    raw = container.assays["proteins"].layers["raw"].X
    np.testing.assert_allclose(
        raw,
        np.array(
            [
                [150.0, 120.0],
                [120.0, 135.0],
                [55.0, np.nan],
            ]
        ),
        equal_nan=True,
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
    container = impute_row_median(
        container,
        assay_name="proteins",
        source_layer="norm",
        new_layer_name="imputed",
    )

    assert set(container.assays) == {"peptides", "proteins"}

    proteins = container.assays["proteins"]
    assert {"raw", "log", "norm", "imputed"}.issubset(proteins.layers)

    imputed = proteins.layers["imputed"]
    assert imputed.X.shape == (3, 2)
    assert np.isfinite(imputed.X).all()
    assert imputed.X[2, 1] == pytest.approx(imputed.X[2, 0])

    mask = imputed.get_m()
    assert int(mask[2, 1]) == MaskCode.IMPUTED.value

    assert [entry.action for entry in container.history[-5:]] == [
        "load_quant_table",
        "aggregate_to_protein",
        "log_transform",
        "normalization_median_centering",
        "impute_row_median",
    ]
    assert container.history[-5].params["software"] == "diann"
    assert container.history[-4].params["target_assay"] == "proteins"
    assert container.history[-1].params["new_layer_name"] == "imputed"
