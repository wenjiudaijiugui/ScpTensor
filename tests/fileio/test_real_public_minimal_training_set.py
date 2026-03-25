"""Real-data regression tests for the curated public minimal training set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scptensor.io import (
    aggregate_to_protein,
    load_diann,
    load_peptide_pivot,
    load_quant_table,
    load_spectronaut,
)
from scptensor.normalization import norm_median, norm_quantile, norm_trqn
from scptensor.transformation.log_transform import log_transform

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MINIMAL_DIR = Path("data/public_minimal_training_set")
DIANN_LONG = MINIMAL_DIR / "01_diann_long_report.tsv"
SPECTRONAUT_PROTEIN = MINIMAL_DIR / "02_spectronaut_protein_matrix.tsv"
SPECTRONAUT_PEPTIDE = MINIMAL_DIR / "03_spectronaut_peptide_matrix.tsv"


def _require_real_training_set() -> None:
    required = [DIANN_LONG, SPECTRONAUT_PROTEIN, SPECTRONAUT_PEPTIDE]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(
            "Real public minimal training set is not available locally. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def test_real_diann_long_report_supports_both_protein_and_peptide_import() -> None:
    _require_real_training_set()

    proteins = load_diann(DIANN_LONG, level="protein", assay_name="proteins")
    peptides = load_diann(DIANN_LONG, level="peptide", assay_name="peptides")

    assert proteins.obs.shape[0] == 24
    assert proteins.assays["proteins"].layers["raw"].X.shape == (24, 2719)
    assert peptides.obs.shape[0] == 24
    assert peptides.assays["peptides"].layers["raw"].X.shape[0] == 24
    assert (
        peptides.assays["peptides"].layers["raw"].X.shape[1]
        > proteins.assays["proteins"].layers["raw"].X.shape[1]
    )


def test_real_spectronaut_protein_matrix_auto_and_explicit_import_agree() -> None:
    _require_real_training_set()

    explicit = load_spectronaut(SPECTRONAUT_PROTEIN, level="protein", assay_name="proteins")
    auto = load_quant_table(
        SPECTRONAUT_PROTEIN,
        software="auto",
        level="protein",
        assay_name="proteins",
    )

    explicit_assay = explicit.assays["proteins"]
    auto_assay = auto.assays["proteins"]

    assert explicit.obs["_index"].to_list() == [
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_1",
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_2",
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_3",
    ]
    assert explicit.obs["_index"].to_list() == auto.obs["_index"].to_list()
    assert explicit_assay.layers["raw"].X.shape == (3, 5683)
    assert auto_assay.layers["raw"].X.shape == (3, 5683)
    np.testing.assert_allclose(
        explicit_assay.layers["raw"].X, auto_assay.layers["raw"].X, equal_nan=True
    )


def test_real_spectronaut_peptide_pivot_aggregates_to_compact_protein_matrix() -> None:
    _require_real_training_set()

    peptides = load_peptide_pivot(
        SPECTRONAUT_PEPTIDE,
        software="spectronaut",
        assay_name="peptides",
    )
    proteins = aggregate_to_protein(
        peptides,
        source_assay="peptides",
        target_assay="proteins",
        method="sum",
    )

    peptide_assay = peptides.assays["peptides"]
    protein_assay = proteins.assays["proteins"]

    assert peptides.obs["_index"].to_list() == [
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_1",
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_2",
        "20231123_DIA_240k_20Th_40ms_FAIMSCV-48_gas3p8_250pgHeLa_CV-48_3",
    ]
    assert peptide_assay.layers["raw"].X.shape == (3, 35622)
    assert protein_assay.layers["raw"].X.shape == (3, 5683)
    assert proteins.links[-1].source_assay == "peptides"
    assert proteins.links[-1].target_assay == "proteins"


@pytest.mark.parametrize(
    ("loader", "path", "expected_shape"),
    [
        ("diann", DIANN_LONG, (24, 2719)),
        ("spectronaut", SPECTRONAUT_PROTEIN, (3, 5683)),
    ],
)
def test_real_protein_matrices_support_log_and_normalization_smoke(
    loader: str,
    path: Path,
    expected_shape: tuple[int, int],
) -> None:
    _require_real_training_set()

    if loader == "diann":
        container = load_diann(path, level="protein", assay_name="proteins")
    else:
        container = load_spectronaut(path, level="protein", assay_name="proteins")

    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log2",
    )
    container = norm_median(
        container,
        assay_name="proteins",
        source_layer="log2",
        new_layer_name="median_norm",
    )
    container = norm_quantile(
        container,
        assay_name="proteins",
        source_layer="log2",
        new_layer_name="quantile_norm",
    )
    container = norm_trqn(
        container,
        assay_name="proteins",
        source_layer="log2",
        new_layer_name="trqn_norm",
    )

    assay = container.assays["proteins"]
    assert assay.layers["log2"].X.shape == expected_shape
    assert assay.layers["median_norm"].X.shape == expected_shape
    assert assay.layers["quantile_norm"].X.shape == expected_shape
    assert assay.layers["trqn_norm"].X.shape == expected_shape
