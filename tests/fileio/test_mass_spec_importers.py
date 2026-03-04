"""Tests for unified DIA-NN / Spectronaut importers."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import MaskCode
from scptensor.io import load_diann, load_peptide_pivot, load_quant_table, load_spectronaut


def _write_tsv(tmp_path: Path, name: str, frame: pl.DataFrame) -> Path:
    path = tmp_path / name
    frame.write_csv(path, separator="\t")
    return path


def test_load_diann_protein_matrix(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_protein_matrix.tsv",
        pl.DataFrame(
            {
                "Protein.Group": ["P1", "P2"],
                "Protein.Ids": ["ID1", "ID2"],
                "Genes": ["G1", "G2"],
                "S1.raw": [100.0, 300.0],
                "S2.raw": [200.0, None],
            }
        ),
    )

    container = load_diann(path, assay_name="proteins", level="protein")
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[100.0, 300.0], [200.0, np.nan]]),
        equal_nan=True,
    )

    m = assay.layers["raw"].get_m()
    assert int(m[0, 0]) == MaskCode.VALID.value
    assert int(m[1, 1]) == MaskCode.LOD.value


def test_load_diann_peptide_matrix_bgs_columns(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_peptide_matrix.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2"],
                "Protein.Group": ["P1", "P1"],
                "[1] SampleA.raw.PEP.Quantity": [10.0, None],
                "[2] SampleB.raw.PEP.Quantity": [None, 30.0],
            }
        ),
    )

    container = load_diann(path, assay_name="peptides", level="peptide")
    assay = container.assays["peptides"]

    assert container.obs["_index"].to_list() == ["SampleA", "SampleB"]
    assert assay.var["_index"].to_list() == ["pep1", "pep2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[10.0, np.nan], [np.nan, 30.0]]),
        equal_nan=True,
    )

    m = assay.layers["raw"].get_m()
    assert int(m[0, 1]) == MaskCode.LOD.value
    assert int(m[1, 0]) == MaskCode.LOD.value


def test_load_diann_peptide_default_assay_name_is_peptides(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_peptide_long.tsv",
        pl.DataFrame(
            {
                "Run": ["S1.raw", "S2.raw"],
                "Precursor.Id": ["pep1", "pep1"],
                "Precursor.Normalised": [10.0, 12.0],
                "Q.Value": [0.001, 0.002],
            }
        ),
    )

    container = load_diann(path, level="peptide")
    assert "peptides" in container.assays
    assert "proteins" not in container.assays


def test_load_spectronaut_protein_long_with_fdr_filter(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_protein_long.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P1", "P2", "P2"],
                "R.FileName": ["S1.raw", "S2.raw", "S1.raw", "S2.raw"],
                "PG.Quantity": [100.0, 120.0, 50.0, 60.0],
                "Q.Value": [0.005, 0.009, 0.02, 0.005],
                "PG.Q.Value": [0.005, 0.009, 0.005, 0.02],
            }
        ),
    )

    container = load_spectronaut(
        path,
        assay_name="proteins",
        level="protein",
        table_format="long",
        quantity_column="PG.Quantity",
        fdr_threshold=0.01,
    )
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[100.0, 50.0], [120.0, np.nan]]),
        equal_nan=True,
    )


def test_load_spectronaut_protein_long_with_pg_qvalue_filter(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_protein_long_qvalue.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P1", "P2", "P2"],
                "R.FileName": ["S1.raw", "S2.raw", "S1.raw", "S2.raw"],
                "PG.Quantity": [100.0, 120.0, 50.0, 60.0],
                "PG.Qvalue": [0.005, 0.009, 0.02, 0.02],
            }
        ),
    )

    container = load_spectronaut(
        path,
        assay_name="proteins",
        level="protein",
        table_format="long",
        quantity_column="PG.Quantity",
        fdr_threshold=0.01,
    )
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["P1"]
    np.testing.assert_allclose(assay.layers["raw"].X[:, 0], np.array([100.0, 120.0]))


def test_load_spectronaut_peptide_matrix_quantity_suffix(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_matrix.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2"],
                "PG.ProteinGroups": ["P1", "P2"],
                "S1.raw_Quantity": [1.5, 2.5],
                "S2.raw_Quantity": [None, 3.5],
            }
        ),
    )

    container = load_spectronaut(path, assay_name="peptides", level="peptide")
    assay = container.assays["peptides"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["pep1", "pep2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[1.5, 2.5], [np.nan, 3.5]]),
        equal_nan=True,
    )

    m = assay.layers["raw"].get_m()
    assert int(m[1, 0]) == MaskCode.LOD.value


def test_load_peptide_pivot_with_protein_aggregation(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_for_agg.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2", "pep3"],
                "PG.ProteinGroups": ["P1", "P1", "P2"],
                "S1.raw_Quantity": [1.0, 3.0, 5.0],
                "S2.raw_Quantity": [2.0, 4.0, None],
            }
        ),
    )

    container = load_peptide_pivot(
        path,
        software="spectronaut",
        assay_name="peptides",
        protein_agg=True,
        protein_assay_name="proteins",
        agg_method="sum",
        layer_name="raw",
    )

    assert set(container.assays) == {"peptides", "proteins"}
    assert len(container.links) == 1
    link = container.links[0]
    assert link.source_assay == "peptides"
    assert link.target_assay == "proteins"
    assert link.linkage.height == 3
    assert set(link.linkage["source_id"].to_list()) == {"pep1", "pep2", "pep3"}
    assert set(link.linkage["target_id"].to_list()) == {"P1", "P2"}

    protein_assay = container.assays["proteins"]
    protein_ids = protein_assay.var["_index"].to_list()
    idx_p1 = protein_ids.index("P1")
    idx_p2 = protein_ids.index("P2")

    x = protein_assay.layers["raw"].X
    assert x[0, idx_p1] == pytest.approx(4.0)
    assert x[1, idx_p1] == pytest.approx(6.0)
    assert x[0, idx_p2] == pytest.approx(5.0)
    assert x[1, idx_p2] == pytest.approx(0.0)

    m = protein_assay.layers["raw"].get_m()
    assert int(m[1, idx_p2]) == MaskCode.LOD.value


def test_load_peptide_pivot_with_top_n_aggregation(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_topn.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2", "pep3"],
                "PG.ProteinGroups": ["P1", "P1", "P2"],
                "S1.raw_Quantity": [1.0, 10.0, 5.0],
                "S2.raw_Quantity": [2.0, 20.0, 6.0],
            }
        ),
    )

    container = load_peptide_pivot(
        path,
        software="spectronaut",
        assay_name="peptides",
        protein_agg=True,
        protein_assay_name="proteins",
        agg_method="top_n",
        agg_top_n=1,
        agg_top_n_aggregate="mean",
        layer_name="raw",
    )

    protein_assay = container.assays["proteins"]
    protein_ids = protein_assay.var["_index"].to_list()
    idx_p1 = protein_ids.index("P1")
    idx_p2 = protein_ids.index("P2")
    x = protein_assay.layers["raw"].X

    # P1 should keep the top-1 peptide (pep2).
    assert x[0, idx_p1] == pytest.approx(10.0)
    assert x[1, idx_p1] == pytest.approx(20.0)
    # P2 has one peptide, unaffected.
    assert x[0, idx_p2] == pytest.approx(5.0)
    assert x[1, idx_p2] == pytest.approx(6.0)


def test_load_quant_table_invalid_fdr_threshold(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_long.tsv",
        pl.DataFrame(
            {
                "Run": ["S1.raw"],
                "Protein.Group": ["P1"],
                "PG.MaxLFQ": [100.0],
            }
        ),
    )

    with pytest.raises(ValidationError, match="fdr_threshold must be within"):
        load_quant_table(
            path, software="diann", level="protein", table_format="long", fdr_threshold=1.5
        )


def test_load_quant_table_software_auto_detect_error_message(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "unknown_matrix.tsv",
        pl.DataFrame({"A": ["f1", "f2"], "X": [1.0, 2.0], "Y": [3.0, 4.0]}),
    )

    with pytest.raises(ValidationError, match="Unable to detect software type"):
        load_quant_table(path, software="auto", level="protein")


def test_load_diann_matrix_missing_feature_column_has_clear_error(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_matrix_missing_feature.tsv",
        pl.DataFrame(
            {
                "SomeOtherId": ["P1", "P2"],
                "S1.raw": [100.0, 200.0],
                "S2.raw": [300.0, 400.0],
            }
        ),
    )

    with pytest.raises(ValidationError, match="Unable to auto-detect feature column"):
        load_diann(path, level="protein", table_format="matrix")


def test_load_quant_table_unsupported_extension_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_format.xlsx"
    path.write_text("fake", encoding="utf-8")

    with pytest.raises(ValidationError, match="Unsupported file extension"):
        load_quant_table(path, software="diann", level="protein")
