"""Tests for unified DIA-NN / Spectronaut importers."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import MaskCode
from scptensor.io import (
    aggregate_to_protein,
    load_diann,
    load_peptide_pivot,
    load_quant_table,
    load_spectronaut,
)


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
            },
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
    assert int(m[1, 1]) == MaskCode.UNCERTAIN.value


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
            },
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
    assert int(m[0, 1]) == MaskCode.UNCERTAIN.value
    assert int(m[1, 0]) == MaskCode.UNCERTAIN.value


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
            },
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
            },
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
    m = assay.layers["raw"].get_m()
    assert int(m[1, 1]) == MaskCode.FILTERED.value


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
            },
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
            },
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
    assert int(m[1, 0]) == MaskCode.UNCERTAIN.value


def test_load_spectronaut_matrix_prefers_suffix_sample_columns_over_other_numeric_columns(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_matrix_suffix_priority.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P2"],
                "ExtraNumeric": [999.0, 888.0],
                "S1.raw_Quantity": [1.5, 2.5],
                "S2.raw_Quantity": [None, 3.5],
            },
        ),
    )

    container = load_spectronaut(
        path,
        assay_name="proteins",
        level="protein",
        table_format="matrix",
    )
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[1.5, 2.5], [np.nan, 3.5]]),
        equal_nan=True,
    )


def test_load_spectronaut_real_style_protein_matrix_prefers_pg_quantity_over_other_runwise_metrics(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_real_style_protein_matrix.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P2"],
                "[1] SampleA.raw.PG.IsSingleHit": [1, 0],
                "[2] SampleB.raw.PG.IsSingleHit": [0, 1],
                "[1] SampleA.raw.PG.Quantity": [10.0, 20.0],
                "[2] SampleB.raw.PG.Quantity": [11.0, None],
                "[1] SampleA.raw.PG.Log2Quantity": [3.3, 4.3],
                "[2] SampleB.raw.PG.Log2Quantity": [3.5, None],
            },
        ),
    )

    container = load_spectronaut(path, assay_name="proteins", level="protein")
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["SampleA", "SampleB"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[10.0, 20.0], [11.0, np.nan]]),
        equal_nan=True,
    )


def test_load_spectronaut_real_style_peptide_matrix_prefers_pep_quantity_over_ms1_ms2_metrics(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_real_style_peptide_matrix.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2"],
                "PG.ProteinGroups": ["P1", "P2"],
                "[1] SampleA.raw.PEP.Quantity": [1.0, 3.0],
                "[2] SampleB.raw.PEP.Quantity": [2.0, None],
                "[1] SampleA.raw.PEP.MS1Quantity": [10.0, 30.0],
                "[2] SampleB.raw.PEP.MS1Quantity": [20.0, None],
                "[1] SampleA.raw.EG.ApexRT": [5.0, 6.0],
                "[2] SampleB.raw.EG.ApexRT": [5.5, None],
            },
        ),
    )

    container = load_peptide_pivot(path, software="spectronaut", assay_name="peptides")
    assay = container.assays["peptides"]

    assert container.obs["_index"].to_list() == ["SampleA", "SampleB"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[1.0, 3.0], [2.0, np.nan]]),
        equal_nan=True,
    )


def test_load_spectronaut_matrix_honors_explicit_quantity_column_for_ms1_matrix(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_real_style_ms1_matrix.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2"],
                "PG.ProteinGroups": ["P1", "P2"],
                "[1] SampleA.raw.PEP.Quantity": [1.0, 3.0],
                "[2] SampleB.raw.PEP.Quantity": [2.0, None],
                "[1] SampleA.raw.PEP.MS1Quantity": [10.0, 30.0],
                "[2] SampleB.raw.PEP.MS1Quantity": [20.0, None],
            },
        ),
    )

    container = load_spectronaut(
        path,
        assay_name="peptides",
        level="peptide",
        quantity_column="PEP.MS1Quantity",
    )
    assay = container.assays["peptides"]

    assert container.obs["_index"].to_list() == ["SampleA", "SampleB"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[10.0, 30.0], [20.0, np.nan]]),
        equal_nan=True,
    )


def test_load_peptide_pivot_keeps_io_and_aggregation_as_separate_stages(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_for_agg.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2", "pep3"],
                "PG.ProteinGroups": ["P1", "P1", "P2"],
                "S1.raw_Quantity": [1.0, 3.0, 5.0],
                "S2.raw_Quantity": [2.0, 4.0, None],
            },
        ),
    )

    peptide_container = load_peptide_pivot(
        path,
        software="spectronaut",
        assay_name="peptides",
        layer_name="raw",
    )

    assert set(peptide_container.assays) == {"peptides"}
    assert peptide_container.links == []

    container = aggregate_to_protein(
        peptide_container,
        source_assay="peptides",
        target_assay="proteins",
        method="sum",
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
    assert np.isnan(x[1, idx_p2])

    m = protein_assay.layers["raw"].get_m()
    assert int(m[1, idx_p2]) == MaskCode.UNCERTAIN.value


def test_load_peptide_pivot_then_aggregate_to_protein_top_n(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_topn.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1", "pep2", "pep3"],
                "PG.ProteinGroups": ["P1", "P1", "P2"],
                "S1.raw_Quantity": [1.0, 10.0, 5.0],
                "S2.raw_Quantity": [2.0, 20.0, 6.0],
            },
        ),
    )

    peptide_container = load_peptide_pivot(
        path,
        software="spectronaut",
        assay_name="peptides",
        layer_name="raw",
    )

    container = aggregate_to_protein(
        peptide_container,
        source_assay="peptides",
        target_assay="proteins",
        method="top_n",
        top_n=1,
        top_n_aggregate="mean",
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


def test_load_peptide_pivot_no_longer_accepts_cross_stage_aggregation_flag(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_peptide_no_cross_stage.tsv",
        pl.DataFrame(
            {
                "EG.PrecursorId": ["pep1"],
                "PG.ProteinGroups": ["P1"],
                "S1.raw_Quantity": [1.0],
            },
        ),
    )

    with pytest.raises(TypeError, match="protein_agg"):
        load_peptide_pivot(  # type: ignore[call-arg]
            path,
            software="spectronaut",
            protein_agg=True,
        )


def test_load_quant_table_diann_matrix_success_path(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_quant_table_matrix.tsv",
        pl.DataFrame(
            {
                "Protein.Group": ["P1", "P2"],
                "S1.raw": [100.0, 300.0],
                "S2.raw": [200.0, None],
            },
        ),
    )

    container = load_quant_table(
        path,
        software="diann",
        level="protein",
        table_format="matrix",
        assay_name="proteins",
        layer_name="raw",
    )
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[100.0, 300.0], [200.0, np.nan]]),
        equal_nan=True,
    )
    assert container.history[-1].action == "load_quant_table"
    assert container.history[-1].params["software"] == "diann"


def test_load_quant_table_rejects_tmt_channelized_matrix_input(tmp_path: Path) -> None:
    channels = [
        "126",
        "127N",
        "127C",
        "128N",
        "128C",
        "129N",
        "129C",
        "130N",
        "130C",
        "131N",
        "131C",
        "132N",
        "132C",
        "133N",
        "133C",
        "134N",
    ]
    matrix = {"Protein.Group": ["P1", "P2"]}
    for channel in channels:
        matrix[f"F1_{channel}"] = [100.0, 200.0]

    path = _write_tsv(
        tmp_path,
        "diann_tmt_like_matrix.tsv",
        pl.DataFrame(matrix),
    )

    with pytest.raises(ValidationError, match="TMT-like channelized sample IDs"):
        load_quant_table(
            path,
            software="diann",
            level="protein",
            table_format="matrix",
            assay_name="proteins",
        )


def test_load_quant_table_auto_detect_spectronaut_long_success(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_quant_table_long.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P1", "P2", "P2"],
                "R.FileName": ["S1.raw", "S2.raw", "S1.raw", "S2.raw"],
                "PG.Quantity": [100.0, 120.0, 50.0, 60.0],
                "PG.Q.Value": [0.005, 0.009, 0.005, 0.02],
            },
        ),
    )

    container = load_quant_table(
        path,
        software="auto",
        level="protein",
        table_format="long",
        quantity_column="PG.Quantity",
        fdr_threshold=0.01,
        assay_name="proteins",
    )
    assay = container.assays["proteins"]

    assert container.obs["_index"].to_list() == ["S1", "S2"]
    assert assay.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[100.0, 50.0], [120.0, np.nan]]),
        equal_nan=True,
    )
    assert container.history[-1].params["software"] == "spectronaut"
    assert container.history[-1].params["format"] == "long"


def test_load_quant_table_invalid_fdr_threshold(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_long.tsv",
        pl.DataFrame(
            {
                "Run": ["S1.raw"],
                "Protein.Group": ["P1"],
                "PG.MaxLFQ": [100.0],
            },
        ),
    )

    with pytest.raises(ValidationError, match="fdr_threshold must be within"):
        load_quant_table(
            path,
            software="diann",
            level="protein",
            table_format="long",
            fdr_threshold=1.5,
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
            },
        ),
    )

    with pytest.raises(ValidationError, match="Unable to auto-detect feature column"):
        load_diann(path, level="protein", table_format="matrix")


def test_load_quant_table_unsupported_extension_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_format.xlsx"
    path.write_text("fake", encoding="utf-8")

    with pytest.raises(ValidationError, match="Unsupported file extension"):
        load_quant_table(path, software="diann", level="protein")


def test_load_quant_table_invalid_table_format_raises_error(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_matrix.tsv",
        pl.DataFrame(
            {
                "Protein.Group": ["P1"],
                "S1.raw": [100.0],
            },
        ),
    )

    with pytest.raises(ValidationError, match="Unsupported table_format"):
        load_quant_table(path, software="diann", level="protein", table_format="bad")


def test_load_quant_table_parquet_matrix_auto_detect(tmp_path: Path) -> None:
    path = tmp_path / "diann_protein_matrix.parquet"
    pl.DataFrame(
        {
            "Protein.Group": ["P1", "P2"],
            "S1.raw": [100.0, 300.0],
            "S2.raw": [200.0, None],
        },
    ).write_parquet(path)

    container = load_diann(path, level="protein", table_format="auto")
    assay = container.assays["proteins"]

    assert container.history[-1].params["format"] == "matrix"
    np.testing.assert_allclose(
        assay.layers["raw"].X,
        np.array([[100.0, 300.0], [200.0, np.nan]]),
        equal_nan=True,
    )


def test_load_quant_table_matrix_applies_fdr_filter(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_matrix_fdr.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P2"],
                "PG.Qvalue": [0.005, 0.02],
                "S1.raw_Quantity": [10.0, 20.0],
                "S2.raw_Quantity": [11.0, 21.0],
            },
        ),
    )

    container = load_spectronaut(path, level="protein", table_format="matrix", fdr_threshold=0.01)
    assay = container.assays["proteins"]

    assert assay.var["_index"].to_list() == ["P1"]
    np.testing.assert_allclose(assay.layers["raw"].X[:, 0], np.array([10.0, 11.0]))


def test_load_diann_protein_default_assay_works_with_protein_alias_downstream(
    tmp_path: Path,
) -> None:
    path = _write_tsv(
        tmp_path,
        "diann_protein_long.tsv",
        pl.DataFrame(
            {
                "Run": ["S1.raw", "S2.raw"],
                "Protein.Group": ["P1", "P1"],
                "PG.MaxLFQ": [100.0, 120.0],
                "PG.Q.Value": [0.001, 0.002],
            },
        ),
    )

    container = load_diann(path, level="protein")
    from scptensor.normalization import norm_median

    result = norm_median(container, assay_name="protein", source_layer="raw", new_layer_name="norm")
    assert "norm" in result.assays["proteins"].layers


def test_load_quant_table_logs_vendor_normalized_input(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_vendor_normalized.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P2"],
                "PG.Normalized": [10.0, 20.0],
                "R.FileName": ["S1.raw", "S1.raw"],
                "PG.Qvalue": [0.001, 0.001],
            },
        ),
    )

    container = load_spectronaut(path, level="protein", table_format="long")
    log = container.history[-1]

    assert log.params["resolved_quantity_column"] == "PG.Normalized"
    assert log.params["input_quantity_is_vendor_normalized"] is True


def test_normalization_warns_on_vendor_normalized_raw_input(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "spectronaut_vendor_normalized_warning.tsv",
        pl.DataFrame(
            {
                "PG.ProteinGroups": ["P1", "P2"],
                "PG.Normalized": [10.0, 20.0],
                "R.FileName": ["S1.raw", "S1.raw"],
                "PG.Qvalue": [0.001, 0.001],
            },
        ),
    )

    container = load_spectronaut(path, level="protein", table_format="long")

    from scptensor.normalization import norm_median

    with pytest.warns(UserWarning, match="vendor-normalized intensities"):
        norm_median(container, assay_name="protein", source_layer="raw", new_layer_name="norm")
