"""Tests for the dedicated peptide->protein aggregation module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.aggregation import aggregate_to_protein
from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.io import aggregate_to_protein as io_aggregate_to_protein


def _build_container(
    x: np.ndarray | sp.spmatrix,
    protein_col: str = "PG.ProteinGroups",
    protein_ids: list[str | None] | None = None,
    feature_ids: list[str] | None = None,
) -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S1", "S2"]})
    var = pl.DataFrame(
        {
            "_index": feature_ids or ["pep1", "pep2", "pep3"],
            protein_col: protein_ids or ["P1", "P1", "P2"],
        }
    )

    m = np.array(
        [
            [MaskCode.VALID.value, MaskCode.VALID.value, MaskCode.VALID.value],
            [MaskCode.VALID.value, MaskCode.VALID.value, MaskCode.LOD.value],
        ],
        dtype=np.int8,
    )

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)}, feature_id_col="_index")
    return ScpContainer(obs=obs, assays={"peptides": assay}, sample_id_col="_index")


def test_aggregate_to_protein_sum_basic() -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, np.nan]], dtype=np.float64))

    out = aggregate_to_protein(container)
    protein = out.assays["proteins"]

    assert protein.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        protein.layers["raw"].X,
        np.array([[4.0, 5.0], [6.0, np.nan]], dtype=np.float64),
        equal_nan=True,
    )

    m = protein.layers["raw"].get_m()
    assert int(m[1, 1]) == MaskCode.LOD.value

    assert out.links[-1].source_assay == "peptides"
    assert out.links[-1].target_assay == "proteins"
    assert out.links[-1].linkage.height == 3
    assert out.history[-1].action == "aggregate_to_protein"
    assert out.history[-1].params["source_assay"] == "peptides"
    assert out.history[-1].params["target_assay"] == "proteins"
    assert out.history[-1].params["method"] == "sum"


def test_aggregate_to_protein_returns_new_container_but_reuses_source_assay_object() -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))
    peptide_assay = container.assays["peptides"]

    out = aggregate_to_protein(container)

    assert out is not container
    assert out.obs is not container.obs
    assert out.assays["peptides"] is peptide_assay
    assert "proteins" not in container.assays
    assert "proteins" in out.assays


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("max", np.array([[3.0, 5.0], [4.0, np.nan]], dtype=np.float64)),
        ("mean", np.array([[2.0, 5.0], [3.0, np.nan]], dtype=np.float64)),
    ],
)
def test_aggregate_to_protein_alt_methods(method: str, expected: np.ndarray) -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, np.nan]], dtype=np.float64))

    out = aggregate_to_protein(container, method=method)
    np.testing.assert_allclose(out.assays["proteins"].layers["raw"].X, expected, equal_nan=True)


def test_aggregate_to_protein_with_explicit_mapping_column() -> None:
    container = _build_container(
        np.array([[10.0, 2.0, 5.0], [1.0, 7.0, 3.0]], dtype=np.float64),
        protein_col="my_protein_map",
    )

    out = aggregate_to_protein(container, protein_column="my_protein_map")
    assert out.assays["proteins"].var.columns == ["_index", "my_protein_map"]


def test_aggregate_to_protein_drop_unmapped() -> None:
    container = _build_container(
        np.array([[10.0, 2.0, 5.0], [1.0, 7.0, 3.0]], dtype=np.float64),
        protein_ids=["P1", None, "P2"],
    )

    out = aggregate_to_protein(container, keep_unmapped=False)
    assert out.assays["proteins"].var["_index"].to_list() == ["P1", "P2"]
    assert out.links[-1].linkage.height == 2


def test_aggregate_to_protein_default_drops_unmapped_for_protein_level_output() -> None:
    container = _build_container(
        np.array([[10.0, 2.0, 5.0], [1.0, 7.0, 3.0]], dtype=np.float64),
        protein_ids=["P1", None, "P2"],
    )

    out = aggregate_to_protein(container)
    assert out.assays["proteins"].var["_index"].to_list() == ["P1", "P2"]


def test_aggregate_to_protein_keep_unmapped_preserves_each_peptide_separately() -> None:
    container = _build_container(
        np.array([[10.0, 2.0, 5.0], [1.0, 7.0, 3.0]], dtype=np.float64),
        protein_ids=["P1", None, None],
    )

    out = aggregate_to_protein(container, keep_unmapped=True)
    protein = out.assays["proteins"]

    feature_ids = protein.var["_index"].to_list()
    feature_to_col = {feature_id: idx for idx, feature_id in enumerate(feature_ids)}

    assert "P1" in feature_to_col
    unmapped_ids = [
        feature_id for feature_id in feature_ids if feature_id.startswith("__UNMAPPED__--")
    ]
    assert len(unmapped_ids) == 2
    assert not any(any(ch in '<>:"/\\\\|?*' for ch in feature_id) for feature_id in unmapped_ids)

    np.testing.assert_allclose(
        protein.layers["raw"].X[:, feature_to_col["P1"]], np.array([10.0, 1.0])
    )
    np.testing.assert_allclose(
        protein.layers["raw"].X[:, feature_to_col["__UNMAPPED__--NA--pep2"]],
        np.array([2.0, 7.0]),
    )
    np.testing.assert_allclose(
        protein.layers["raw"].X[:, feature_to_col["__UNMAPPED__--NA--pep3"]],
        np.array([5.0, 3.0]),
    )
    assert protein.var["PG.ProteinGroups"].to_list().count(None) == 2


def test_aggregate_to_protein_keep_unmapped_encodes_reserved_characters() -> None:
    container = _build_container(
        np.array([[10.0, 2.0, 5.0], [1.0, 7.0, 3.0]], dtype=np.float64),
        protein_ids=["P1", None, None],
        feature_ids=["pep1", "pep:2", "pep/3"],
    )

    out = aggregate_to_protein(container, keep_unmapped=True, unmapped_label="NA:missing")
    feature_ids = out.assays["proteins"].var["_index"].to_list()

    assert "__UNMAPPED__--NA%3Amissing--pep%3A2" in feature_ids
    assert "__UNMAPPED__--NA%3Amissing--pep%2F3" in feature_ids
    assert not any(any(ch in '<>:"/\\\\|?*' for ch in feature_id) for feature_id in feature_ids)


def test_aggregate_to_protein_existing_target_assay_is_silently_overwritten() -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))
    existing = Assay(
        var=pl.DataFrame({"_index": ["old_protein"]}),
        layers={"raw": ScpMatrix(X=np.array([[9.0], [8.0]], dtype=np.float64), M=None)},
    )
    container.assays["proteins"] = existing

    out = aggregate_to_protein(container, target_assay="proteins")

    assert out.assays["proteins"] is not existing
    assert out.assays["proteins"].var["_index"].to_list() == ["P1", "P2"]


def test_aggregate_to_protein_same_source_and_target_assay_currently_errors() -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))

    with pytest.raises(ValueError, match="Link source_id values not found in assay 'peptides'"):
        aggregate_to_protein(container, source_assay="peptides", target_assay="peptides")


def test_aggregate_to_protein_noncontiguous_mapping_groups_correctly() -> None:
    container = _build_container(
        np.array([[1.0, 10.0, 3.0], [2.0, 20.0, 4.0]], dtype=np.float64),
        protein_ids=["P2", "P1", "P2"],
    )

    out = aggregate_to_protein(container, method="sum")
    protein = out.assays["proteins"]

    assert protein.var["_index"].to_list() == ["P1", "P2"]
    np.testing.assert_allclose(
        protein.layers["raw"].X,
        np.array([[10.0, 4.0], [20.0, 6.0]], dtype=np.float64),
    )
    assert out.links[-1].linkage.height == 3


def test_aggregate_to_protein_sparse_input_supported() -> None:
    x_sparse = sp.csr_matrix(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))
    container = _build_container(x_sparse)

    out = aggregate_to_protein(container)
    np.testing.assert_allclose(
        out.assays["proteins"].layers["raw"].X,
        np.array([[4.0, 5.0], [6.0, 6.0]], dtype=np.float64),
    )
    assert not sp.issparse(out.assays["proteins"].layers["raw"].X)


def test_aggregate_to_protein_median_and_weighted_mean() -> None:
    container = _build_container(np.array([[1.0, 9.0, 5.0], [3.0, 7.0, 1.0]], dtype=np.float64))

    median_out = aggregate_to_protein(container, method="median")
    weighted_out = aggregate_to_protein(container, method="weighted_mean")

    # P1 comes from pep1+pep2
    np.testing.assert_allclose(
        median_out.assays["proteins"].layers["raw"].X[:, 0],
        np.array([5.0, 5.0], dtype=np.float64),
    )

    # weighted mean should stay within min/max peptide intensities for each sample
    weighted_p1 = weighted_out.assays["proteins"].layers["raw"].X[:, 0]
    assert np.all(weighted_p1 >= np.array([1.0, 3.0]))
    assert np.all(weighted_p1 <= np.array([9.0, 7.0]))


def test_aggregate_to_protein_top_n() -> None:
    container = _build_container(np.array([[1.0, 100.0, 5.0], [2.0, 200.0, 6.0]], dtype=np.float64))

    out = aggregate_to_protein(container, method="top_n", top_n=1, top_n_aggregate="mean")
    # For P1, top-1 should keep the most abundant peptide (pep2).
    np.testing.assert_allclose(
        out.assays["proteins"].layers["raw"].X[:, 0],
        np.array([100.0, 200.0], dtype=np.float64),
    )


def test_aggregate_to_protein_maxlfq() -> None:
    container = _build_container(
        np.array([[10.0, 20.0, 5.0], [20.0, 40.0, 10.0]], dtype=np.float64)
    )

    out = aggregate_to_protein(container, method="maxlfq")
    x = out.assays["proteins"].layers["raw"].X

    # Relative fold-change should be preserved (S2 ~= 2 * S1).
    assert x[1, 0] == pytest.approx(2.0 * x[0, 0], rel=1e-6)
    assert x[1, 1] == pytest.approx(2.0 * x[0, 1], rel=1e-6)


def test_aggregate_to_protein_maxlfq_fallback_preserves_intensity_scale() -> None:
    container = _build_container(
        np.array([[5.0, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float64)
    )

    out = aggregate_to_protein(container, method="maxlfq")
    x = out.assays["proteins"].layers["raw"].X

    assert x[0, 0] == pytest.approx(5.0)
    assert np.isnan(x[1, 0])


def test_aggregate_to_protein_tmp() -> None:
    container = _build_container(
        np.array([[10.0, 100.0, 6.0], [20.0, 200.0, 12.0]], dtype=np.float64)
    )

    out = aggregate_to_protein(container, method="tmp")
    x = out.assays["proteins"].layers["raw"].X

    assert np.isfinite(x).all()
    assert x[1, 0] == pytest.approx(2.0 * x[0, 0], rel=1e-6)
    assert x[1, 1] == pytest.approx(2.0 * x[0, 1], rel=1e-6)


def test_aggregate_to_protein_ibaq_with_denominator() -> None:
    container = _build_container(np.array([[10.0, 20.0, 30.0], [5.0, 5.0, 5.0]], dtype=np.float64))

    out = aggregate_to_protein(
        container,
        method="ibaq",
        ibaq_denominator={"P1": 6, "P2": 2},
    )
    x = out.assays["proteins"].layers["raw"].X
    np.testing.assert_allclose(x[:, 0], np.array([5.0, 10.0 / 6.0], dtype=np.float64))
    np.testing.assert_allclose(x[:, 1], np.array([15.0, 2.5], dtype=np.float64))


def test_aggregate_to_protein_ibaq_preserves_all_missing_as_nan() -> None:
    container = _build_container(
        np.array([[10.0, 20.0, 5.0], [5.0, 5.0, np.nan]], dtype=np.float64)
    )

    out = aggregate_to_protein(
        container,
        method="ibaq",
        ibaq_denominator={"P1": 6, "P2": 2},
    )
    x = out.assays["proteins"].layers["raw"].X

    assert np.isnan(x[1, 1])


def test_aggregate_to_protein_ibaq_missing_denominator_error() -> None:
    container = _build_container(np.array([[10.0, 20.0, 30.0], [5.0, 5.0, 5.0]], dtype=np.float64))

    with pytest.raises(ValidationError, match="Missing iBAQ denominator"):
        aggregate_to_protein(container, method="ibaq", ibaq_denominator={"P1": 6})


def test_aggregate_to_protein_missing_mapping_column_error() -> None:
    container = _build_container(
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64),
        protein_col="mapping",
    )

    with pytest.raises(ValidationError, match="No protein mapping column"):
        aggregate_to_protein(container)


def test_io_aggregate_wrapper_matches_core_for_shared_parameters() -> None:
    container = _build_container(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))

    out_core = aggregate_to_protein(
        container,
        source_assay="peptides",
        source_layer="raw",
        target_assay="proteins_core",
        method="top_n",
        top_n=1,
        top_n_aggregate="mean",
        keep_unmapped=True,
    )
    out_io = io_aggregate_to_protein(
        container,
        source_assay="peptides",
        source_layer="raw",
        target_assay="proteins_io",
        method="top_n",
        top_n=1,
        top_n_aggregate="mean",
        keep_unmapped=True,
    )

    np.testing.assert_allclose(
        out_core.assays["proteins_core"].layers["raw"].X,
        out_io.assays["proteins_io"].layers["raw"].X,
    )
    assert out_io.history[-1].params["target_assay"] == "proteins_io"
