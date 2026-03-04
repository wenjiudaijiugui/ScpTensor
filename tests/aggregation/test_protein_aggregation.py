"""Tests for the dedicated peptide->protein aggregation module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.aggregation import aggregate_to_protein
from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix


def _build_container(
    x: np.ndarray | sp.spmatrix,
    protein_col: str = "PG.ProteinGroups",
    protein_ids: list[str | None] | None = None,
) -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S1", "S2"]})
    var = pl.DataFrame(
        {
            "_index": ["pep1", "pep2", "pep3"],
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
        np.array([[4.0, 5.0], [6.0, 0.0]], dtype=np.float64),
        equal_nan=True,
    )

    m = protein.layers["raw"].get_m()
    assert int(m[1, 1]) == MaskCode.LOD.value

    assert out.links[-1].source_assay == "peptides"
    assert out.links[-1].target_assay == "proteins"
    assert out.links[-1].linkage.height == 3
    assert out.history[-1].action == "aggregate_to_protein"


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


def test_aggregate_to_protein_sparse_input_supported() -> None:
    x_sparse = sp.csr_matrix(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64))
    container = _build_container(x_sparse)

    out = aggregate_to_protein(container)
    np.testing.assert_allclose(
        out.assays["proteins"].layers["raw"].X,
        np.array([[4.0, 5.0], [6.0, 6.0]], dtype=np.float64),
    )


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
