"""Tests for cluster base helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from scptensor.cluster.base import _prepare_matrix, _validate_assay_layer
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S0", "S1"]})
    var = pl.DataFrame({"_index": ["F0", "F1"]})
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assay = Assay(var=var, layers={"X": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_validate_assay_layer_resolves_alias() -> None:
    container = _make_container()
    assay, x = _validate_assay_layer(container, "protein", "X")

    assert assay is container.assays["proteins"]
    assert x is container.assays["proteins"].layers["X"].X


def test_prepare_matrix_dense_passthrough_dtype() -> None:
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    out = _prepare_matrix(x)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert np.array_equal(out, x)


def test_prepare_matrix_sparse_to_dense() -> None:
    x_sparse = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    out = _prepare_matrix(x_sparse)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert np.array_equal(out, x_sparse.toarray())
