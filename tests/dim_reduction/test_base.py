"""Tests for dim-reduction base helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction.base import _prepare_matrix, _validate_assay_layer


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S0", "S1"]})
    var = pl.DataFrame({"_index": ["F0", "F1"]})
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assay = Assay(var=var, layers={"imputed": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_validate_assay_layer_resolves_alias() -> None:
    container = _make_container()
    assay, x = _validate_assay_layer(container, "protein", "imputed")

    assert assay is container.assays["proteins"]
    assert x is container.assays["proteins"].layers["imputed"].X


def test_prepare_matrix_default_casts_to_float64() -> None:
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    out = _prepare_matrix(x)

    assert out.dtype == np.float64
    assert np.array_equal(out, x.astype(np.float64))


def test_prepare_matrix_honors_dtype_for_sparse_input() -> None:
    x_sparse = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    out = _prepare_matrix(x_sparse, dtype=np.dtype(np.float32))

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert np.array_equal(out, x_sparse.toarray().astype(np.float32))
